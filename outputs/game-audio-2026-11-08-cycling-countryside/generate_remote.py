"""Remote inference & post-processing template for kanaworksai-hue/audio-studio-mcp on Colab L4."""
import gc
import hashlib
import json
from pathlib import Path
import time
import zipfile

import torch
import soundfile as sf

from audio_studio_mcp.backends import create_backend, GenerationRequest
from audio_studio_mcp import prompts, post

ROOT = Path('/content/colab_job')
OUT = ROOT / 'results'
PACKS = ROOT / 'packs'
OUT.mkdir(parents=True, exist_ok=True)
PACKS.mkdir(parents=True, exist_ok=True)

ITEMS = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
SETUP_TIMINGS = (
    json.loads((ROOT / 'setup_timings.json').read_text(encoding='utf-8'))
    if (ROOT / 'setup_timings.json').exists()
    else {}
)


def update_progress(current: str):
    entries = [{'name': f.name, 'size': f.stat().st_size} for f in sorted(PACKS.glob('*.zip'))]
    tmp = ROOT / 'progress.tmp.json'
    tmp.write_text(json.dumps({
        'current': current,
        'packs': entries,
        'time': time.time(),
    }), encoding='utf-8')
    tmp.replace(ROOT / 'progress.json')


def apply_post_options(audio, sr: int, post_cfg: dict) -> tuple:
    """Apply optional post-processing (trim_silence, mono, resample, EBU R128 LUFS / peak normalize, loop, fade)."""
    info = {}
    if post_cfg.get('trim_silence'):
        audio = post.trim_silence(
            audio,
            sr,
            threshold_db=float(post_cfg.get('silence_threshold_db', -50.0)),
            pad_ms=float(post_cfg.get('silence_pad_ms', 10.0)),
        )
    if post_cfg.get('mono'):
        audio = post.to_mono(audio)

    out_sr = int(post_cfg.get('out_sr', sr))
    target_lufs = post_cfg.get('target_lufs')
    peak_ceiling_db = float(post_cfg.get('peak_ceiling_db', -1.0))

    if out_sr != sr or target_lufs is not None:
        audio, lufs_info = post.ffmpeg_process(
            audio,
            sr,
            out_sr,
            target_lufs=float(target_lufs) if target_lufs is not None else None,
            true_peak_db=peak_ceiling_db,
        )
        sr = out_sr
        info.update(lufs_info)

    if post_cfg.get('normalize') == 'peak':
        audio, gain_db = post.peak_normalize(audio, peak_ceiling_db)
        info['peak_normalize_gain_db'] = round(gain_db, 2)
    else:
        audio, gain_db = post.peak_ceiling(audio, peak_ceiling_db)
        if gain_db != 0.0:
            info['peak_ceiling_gain_db'] = round(gain_db, 2)

    if post_cfg.get('loop_crossfade_ms'):
        audio = post.make_loop(audio, sr, float(post_cfg['loop_crossfade_ms']))

    fade_in = float(post_cfg.get('fade_in_ms', 0.0))
    fade_out = float(post_cfg.get('fade_out_ms', 0.0))
    if fade_in > 0 or fade_out > 0:
        audio = post.fade(audio, sr, fade_in, fade_out)

    return audio, sr, info


def resolve_source_file(source_id: str) -> Path:
    for ext in ('.wav', '.ogg', '.flac'):
        cand = OUT / f"{source_id}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Source audio for {source_id!r} not found in {OUT}")


def save_artifact_pack(item: dict, output_file: Path, wall_seconds: float, extra_meta: dict | None = None):
    """Package output file + SHA-256 metadata sidecar into an atomic ZIP archive."""
    pack_name = f"{item['id']}.zip"
    file_sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    info = sf.info(str(output_file))
    meta = {
        'item': item,
        'pack_name': pack_name,
        'file_size_bytes': output_file.stat().st_size,
        'audio_duration_seconds': round(info.duration, 2),
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'subtype': info.subtype,
        'generation_wall_seconds': round(wall_seconds, 2),
        'gpu': torch.cuda.get_device_name(),
        'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'peak_gpu_reserved_bytes': int(torch.cuda.max_memory_reserved()),
        'setup_timings': SETUP_TIMINGS,
        'files': {
            output_file.name: file_sha256,
        },
        **(extra_meta or {}),
    }
    sidecar = output_file.with_suffix(output_file.suffix + '.json')
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    temp_zip = PACKS / (pack_name + '.tmp')
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in (output_file, sidecar):
            z.write(f, f.relative_to(OUT))
    temp_zip.replace(PACKS / pack_name)

    update_progress(f"saved {item['id']}")
    print(
        f"SAVED {item['id']} size={meta['file_size_bytes']/1024/1024:.2f}MiB "
        f"generation={wall_seconds:.1f}s peak={meta['peak_gpu_allocated_bytes']/1024**3:.2f}GiB",
        flush=True,
    )


def main():
    update_progress('loading_model')
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.monotonic()

    backend = create_backend('sa3-torch')
    # Disable flex_attention torch.compile autotune on L4 so SDPA runs immediately without 60s compile warmup
    from stable_audio_3.models import transformer
    transformer.flex_attention_compiled = None

    backend._load()
    load_wall_seconds = round(time.monotonic() - t_load0, 2)
    load_vram_allocated = int(torch.cuda.max_memory_allocated())
    load_vram_reserved = int(torch.cuda.max_memory_reserved())
    print(
        f"MODEL_LOADED wall={load_wall_seconds}s "
        f"vram_alloc={load_vram_allocated/1024**3:.2f}GiB "
        f"vram_res={load_vram_reserved/1024**3:.2f}GiB",
        flush=True,
    )

    for item in ITEMS:
        if (ROOT / 'STOP').exists():
            return
        pack_name = f"{item['id']}.zip"
        if (PACKS / pack_name).exists():
            continue

        update_progress(f"generating {item['id']}")
        print(f"GENERATING {item['id']}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        extra_meta = {
            'model_load_wall_seconds': load_wall_seconds,
            'model_load_vram_allocated_bytes': load_vram_allocated,
        }

        post_cfg = item.get('post') or {}
        fmt = post_cfg.get('format', item.get('format', 'wav')).lower()
        ext = '.wav' if fmt in ('wav', 'wav16') else f'.{fmt}'
        output_file = OUT / f"{item['id']}{ext}"

        steps = int(item.get('steps', 8))
        cfg = float(item.get('cfg', 1.0))
        neg_prompt = item.get('negative_prompt', '')
        seed = int(item.get('seed', 42))

        if item['type'] == 'generate':
            prompt_used, dur = prompts.prepare_prompt(
                item.get('prompt', ''),
                item.get('audio_type', 'music'),
                item.get('duration_seconds'),
                bpm=item.get('bpm'),
                sfx_fields=tuple(item.get('sfx_fields', ())),
            )
            req = GenerationRequest(
                prompt=prompt_used,
                seconds=float(dur or prompts.DEFAULT_SECONDS.get(item.get('audio_type', 'music'), 30.0)),
                negative_prompt=neg_prompt,
                steps=steps,
                cfg=cfg,
            )
            res = backend._generate([seed], req)
            audio, sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used

        elif item['type'] == 'process':
            src_path = resolve_source_file(item['source_id'])
            audio, sr = post.read(src_path)
            post_cfg = {**item, **post_cfg}

        elif item['type'] == 'edit_continue':
            src_path = resolve_source_file(item['source_id'])
            src_audio, src_sr = post.read(src_path)
            end_sec = post.audible_end(src_audio, src_sr)
            if end_sec <= 0.1:
                end_sec = src_audio.shape[-1] / src_sr
            target_sec = float(item['duration_seconds'])
            prompt_used, _ = prompts.prepare_prompt(
                item['prompt'],
                item.get('audio_type', 'music'),
                target_sec,
                bpm=item.get('bpm'),
            )
            req = GenerationRequest(
                prompt=prompt_used,
                seconds=target_sec,
                negative_prompt=neg_prompt,
                steps=steps,
                cfg=cfg,
                inpaint_audio=str(src_path),
                inpaint_range=(max(0.0, end_sec - 1.0), target_sec),
            )
            res = backend._generate([seed], req)
            audio, sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used
            extra_meta['inpaint_range'] = [round(max(0.0, end_sec - 1.0), 2), target_sec]

        elif item['type'] == 'edit_inpaint':
            src_path = resolve_source_file(item['source_id'])
            src_audio, src_sr = post.read(src_path)
            total_sec = float(item.get('duration_seconds', src_audio.shape[-1] / src_sr))
            start_s = float(item['start_seconds'])
            end_s = float(item['end_seconds'])
            prompt_used, _ = prompts.prepare_prompt(
                item['prompt'],
                item.get('audio_type', 'music'),
                total_sec,
                bpm=item.get('bpm'),
            )
            req = GenerationRequest(
                prompt=prompt_used,
                seconds=total_sec,
                negative_prompt=neg_prompt,
                steps=steps,
                cfg=cfg,
                inpaint_audio=str(src_path),
                inpaint_range=(start_s, end_s),
            )
            res = backend._generate([seed], req)
            audio, sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used
            extra_meta['inpaint_range'] = [start_s, end_s]

        elif item['type'] == 'edit_variation':
            src_path = resolve_source_file(item['source_id'])
            src_audio, src_sr = post.read(src_path)
            total_sec = float(item.get('duration_seconds', src_audio.shape[-1] / src_sr))
            strength = float(item.get('strength', 0.5))
            prompt_used, _ = prompts.prepare_prompt(
                item['prompt'],
                item.get('audio_type', 'music'),
                total_sec,
                bpm=item.get('bpm'),
            )
            req = GenerationRequest(
                prompt=prompt_used,
                seconds=total_sec,
                negative_prompt=neg_prompt,
                steps=steps,
                cfg=cfg,
                init_audio=str(src_path),
                init_strength=strength,
            )
            res = backend._generate([seed], req)
            audio, sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used
            extra_meta['init_strength'] = strength

        else:
            raise ValueError(f"Unknown item type: {item['type']}")

        if post_cfg:
            audio, sr, post_info = apply_post_options(audio, sr, post_cfg)
            extra_meta['post_info'] = post_info

        post.write(output_file, audio, sr, fmt)
        extra_meta['audio_stats'] = post.stats(audio)

        elapsed = time.monotonic() - t0
        save_artifact_pack(item, output_file, elapsed, extra_meta)

        gc.collect()
        torch.cuda.empty_cache()

    backend._unload()
    gc.collect()
    torch.cuda.empty_cache()
    print('MODEL_DONE', flush=True)


if __name__ == '__main__':
    main()
