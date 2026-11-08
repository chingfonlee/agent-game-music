"""Remote inference & smoke test for kanaworksai-hue/audio-studio-mcp on Colab L4."""
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
SETUP_TIMINGS = json.loads((ROOT / 'setup_timings.json').read_text(encoding='utf-8')) if (ROOT / 'setup_timings.json').exists() else {}


def update_progress(current: str):
    entries = [{'name': f.name, 'size': f.stat().st_size} for f in sorted(PACKS.glob('*.zip'))]
    tmp = ROOT / 'progress.tmp.json'
    tmp.write_text(json.dumps({
        'current': current,
        'packs': entries,
        'time': time.time(),
    }), encoding='utf-8')
    tmp.replace(ROOT / 'progress.json')


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
        output_file = OUT / f"{item['id']}.wav"
        if (PACKS / pack_name).exists() and output_file.exists():
            continue

        update_progress(f"generating {item['id']}")
        print(f"GENERATING {item['id']}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        extra_meta = {
            'model_load_wall_seconds': load_wall_seconds,
            'model_load_vram_allocated_bytes': load_vram_allocated,
        }

        if item['type'] == 'generate':
            prompt_used, dur = prompts.prepare_prompt(
                item.get('prompt', ''),
                item['audio_type'],
                item['duration_seconds'],
                bpm=item.get('bpm'),
                sfx_fields=tuple(item.get('sfx_fields', ())),
            )
            req = GenerationRequest(
                prompt=prompt_used,
                seconds=float(dur),
                steps=8,
                cfg=1.0,
            )
            res = backend._generate([int(item['seed'])], req)
            audio = res.audios[0]
            post.write(output_file, audio, backend.sample_rate, 'wav')
            extra_meta['prompt_used'] = prompt_used
            extra_meta['audio_stats'] = post.stats(audio)

        elif item['type'] == 'process':
            src_path = OUT / f"{item['source_id']}.wav"
            audio, sr = post.read(src_path)
            audio, lufs_info = post.ffmpeg_process(
                audio,
                sr,
                int(item.get('out_sr', 48000)),
                target_lufs=float(item.get('target_lufs', -16.0)),
                true_peak_db=-1.0,
            )
            audio, _ = post.peak_ceiling(audio, -1.0)
            audio = post.make_loop(
                audio,
                int(item.get('out_sr', 48000)),
                float(item.get('loop_crossfade_ms', 200.0)),
            )
            post.write(output_file, audio, int(item.get('out_sr', 48000)), 'wav')
            extra_meta['loudnorm_info'] = lufs_info
            extra_meta['audio_stats'] = post.stats(audio)

        elif item['type'] == 'edit_continue':
            src_path = OUT / f"{item['source_id']}.wav"
            src_audio, src_sr = post.read(src_path)
            in_sec = src_audio.shape[-1] / src_sr
            target_sec = float(item['duration_seconds'])
            prompt_used, _ = prompts.prepare_prompt(
                item['prompt'],
                item['audio_type'],
                target_sec,
            )
            req = GenerationRequest(
                prompt=prompt_used,
                seconds=target_sec,
                steps=8,
                cfg=1.0,
                inpaint_audio=str(src_path),
                inpaint_range=(max(0.0, in_sec - 1.0), target_sec),
            )
            res = backend._generate([int(item['seed'])], req)
            audio = res.audios[0]
            post.write(output_file, audio, backend.sample_rate, 'wav')
            extra_meta['prompt_used'] = prompt_used
            extra_meta['inpaint_range'] = [max(0.0, in_sec - 1.0), target_sec]
            extra_meta['audio_stats'] = post.stats(audio)

        else:
            raise ValueError(f"Unknown item type: {item['type']}")

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
