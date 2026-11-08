"""Generate full-length tracks with MiniMax-Music3 on Colab L4 + optional HDemucs vocal stem removal and direct MP3 export (v2.3)."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

import numpy as np
import soundfile as sf
import torch

# L4 (Ada Lovelace) Tensor Core TF32 optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path('/content/funky_house_mp3')
OUT = ROOT / 'results'
PACKS = ROOT / 'packs'
OUT.mkdir(parents=True, exist_ok=True)
PACKS.mkdir(parents=True, exist_ok=True)

ITEMS = json.loads((ROOT / 'music_manifest.json').read_text(encoding='utf-8'))


def db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -60.0, pad_ms: float = 100.0) -> np.ndarray:
    win = max(1, int(sr * 0.005))
    mono = np.abs(audio).max(axis=0)
    n = len(mono) // win
    if n == 0:
        return audio
    env = np.sqrt((mono[: n * win].reshape(n, win) ** 2).mean(axis=1))
    loud = np.nonzero(env > db_to_lin(threshold_db))[0]
    if loud.size == 0:
        return audio
    pad = int(sr * pad_ms / 1000.0)
    s = max(0, int(loud[0]) * win - pad)
    e = min(audio.shape[-1], (int(loud[-1]) + 1) * win + pad)
    return audio[:, s:e]


def fade(audio: np.ndarray, sr: int, fade_in_ms: float = 15.0, fade_out_ms: float = 1500.0) -> np.ndarray:
    out = audio.copy()
    n_in = min(out.shape[-1], int(sr * fade_in_ms / 1000.0))
    n_out = min(out.shape[-1], int(sr * fade_out_ms / 1000.0))
    if n_in > 0:
        out[:, :n_in] *= np.sin(np.linspace(0, np.pi / 2, n_in, dtype=np.float32)) ** 2
    if n_out > 0:
        out[:, -n_out:] *= np.cos(np.linspace(0, np.pi / 2, n_out, dtype=np.float32)) ** 2
    return out


def remove_vocals_hdemucs(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Run TorchAudio HDEMUCS_HIGH_MUSDB_PLUS on L4 GPU to strip any vocal stem and keep 100% of drums + bass + other."""
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    t0 = time.monotonic()
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to('cuda').eval()

    mix = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)  # (1, 2, T)
    mix = mix - mix.mean(dim=-1, keepdim=True)
    ref = mix.mean(dim=1, keepdim=True)  # (1, 1, T)
    ref_std = float(ref.std().item()) + 1e-8
    mix_norm = mix / ref_std

    total_len = mix_norm.shape[-1]
    chunk_len = int(24 * sr)
    overlap_len = int(2 * sr)
    step = chunk_len - overlap_len

    inst_acc = torch.zeros((2, total_len), dtype=torch.float32)
    voc_acc = torch.zeros((2, total_len), dtype=torch.float32)
    weight_acc = torch.zeros((1, total_len), dtype=torch.float32)

    ramp = torch.linspace(0.0, 1.0, overlap_len, dtype=torch.float32)

    with torch.no_grad():
        pos = 0
        while pos < total_len:
            end = min(pos + chunk_len, total_len)
            seg = mix_norm[:, :, pos:end]
            seg_len = seg.shape[-1]
            if seg_len < sr:
                pad_amt = sr - seg_len
                seg = torch.nn.functional.pad(seg, (0, pad_amt))
            stems = model(seg.to('cuda')).cpu()[0, :, :, :seg_len]  # (4, 2, seg_len): drums, bass, other, vocals
            stems = stems * ref_std
            inst_seg = stems[0] + stems[1] + stems[2]
            voc_seg = stems[3]

            w = torch.ones(seg_len, dtype=torch.float32)
            if pos > 0 and seg_len >= overlap_len:
                w[:overlap_len] = ramp
            if end < total_len and seg_len >= overlap_len:
                w[-overlap_len:] = 1.0 - ramp

            inst_acc[:, pos:end] += inst_seg * w.unsqueeze(0)
            voc_acc[:, pos:end] += voc_seg * w.unsqueeze(0)
            weight_acc[:, pos:end] += w.unsqueeze(0)

            if end >= total_len:
                break
            pos += step

    del model
    gc.collect()
    torch.cuda.empty_cache()

    weight_acc = weight_acc.clamp_min(1e-6)
    inst_np = (inst_acc / weight_acc).numpy().astype(np.float32)
    voc_np = (voc_acc / weight_acc).numpy().astype(np.float32)

    seconds = int(total_len / sr)
    voc_sec_rms = np.sqrt(np.mean(voc_np[:, : seconds * sr].reshape(2, seconds, sr) ** 2, axis=(0, 2)))
    voc_total_rms = float(np.sqrt(np.mean(voc_np ** 2)))
    voc_peak = float(np.max(np.abs(voc_np)))
    active_vocal_secs = [int(i) for i, r in enumerate(voc_sec_rms) if lin_to_db(float(r)) > -32.0]

    report = {
        'demucs_wall_seconds': round(time.monotonic() - t0, 2),
        'removed_vocal_stem_rms_dbfs': round(lin_to_db(voc_total_rms), 2),
        'removed_vocal_stem_peak_dbfs': round(lin_to_db(voc_peak), 2),
        'removed_active_vocal_seconds_count': len(active_vocal_secs),
        'instrumental_stem_rms_dbfs': round(lin_to_db(float(np.sqrt(np.mean(inst_np ** 2)))), 2),
    }
    print(
        f"DEMUCS_REPORT removed_vocal_rms={report['removed_vocal_stem_rms_dbfs']}dBFS "
        f"removed_active_vocal_secs={report['removed_active_vocal_seconds_count']}s "
        f"inst_rms={report['instrumental_stem_rms_dbfs']}dBFS "
        f"wall={report['demucs_wall_seconds']}s",
        flush=True,
    )
    return inst_np, report


def encode_mp3_direct(audio: np.ndarray, sr: int, mp3_path: Path, out_sr: int = 44100,
                      target_lufs: float = -16.0, true_peak_db: float = -1.5, bitrate: str = '192k') -> dict:
    """Two-pass EBU R128 loudness normalization and direct MP3 encoding via ffmpeg stdin pipe (zero WAV files)."""
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('ffmpeg not found on remote VM')
    pcm_bytes = np.ascontiguousarray(audio.T, dtype=np.float32).tobytes()
    channels = audio.shape[0]
    in_args = ['-f', 'f32le', '-ar', str(sr), '-ac', str(channels), '-i', 'pipe:0']
    base_loudnorm = f'loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=14'

    r1 = subprocess.run(
        [exe, '-hide_banner', '-nostdin', '-y', *in_args, '-af', base_loudnorm + ':print_format=json', '-f', 'null', '-'],
        input=pcm_bytes,
        capture_output=True,
    )
    err1 = r1.stderr.decode('utf-8', errors='replace')
    if r1.returncode != 0:
        raise RuntimeError(f'ffmpeg pass 1 failed: {err1[-800:]}')
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', err1, re.S)
    if not m:
        raise RuntimeError('ffmpeg loudnorm analysis produced no JSON stats')
    st = json.loads(m.group(0))
    info = {
        'input_lufs': round(float(st['input_i']), 2),
        'input_true_peak_db': round(float(st['input_tp']), 2),
    }

    af_chain = (
        f"{base_loudnorm}:measured_I={st['input_i']}:measured_TP={st['input_tp']}"
        f":measured_LRA={st['input_lra']}:measured_thresh={st['input_thresh']}"
        f":offset={st['target_offset']}:linear=true:print_format=json,"
        f"aresample={out_sr}"
    )
    r2 = subprocess.run(
        [
            exe, '-hide_banner', '-nostdin', '-y', *in_args,
            '-af', af_chain,
            '-c:a', 'libmp3lame', '-b:a', bitrate,
            '-ar', str(out_sr), '-ac', '2',
            str(mp3_path),
        ],
        input=pcm_bytes,
        capture_output=True,
    )
    err2 = r2.stderr.decode('utf-8', errors='replace')
    if r2.returncode != 0:
        raise RuntimeError(f'ffmpeg pass 2 MP3 encoding failed: {err2[-800:]}')
    m2 = re.search(r'\{[^{}]*"output_i"[^{}]*\}', err2, re.S)
    if m2:
        st2 = json.loads(m2.group(0))
        info['output_lufs'] = round(float(st2['output_i']), 2)
        info['loudnorm_mode'] = st2.get('normalization_type', '')
    return info


def update_progress(current: str):
    entries = []
    for f in sorted(PACKS.glob('*.zip')):
        entries.append({'name': f.name, 'size': f.stat().st_size})
    tmp = ROOT / 'progress.tmp.json'
    tmp.write_text(json.dumps({
        'model': 'minimax-music-3',
        'current': current,
        'packs': entries,
        'time': time.time(),
    }), encoding='utf-8')
    tmp.replace(ROOT / 'progress.json')


def save_mp3_pack(item: dict, requested_seed: int, actual_seed: int, audio: np.ndarray, sr: int,
                  wall_seconds: float, flow_steps: int, flow_cfg: float):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] != 2 or not np.isfinite(audio).all():
        raise ValueError('Invalid stereo audio tensor')

    content = trim_silence(audio, sr, threshold_db=-60.0, pad_ms=100.0)
    if content.shape[-1] < 120 * sr:
        raise ValueError(f'Short sample: only {content.shape[-1] / sr:.2f}s (< 120s minimum)')

    seconds = int(content.shape[-1] / sr)
    second_rms = np.sqrt(np.mean(content[:, : seconds * sr].reshape(2, seconds, sr) ** 2, axis=(0, 2)))
    if any(np.all(second_rms[i : i + 5] < 1e-5) for i in range(0, max(0, seconds - 9))):
        raise ValueError('Generated music contains a sustained near-silent gap')

    demucs_report = {}
    if item.get('remove_vocals', True):
        content, demucs_report = remove_vocals_hdemucs(content, sr)

    faded = fade(content, sr, fade_in_ms=15.0, fade_out_ms=1500.0)
    dest = OUT / 'minimax-music-3'
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{item['id']}_seed{requested_seed}"
    mp3_path = dest / f'{stem}.mp3'

    loudness_info = encode_mp3_direct(
        faded, sr, mp3_path, out_sr=44100, target_lufs=-16.0, true_peak_db=-1.5, bitrate='192k'
    )

    mp3_info = sf.info(str(mp3_path))
    decoded, _ = sf.read(str(mp3_path), dtype='float32', always_2d=True)
    peak_val = float(np.abs(decoded).max())
    rms_val = float(np.sqrt((decoded ** 2).mean()))
    if mp3_info.format != 'MP3' or mp3_info.channels != 2 or mp3_info.duration < 120.0:
        raise ValueError(f'MP3 verification failed on remote: {mp3_info}')

    pack_name = f"minimax-music-3__{item['id']}__{requested_seed}.zip"
    mp3_sha256 = hashlib.sha256(mp3_path.read_bytes()).hexdigest()
    meta = {
        'model': 'minimax-music-3',
        'format': 'mp3',
        'bitrate': '192k',
        'item': item,
        'seed': actual_seed,
        'requested_seed': requested_seed,
        'pack_name': pack_name,
        'native_sample_rate': sr,
        'exported_sample_rate': int(mp3_info.samplerate),
        'duration_seconds': round(float(mp3_info.duration), 3),
        'file_size_bytes': mp3_path.stat().st_size,
        'generation_wall_seconds': round(wall_seconds, 2),
        'loudness': loudness_info,
        'demucs_vocal_check': demucs_report,
        'stats': {
            'peak': round(peak_val, 4),
            'peak_dbfs': round(lin_to_db(peak_val), 2),
            'rms': round(rms_val, 4),
        },
        'gpu': torch.cuda.get_device_name(),
        'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'peak_gpu_reserved_bytes': int(torch.cuda.max_memory_reserved()),
        'settings': {
            'dtype': 'bfloat16',
            'allow_tf32': True,
            'cpu_offload': 'automatic_stage_level',
            'flow_steps_per_chunk': flow_steps,
            'flow_cfg_scale': flow_cfg,
            'vocal_removal': 'HDEMUCS_HIGH_MUSDB_PLUS' if item.get('remove_vocals', True) else 'none',
        },
        'files': {
            mp3_path.name: mp3_sha256,
        },
    }
    sidecar = dest / f'{stem}.mp3.json'
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    temp_zip = PACKS / (pack_name + '.tmp')
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in (mp3_path, sidecar):
            z.write(f, f.relative_to(OUT))
    temp_zip.replace(PACKS / pack_name)

    update_progress(f"saved {item['id']} seed={requested_seed}")
    print(
        f"SAVED minimax-music-3 {item['id']} seed={actual_seed} "
        f"duration={meta['duration_seconds']:.2f}s size={meta['file_size_bytes']/1024/1024:.2f}MiB "
        f"generation={wall_seconds:.1f}s peak={meta['peak_gpu_allocated_bytes']/1024**3:.2f}GiB",
        flush=True,
    )


def main():
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.guiders import ClassifierFreeGuidance

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device='cuda')
    pipe = ModularPipeline.from_pretrained(
        'MiniMaxAI/MiniMax-Music3',
        components_manager=manager,
        revision='1b525ccb685ae809f33027c51c351eeb5833cf72',
    )
    pipe.load_components(dtype=torch.bfloat16)

    if os.environ.get('MINIMAX_COMPILE_DIT') == '1' or len(ITEMS) >= 3:
        print('COMPILING Flow-Matching DiT transformer (regional torch.compile)...', flush=True)
        pipe.transformer = torch.compile(pipe.transformer, mode='reduce-overhead')

    current_cfg = 1.7
    for item in ITEMS:
        if (ROOT / 'STOP').exists():
            return
        requested_seed = int(item['seed'])
        pack_name = f"minimax-music-3__{item['id']}__{requested_seed}.zip"
        if (PACKS / pack_name).exists():
            continue

        flow_steps = int(item.get('flow_steps', 30))
        flow_cfg = float(item.get('flow_cfg', 1.7))
        if abs(flow_cfg - current_cfg) > 1e-4:
            pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=flow_cfg))
            current_cfg = flow_cfg

        update_progress(f"generating {item['id']} seed={requested_seed}")
        for attempt in range(3):
            actual_seed = requested_seed + 100000 * attempt
            print(
                f"GENERATING minimax-music-3 {item['id']} bpm={item['bpm']} "
                f"seed={actual_seed} target={item['minimax_max_seconds']}s steps={flow_steps} cfg={flow_cfg} format=mp3",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            result = pipe(
                prompt=item['prompt'],
                lyrics=item['lyrics'],
                audio_duration=float(item['minimax_max_seconds']),
                num_inference_steps=flow_steps,
                generator=torch.Generator('cuda').manual_seed(actual_seed),
                output='audios',
            )[0]
            if hasattr(result, 'detach'):
                audio_np = result.detach().float().cpu().numpy()
            else:
                audio_np = np.asarray(result, dtype=np.float32)
            del result
            if audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
                audio_np = audio_np.T

            elapsed = time.monotonic() - t0
            sr = int(pipe.sampling_rate)
            dur = audio_np.shape[-1] / sr
            if dur < 120.0:
                print(f"SHORT_SAMPLE {item['id']} seed={actual_seed} seconds={dur:.2f}", flush=True)
                del audio_np
                gc.collect()
                torch.cuda.empty_cache()
                if attempt == 2:
                    raise RuntimeError('Three attempts produced <120s audio; stopping for diagnosis')
                continue

            if len(ITEMS) == 1:
                del pipe
                pipe = None
                gc.collect()
                torch.cuda.empty_cache()

            save_mp3_pack(item, requested_seed, actual_seed, audio_np, sr, elapsed, flow_steps, flow_cfg)
            del audio_np
            gc.collect()
            torch.cuda.empty_cache()
            break

    if pipe is not None:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print('MODEL_DONE minimax-music-3', flush=True)


if __name__ == '__main__':
    main()
