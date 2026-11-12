"""Remote inference, post-processing & Pre-Download Auto-QA/Repair v3 template for colab-audio-studio-mcp.

Includes:
  1. Optimization A: Best-of-N Multi-Candidate Generation & Multi-Objective Reranking (`candidates: N`)
  2. Optimization B: pymusiclooper-inspired Smart Downbeat, Chroma Similarity & Zero-Crossing Loop Alignment (`smart_align_loop`)
  3. Optimization C: 100% Sample-Locked Vertical Stem Layering via built-in GPU `HDemucs` (`type: "stem_layer"`)
  4. 4-Gate Pre-Download Bar-by-Bar Diagnostics & On-GPU Auto-Repair (`run_qa_and_auto_repair`)
"""
import gc
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import soundfile as sf
import torch

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


def is_channels_first(audio: np.ndarray) -> bool:
    return audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]


def to_cf(audio: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    """Convert audio to (C, T) float32 array and return (cf_audio, was_1d, was_cf)."""
    if audio.ndim == 1:
        return audio[np.newaxis, :].astype(np.float32), True, True
    if is_channels_first(audio):
        return audio.astype(np.float32), False, True
    return audio.T.astype(np.float32), False, False


def from_cf(cf_audio: np.ndarray, was_1d: bool, was_cf: bool) -> np.ndarray:
    if was_1d:
        return cf_audio[0]
    return cf_audio if was_cf else cf_audio.T


def to_mono_1d(audio: np.ndarray) -> np.ndarray:
    cf, _, _ = to_cf(audio)
    return cf.mean(axis=0)


def num_samples(audio: np.ndarray) -> int:
    cf, _, _ = to_cf(audio)
    return cf.shape[1]


def slice_samples(audio: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    cf, was_1d, was_cf = to_cf(audio)
    return from_cf(cf[:, start_idx:end_idx].copy(), was_1d, was_cf)


def concat_samples(chunks: list[np.ndarray]) -> np.ndarray:
    cf_chunks = [to_cf(c)[0] for c in chunks]
    _, was_1d, was_cf = to_cf(chunks[0])
    return from_cf(np.concatenate(cf_chunks, axis=1), was_1d, was_cf)


def attenuate_band_fft(audio: np.ndarray, sr: int, f_low: float, f_high: float, gain_db: float) -> np.ndarray:
    """Apply smooth frequency-domain attenuation in [f_low, f_high] Hz to protect SFX frequency space."""
    cf, was_1d, was_cf = to_cf(audio)
    factor = 10.0 ** (gain_db / 20.0)
    n = cf.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = np.ones_like(freqs)
    in_band = (freqs >= f_low) & (freqs <= f_high)
    mask[in_band] = factor
    out = np.empty_like(cf)
    for c in range(cf.shape[0]):
        spec = np.fft.rfft(cf[c])
        out[c] = np.fft.irfft(spec * mask, n=n).astype(cf.dtype)
    return from_cf(out, was_1d, was_cf)


def lowpass_fft(audio: np.ndarray, sr: int, cutoff_hz: float, rolloff_hz: float = 600.0) -> np.ndarray:
    """Apply smooth cosine-tapered low-pass filter in frequency domain for stem layering."""
    cf, was_1d, was_cf = to_cf(audio)
    n = cf.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = np.ones_like(freqs)
    trans = (freqs > cutoff_hz) & (freqs < cutoff_hz + rolloff_hz)
    mask[trans] = 0.5 * (1.0 + np.cos(np.pi * (freqs[trans] - cutoff_hz) / rolloff_hz))
    mask[freqs >= cutoff_hz + rolloff_hz] = 0.08
    out = np.empty_like(cf)
    for c in range(cf.shape[0]):
        spec = np.fft.rfft(cf[c])
        out[c] = np.fft.irfft(spec * mask, n=n).astype(cf.dtype)
    return from_cf(out, was_1d, was_cf)


def compute_chroma_vector(mono_seg: np.ndarray, sr: int) -> np.ndarray:
    """Compute a 12-bin pitch class chroma vector using FFT bins (80 Hz - 4000 Hz) for loop harmonic matching."""
    if len(mono_seg) < 256:
        return np.ones(12, dtype=np.float64) / np.sqrt(12)
    win = np.hanning(len(mono_seg))
    spec = np.abs(np.fft.rfft(mono_seg * win)) ** 2
    freqs = np.fft.rfftfreq(len(mono_seg), 1.0 / sr)
    valid = (freqs >= 80.0) & (freqs <= 4000.0)
    f_val = freqs[valid]
    s_val = spec[valid]
    midi = 69.0 + 12.0 * np.log2(np.maximum(f_val, 1.0) / 440.0)
    pitch_class = np.mod(np.round(midi).astype(int), 12)
    chroma = np.zeros(12, dtype=np.float64)
    for pc in range(12):
        chroma[pc] = np.sum(s_val[pitch_class == pc])
    norm = np.linalg.norm(chroma) + 1e-12
    return chroma / norm


def find_zero_crossing(mono: np.ndarray, target_idx: int, radius: int = 480) -> int:
    """Snap sample index to nearest zero-crossing with positive slope (or minimum amplitude) within radius."""
    n = len(mono)
    lo = max(1, target_idx - radius)
    hi = min(n - 2, target_idx + radius)
    if lo >= hi:
        return max(0, min(n - 1, target_idx))
    seg0 = mono[lo:hi]
    seg1 = mono[lo + 1:hi + 1]
    zc = np.where((seg0 <= 0.0) & (seg1 > 0.0))[0]
    if len(zc) == 0:
        zc = np.where(seg0 * seg1 <= 0.0)[0]
    if len(zc) > 0:
        best = zc[np.argmin(np.abs((lo + zc) - target_idx))]
        return int(lo + best)
    return int(lo + np.argmin(np.abs(seg0)))


def smart_align_loop_window(
    audio: np.ndarray,
    sr: int,
    target_loop_sec: float,
    crossfade_ms: float,
) -> tuple[np.ndarray, dict]:
    """Optimization B: pymusiclooper-inspired downbeat onset + Chroma SSM + Zero-Crossing loop window alignment."""
    cf, was_1d, was_cf = to_cf(audio)
    mono = cf.mean(axis=0)
    n_total = cf.shape[1]
    n_needed = int(round((target_loop_sec + crossfade_ms / 1000.0) * sr))
    if n_total <= n_needed:
        return audio, {'aligned_start_ms': 0.0, 'seam_chroma_similarity': 1.0, 'zero_crossing_snapped': False}

    max_slack = n_total - n_needed
    # 1. Detect true first rhythmic onset within available slack (up to 350ms)
    hop = max(1, int(sr * 0.005))  # 5ms hop
    search_onset_max = min(max_slack, int(sr * 0.35))
    n_hops = max(1, search_onset_max // hop)
    envs = [float(np.sqrt(np.mean(mono[i * hop:(i + 2) * hop] ** 2)) + 1e-9) for i in range(n_hops)]
    peak_env = max(envs) if envs else 1e-3
    onset_hop = 0
    for i, e in enumerate(envs):
        if e >= 0.25 * peak_env and 20.0 * np.log10(e) > -34.0:
            onset_hop = i
            break
    base_start = min(max_slack, onset_hop * hop)

    # 2. Search candidate start offsets around base_start to maximize head/tail Chroma + RMS match
    # Expand search radius up to +/- 1 beat (e.g. 500ms at 120 BPM) if available slack allows
    xfade_samples = max(256, int(round(sr * (crossfade_ms / 1000.0))))
    step = max(1, int(sr * 0.010))  # 10ms search step
    lo_start = max(0, base_start - int(sr * 0.25))
    hi_start = min(max_slack, base_start + int(sr * 0.40))

    best_start = base_start
    best_sim = -999.0
    best_chroma_sim = 0.0

    for cand_s in range(lo_start, hi_start + 1, step):
        cand_e = cand_s + n_needed
        if cand_e > n_total:
            break
        head_seg = mono[cand_s:cand_s + xfade_samples]
        tail_seg = mono[cand_e - xfade_samples:cand_e]
        c_head = compute_chroma_vector(head_seg, sr)
        c_tail = compute_chroma_vector(tail_seg, sr)
        chroma_sim = float(np.dot(c_head, c_tail))
        rms_head = float(20.0 * np.log10(np.sqrt(np.mean(head_seg ** 2)) + 1e-9))
        rms_tail = float(20.0 * np.log10(np.sqrt(np.mean(tail_seg ** 2)) + 1e-9))
        rms_penalty = 0.04 * abs(rms_head - rms_tail)
        score = chroma_sim - rms_penalty
        if score > best_sim:
            best_sim = score
            best_start = cand_s
            best_chroma_sim = chroma_sim

    # 3. Snap best_start to nearest stereo zero-crossing while keeping exact n_needed length
    zc_start = find_zero_crossing(mono, best_start, radius=min(int(sr * 0.015), max_slack - best_start if max_slack > best_start else 1))
    zc_start = max(0, min(max_slack, zc_start))
    sliced_cf = cf[:, zc_start:zc_start + n_needed].copy()

    align_info = {
        'aligned_start_ms': round(1000.0 * zc_start / sr, 1),
        'seam_chroma_similarity': round(best_chroma_sim, 4),
        'zero_crossing_snapped': True,
    }
    return from_cf(sliced_cf, was_1d, was_cf), align_info


def run_hdemucs_stem_layer(
    src_audio: np.ndarray,
    src_sr: int,
    stem_weights: dict,
    other_lowpass_hz: float | None = None,
    export_stems_dir: Path | None = None,
    stem_prefix: str = '',
) -> tuple[np.ndarray, dict]:
    """Optimization C: 100% Sample-Locked Vertical Layering via TorchAudio HDEMUCS_HIGH_MUSDB_PLUS on L4 GPU."""
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    import torchaudio.functional as taF

    t0 = time.monotonic()
    cf, was_1d, was_cf = to_cf(src_audio)
    if cf.shape[0] == 1:
        cf = np.repeat(cf, 2, axis=0)
    orig_len = cf.shape[1]

    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model_sr = int(bundle.sample_rate)  # 44100 Hz (HDemucs training rate)
    model = bundle.get_model().to('cuda').eval()

    mix = torch.from_numpy(np.ascontiguousarray(cf, dtype=np.float32)).unsqueeze(0)  # (1, 2, T)
    if src_sr != model_sr:
        mix = taF.resample(mix, src_sr, model_sr)
    work_sr = model_sr
    mix_mean = mix.mean(dim=-1, keepdim=True)
    mix_zm = mix - mix_mean
    ref = mix_zm.mean(dim=1, keepdim=True)
    ref_std = float(ref.std().item()) + 1e-8
    mix_norm = mix_zm / ref_std

    total_len = mix_norm.shape[-1]
    chunk_len = int(26 * work_sr)
    overlap_len = int(2 * work_sr)
    step = chunk_len - overlap_len

    stems_acc = torch.zeros((4, 2, total_len), dtype=torch.float32)
    weight_acc = torch.zeros((1, 1, total_len), dtype=torch.float32)
    ramp = torch.linspace(0.0, 1.0, overlap_len, dtype=torch.float32)

    with torch.no_grad():
        pos = 0
        while pos < total_len:
            end = min(pos + chunk_len, total_len)
            seg = mix_norm[:, :, pos:end]
            seg_len = seg.shape[-1]
            if seg_len < work_sr:
                seg = torch.nn.functional.pad(seg, (0, work_sr - seg_len))
            out_stems = model(seg.to('cuda')).cpu()[0, :, :, :seg_len] * ref_std  # (4, 2, seg_len): drums, bass, other, vocals
            w = torch.ones(seg_len, dtype=torch.float32)
            if pos > 0 and seg_len >= overlap_len:
                w[:overlap_len] = ramp
            if end < total_len and seg_len >= overlap_len:
                w[-overlap_len:] = 1.0 - ramp
            stems_acc[:, :, pos:end] += out_stems * w.view(1, 1, -1)
            weight_acc[:, :, pos:end] += w.view(1, 1, -1)
            if end >= total_len:
                break
            pos += step

    del model
    gc.collect()
    torch.cuda.empty_cache()

    stems_t = stems_acc / torch.clamp(weight_acc, min=1e-6)  # (4, 2, T_work)
    # Distribute the removed DC mean back to the drums/bass stem so remix sum stays faithful
    stems_t[0] = stems_t[0] + mix_mean[0]
    if work_sr != src_sr:
        stems_t = taF.resample(stems_t, work_sr, src_sr)
    # Enforce EXACT sample length of the source loop so the layer stays 100% sample-locked
    cur_len = stems_t.shape[-1]
    if cur_len > orig_len:
        stems_t = stems_t[..., :orig_len]
    elif cur_len < orig_len:
        stems_t = torch.nn.functional.pad(stems_t, (0, orig_len - cur_len))
    stems_np = stems_t.numpy().astype(np.float32)
    drums, bass, other, vocals = stems_np[0], stems_np[1], stems_np[2], stems_np[3]

    if export_stems_dir is not None and stem_prefix:
        rhythm_stem = np.clip(drums + bass, -1.0, 1.0)
        melodic_stem = np.clip(other + vocals, -1.0, 1.0)
        post.write(export_stems_dir / f"{stem_prefix}_stem_drums_bass.wav", from_cf(rhythm_stem, was_1d, was_cf), src_sr, 'wav')
        post.write(export_stems_dir / f"{stem_prefix}_stem_guitars_synths.wav", from_cf(melodic_stem, was_1d, was_cf), src_sr, 'wav')

    if other_lowpass_hz is not None and other_lowpass_hz > 0:
        other = lowpass_fft(other, src_sr, float(other_lowpass_hz))

    w_drums = float(stem_weights.get('drums', 1.0))
    w_bass = float(stem_weights.get('bass', 1.0))
    w_other = float(stem_weights.get('other', 0.30))
    w_vocals = float(stem_weights.get('vocals', 0.0))

    remixed_cf = w_drums * drums + w_bass * bass + w_other * other + w_vocals * vocals

    # Measure real sync: cross-correlate onset envelopes (1 ms hop) of main loop vs layer within +/-50 ms
    orig_mono = cf.mean(axis=0)
    remix_mono = remixed_cf.mean(axis=0)
    drift_ms = measure_onset_lag_ms(orig_mono, remix_mono, src_sr, max_lag_ms=50.0)
    corr = float(np.corrcoef(orig_mono, remix_mono)[0, 1]) if len(orig_mono) > 10 else 1.0

    stem_info = {
        'hdemucs_wall_seconds': round(time.monotonic() - t0, 2),
        'hdemucs_model_sr': work_sr,
        'stem_weights': {'drums': w_drums, 'bass': w_bass, 'other': w_other, 'vocals': w_vocals},
        'other_lowpass_hz': other_lowpass_hz,
        'source_samples': int(orig_len),
        'layer_samples': int(remixed_cf.shape[-1]),
        'waveform_correlation_with_main_loop': round(corr, 4),
        'measured_onset_lag_ms': drift_ms,
    }
    return from_cf(remixed_cf, was_1d, was_cf), stem_info


def measure_onset_lag_ms(ref_mono: np.ndarray, test_mono: np.ndarray, sr: int, max_lag_ms: float = 50.0) -> float:
    """Return lag (ms) maximizing cross-correlation of half-wave-rectified onset envelopes (positive = test is late)."""
    hop = max(1, int(sr * 0.001))
    n = min(len(ref_mono), len(test_mono)) // hop
    if n < 200:
        return 0.0

    def env(x):
        e = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
        d = np.maximum(0.0, np.diff(np.log(e)))
        return (d - d.mean()) / (d.std() + 1e-9)

    a, b = env(ref_mono), env(test_mono)
    max_lag = int(max_lag_ms)
    best_lag, best_val = 0, -1e18
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            v = float(np.dot(a[:len(a) - lag], b[lag:]))
        else:
            v = float(np.dot(a[-lag:], b[:len(b) + lag]))
        if v > best_val:
            best_val, best_lag = v, lag
    return float(best_lag)


def compute_diagnostics(audio: np.ndarray, sr: int, bar_sec: float = 2.0) -> dict:
    """Compute whole-track and bar-by-bar acoustic, spectral, onset clarity & Chroma loop telemetry."""
    cf, _, _ = to_cf(audio)
    mono = cf.mean(axis=0).astype(np.float64)
    n = len(mono)
    dur = n / float(sr)
    overall_rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
    overall_rms_db = round(float(20.0 * np.log10(overall_rms)), 2)
    overall_peak_db = round(float(20.0 * np.log10(np.max(np.abs(mono)) + 1e-9)), 2)

    spec_sq = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    tot_e = float(np.sum(spec_sq)) + 1e-12
    low_pct = round(float(100.0 * np.sum(spec_sq[freqs < 250.0]) / tot_e), 2)
    mid_pct = round(float(100.0 * np.sum(spec_sq[(freqs >= 250.0) & (freqs < 3000.0)]) / tot_e), 2)
    sfx_zone_pct = round(float(100.0 * np.sum(spec_sq[(freqs >= 3000.0) & (freqs < 6000.0)]) / tot_e), 2)

    seam_jump = round(float(np.max(np.abs(cf[:, 0] - cf[:, -1]))), 4)

    # Compute rhythmic onset clarity (90th vs 50th percentile of 20ms frame RMS differences)
    frame_len = max(1, int(sr * 0.02))
    n_frames = n // frame_len
    if n_frames > 4:
        f_rms = np.sqrt(np.mean(mono[:n_frames * frame_len].reshape(n_frames, frame_len) ** 2, axis=1) + 1e-12)
        f_db = 20.0 * np.log10(f_rms)
        onset_flux = np.maximum(0.0, np.diff(f_db))
        onset_clarity_db = round(float(np.percentile(onset_flux, 90)), 2)
    else:
        onset_clarity_db = 0.0

    # Compute head-vs-tail Chroma harmonic similarity across the first/last bar
    win_chroma = min(n // 2, int(round(bar_sec * sr)))
    if win_chroma > 512:
        c_head = compute_chroma_vector(mono[:win_chroma], sr)
        c_tail = compute_chroma_vector(mono[-win_chroma:], sr)
        bar_level_chroma_sim = round(float(np.dot(c_head, c_tail)), 4)
    else:
        bar_level_chroma_sim = 1.0

    n_bars = max(1, int(round(dur / bar_sec)))
    bars = []
    for b in range(n_bars):
        i0 = int(round(b * bar_sec * sr))
        i1 = min(n, int(round((b + 1) * bar_sec * sr)))
        seg = mono[i0:i1]
        if len(seg) < 64:
            continue
        b_rms_db = round(float(20.0 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)), 2)
        b_spec = np.abs(np.fft.rfft(seg)) ** 2
        b_freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        b_tot = float(np.sum(b_spec)) + 1e-12
        b_mid = round(float(100.0 * np.sum(b_spec[(b_freqs >= 250.0) & (b_freqs < 3000.0)]) / b_tot), 1)
        b_sfx = round(float(100.0 * np.sum(b_spec[(b_freqs >= 3000.0) & (b_freqs < 6000.0)]) / b_tot), 1)
        bars.append({
            'bar': b + 1,
            'start_s': round(b * bar_sec, 1),
            'end_s': round(i1 / sr, 1),
            'rms_db': b_rms_db,
            'mid_pct': b_mid,
            'sfx_zone_pct': b_sfx,
        })

    intro_drop_db = 0.0
    max_bar_spread_db = 0.0
    if len(bars) >= 6:
        first2_rms = float(np.mean([bars[0]['rms_db'], bars[1]['rms_db']]))
        last4_rms = float(np.mean([x['rms_db'] for x in bars[-4:]]))
        intro_drop_db = round(last4_rms - first2_rms, 2)
        all_rms = [x['rms_db'] for x in bars]
        max_bar_spread_db = round(max(all_rms) - min(all_rms), 2)

    return {
        'duration_seconds': round(dur, 2),
        'overall_rms_db': overall_rms_db,
        'overall_peak_db': overall_peak_db,
        'low_lt250_pct': low_pct,
        'mid_250to3k_pct': mid_pct,
        'sfx_zone_3to6k_pct': sfx_zone_pct,
        'seam_jump': seam_jump,
        'onset_clarity_db': onset_clarity_db,
        'bar_chroma_similarity': bar_level_chroma_sim,
        'intro_drop_db': intro_drop_db,
        'max_bar_spread_db': max_bar_spread_db,
        'bars': bars,
    }


def score_candidate_for_reranking(diag: dict, role: str, qa_cfg: dict) -> float:
    """Optimization A: Multi-objective candidate scoring function for Best-of-N selection."""
    if role == 'bgm_loop':
        score = (
            100.0
            - 4.0 * abs(diag['intro_drop_db'])
            - 2.5 * diag['max_bar_spread_db']
            - 3.0 * max(0.0, diag['sfx_zone_3to6k_pct'] - 6.0)
            + 12.0 * diag['bar_chroma_similarity']
            + 1.5 * diag['onset_clarity_db']
        )
        return round(float(score), 2)
    elif role == 'bgm_outro':
        c_start = float(qa_cfg.get('climax_start_sec', 24.0))
        c_end = float(qa_cfg.get('climax_end_sec', 28.0))
        body = [b['rms_db'] for b in diag['bars'] if b['end_s'] <= c_start]
        climax = [b for b in diag['bars'] if c_start <= b['start_s'] < c_end]
        b_rms = float(np.mean(body)) if body else -18.0
        c_rms = float(np.mean([b['rms_db'] for b in climax])) if climax else -30.0
        c_mid = float(np.mean([b['mid_pct'] for b in climax])) if climax else 0.0
        drop = max(0.0, b_rms - c_rms)
        return round(float(100.0 - 4.0 * drop + 1.0 * c_mid), 2)
    else:
        return round(float(100.0 + diag['overall_rms_db'] + 2.0 * diag['onset_clarity_db']), 2)


def apply_post_options(audio, sr: int, post_cfg: dict) -> tuple:
    """Apply post-processing including Smart Downbeat/Chroma/Zero-Crossing Loop Alignment (Optimization B)."""
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
        xfade_ms = float(post_cfg['loop_crossfade_ms'])
        target_loop_sec = post_cfg.get('target_loop_seconds')
        if target_loop_sec is not None:
            audio, align_info = smart_align_loop_window(audio, sr, float(target_loop_sec), xfade_ms)
            info.update(align_info)
        audio = post.make_loop(audio, sr, xfade_ms)

    fade_in = float(post_cfg.get('fade_in_ms', 0.0))
    fade_out = float(post_cfg.get('fade_out_ms', 0.0))
    if fade_in > 0 or fade_out > 0:
        audio = post.fade(audio, sr, fade_in, fade_out)

    return audio, sr, info


def resolve_source_file(source_id: str, prefer_looped: bool = False) -> Path:
    if not prefer_looped:
        cand_raw_fixed = OUT / f"{source_id}_unlooped_fixed.wav"
        if cand_raw_fixed.exists():
            return cand_raw_fixed
    for ext in ('.wav', '.ogg', '.flac'):
        cand = OUT / f"{source_id}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Source audio for {source_id!r} not found in {OUT}")


def run_qa_and_auto_repair(
    backend,
    item: dict,
    raw_audio: np.ndarray,
    raw_sr: int,
    post_cfg: dict,
    extra_files_to_pack: list[Path] | None = None,
) -> tuple[np.ndarray, int, dict, dict]:
    """Run bar-by-bar QA diagnostics before packaging; auto-repair on GPU or via studio DSP if any gate fails."""
    qa_cfg = item.get('qa') or {}
    role = qa_cfg.get('role', 'generic')
    bar_sec = float(qa_cfg.get('bar_seconds', 2.0))
    repairs = []

    proc_audio, proc_sr, post_info = apply_post_options(raw_audio.copy(), raw_sr, post_cfg)
    initial_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
    current_raw = raw_audio.copy()
    current_raw_sr = raw_sr

    print(
        f"QA_INITIAL {item['id']} role={role} dur={initial_diag['duration_seconds']}s "
        f"intro_drop={initial_diag['intro_drop_db']}dB bar_chroma={initial_diag['bar_chroma_similarity']} "
        f"sfx_zone={initial_diag['sfx_zone_3to6k_pct']}% seam={initial_diag['seam_jump']}",
        flush=True,
    )

    # ─── GATE A: Countdown SFX 3-Window Activity Check ───
    if role == 'countdown_sfx':
        mono = to_mono_1d(proc_audio)
        win_rms = []
        for w0, w1 in ((0.0, 1.0), (1.0, 2.0), (2.0, min(3.4, len(mono) / proc_sr))):
            seg = mono[int(w0 * proc_sr):int(w1 * proc_sr)]
            r_db = float(20.0 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)) if len(seg) > 0 else -90.0
            win_rms.append(round(r_db, 1))
        active_count = sum(1 for r in win_rms if r > -32.0)
        if active_count < int(qa_cfg.get('min_active_windows', 3)):
            for retry_idx in range(1, 3):
                new_seed = int(item.get('seed', 101)) + retry_idx * 313
                print(f"QA_REPAIR {item['id']}: countdown windows={win_rms} (<3 active), retrying seed={new_seed}", flush=True)
                prompt_used, dur = prompts.prepare_prompt(
                    item.get('prompt', ''), 'sfx', item.get('duration_seconds', 3.5),
                    sfx_fields=tuple(item.get('sfx_fields', ())),
                )
                req = GenerationRequest(prompt=prompt_used, seconds=float(dur), steps=int(item.get('steps', 8)), cfg=float(item.get('cfg', 1.0)))
                res = backend._generate([new_seed], req)
                cand_raw, cand_sr = res.audios[0], backend.sample_rate
                cand_proc, cand_psr, cand_pinfo = apply_post_options(cand_raw.copy(), cand_sr, post_cfg)
                c_mono = to_mono_1d(cand_proc)
                c_wins = [
                    round(float(20.0 * np.log10(np.sqrt(np.mean(c_mono[int(w0 * cand_psr):int(w1 * cand_psr)] ** 2)) + 1e-9)), 1)
                    for w0, w1 in ((0.0, 1.0), (1.0, 2.0), (2.0, min(3.4, len(c_mono) / cand_psr)))
                ]
                repairs.append(f"retry_countdown_seed_{new_seed}_windows_{c_wins}")
                if sum(1 for r in c_wins if r > -32.0) >= active_count:
                    proc_audio, proc_sr, post_info = cand_proc, cand_psr, cand_pinfo
                    current_raw, current_raw_sr = cand_raw, cand_sr
                    if sum(1 for r in c_wins if r > -32.0) >= 3:
                        break

    # ─── GATE B: BGM Loop Continuity & Intro-Trap Repair ───
    elif role == 'bgm_loop':
        max_drop = float(qa_cfg.get('max_intro_drop_db', 2.5))
        cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
        bar1_drop = (cur_diag['bars'][-1]['rms_db'] - cur_diag['bars'][0]['rms_db']) if len(cur_diag['bars']) >= 6 else 0.0

        if cur_diag['intro_drop_db'] > max_drop or bar1_drop > max_drop:
            # Optional Horizontal Resequencing bonus: if raw audio has a distinct 4-bar intro, export it as a play-once intro file first!
            if qa_cfg.get('export_intro_segment', False) and extra_files_to_pack is not None:
                intro_samples = int(round(4.0 * bar_sec * proc_sr))
                if num_samples(proc_audio) >= intro_samples:
                    intro_seg = slice_samples(proc_audio, 0, intro_samples)
                    intro_file = OUT / f"{item['id']}_intro_4bars.wav"
                    post.write(intro_file, intro_seg, proc_sr, 'wav')
                    extra_files_to_pack.append(intro_file)
                    repairs.append(f"exported_horizontal_intro_4bars({intro_file.name})")

            print(
                f"QA_REPAIR {item['id']}: detected Intro Trap (intro_drop={cur_diag['intro_drop_db']}dB, "
                f"bar12_vs_bar1={bar1_drop:.2f}dB > {max_drop}dB). Running GPU Inpaint on Bars 1-4 [0..8s]...",
                flush=True,
            )
            tmp_inpaint_src = OUT / f"_tmp_{item['id']}_inpaint_src.wav"
            post.write(tmp_inpaint_src, current_raw, current_raw_sr, 'wav')
            fix_prompt_raw = qa_cfg.get('inpaint_fix_prompt') or item.get('prompt', '')
            fix_prompt, _ = prompts.prepare_prompt(
                fix_prompt_raw, 'music', float(item.get('duration_seconds', 24.4)), bpm=item.get('bpm', 120)
            )
            inpaint_seed = int(item.get('seed', 42)) + 777
            req = GenerationRequest(
                prompt=fix_prompt,
                seconds=float(item.get('duration_seconds', 24.4)),
                steps=int(item.get('steps', 8)),
                cfg=float(item.get('cfg', 1.0)),
                inpaint_audio=str(tmp_inpaint_src),
                inpaint_range=(0.0, 4.0 * bar_sec),
            )
            res = backend._generate([inpaint_seed], req)
            inp_raw, inp_sr = res.audios[0], backend.sample_rate
            if tmp_inpaint_src.exists():
                tmp_inpaint_src.unlink()

            inp_proc, inp_psr, inp_pinfo = apply_post_options(inp_raw.copy(), inp_sr, post_cfg)
            inp_diag = compute_diagnostics(inp_proc, inp_psr, bar_sec=bar_sec)
            inp_bar1_drop = inp_diag['bars'][-1]['rms_db'] - inp_diag['bars'][0]['rms_db']
            repairs.append(
                f"gpu_inpaint_0to8s_seed_{inpaint_seed}(intro_drop:{cur_diag['intro_drop_db']}->{inp_diag['intro_drop_db']}dB)"
            )
            if abs(inp_diag['intro_drop_db']) <= abs(cur_diag['intro_drop_db']):
                current_raw, current_raw_sr = inp_raw, inp_sr
                proc_audio, proc_sr, post_info = inp_proc, inp_psr, inp_pinfo
                cur_diag = inp_diag
                bar1_drop = inp_bar1_drop

            if cur_diag['intro_drop_db'] > max_drop or bar1_drop > max_drop:
                print(
                    f"QA_REPAIR {item['id']}: aligning Bars 1-4 with full-energy Bars 5-8 groove to guarantee zero loop energy drop...",
                    flush=True,
                )
                s4b = int(round(4.0 * bar_sec * current_raw_sr))
                s8b = int(round(8.0 * bar_sec * current_raw_sr))
                n_tot = num_samples(current_raw)
                if n_tot >= s8b:
                    bars_5_to_8 = slice_samples(current_raw, s4b, s8b)
                    bars_5_to_end = slice_samples(current_raw, s4b, n_tot)
                    current_raw = concat_samples([bars_5_to_8, bars_5_to_end])
                    proc_audio, proc_sr, post_info = apply_post_options(current_raw.copy(), current_raw_sr, post_cfg)
                    cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
                    repairs.append(
                        f"bar_aligned_groove_promotion_bars5to8(final_intro_drop:{cur_diag['intro_drop_db']}dB)"
                    )

        max_sfx_pct = float(qa_cfg.get('max_sfx_zone_pct', 8.0))
        if cur_diag['sfx_zone_3to6k_pct'] > max_sfx_pct:
            print(
                f"QA_REPAIR {item['id']}: 3-6kHz SFX zone={cur_diag['sfx_zone_3to6k_pct']}% > {max_sfx_pct}%. "
                f"Applying surgical -4.0dB 3-6kHz pocket carve...",
                flush=True,
            )
            current_raw = attenuate_band_fft(current_raw, current_raw_sr, 3000.0, 6000.0, -4.0)
            proc_audio, proc_sr, post_info = apply_post_options(current_raw.copy(), current_raw_sr, post_cfg)
            cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
            repairs.append(f"surgical_sfx_pocket_eq_3to6k(sfx_zone:{cur_diag['sfx_zone_3to6k_pct']}%)")

        post.write(OUT / f"{item['id']}_unlooped_fixed.wav", current_raw, current_raw_sr, 'wav')

    # ─── GATE C: BGM Finish Outro Climax & Resolution Check ───
    elif role == 'bgm_outro':
        cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
        c_start = float(qa_cfg.get('climax_start_sec', 24.0))
        c_end = float(qa_cfg.get('climax_end_sec', 28.0))
        body_bars = [b for b in cur_diag['bars'] if b['end_s'] <= c_start]
        climax_bars = [b for b in cur_diag['bars'] if c_start <= b['start_s'] < c_end]
        body_rms = float(np.mean([b['rms_db'] for b in body_bars])) if body_bars else -18.0
        climax_rms = float(np.mean([b['rms_db'] for b in climax_bars])) if climax_bars else -30.0
        body_mid = float(np.mean([b['mid_pct'] for b in body_bars])) if body_bars else 10.0
        climax_mid = float(np.mean([b['mid_pct'] for b in climax_bars])) if climax_bars else 0.0
        climax_drop = round(body_rms - climax_rms, 2)
        relative_mid_ratio = round(climax_mid / max(1.0, body_mid), 2)

        max_c_drop = float(qa_cfg.get('max_climax_drop_db', 3.5))
        min_mid_ratio = float(qa_cfg.get('min_climax_relative_mid_ratio', 1.15))  # Require >= +15% to +20% relative mid boost

        if climax_drop > max_c_drop or relative_mid_ratio < min_mid_ratio:
            print(
                f"QA_REPAIR {item['id']}: Outro climax ({c_start}-{c_end}s) drop={climax_drop}dB (max {max_c_drop}), "
                f"mid_ratio={relative_mid_ratio}x (min {min_mid_ratio}x relative to body mid {body_mid:.1f}%). Retrying edit_continue on GPU...",
                flush=True,
            )
            src_path = resolve_source_file(item['source_id'])
            best_raw, best_sr = current_raw, current_raw_sr
            best_proc, best_psr, best_pinfo = proc_audio, proc_sr, post_info
            best_score = climax_rms + 5.0 * relative_mid_ratio

            for retry_idx in range(1, 3):
                new_seed = int(item.get('seed', 606)) + retry_idx * 111
                target_sec = float(item['duration_seconds'])
                prompt_used, _ = prompts.prepare_prompt(
                    item['prompt'], 'music', target_sec, bpm=item.get('bpm', 120)
                )
                req = GenerationRequest(
                    prompt=prompt_used,
                    seconds=target_sec,
                    steps=int(item.get('steps', 8)),
                    cfg=float(item.get('cfg', 1.0)),
                    inpaint_audio=str(src_path),
                    inpaint_range=(max(0.0, c_start - 2.0), target_sec),
                )
                res = backend._generate([new_seed], req)
                cand_raw, cand_sr = res.audios[0], backend.sample_rate
                cand_proc, cand_psr, cand_pinfo = apply_post_options(cand_raw.copy(), cand_sr, post_cfg)
                c_diag = compute_diagnostics(cand_proc, cand_psr, bar_sec=bar_sec)
                c_climax = [b for b in c_diag['bars'] if c_start <= b['start_s'] < c_end]
                c_rms = float(np.mean([b['rms_db'] for b in c_climax])) if c_climax else -30.0
                c_mid = float(np.mean([b['mid_pct'] for b in c_climax])) if c_climax else 0.0
                c_ratio = round(c_mid / max(1.0, body_mid), 2)
                score = c_rms + 5.0 * c_ratio
                repairs.append(f"retry_outro_seed_{new_seed}(climax_rms:{c_rms:.1f}dB,mid_ratio:{c_ratio}x)")
                if score > best_score:
                    best_score = score
                    best_raw, best_sr = cand_raw, cand_sr
                    best_proc, best_psr, best_pinfo = cand_proc, cand_psr, cand_pinfo
                    climax_drop = round(body_rms - c_rms, 2)
                    relative_mid_ratio = c_ratio
                if climax_drop <= max_c_drop and relative_mid_ratio >= min_mid_ratio:
                    break

            current_raw, current_raw_sr = best_raw, best_sr
            proc_audio, proc_sr, post_info = best_proc, best_psr, best_pinfo

            cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
            c_climax = [b for b in cur_diag['bars'] if c_start <= b['start_s'] < c_end]
            c_rms = float(np.mean([b['rms_db'] for b in c_climax])) if c_climax else -30.0
            c_mid = float(np.mean([b['mid_pct'] for b in c_climax])) if c_climax else 0.0
            c_ratio = round(c_mid / max(1.0, body_mid), 2)
            if (body_rms - c_rms) > max_c_drop or c_ratio < min_mid_ratio:
                s16 = int(round((c_start - 4.0 * bar_sec) * current_raw_sr))
                s24 = int(round(c_start * current_raw_sr))
                body_part = slice_samples(current_raw, 0, s24)
                climax_part = slice_samples(current_raw, s16, s24)
                combined = concat_samples([body_part, climax_part])
                outro_post = {**post_cfg, 'fade_out_ms': 1800.0}
                proc_audio, proc_sr, post_info = apply_post_options(combined, current_raw_sr, outro_post)
                repairs.append("climax_4bar_summit_cadence_extension_with_1800ms_fade")

    # ─── UNIVERSAL GATE: SFX Frequency Pocket Protection (3-6 kHz) for ALL BGM roles ───
    if role in ('bgm_loop', 'bgm_outro', 'bgm_stem_layer'):
        cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
        max_sfx_pct = float(qa_cfg.get('max_sfx_zone_pct', 8.0))
        if cur_diag['sfx_zone_3to6k_pct'] > max_sfx_pct:
            print(
                f"QA_REPAIR {item['id']}: 3-6kHz SFX zone={cur_diag['sfx_zone_3to6k_pct']}% > {max_sfx_pct}%. "
                f"Applying surgical -4.0dB 3-6kHz pocket carve...",
                flush=True,
            )
            current_raw = attenuate_band_fft(current_raw, current_raw_sr, 3000.0, 6000.0, -4.0)
            proc_audio, proc_sr, post_info = apply_post_options(current_raw.copy(), current_raw_sr, post_cfg)
            cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
            repairs.append(f"surgical_sfx_pocket_eq_3to6k(sfx_zone:{cur_diag['sfx_zone_3to6k_pct']}%)")

    # ─── THREE-TIER QA STATUS: passed | warning | failed ───
    final_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
    status = 'passed'
    status_reasons = []

    if role == 'bgm_loop':
        max_drop = float(qa_cfg.get('max_intro_drop_db', 2.5))
        max_sfx = float(qa_cfg.get('max_sfx_zone_pct', 8.0))
        max_seam = float(qa_cfg.get('max_seam_jump', 0.02))
        if final_diag['intro_drop_db'] > max_drop:
            status = 'failed'
            status_reasons.append(f"intro_drop {final_diag['intro_drop_db']}dB > {max_drop}dB")
        if final_diag['seam_jump'] > max_seam:
            status = 'failed' if status == 'failed' else 'warning'
            status_reasons.append(f"seam_jump {final_diag['seam_jump']} > {max_seam}")
        if final_diag['sfx_zone_3to6k_pct'] > max_sfx:
            status = 'warning' if status == 'passed' else status
            status_reasons.append(f"sfx_zone {final_diag['sfx_zone_3to6k_pct']}% > {max_sfx}%")

    elif role == 'bgm_outro':
        c_start = float(qa_cfg.get('climax_start_sec', 24.0))
        c_end = float(qa_cfg.get('climax_end_sec', 28.0))
        body_bars = [b for b in final_diag['bars'] if b['end_s'] <= c_start]
        climax_bars = [b for b in final_diag['bars'] if c_start <= b['start_s'] < c_end]
        body_rms = float(np.mean([b['rms_db'] for b in body_bars])) if body_bars else -18.0
        climax_rms = float(np.mean([b['rms_db'] for b in climax_bars])) if climax_bars else -30.0
        max_c_drop = float(qa_cfg.get('max_climax_drop_db', 3.5))
        drop = round(body_rms - climax_rms, 2)
        if drop > max_c_drop:
            status = 'warning'
            status_reasons.append(f"climax drop {drop}dB > {max_c_drop}dB")

    elif role == 'countdown_sfx':
        mono = to_mono_1d(proc_audio)
        win_rms = []
        for w0, w1 in ((0.0, 1.0), (1.0, 2.0), (2.0, min(3.4, len(mono) / proc_sr))):
            seg = mono[int(w0 * proc_sr):int(w1 * proc_sr)]
            r_db = float(20.0 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)) if len(seg) > 0 else -90.0
            win_rms.append(round(r_db, 1))
        active_count = sum(1 for r in win_rms if r > -32.0)
        req_active = int(qa_cfg.get('min_active_windows', 3))
        if active_count < req_active:
            status = 'failed' if active_count < req_active - 1 else 'warning'
            status_reasons.append(f"active windows {active_count} < {req_active}")

    qa_report = {
        'status': status,
        'passed': (status in ('passed', 'warning')),  # True for clean pass or acceptable warning
        'status_reasons': status_reasons,
        'role': role,
        'repairs_applied': repairs,
        'initial_diagnostics': initial_diag,
        'final_diagnostics': final_diag,
    }
    print(
        f"QA_RESULT {item['id']} status={status.upper()} repairs={len(repairs)} "
        f"reasons={status_reasons} intro_drop={final_diag['intro_drop_db']}dB "
        f"bar_chroma={final_diag['bar_chroma_similarity']} "
        f"final_sfx_zone={final_diag['sfx_zone_3to6k_pct']}% "
        f"final_seam={final_diag['seam_jump']}",
        flush=True,
    )
    return proc_audio, proc_sr, post_info, qa_report


def save_artifact_pack(
    item: dict,
    output_file: Path,
    wall_seconds: float,
    extra_meta: dict | None = None,
    extra_files: list[Path] | None = None,
):
    """Package QA-verified output file (+ any bonus stem/intro files) + SHA-256 metadata sidecar into an atomic ZIP archive."""
    pack_name = f"{item['id']}.zip"
    all_audio_files = [output_file] + [f for f in (extra_files or []) if f.exists()]
    files_map = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in all_audio_files}

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
        'files': files_map,
        **(extra_meta or {}),
    }
    sidecar = output_file.with_suffix(output_file.suffix + '.json')
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    temp_zip = PACKS / (pack_name + '.tmp')
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in [*all_audio_files, sidecar]:
            z.write(f, f.relative_to(OUT))
    temp_zip.replace(PACKS / pack_name)

    update_progress(f"saved {item['id']}")
    print(
        f"SAVED {item['id']} size={meta['file_size_bytes']/1024/1024:.2f}MiB "
        f"files={len(all_audio_files)} generation={wall_seconds:.1f}s peak={meta['peak_gpu_allocated_bytes']/1024**3:.2f}GiB",
        flush=True,
    )


def main():
    update_progress('loading_model')
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.monotonic()

    backend = create_backend('sa3-torch')
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
        extra_files_to_pack: list[Path] = []

        post_cfg = item.get('post') or {}
        qa_cfg = item.get('qa') or {}
        role = qa_cfg.get('role', 'generic')
        bar_sec = float(qa_cfg.get('bar_seconds', 2.0))
        fmt = post_cfg.get('format', item.get('format', 'wav')).lower()
        ext = '.wav' if fmt in ('wav', 'wav16') else f'.{fmt}'
        output_file = OUT / f"{item['id']}{ext}"

        steps = int(item.get('steps', 8))
        cfg = float(item.get('cfg', 1.0))
        neg_prompt = item.get('negative_prompt', '')
        seed = int(item.get('seed', 42))
        n_candidates = int(item.get('candidates', 1))

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
            # Optimization A: Best-of-N candidate sampling & multi-objective reranking
            if n_candidates > 1:
                cand_seeds = [seed + i * 101 for i in range(n_candidates)]
                best_score = -1e9
                best_seed = seed
                best_raw = None
                cand_scores = {}
                for c_seed in cand_seeds:
                    res = backend._generate([c_seed], req)
                    c_audio, c_sr = res.audios[0], backend.sample_rate
                    c_proc, c_psr, _ = apply_post_options(c_audio.copy(), c_sr, post_cfg)
                    c_diag = compute_diagnostics(c_proc, c_psr, bar_sec=bar_sec)
                    c_score = score_candidate_for_reranking(c_diag, role, qa_cfg)
                    cand_scores[str(c_seed)] = {
                        'score': c_score,
                        'intro_drop_db': c_diag['intro_drop_db'],
                        'max_bar_spread_db': c_diag['max_bar_spread_db'],
                        'bar_chroma_similarity': c_diag['bar_chroma_similarity'],
                        'onset_clarity_db': c_diag['onset_clarity_db'],
                    }
                    if c_score > best_score:
                        best_score = c_score
                        best_seed = c_seed
                        best_raw = c_audio
                raw_audio, raw_sr = best_raw, backend.sample_rate
                extra_meta['best_of_n'] = {
                    'candidates_evaluated': n_candidates,
                    'winner_seed': best_seed,
                    'winner_score': best_score,
                    'candidate_metrics': cand_scores,
                }
                print(
                    f"BEST_OF_N {item['id']} n={n_candidates} winner_seed={best_seed} "
                    f"score={best_score} candidates={json.dumps(cand_scores)}",
                    flush=True,
                )
            else:
                res = backend._generate([seed], req)
                raw_audio, raw_sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used

        elif item['type'] == 'stem_layer':
            # Optimization C: 100% Sample-Locked Vertical Layering via HDemucs on L4 GPU
            # Prefer already-looped master so the stem layer is 100% sample-synchronized with the final loop!
            src_path = resolve_source_file(item['source_id'], prefer_looped=True)
            src_audio, src_sr = post.read(src_path)
            stem_weights = item.get('stem_weights') or {'drums': 1.0, 'bass': 1.0, 'other': 0.28, 'vocals': 0.0}
            other_lp = item.get('other_lowpass_hz', 2200.0)
            export_dir = OUT if item.get('export_all_stems', False) else None
            raw_audio, stem_info = run_hdemucs_stem_layer(
                src_audio,
                src_sr,
                stem_weights=stem_weights,
                other_lowpass_hz=other_lp,
                export_stems_dir=export_dir,
                stem_prefix=item['id'],
            )
            raw_sr = src_sr
            # A stem layer must never be re-looped/trimmed, or it loses sample-lock with the main loop
            post_cfg = {k: v for k, v in post_cfg.items() if k not in ('loop_crossfade_ms', 'target_loop_seconds', 'trim_silence', 'fade_in_ms', 'fade_out_ms')}
            extra_meta['hdemucs_stem_layer'] = stem_info
            if export_dir is not None:
                for st_name in (f"{item['id']}_stem_drums_bass.wav", f"{item['id']}_stem_guitars_synths.wav"):
                    st_p = OUT / st_name
                    if st_p.exists():
                        extra_files_to_pack.append(st_p)
            print(
                f"HDEMUCS_STEM_LAYER {item['id']} corr={stem_info['waveform_correlation_with_main_loop']} "
                f"onset_lag={stem_info['measured_onset_lag_ms']}ms samples={stem_info['layer_samples']}/{stem_info['source_samples']} "
                f"wall={stem_info['hdemucs_wall_seconds']}s",
                flush=True,
            )

        elif item['type'] == 'process':
            src_path = resolve_source_file(item['source_id'])
            raw_audio, raw_sr = post.read(src_path)
            post_cfg = {**item, **post_cfg}

        elif item['type'] == 'edit_continue':
            src_path = resolve_source_file(item['source_id'])
            src_audio, src_sr = post.read(src_path)
            end_sec = post.audible_end(src_audio, src_sr)
            if end_sec <= 0.1:
                end_sec = num_samples(src_audio) / src_sr
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
            raw_audio, raw_sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used
            extra_meta['inpaint_range'] = [round(max(0.0, end_sec - 1.0), 2), target_sec]

        elif item['type'] == 'edit_inpaint':
            src_path = resolve_source_file(item['source_id'])
            src_audio, src_sr = post.read(src_path)
            total_sec = float(item.get('duration_seconds', num_samples(src_audio) / src_sr))
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
            raw_audio, raw_sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used
            extra_meta['inpaint_range'] = [start_s, end_s]

        elif item['type'] == 'edit_variation':
            src_path = resolve_source_file(item['source_id'])
            src_audio, src_sr = post.read(src_path)
            total_sec = float(item.get('duration_seconds', num_samples(src_audio) / src_sr))
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
            raw_audio, raw_sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used
            extra_meta['init_strength'] = strength

        else:
            raise ValueError(f"Unknown item type: {item['type']}")

        audio, sr, post_info, qa_report = run_qa_and_auto_repair(
            backend, item, raw_audio, raw_sr, post_cfg, extra_files_to_pack
        )
        extra_meta['post_info'] = post_info
        extra_meta['qa_report'] = qa_report

        post.write(output_file, audio, sr, fmt)
        extra_meta['audio_stats'] = post.stats(audio)

        elapsed = time.monotonic() - t0
        save_artifact_pack(item, output_file, elapsed, extra_meta, extra_files_to_pack)

        gc.collect()
        torch.cuda.empty_cache()

    backend._unload()
    gc.collect()
    torch.cuda.empty_cache()
    print('MODEL_DONE', flush=True)


if __name__ == '__main__':
    main()
