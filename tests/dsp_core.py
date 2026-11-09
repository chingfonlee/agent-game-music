"""Core DSP and analytical functions extracted for offline verification and zero-CU unit testing.

Pure NumPy / SciPy implementation with zero PyTorch or GPU dependencies.
"""

import re
import numpy as np


def is_channels_first(audio: np.ndarray) -> bool:
    if audio.ndim == 1:
        return True
    return audio.shape[0] in (1, 2) and audio.shape[0] < audio.shape[1]


def to_cf(audio: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    was_1d = audio.ndim == 1
    if was_1d:
        return audio[np.newaxis, :].astype(np.float32), True, True
    was_cf = is_channels_first(audio)
    if was_cf:
        return audio.astype(np.float32), False, True
    return audio.T.astype(np.float32), False, False


def num_samples(audio: np.ndarray) -> int:
    cf, _, _ = to_cf(audio)
    return cf.shape[1]


def peak_normalize(audio: np.ndarray, ceiling_db: float = -1.0) -> tuple[np.ndarray, float]:
    """Normalize peak amplitude to ceiling_db (safe fallback for ultra-short SFX < 3s)."""
    cf, was_1d, was_cf = to_cf(audio)
    target_peak = 10.0 ** (ceiling_db / 20.0)
    current_peak = float(np.max(np.abs(cf)))
    if current_peak < 1e-9:
        return audio, 0.0
    gain = target_peak / current_peak
    gain_db = float(20.0 * np.log10(gain))
    return from_cf((cf * gain).astype(cf.dtype), was_1d, was_cf), gain_db


def from_cf(cf_audio: np.ndarray, was_1d: bool, was_cf: bool) -> np.ndarray:
    if was_1d:
        return cf_audio[0]
    if was_cf:
        return cf_audio
    return cf_audio.T


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
    """Smart onset + Chroma SSM + Zero-Crossing loop window alignment."""
    cf, was_1d, was_cf = to_cf(audio)
    mono = cf.mean(axis=0)
    n_total = cf.shape[1]
    n_needed = int(round((target_loop_sec + crossfade_ms / 1000.0) * sr))
    if n_total <= n_needed:
        return audio, {'aligned_start_ms': 0.0, 'seam_chroma_similarity': 1.0, 'zero_crossing_snapped': False}

    max_slack = n_total - n_needed
    # 1. Detect true first rhythmic onset within available slack
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

    zc_start = find_zero_crossing(mono, best_start, radius=min(int(sr * 0.015), max_slack - best_start if max_slack > best_start else 1))
    zc_start = max(0, min(max_slack, zc_start))
    sliced_cf = cf[:, zc_start:zc_start + n_needed].copy()

    align_info = {
        'aligned_start_ms': round(1000.0 * zc_start / sr, 1),
        'seam_chroma_similarity': round(best_chroma_sim, 4),
        'zero_crossing_snapped': True,
    }
    return from_cf(sliced_cf, was_1d, was_cf), align_info


def measure_onset_lag_ms(ref_mono: np.ndarray, test_mono: np.ndarray, sr: int, max_lag_ms: float = 50.0) -> float:
    """Measure delay between reference track and layer track using normalized cross-correlation of onset envelopes."""
    hop = max(1, int(sr * 0.002))  # 2 ms hop
    n_frames = min(len(ref_mono), len(test_mono)) // hop
    if n_frames < 20:
        return 0.0
    ref_env = np.array([np.sqrt(np.mean(ref_mono[i * hop:(i + 2) * hop] ** 2) + 1e-12) for i in range(n_frames - 1)])
    test_env = np.array([np.sqrt(np.mean(test_mono[i * hop:(i + 2) * hop] ** 2) + 1e-12) for i in range(n_frames - 1)])
    ref_diff = np.maximum(0.0, np.diff(ref_env))
    test_diff = np.maximum(0.0, np.diff(test_env))
    ref_std = np.std(ref_diff)
    test_std = np.std(test_diff)
    if ref_std < 1e-6 or test_std < 1e-6:
        return 0.0
    r_norm = (ref_diff - np.mean(ref_diff)) / ref_std
    t_norm = (test_diff - np.mean(test_diff)) / test_std
    max_lag_frames = int((max_lag_ms / 1000.0) / (hop / sr))
    corr = np.correlate(t_norm, r_norm, mode='full')
    center = len(r_norm) - 1
    lo = max(0, center - max_lag_frames)
    hi = min(len(corr), center + max_lag_frames + 1)
    best_idx = lo + int(np.argmax(corr[lo:hi]))
    lag_frames = best_idx - center
    return round(float(lag_frames * (hop / sr) * 1000.0), 2)


def compute_diagnostics(audio: np.ndarray, sr: int, bar_sec: float = 2.0) -> dict:
    """Compute comprehensive quantitative audio diagnostics including bar-by-bar dynamics and spectral energy."""
    cf, _, _ = to_cf(audio)
    mono = cf.mean(axis=0)
    dur = len(mono) / sr
    rms_overall = float(20.0 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9))
    peak_overall = float(20.0 * np.log10(np.max(np.abs(mono)) + 1e-9))

    # Spectral balance
    n_fft = min(len(mono), int(sr * 4.0))
    if n_fft >= 256:
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        psd = np.abs(np.fft.rfft(mono[:n_fft])) ** 2
        total_p = float(np.sum(psd) + 1e-12)
        low_pct = float(np.sum(psd[freqs < 250.0]) / total_p * 100.0)
        mid_pct = float(np.sum(psd[(freqs >= 250.0) & (freqs < 3000.0)]) / total_p * 100.0)
        sfx_pct = float(np.sum(psd[(freqs >= 3000.0) & (freqs <= 6000.0)]) / total_p * 100.0)
    else:
        low_pct, mid_pct, sfx_pct = 33.3, 33.3, 33.3

    # Seam continuity check: maximum boundary sample jump between first and last sample
    seam_jump = round(float(np.max(np.abs(cf[:, 0] - cf[:, -1]))), 4)

    # Bar dynamics
    samples_per_bar = max(1, int(round(bar_sec * sr)))
    n_bars = max(1, int(len(mono) // samples_per_bar))
    bars = []
    bar_rms_list = []
    for b in range(n_bars):
        s = b * samples_per_bar
        e = min(len(mono), s + samples_per_bar)
        bar_audio = mono[s:e]
        b_rms = float(20.0 * np.log10(np.sqrt(np.mean(bar_audio ** 2)) + 1e-9))
        bar_rms_list.append(b_rms)
        # Bar mid & sfx pct
        if len(bar_audio) >= 256:
            bfreqs = np.fft.rfftfreq(len(bar_audio), d=1.0 / sr)
            bpsd = np.abs(np.fft.rfft(bar_audio)) ** 2
            btotal = float(np.sum(bpsd) + 1e-12)
            b_mid = float(np.sum(bpsd[(bfreqs >= 250.0) & (bfreqs < 3000.0)]) / btotal * 100.0)
            b_sfx = float(np.sum(bpsd[(bfreqs >= 3000.0) & (bfreqs <= 6000.0)]) / btotal * 100.0)
        else:
            b_mid, b_sfx = 0.0, 0.0
        bars.append({
            'bar': b + 1,
            'start_s': round(s / sr, 2),
            'end_s': round(e / sr, 2),
            'rms_db': round(b_rms, 2),
            'mid_pct': round(b_mid, 1),
            'sfx_zone_pct': round(b_sfx, 1),
        })

    intro_drop_db = 0.0
    if len(bar_rms_list) >= 4:
        intro_drop_db = float(np.mean(bar_rms_list[-4:]) - np.mean(bar_rms_list[:2]))
    max_bar_spread_db = float(max(bar_rms_list) - min(bar_rms_list)) if bar_rms_list else 0.0

    return {
        'duration_seconds': round(dur, 2),
        'overall_rms_db': round(rms_overall, 2),
        'overall_peak_db': round(peak_overall, 2),
        'low_lt250_pct': round(low_pct, 2),
        'mid_250to3k_pct': round(mid_pct, 2),
        'sfx_zone_3to6k_pct': round(sfx_pct, 2),
        'seam_jump': round(seam_jump, 4),
        'intro_drop_db': round(intro_drop_db, 2),
        'max_bar_spread_db': round(max_bar_spread_db, 2),
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
            + 12.0 * diag.get('bar_chroma_similarity', 0.8)
            + 1.5 * diag.get('onset_clarity_db', 5.0)
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
        return round(float(100.0 + diag['overall_rms_db'] + 2.0 * diag.get('onset_clarity_db', 5.0)), 2)


def calculate_loop_math(bpm: float, bars: int, crossfade_ms: float = 200.0, slack_sec: float = 0.4) -> dict:
    """Calculate exact bar timing, seamless loop length, and candidate generation length."""
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")
    bar_seconds = 240.0 / bpm
    target_loop_seconds = bars * bar_seconds
    min_generation_seconds = target_loop_seconds + (crossfade_ms / 1000.0) + slack_sec
    return {
        'bpm': bpm,
        'bars': bars,
        'bar_seconds': round(bar_seconds, 4),
        'target_loop_seconds': round(target_loop_seconds, 4),
        'duration_seconds': round(min_generation_seconds, 4),
        'crossfade_ms': crossfade_ms,
    }


def sanitize_prompt_for_instrumental(prompt: str) -> tuple[str, list[str]]:
    """Sanitize prompt to remove prohibited vocal triggers and ensure 100% instrumental delivery."""
    vocal_keywords = [
        r'\bvocals?\b',
        r'\bsingers?\b',
        r'\blyrics?\b',
        r'\bchoirs?\b',
        r'\bvoices?\b',
        r'\bscream(?:ing)?\b',
        r'\bspoken\b',
        r'\bspeech\b',
        r'\bchant(?:ing)?\b',
    ]
    detected = []
    clean = prompt
    for pat in vocal_keywords:
        found = re.findall(pat, clean, flags=re.IGNORECASE)
        if found:
            detected.extend(found)
            clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
    # Clean multiple spaces and commas
    clean = re.sub(r'\s+', ' ', clean)
    clean = re.sub(r'\s*,\s*,', ',', clean).strip(' ,')
    return clean, detected
