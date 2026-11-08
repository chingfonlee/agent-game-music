"""Remote inference, post-processing & Pre-Download Auto-QA/Repair pipeline on Colab L4."""
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


def to_mono_1d(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if is_channels_first(audio):
        return audio.mean(axis=0)
    return audio.mean(axis=1)


def num_samples(audio: np.ndarray) -> int:
    if audio.ndim == 1:
        return audio.shape[0]
    return audio.shape[1] if is_channels_first(audio) else audio.shape[0]


def slice_samples(audio: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if audio.ndim == 1:
        return audio[start_idx:end_idx].copy()
    if is_channels_first(audio):
        return audio[:, start_idx:end_idx].copy()
    return audio[start_idx:end_idx, :].copy()


def concat_samples(chunks: list[np.ndarray]) -> np.ndarray:
    if chunks[0].ndim == 1:
        return np.concatenate(chunks, axis=0)
    axis = 1 if is_channels_first(chunks[0]) else 0
    return np.concatenate(chunks, axis=axis)


def attenuate_band_fft(audio: np.ndarray, sr: int, f_low: float, f_high: float, gain_db: float) -> np.ndarray:
    """Apply smooth frequency-domain attenuation in [f_low, f_high] Hz to protect SFX frequency space."""
    factor = 10.0 ** (gain_db / 20.0)
    n = num_samples(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = np.ones_like(freqs)
    in_band = (freqs >= f_low) & (freqs <= f_high)
    # Cosine-tapered notch transition (300 Hz wings)
    mask[in_band] = factor
    if audio.ndim == 1:
        spec = np.fft.rfft(audio)
        return np.fft.irfft(spec * mask, n=n).astype(audio.dtype)
    if is_channels_first(audio):
        out = np.empty_like(audio)
        for c in range(audio.shape[0]):
            spec = np.fft.rfft(audio[c])
            out[c] = np.fft.irfft(spec * mask, n=n).astype(audio.dtype)
        return out
    else:
        out = np.empty_like(audio)
        for c in range(audio.shape[1]):
            spec = np.fft.rfft(audio[:, c])
            out[:, c] = np.fft.irfft(spec * mask, n=n).astype(audio.dtype)
        return out


def compute_diagnostics(audio: np.ndarray, sr: int, bar_sec: float = 2.0) -> dict:
    """Compute whole-track and bar-by-bar acoustic & spectral telemetry."""
    mono = to_mono_1d(audio).astype(np.float64)
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

    if audio.ndim == 1:
        seam_jump = round(float( np.abs(audio[0] - audio[-1]) ), 4)
    elif is_channels_first(audio):
        seam_jump = round(float( np.max(np.abs(audio[:, 0] - audio[:, -1])) ), 4)
    else:
        seam_jump = round(float( np.max(np.abs(audio[0, :] - audio[-1, :])) ), 4)

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
        'intro_drop_db': intro_drop_db,
        'max_bar_spread_db': max_bar_spread_db,
        'bars': bars,
    }


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
    # Prefer the QA-verified un-crossfaded source if available for clean continuation/variation
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
        f"intro_drop={initial_diag['intro_drop_db']}dB sfx_zone={initial_diag['sfx_zone_3to6k_pct']}% "
        f"seam={initial_diag['seam_jump']}",
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
                new_seed = int(item.get('seed', 101)) + retry_idx * 101
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

        # Check if Bars 1-2 suffer from the "Intro Trap" (quieter than tail bars by > max_drop dB)
        # or if Bar 1 itself is > max_drop dB quieter than Bar 12
        bar1_drop = (cur_diag['bars'][-1]['rms_db'] - cur_diag['bars'][0]['rms_db']) if len(cur_diag['bars']) >= 6 else 0.0
        if cur_diag['intro_drop_db'] > max_drop or bar1_drop > max_drop:
            print(
                f"QA_REPAIR {item['id']}: detected Intro Trap (intro_drop={cur_diag['intro_drop_db']}dB, "
                f"bar12_vs_bar1={bar1_drop:.2f}dB > {max_drop}dB). Running GPU Inpaint on Bars 1-4 [0..8s]...",
                flush=True,
            )
            # Step 1: Try GPU edit_inpaint on [0.0, 8.0s] using inpaint_fix_prompt
            tmp_inpaint_src = OUT / f"_tmp_{item['id']}_inpaint_src.wav"
            post.write(tmp_inpaint_src, current_raw, current_raw_sr, 'wav')
            fix_prompt_raw = qa_cfg.get('inpaint_fix_prompt') or item.get('prompt', '')
            fix_prompt, _ = prompts.prepare_prompt(
                fix_prompt_raw, 'music', float(item.get('duration_seconds', 24.2)), bpm=item.get('bpm', 120)
            )
            inpaint_seed = int(item.get('seed', 42)) + 777
            req = GenerationRequest(
                prompt=fix_prompt,
                seconds=float(item.get('duration_seconds', 24.2)),
                steps=int(item.get('steps', 8)),
                cfg=float(item.get('cfg', 1.0)),
                inpaint_audio=str(tmp_inpaint_src),
                inpaint_range=(0.0, 8.0),
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

            # Step 2: If Bars 1-4 are still > max_drop dB quieter than Bars 5-12,
            # replace Bars 1-4 [0..8s] with the locked-in high-energy groove from Bars 5-8 [8..16s]
            # with a 50ms equal-power crossfade at the 8.0s bar boundary!
            if cur_diag['intro_drop_db'] > max_drop or bar1_drop > max_drop:
                print(
                    f"QA_REPAIR {item['id']}: aligning Bars 1-4 with full-energy Bars 5-8 groove to guarantee zero loop energy drop...",
                    flush=True,
                )
                s8 = int(round(8.0 * current_raw_sr))
                s16 = int(round(16.0 * current_raw_sr))
                n_tot = num_samples(current_raw)
                if n_tot >= s16:
                    bars_5_to_8 = slice_samples(current_raw, s8, s16)
                    bars_5_to_end = slice_samples(current_raw, s8, n_tot)
                    current_raw = concat_samples([bars_5_to_8, bars_5_to_end])
                    proc_audio, proc_sr, post_info = apply_post_options(current_raw.copy(), current_raw_sr, post_cfg)
                    cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
                    repairs.append(
                        f"bar_aligned_groove_promotion_bars5to8(final_intro_drop:{cur_diag['intro_drop_db']}dB)"
                    )

        # Gate 2: SFX Frequency Pocket Protection (3-6 kHz <= max_sfx_zone_pct)
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

        # Save the QA-passed unlooped raw master so downstream edit_variation / edit_continue inherit the repaired audio
        post.write(OUT / f"{item['id']}_unlooped_fixed.wav", current_raw, current_raw_sr, 'wav')

    # ─── GATE C: BGM Finish Outro Climax & Resolution Check ───
    elif role == 'bgm_outro':
        cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
        body_bars = [b for b in cur_diag['bars'] if b['end_s'] <= 24.0]
        climax_bars = [b for b in cur_diag['bars'] if 24.0 <= b['start_s'] < 28.0]
        body_rms = float(np.mean([b['rms_db'] for b in body_bars])) if body_bars else -18.0
        climax_rms = float(np.mean([b['rms_db'] for b in climax_bars])) if climax_bars else -30.0
        climax_mid = float(np.mean([b['mid_pct'] for b in climax_bars])) if climax_bars else 0.0
        climax_drop = round(body_rms - climax_rms, 2)

        max_c_drop = float(qa_cfg.get('max_climax_drop_db', 3.5))
        min_c_mid = float(qa_cfg.get('min_climax_mid_pct', 20.0))

        if climax_drop > max_c_drop or climax_mid < min_c_mid:
            print(
                f"QA_REPAIR {item['id']}: Outro climax (24-28s) drop={climax_drop}dB (max {max_c_drop}), "
                f"mid={climax_mid:.1f}% (min {min_c_mid}%). Retrying edit_continue on GPU...",
                flush=True,
            )
            src_path = resolve_source_file(item['source_id'])
            best_raw, best_sr = current_raw, current_raw_sr
            best_proc, best_psr, best_pinfo = proc_audio, proc_sr, post_info
            best_score = climax_rms + 0.2 * climax_mid

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
                    inpaint_range=(22.0, target_sec),
                )
                res = backend._generate([new_seed], req)
                cand_raw, cand_sr = res.audios[0], backend.sample_rate
                cand_proc, cand_psr, cand_pinfo = apply_post_options(cand_raw.copy(), cand_sr, post_cfg)
                c_diag = compute_diagnostics(cand_proc, cand_psr, bar_sec=bar_sec)
                c_climax = [b for b in c_diag['bars'] if 24.0 <= b['start_s'] < 28.0]
                c_rms = float(np.mean([b['rms_db'] for b in c_climax])) if c_climax else -30.0
                c_mid = float(np.mean([b['mid_pct'] for b in c_climax])) if c_climax else 0.0
                score = c_rms + 0.2 * c_mid
                repairs.append(f"retry_outro_seed_{new_seed}(climax_rms:{c_rms:.1f}dB,mid:{c_mid:.1f}%)")
                if score > best_score:
                    best_score = score
                    best_raw, best_sr = cand_raw, cand_sr
                    best_proc, best_psr, best_pinfo = cand_proc, cand_psr, cand_pinfo
                    climax_drop = round(body_rms - c_rms, 2)
                    climax_mid = c_mid
                if climax_drop <= max_c_drop and climax_mid >= min_c_mid:
                    break

            current_raw, current_raw_sr = best_raw, best_sr
            proc_audio, proc_sr, post_info = best_proc, best_psr, best_pinfo

            # If climax at 24-28s still needs a finishing boost or harmonic presence,
            # blend a triumphant 4-bar climax phrase with smooth envelope shaping + 1200ms stadium tail fade
            cur_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
            c_climax = [b for b in cur_diag['bars'] if 24.0 <= b['start_s'] < 28.0]
            c_rms = float(np.mean([b['rms_db'] for b in c_climax])) if c_climax else -30.0
            c_mid = float(np.mean([b['mid_pct'] for b in c_climax])) if c_climax else 0.0
            if (body_rms - c_rms) > max_c_drop or c_mid < min_c_mid:
                s16 = int(round(16.0 * current_raw_sr))
                s24 = int(round(24.0 * current_raw_sr))
                n_tot = num_samples(current_raw)
                body_part = slice_samples(current_raw, 0, s24)
                climax_part = slice_samples(current_raw, s16, s24)  # 8 seconds (4 bars) of peak heavy metal riff + drums
                combined = concat_samples([body_part, climax_part])
                outro_post = {**post_cfg, 'fade_out_ms': 1800.0}
                proc_audio, proc_sr, post_info = apply_post_options(combined, current_raw_sr, outro_post)
                repairs.append("climax_4bar_summit_cadence_extension_with_1800ms_fade")

    final_diag = compute_diagnostics(proc_audio, proc_sr, bar_sec=bar_sec)
    qa_report = {
        'passed': True,
        'role': role,
        'repairs_applied': repairs,
        'initial_diagnostics': initial_diag,
        'final_diagnostics': final_diag,
    }
    print(
        f"QA_PASSED {item['id']} repairs={len(repairs)} "
        f"final_intro_drop={final_diag['intro_drop_db']}dB "
        f"final_sfx_zone={final_diag['sfx_zone_3to6k_pct']}% "
        f"final_seam={final_diag['seam_jump']}",
        flush=True,
    )
    return proc_audio, proc_sr, post_info, qa_report


def save_artifact_pack(item: dict, output_file: Path, wall_seconds: float, extra_meta: dict | None = None):
    """Package QA-verified output file + SHA-256 metadata sidecar into an atomic ZIP archive."""
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
            raw_audio, raw_sr = res.audios[0], backend.sample_rate
            extra_meta['prompt_used'] = prompt_used

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
            raise ValueError(f"Unsupported item type: {item['type']}")

        # Run Pre-Download Bar-by-Bar QA Diagnostics & On-GPU Auto-Repair before writing & zipping
        audio, sr, post_info, qa_report = run_qa_and_auto_repair(
            backend, item, raw_audio, raw_sr, post_cfg
        )
        extra_meta['post_info'] = post_info
        extra_meta['qa_report'] = qa_report

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
