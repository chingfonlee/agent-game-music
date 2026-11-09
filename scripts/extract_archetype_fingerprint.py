#!/usr/bin/env python3
"""
extract_archetype_fingerprint.py
================================================================================
agent-game-music — 離線聲紋特徵提取、三軸正交門檻合成與閉環調參引擎 (Zero-CU / Pure CPU)
Reference Repository: https://github.com/chingfonlee/agent-game-music

功能總覽 (v2.1 — 已強化 24-bit PCM、Windows 跨平台暫存檔與 outputs/ 角色分流過濾)：
1. 【真實音訊聲紋提取 (Reference & Golden Track Ingestion)】：
   - 完整支援 8/16/24/32-bit PCM WAV（優先使用 soundfile，並內建純 NumPy 24-bit 3-byte 符號擴展解碼 Fallback，零系統 C 庫依賴也能讀 24-bit 母帶）。
   - 掃描 outputs/ 資料夾時自動依檔名角色分流（預設僅聚合 BGM 主循環 *bgm*loop*.wav，自動排除 01_sfx_* 與 2200Hz 低通濾波後的 05_*pause* 分軌，防止統計污染）。
2. 【前景 SFX 主頻段自動偵測 (Dynamic SFX Collision Detector)】：
   - 直接分析同任務生成的 SFX .wav，找出能量最集中的頻段峰值，自動覆寫 BGM 的 Notch 避讓中心頻率。
3. 【三軸正交門檻合成器 (3-Axis Profile Resolver)】：
   - 讀取 composable_archetypes.yaml，將 (timbre_base × state_envelope × sfx_mask) 合成為單一 QA Profile。
4. 【第四層閉環處方籤生成 (Closed-Loop Remediation Advisor)】：
   - 比對實測聲紋、三軸門檻與人工試聽標籤 (如 fatigue_risk:high)，輸出 Stage 2/3/4 確定性修正指令。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import tempfile
import wave
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import soundfile as sf  # type: ignore
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False


# ------------------------------------------------------------------------------
# 資料結構定義
# ------------------------------------------------------------------------------
@dataclass
class TrackDiagnostics:
    file_path: str
    detected_role: str
    sample_rate: int
    bit_depth: int
    duration_sec: float
    bpm: float
    bars: int
    # 全域頻譜能量佔比 (%)
    low_lt250_hz_pct: float
    mid_250to3k_hz_pct: float
    presence_3to6k_hz_pct: float
    air_gt6k_hz_pct: float
    # 小節滑動視窗高頻峰值 (防止前後半首失衡被平均值掩蓋)
    bar_window_3to6k_max_pct: float
    bar_window_3to6k_std_pct: float
    spectral_centroid_hz: float
    spectral_flatness: float
    # 動態與時間包絡指標
    crest_factor_db: float
    intro_drop_db: float
    max_bar_spread_db: float
    bar_chroma_similarity: float
    seam_jump: float
    tail_last_100ms_rms_dbfs: float
    transient_density_per_sec: float
    climax_relative_mid_ratio: float
    dominant_sfx_peak_hz: float


# ------------------------------------------------------------------------------
# 1. 角色識別與跨平台 16/24/32-bit WAV 讀取工具
# ------------------------------------------------------------------------------
def classify_cue_role(wav_path: pathlib.Path) -> str:
    """
    依據 agent-game-music 命名慣例自動識別音軌角色，防止掃描 outputs/ 時把 SFX 或低通 Pause 軌混入 BGM 統計。
    - 'sfx': 01_sfx_countdown.wav 等音效檔
    - 'pause_stem': 05_bgm_pause_swap_stem_layer.wav 等經低通濾波的分層軌
    - 'outro': 04_bgm_finish_outro.wav 等終止式
    - 'bgm_loop': 02_bgm_main_loop_A.wav 等主循環 BGM
    """
    name = wav_path.stem.lower()
    if "sfx" in name:
        return "sfx"
    if "pause" in name or "stem" in name:
        return "pause_stem"
    if "outro" in name or "stinger" in name or "finish" in name:
        return "outro"
    return "bgm_loop"


def _decode_pcm_bytes(raw: bytes, sampwidth: int) -> np.ndarray:
    """純 NumPy 解碼 8/16/24/32-bit PCM bytes 為 [-1.0, 1.0] float32 陣列。"""
    if sampwidth == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sampwidth == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sampwidth == 3:
        # 24-bit Little-Endian PCM 向量化符號擴展 (Sign-extension to int32)
        raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed_int32 = (
            raw_bytes[:, 0].astype(np.int32)
            | (raw_bytes[:, 1].astype(np.int32) << 8)
            | (raw_bytes[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        return signed_int32.astype(np.float32) / 8388608.0
    if sampwidth == 4:
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sampwidth} bytes")


def load_wav_mono(path: pathlib.Path) -> Tuple[np.ndarray, int, int]:
    """
    讀取 8/16/24/32-bit WAV 並轉為 [-1.0, 1.0] float32 單聲道。
    雙重保障：先透過 wave 取得精確 bit_depth 與 PCM bytes（原生支援 24-bit sampwidth==3）；
    若遇非標準 WAV header (如 IEEE Float)，自動回退至 soundfile.read()。
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        data = _decode_pcm_bytes(raw, sampwidth)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, sr, sampwidth * 8
    except wave.Error:
        if _HAS_SOUNDFILE:
            info = sf.info(str(path))
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            bit_depth = 24 if "24" in str(info.subtype) else (16 if "16" in str(info.subtype) else 32)
            return mono, int(sr), bit_depth
        raise


def compute_band_energy_ratios(signal: np.ndarray, sr: int) -> Dict[str, float]:
    """利用漢寧窗與 rFFT 計算四個核心頻段的功率佔比 (%)、頻譜質心與平坦度。"""
    if len(signal) == 0:
        return {
            "low_lt250_hz_pct": 0.0,
            "mid_250to3k_hz_pct": 0.0,
            "presence_3to6k_hz_pct": 0.0,
            "air_gt6k_hz_pct": 0.0,
            "centroid_hz": 0.0,
            "flatness": 0.0,
            "peak_sfx_hz": 4000.0,
        }

    windowed = signal * np.hanning(len(signal))
    spec = np.fft.rfft(windowed)
    power = np.abs(spec) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)

    total_power = float(np.sum(power))
    low_mask = freqs < 250.0
    mid_mask = (freqs >= 250.0) & (freqs < 3000.0)
    pres_mask = (freqs >= 3000.0) & (freqs <= 6000.0)
    air_mask = freqs > 6000.0

    # 計算 800Hz ~ 8000Hz 內最突出的窄頻峰值 (供 SFX 主頻段偵測使用)
    sfx_search_mask = (freqs >= 800.0) & (freqs <= 8000.0)
    if np.any(sfx_search_mask):
        sub_freqs = freqs[sfx_search_mask]
        sub_power = power[sfx_search_mask]
        peak_sfx_hz = float(sub_freqs[int(np.argmax(sub_power))])
    else:
        peak_sfx_hz = 4000.0

    centroid = float(np.sum(freqs * power) / total_power)
    geometric_mean = float(np.exp(np.mean(np.log(power))))
    arithmetic_mean = float(np.mean(power))
    flatness = geometric_mean / max(arithmetic_mean, 1e-12)

    return {
        "low_lt250_hz_pct": round(float(np.sum(power[low_mask]) / total_power * 100.0), 2),
        "mid_250to3k_hz_pct": round(float(np.sum(power[mid_mask]) / total_power * 100.0), 2),
        "presence_3to6k_hz_pct": round(float(np.sum(power[pres_mask]) / total_power * 100.0), 2),
        "air_gt6k_hz_pct": round(float(np.sum(power[air_mask]) / total_power * 100.0), 2),
        "centroid_hz": round(centroid, 1),
        "flatness": round(flatness, 4),
        "peak_sfx_hz": round(peak_sfx_hz, 1),
    }


def compute_12bin_chroma(segment: np.ndarray, sr: int) -> np.ndarray:
    """純 NumPy 計算 12 維度色度向量 (Chroma Vector)，用於無鼓點氛圍音樂的循環相似度檢驗。"""
    if len(segment) < 256:
        return np.ones(12, dtype=np.float32) / np.sqrt(12)
    spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    valid = (freqs >= 65.0) & (freqs <= 2000.0)
    freqs = freqs[valid]
    spec = spec[valid]
    chroma = np.zeros(12, dtype=np.float64)
    if len(freqs) > 0:
        midi_notes = 69.0 + 12.0 * np.log2(freqs / 440.0)
        pitch_classes = np.round(midi_notes).astype(int) % 12
        for pc in range(12):
            chroma[pc] = np.sum(spec[pitch_classes == pc])
    norm = np.linalg.norm(chroma)
    return (chroma / norm) if norm > 1e-9 else (np.ones(12) / np.sqrt(12))


def analyze_audio_file(
    wav_path: pathlib.Path,
    bpm: float = 120.0,
    beats_per_bar: int = 4,
) -> TrackDiagnostics:
    """對單一音訊檔執行全方位聲學診斷（含 24-bit 支援與逐小節滑動視窗分析）。"""
    audio, sr, bit_depth = load_wav_mono(wav_path)
    detected_role = classify_cue_role(wav_path)
    duration_sec = len(audio) / float(sr)
    bar_sec = (beats_per_bar * 60.0) / bpm
    samples_per_bar = max(1, int(round(bar_sec * sr)))
    bars = max(1, int(len(audio) // samples_per_bar))

    global_spec = compute_band_energy_ratios(audio, sr)

    # 逐小節能量與 3-6kHz 峰值追蹤
    bar_rms_db_list: List[float] = []
    bar_3to6k_list: List[float] = []
    bar_mid_rms_list: List[float] = []
    bar_chromas: List[np.ndarray] = []

    for b in range(bars):
        seg = audio[b * samples_per_bar : (b + 1) * samples_per_bar]
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
        bar_rms_db_list.append(20.0 * math.log10(rms))
        b_spec = compute_band_energy_ratios(seg, sr)
        bar_3to6k_list.append(b_spec["presence_3to6k_hz_pct"])
        bar_mid_rms_list.append(rms * (b_spec["mid_250to3k_hz_pct"] / 100.0))
        bar_chromas.append(compute_12bin_chroma(seg, sr))

    # 前奏陷阱 (Intro-Trap) 計算：前 25% 小節 vs 後 75% 小節 dB 落差
    split_bar = max(1, bars // 4)
    intro_mean_db = float(np.mean(bar_rms_db_list[:split_bar]))
    body_mean_db = float(np.mean(bar_rms_db_list[split_bar:])) if bars > 1 else intro_mean_db
    intro_drop_db = round(max(0.0, body_mean_db - intro_mean_db), 2)

    # 終止式高潮中頻爆發比 (最後 25% 小節 vs 主體小節)
    climax_start = max(1, int(bars * 0.75))
    body_mid = float(np.mean(bar_mid_rms_list[:climax_start])) if climax_start > 0 else 1e-6
    climax_mid = float(np.mean(bar_mid_rms_list[climax_start:])) if climax_start < bars else body_mid
    climax_ratio = round(climax_mid / max(body_mid, 1e-6), 3)

    # 首尾小節 Chroma 餘弦相似度
    if len(bar_chromas) >= 2:
        chroma_sim = round(float(np.dot(bar_chromas[0], bar_chromas[-1])), 3)
    else:
        chroma_sim = 1.0

    # Crest Factor (Peak dB - RMS dB)
    peak_amp = float(np.max(np.abs(audio)) + 1e-12)
    global_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    crest_factor_db = round(20.0 * math.log10(peak_amp / global_rms), 2)

    # Loop 接縫振幅跳變與結尾 100ms 尾音 RMS
    seam_jump = round(float(abs(audio[0] - audio[-1])), 5) if len(audio) > 1 else 0.0
    tail_samples = max(1, int(0.1 * sr))
    tail_rms = float(np.sqrt(np.mean(audio[-tail_samples:] ** 2) + 1e-12))
    tail_rms_dbfs = round(20.0 * math.log10(tail_rms), 2)

    # 瞬態密度 (每 20ms 能量包絡差分超過閾值的起音次數 / 秒)
    hop = max(1, int(0.02 * sr))
    frames = len(audio) // hop
    if frames > 2:
        env = np.array([np.sqrt(np.mean(audio[i * hop : (i + 1) * hop] ** 2)) for i in range(frames)])
        flux = np.maximum(0.0, np.diff(env))
        thresh = max(float(np.median(flux) * 2.2 + np.std(flux) * 0.8), 1e-4)
        peaks = np.where((flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > thresh))[0]
        transient_density = round(len(peaks) / max(duration_sec, 0.1), 2)
    else:
        transient_density = 0.0

    return TrackDiagnostics(
        file_path=str(wav_path),
        detected_role=detected_role,
        sample_rate=sr,
        bit_depth=bit_depth,
        duration_sec=round(duration_sec, 3),
        bpm=bpm,
        bars=bars,
        low_lt250_hz_pct=global_spec["low_lt250_hz_pct"],
        mid_250to3k_hz_pct=global_spec["mid_250to3k_hz_pct"],
        presence_3to6k_hz_pct=global_spec["presence_3to6k_hz_pct"],
        air_gt6k_hz_pct=global_spec["air_gt6k_hz_pct"],
        bar_window_3to6k_max_pct=round(float(np.max(bar_3to6k_list)), 2),
        bar_window_3to6k_std_pct=round(float(np.std(bar_3to6k_list)), 2),
        spectral_centroid_hz=global_spec["centroid_hz"],
        spectral_flatness=global_spec["flatness"],
        crest_factor_db=crest_factor_db,
        intro_drop_db=intro_drop_db,
        max_bar_spread_db=round(float(np.max(bar_rms_db_list) - np.min(bar_rms_db_list)), 2),
        bar_chroma_similarity=chroma_sim,
        seam_jump=seam_jump,
        tail_last_100ms_rms_dbfs=tail_rms_dbfs,
        transient_density_per_sec=transient_density,
        climax_relative_mid_ratio=climax_ratio,
        dominant_sfx_peak_hz=global_spec["peak_sfx_hz"],
    )


# ------------------------------------------------------------------------------
# 2. 多首標竿音訊統計聚合器 (支援依角色分流，避免 SFX 與 Pause 軌污染 BGM 指紋)
# ------------------------------------------------------------------------------
def aggregate_reference_fingerprints(
    diagnostics_list: List[TrackDiagnostics],
    role_filter: str = "bgm_loop",
) -> Dict[str, Any]:
    """
    將多首標竿音訊的診斷結果聚合為 p10 / p50 / p90 統計指紋。
    預設僅統計 role_filter=='bgm_loop' 的主循環音軌，避免 01_sfx 或 05_pause_stem 扭曲分佈。
    """
    if role_filter != "all":
        filtered = [d for d in diagnostics_list if d.detected_role == role_filter]
    else:
        filtered = list(diagnostics_list)

    if not filtered:
        return {
            "role_filter_applied": role_filter,
            "sample_count": 0,
            "warning": f"No tracks matched role_filter='{role_filter}'. Use --role-filter all to override.",
        }

    def pct_stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        return {
            "p10": round(float(np.percentile(arr, 10)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
        }

    return {
        "role_filter_applied": role_filter,
        "sample_count": len(filtered),
        "included_files": [pathlib.Path(d.file_path).name for d in filtered],
        "spectral_bands_pct": {
            "low_lt250_hz": pct_stats([d.low_lt250_hz_pct for d in filtered]),
            "mid_250to3k_hz": pct_stats([d.mid_250to3k_hz_pct for d in filtered]),
            "presence_3to6k_hz": pct_stats([d.presence_3to6k_hz_pct for d in filtered]),
            "air_gt6k_hz": pct_stats([d.air_gt6k_hz_pct for d in filtered]),
        },
        "spectral_centroid_hz": pct_stats([d.spectral_centroid_hz for d in filtered]),
        "crest_factor_db": pct_stats([d.crest_factor_db for d in filtered]),
        "transient_density_per_sec": pct_stats([d.transient_density_per_sec for d in filtered]),
    }


# ------------------------------------------------------------------------------
# 3. 三軸正交門檻合成器 (Composable 3-Axis Profile Resolver)
# ------------------------------------------------------------------------------
def resolve_composable_profile(
    config: Dict[str, Any],
    timbre_key: str,
    state_key: str,
    sfx_mask_key: str,
    measured_sfx_wav_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    將 (軸1: 音色基底) × (軸2: 遊戲狀態包絡) × (軸3: 前景 SFX 遮罩) 動態合成為最終 QA 執行標準。
    若提供同任務生成的 measured_sfx_wav_path，將自動測定 SFX 峰值頻率並覆寫 Notch 中心頻率！
    """
    timbre = config["timbre_bases"][timbre_key]
    state = config["state_envelopes"][state_key]
    sfx_mask = config["sfx_collision_masks"][sfx_mask_key]

    natural_3to6k_p90 = float(timbre["spectral_bands_pct"]["presence_3to6k_hz"]["p90"])
    attenuation_ratio = float(sfx_mask["attenuation_ratio"])
    hard_ceiling = float(sfx_mask["hard_ceiling_pct"])

    # 核心公式：兼顧音色自然泛音上限與 SFX 避讓天花板
    effective_pocket_max_pct = round(min(natural_3to6k_p90 * attenuation_ratio, hard_ceiling), 2)
    # 保護上限：防止對高泛音音色 (如 Chiptune/鋼琴) 下過重的 Notch EQ
    effective_notch_cut_db = round(
        max(float(timbre["max_safe_notch_cut_db"]), float(sfx_mask["target_cut_db"])), 2
    )

    notch_center_hz = float(sfx_mask["notch_center_hz"])
    sfx_source_mode = "static_mask_preset"
    if measured_sfx_wav_path is not None and measured_sfx_wav_path.exists():
        sfx_diag = analyze_audio_file(measured_sfx_wav_path)
        notch_center_hz = sfx_diag.dominant_sfx_peak_hz
        sfx_source_mode = f"dynamic_measured_from_{measured_sfx_wav_path.name}"

    return {
        "composed_id": f"{timbre_key}__{state_key}__{sfx_mask_key}",
        "timbre_base": timbre_key,
        "state_envelope": state_key,
        "sfx_collision_mask": sfx_mask_key,
        "sfx_Notch_routing_mode": sfx_source_mode,
        "gates": {
            "low_lt250_hz": timbre["spectral_bands_pct"]["low_lt250_hz"],
            "mid_250to3k_hz": timbre["spectral_bands_pct"]["mid_250to3k_hz"],
            "presence_3to6k_hz": timbre["spectral_bands_pct"]["presence_3to6k_hz"],
            "effective_pocket_max_pct": effective_pocket_max_pct,
            "max_intro_drop_db": state["max_intro_drop_db"],
            "crest_factor_db": state["crest_factor_db"],
            "max_bar_spread_db": state["max_bar_spread_db"],
            "min_bar_chroma_similarity": state["min_bar_chroma_similarity"],
            "max_seam_jump": state["max_seam_jump"],
            "require_climax_mid_boost": state.get("require_climax_mid_boost", False),
            "min_climax_relative_mid_ratio": state.get("min_climax_relative_mid_ratio", 1.0),
        },
        "dsp_repair_parameters": {
            "notch_center_hz": notch_center_hz,
            "notch_q_factor": sfx_mask["notch_q_factor"],
            "effective_notch_cut_db": effective_notch_cut_db,
            "other_stem_highpass_hz": timbre["stem_separation_hints"]["other_highpass_hz"],
            "onset_align_mode": state.get("onset_align_mode", "transient_first"),
        },
    }


# ------------------------------------------------------------------------------
# 4. 第四層閉環評估與自動修正建議器 (Closed-Loop Remediation Advisor)
# ------------------------------------------------------------------------------
def evaluate_and_recommend_remediation(
    diag: TrackDiagnostics,
    resolved_profile: Dict[str, Any],
    config: Dict[str, Any],
    listening_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """比對音訊實測值、三軸門檻與人工試聽標籤，產出 PASS/FAIL 與 Stage 2/3/4 自動修正處方。"""
    listening_tags = listening_tags or []
    gates = resolved_profile["gates"]
    violations: List[str] = []
    triggered_remediations: Dict[str, Any] = {}

    # 1. 檢查前奏能量落差
    if diag.intro_drop_db > gates["max_intro_drop_db"]:
        violations.append(
            f"intro_drop_db ({diag.intro_drop_db} dB) > gate ({gates['max_intro_drop_db']} dB)"
        )

    # 2. 檢查全域與逐小節滑動視窗 SFX 負空間
    if diag.presence_3to6k_hz_pct > gates["effective_pocket_max_pct"]:
        violations.append(
            f"presence_3to6k_hz_pct ({diag.presence_3to6k_hz_pct}%) > gate ({gates['effective_pocket_max_pct']}%)"
        )
    if diag.bar_window_3to6k_max_pct > gates["effective_pocket_max_pct"] * 1.25:
        violations.append(
            f"bar_window_3to6k_max_pct ({diag.bar_window_3to6k_max_pct}%) exceeds windowed ceiling "
            f"({round(gates['effective_pocket_max_pct'] * 1.25, 2)}%)"
        )

    # 3. 檢查動態峰均比 (過壓縮或動態過大)
    cf_p10 = gates["crest_factor_db"]["p10"]
    if diag.crest_factor_db < cf_p10:
        violations.append(
            f"crest_factor_db ({diag.crest_factor_db} dB) < p10 ({cf_p10} dB) [Over-compressed]"
        )

    # 4. 檢查低頻混濁度
    low_p90 = gates["low_lt250_hz"]["p90"]
    if diag.low_lt250_hz_pct > low_p90:
        violations.append(
            f"low_lt250_hz_pct ({diag.low_lt250_hz_pct}%) > p90 ({low_p90}%) [Bass muddy risk]"
        )

    # 5. 檢查循環接縫與和聲一致性
    if diag.seam_jump > gates["max_seam_jump"]:
        violations.append(f"seam_jump ({diag.seam_jump}) > gate ({gates['max_seam_jump']})")
    if diag.bar_chroma_similarity < gates["min_bar_chroma_similarity"]:
        violations.append(
            f"bar_chroma_similarity ({diag.bar_chroma_similarity}) < gate ({gates['min_bar_chroma_similarity']})"
        )

    # 6. 映射至第四層閉環處方籤 (remediation_mappings)
    rem_map = config.get("remediation_mappings", {})
    if (
        "fatigue_risk:high" in listening_tags
        or diag.crest_factor_db < cf_p10
        or diag.bar_window_3to6k_max_pct > gates["effective_pocket_max_pct"] * 1.25
    ):
        triggered_remediations["fatigue_or_overcompressed"] = rem_map.get("fatigue_or_overcompressed", {})

    if "bass_muddy:true" in listening_tags or diag.low_lt250_hz_pct > low_p90:
        triggered_remediations["bass_muddy_or_boomy"] = rem_map.get("bass_muddy_or_boomy", {})

    if (
        "sfx_masked:true" in listening_tags
        or diag.presence_3to6k_hz_pct > gates["effective_pocket_max_pct"]
    ):
        triggered_remediations["sfx_masked_in_gameplay"] = rem_map.get("sfx_masked_in_gameplay", {})

    if (
        "loop_felt_abrupt:true" in listening_tags
        or diag.seam_jump > gates["max_seam_jump"]
        or diag.bar_chroma_similarity < gates["min_bar_chroma_similarity"]
    ):
        triggered_remediations["loop_seam_abrupt"] = rem_map.get("loop_seam_abrupt", {})

    return {
        "status": "PASSED" if not violations and not triggered_remediations else "NEEDS_REMEDIATION",
        "composed_profile_id": resolved_profile["composed_id"],
        "violations": violations,
        "listening_tags_applied": listening_tags,
        "closed_loop_remediation_plan": triggered_remediations,
    }


# ------------------------------------------------------------------------------
# 5. CLI 入口與內建 24-Bit 合成音訊跨平台自檢模式 (--self-test)
# ------------------------------------------------------------------------------
def _generate_synthetic_24bit_wav(path: pathlib.Path, sr: int = 44100, duration: float = 8.0) -> None:
    """生成一段 24-bit PCM (sampwidth==3) 合成測試 WAV，驗證 24-bit 母帶相容性與跨平台運作。"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (
        0.45 * np.sin(2 * np.pi * 110.0 * t)
        + 0.30 * np.sin(2 * np.pi * 440.0 * t)
        + 0.35 * np.sin(2 * np.pi * 4200.0 * t)
    )
    beat_env = 1.0 + 0.35 * np.maximum(0.0, np.sin(2 * np.pi * 2.0 * t)) ** 8
    sig = np.clip(sig * beat_env, -0.95, 0.95)
    sig[0] = 0.0
    sig[-1] = 0.0

    # 編碼為 24-bit Little-Endian (3 bytes per sample)
    int24 = np.clip(np.round(sig * 8388607.0), -8388608, 8388607).astype(np.int32)
    b0 = (int24 & 0xFF).astype(np.uint8)
    b1 = ((int24 >> 8) & 0xFF).astype(np.uint8)
    b2 = ((int24 >> 16) & 0xFF).astype(np.uint8)
    pcm24_bytes = np.stack([b0, b1, b2], axis=1).tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)  # 24-bit PCM
        wf.setframerate(sr)
        wf.writeframes(pcm24_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="agent-game-music 三軸正交聲紋提取器與閉環調參引擎 (v2.1 支援 24-bit PCM & 角色過濾)"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "configs" / "composable_archetypes.yaml",
        help="composable_archetypes.yaml 路徑",
    )
    parser.add_argument("--wav", type=pathlib.Path, help="要分析的單一 .wav 檔案或資料夾路徑")
    parser.add_argument("--sfx-wav", type=pathlib.Path, help="同任務的前景 SFX .wav（用於自動對準 Notch 中心頻率）")
    parser.add_argument("--bpm", type=float, default=140.0, help="音訊 BPM (預設 140)")
    parser.add_argument("--timbre", type=str, default="chiptune_pulse_fm", help="軸1：音色基底 ID")
    parser.add_argument("--state", type=str, default="state_combat_driving_loop", help="軸2：遊戲狀態包絡 ID")
    parser.add_argument("--sfx-mask", type=str, default="sfx_retro_coin_jump", help="軸3：前景 SFX 碰撞遮罩 ID")
    parser.add_argument(
        "--role-filter",
        type=str,
        default="bgm_loop",
        choices=["bgm_loop", "sfx", "outro", "pause_stem", "all"],
        help="掃描資料夾時僅聚合指定角色的音軌（預設 bgm_loop，避免 SFX 或低通 Pause 軌污染 BGM 統計）",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="附加實機試聽標籤，如 --tag fatigue_risk:high --tag sfx_masked:true",
    )
    parser.add_argument("--self-test", action="store_true", help="執行 0-CU 24-bit 合成音訊跨平台自檢")
    args = parser.parse_args()

    if not args.config.exists():
        # Fallback search
        candidate = pathlib.Path("configs/composable_archetypes.yaml")
        if candidate.exists():
            args.config = candidate

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.self_test:
        tmp_wav = pathlib.Path(tempfile.gettempdir()) / "02_bgm_main_loop_self_test_24bit.wav"
        try:
            _generate_synthetic_24bit_wav(tmp_wav)
            diag = analyze_audio_file(tmp_wav, bpm=args.bpm)
            resolved = resolve_composable_profile(
                config, args.timbre, args.state, args.sfx_mask, measured_sfx_wav_path=tmp_wav
            )
            evaluation = evaluate_and_recommend_remediation(
                diag, resolved, config, listening_tags=args.tag or ["fatigue_risk:high"]
            )
            print(
                json.dumps(
                    {
                        "diagnostics": asdict(diag),
                        "resolved_3axis_profile": resolved,
                        "evaluation": evaluation,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        finally:
            tmp_wav.unlink(missing_ok=True)
        return

    if not args.wav:
        parser.error("請提供 --wav <路徑> 或使用 --self-test 進行自檢。")

    if args.wav.is_dir():
        wav_files = sorted(args.wav.rglob("*.wav"))
        diags = [analyze_audio_file(w, bpm=args.bpm) for w in wav_files]
        summary = aggregate_reference_fingerprints(diags, role_filter=args.role_filter)
        print(yaml.safe_dump({"extracted_archetype_profile": summary}, sort_keys=False, allow_unicode=True))
    else:
        diag = analyze_audio_file(args.wav, bpm=args.bpm)
        resolved = resolve_composable_profile(
            config, args.timbre, args.state, args.sfx_mask, measured_sfx_wav_path=args.sfx_wav
        )
        evaluation = evaluate_and_recommend_remediation(diag, resolved, config, listening_tags=args.tag)
        print(
            json.dumps(
                {
                    "diagnostics": asdict(diag),
                    "resolved_3axis_profile": resolved,
                    "evaluation": evaluation,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
