"""Offline Unit & Edge-Case Stability Test Suite for agent-game-music.

Covers:
- Dimension 1: Offline DSP & Repair Logic Unit Tests (T1.1, T1.2, T1.3, T1.4)
- Dimension 2: Edge-Case & Stress Tests (T2.1, T2.3, T2.4)
- Dimension 4: Contract Schema Compliance (T4.2)
Zero GPU spend (0 Compute Units), runs locally with pure NumPy / SciPy.
"""

import json
from pathlib import Path
import unittest
import numpy as np

from tests.dsp_core import (
    attenuate_band_fft,
    calculate_loop_math,
    compute_diagnostics,
    lowpass_fft,
    measure_onset_lag_ms,
    num_samples,
    sanitize_prompt_for_instrumental,
    smart_align_loop_window,
    to_cf,
)


class TestDimension1OfflineDSP(unittest.TestCase):
    """【維度一】本機離線 DSP 與修復邏輯單元測試（Zero-CU，0 成本）"""

    def setUp(self):
        self.sr = 44100

    def test_T1_1_intro_trap_detection(self):
        """T1.1: 前奏陷阱 (Intro-Trap) 診斷測試：模擬前 4 小節 -8dB 衰減，確認精確捕捉。"""
        # 120 BPM -> 2.0s per bar. 12 bars = 24.0s = 1,058,400 samples
        dur_sec = 24.0
        n_samples = int(self.sr * dur_sec)
        t = np.linspace(0, dur_sec, n_samples, endpoint=False)

        # Base signal: dense stereo harmonic tone
        sig_l = 0.4 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
        sig_r = 0.4 * np.sin(2 * np.pi * 165 * t) + 0.2 * np.sin(2 * np.pi * 554 * t)
        audio = np.stack([sig_l, sig_r], axis=0)

        # Inject Intro Trap: first 4 bars (8 seconds) attenuated by -8 dB (~0.398x)
        intro_samples = int(self.sr * 8.0)
        audio[:, :intro_samples] *= 0.398

        diag = compute_diagnostics(audio, self.sr, bar_sec=2.0)

        # Verify intro_drop_db >= 2.5 dB gate trigger
        self.assertGreater(diag['intro_drop_db'], 2.5, f"Expected intro drop > 2.5 dB, got {diag['intro_drop_db']} dB")
        self.assertEqual(len(diag['bars']), 12, "Should divide 24s into exactly 12 bars at 120 BPM")
        # First bar RMS should be significantly lower than 12th bar
        bar1_rms = diag['bars'][0]['rms_db']
        bar12_rms = diag['bars'][11]['rms_db']
        self.assertLess(bar1_rms, bar12_rms - 5.0, "Bar 1 RMS must be at least 5 dB below Bar 12")

    def test_T1_2_sfx_pocket_notch_filtering(self):
        """T1.2: SFX 3–6 kHz 負空間手術級 EQ 測試：確認超標頻段被精確壓制且不傷低頻。"""
        n_samples = int(self.sr * 5.0)  # 5 seconds
        t = np.linspace(0, 5.0, n_samples, endpoint=False)

        # Inject loud 4500 Hz resonance (inside 3-6 kHz SFX band) + 150 Hz sub bass
        sub_bass = 0.5 * np.sin(2 * np.pi * 150 * t)
        sfx_resonance = 0.6 * np.sin(2 * np.pi * 4500 * t)  # heavy SFX band energy
        raw_audio = np.stack([sub_bass + sfx_resonance, sub_bass + sfx_resonance], axis=0)

        initial_diag = compute_diagnostics(raw_audio, self.sr)
        self.assertGreater(initial_diag['sfx_zone_3to6k_pct'], 8.0, "Initial audio must exceed 8% SFX threshold")

        # Apply surgical -4.0 dB notch in 3000 - 6000 Hz
        filtered = attenuate_band_fft(raw_audio, self.sr, 3000.0, 6000.0, -4.0)
        final_diag = compute_diagnostics(filtered, self.sr)

        # Verify SFX zone energy dropped significantly
        self.assertLess(
            final_diag['sfx_zone_3to6k_pct'],
            initial_diag['sfx_zone_3to6k_pct'],
            "Filtered audio must reduce 3-6 kHz percentage",
        )
        # Verify sub-bass (<250 Hz) was preserved (within 0.5 dB)
        initial_low_pct = initial_diag['low_lt250_pct']
        final_low_pct = final_diag['low_lt250_pct']
        self.assertGreaterEqual(final_low_pct, initial_low_pct, "Low-end ratio must not be degraded by SFX notch")

    def test_T1_3_zero_crossing_and_smart_alignment(self):
        """T1.3: ±1 拍零交差智慧循環對齊測試：確認跳過前置靜音並於零交差点裁切。"""
        target_loop_sec = 8.0
        crossfade_ms = 200.0
        sr = self.sr

        # Generate audio with 80ms silence lead-in + 8.6s of tonal rhythm
        silence_samples = int(sr * 0.08)
        content_samples = int(sr * 9.0)
        t = np.linspace(0, 9.0, content_samples, endpoint=False)
        content = 0.5 * np.sin(2 * np.pi * 220 * t)

        full_mono = np.concatenate([np.zeros(silence_samples), content])
        stereo_audio = np.stack([full_mono, full_mono], axis=0)

        aligned, info = smart_align_loop_window(stereo_audio, sr, target_loop_sec, crossfade_ms)

        # Verify alignment detected lead-in silence and skipped it
        self.assertGreater(info['aligned_start_ms'], 40.0, "Should skip leading silence")
        self.assertTrue(info['zero_crossing_snapped'], "Must snap to zero crossing")

        # Verify exact sliced sample length: (target_loop_sec + crossfade_ms/1000) * sr
        expected_samples = int(round((target_loop_sec + crossfade_ms / 1000.0) * sr))
        self.assertEqual(aligned.shape[1], expected_samples, "Aligned length must match loop + crossfade exactly")

        # Verify first sample is very close to zero amplitude (zero-crossing property)
        start_amp = abs(aligned[0, 0])
        self.assertLess(start_amp, 0.05, f"Start amplitude should be near zero, got {start_amp}")

    def test_T1_4_sample_locked_stem_layer_synchronization(self):
        """T1.4: HDemucs 取樣鎖定相容性測試：驗證分軌點數 100% 吻合且起音延遲嚴格為 0.0 ms。"""
        # Exactly 1,058,400 samples (24s @ 44.1k)
        target_samples = 1058400
        dur_sec = target_samples / self.sr
        t = np.linspace(0, dur_sec, target_samples, endpoint=False)

        # Master loop containing rhythm transient pulses + melodic synth
        drums_bass = np.zeros(target_samples, dtype=np.float32)
        for beat in np.arange(0.2, dur_sec - 0.2, 0.5):
            idx = int(beat * self.sr)
            pulse_len = int(self.sr * 0.05)
            drums_bass[idx:idx + pulse_len] += np.sin(2 * np.pi * 100 * np.linspace(0, 0.05, pulse_len)) * np.exp(-np.linspace(0, 5, pulse_len))

        leads = 0.3 * np.sin(2 * np.pi * 1500 * t).astype(np.float32)
        master_loop = np.stack([drums_bass + leads, drums_bass + leads], axis=0)

        # Generate lowpass stem layer (retaining drums & bass, rolling off melody)
        stem_layer = lowpass_fft(master_loop, self.sr, cutoff_hz=800.0, rolloff_hz=300.0)

        # 1. Sample count match check
        self.assertEqual(num_samples(stem_layer), target_samples, "Stem layer samples must strictly equal master loop")

        # 2. Onset lag cross-correlation check
        lag_ms = measure_onset_lag_ms(master_loop[0], stem_layer[0], self.sr)
        self.assertEqual(lag_ms, 0.0, f"Stem layer onset lag must strictly be 0.0 ms, got {lag_ms}")

    def test_T1_5_best_of_n_candidate_reranking(self):
        """T1.5: Best-of-N 候選重排評分測試：確認演算法懲罰前奏落差並獎勵和聲一致性。"""
        from tests.dsp_core import score_candidate_for_reranking

        # Candidate 1: Flawed take with severe intro drop (3.5 dB) and high SFX bleed (9.5%)
        flawed_cand = {
            'intro_drop_db': 3.5,
            'max_bar_spread_db': 3.0,
            'sfx_zone_3to6k_pct': 9.5,
            'bar_chroma_similarity': 0.82,
            'onset_clarity_db': 4.5,
        }
        # Candidate 2: Superior take with punchy intro (0.3 dB), tight dynamics, and clean SFX pocket (5.0%)
        superior_cand = {
            'intro_drop_db': 0.3,
            'max_bar_spread_db': 1.2,
            'sfx_zone_3to6k_pct': 5.0,
            'bar_chroma_similarity': 0.95,
            'onset_clarity_db': 6.8,
        }

        score_flawed = score_candidate_for_reranking(flawed_cand, role='bgm_loop', qa_cfg={})
        score_superior = score_candidate_for_reranking(superior_cand, role='bgm_loop', qa_cfg={})

        self.assertGreater(
            score_superior,
            score_flawed + 20.0,
            f"Superior candidate should win by a large margin (got {score_superior} vs {score_flawed})",
        )


class TestDimension2EdgeCases(unittest.TestCase):
    """【維度二】邊界與極端值壓力測試（Edge-Case & Robustness）"""

    def test_T2_1_extreme_bpm_and_bar_math(self):
        """T2.1: 極端 BPM 與小節數學精度測試 (60 BPM, 200 BPM, 137.5 BPM)。"""
        test_cases = [
            (60.0, 8, 4.0, 32.0),
            (200.0, 16, 1.2, 19.2),
            (137.5, 12, 1.7455, 20.9455),
        ]
        for bpm, bars, expected_bar_sec, expected_loop_sec in test_cases:
            res = calculate_loop_math(bpm, bars)
            self.assertAlmostEqual(res['bar_seconds'], expected_bar_sec, places=3)
            self.assertAlmostEqual(res['target_loop_seconds'], expected_loop_sec, places=3)
            self.assertGreater(res['duration_seconds'], res['target_loop_seconds'])
            # Verify positive slack
            slack = res['duration_seconds'] - res['target_loop_seconds']
            self.assertGreaterEqual(slack, 0.4)

    def test_T2_2_short_sfx_and_long_loop_bounds(self):
        """T2.2: 極短 (0.5s) SFX 峰值正規化與超長 (60s) 主循環動態負載邊界測試。"""
        from tests.dsp_core import peak_normalize

        # 1. Ultra-short 0.5s SFX (22,050 samples)
        sr = 44100
        n_short = int(sr * 0.5)
        t_short = np.linspace(0, 0.5, n_short, endpoint=False)
        short_sfx = 0.1 * np.sin(2 * np.pi * 1000 * t_short)
        normalized, gain_db = peak_normalize(short_sfx, ceiling_db=-1.0)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 10.0 ** (-1.0 / 20.0), places=3)
        self.assertGreater(gain_db, 10.0, "Should amplify weak 0.1 peak to -1.0 dBFS")

        # 2. Ultra-long 60s loop (120 BPM, 30 bars = 2,646,000 samples)
        n_long = int(sr * 60.0)
        t_long = np.linspace(0, 60.0, n_long, endpoint=False)
        long_loop = np.stack([0.3 * np.sin(2 * np.pi * 120 * t_long), 0.3 * np.sin(2 * np.pi * 120 * t_long)], axis=0)
        long_diag = compute_diagnostics(long_loop, sr, bar_sec=2.0)
        self.assertEqual(len(long_diag['bars']), 30, "Should compute exactly 30 bars for 60s at 120 BPM")
        self.assertEqual(long_diag['duration_seconds'], 60.0)

    def test_T2_3_ambient_drone_without_transients(self):
        """T2.3: 純長音氛圍音樂 (Drone/Ambient) 無明顯鼓點起音容錯測試。"""
        # Continuous smooth sine wave with zero transients
        sr = 44100
        n_samples = int(sr * 10.0)
        t = np.linspace(0, 10.0, n_samples, endpoint=False)
        drone = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 220 * t)], axis=0)

        # Must not throw ValueError (e.g. empty sequence in max)
        aligned, info = smart_align_loop_window(drone, sr, target_loop_sec=8.0, crossfade_ms=200.0)
        self.assertIsNotNone(aligned)
        self.assertIn('aligned_start_ms', info)
        self.assertGreaterEqual(info['aligned_start_ms'], 0.0)

    def test_T2_4_prompt_sanitization_removes_vocals(self):
        """T2.4: 提示詞防污染與人聲關鍵字過濾測試。"""
        dirty_prompt = (
            "High energy heavy metal with powerful lead singer, aggressive choir vocals, "
            "screaming guitar solo, and driving drums with spoken lyrics"
        )
        clean_prompt, detected = sanitize_prompt_for_instrumental(dirty_prompt)

        # All forbidden vocal cues must be caught
        self.assertTrue(any("singer" in d.lower() for d in detected))
        self.assertTrue(any("vocals" in d.lower() for d in detected))
        self.assertTrue(any("screaming" in d.lower() for d in detected))
        self.assertTrue(any("lyrics" in d.lower() for d in detected))

        # Clean prompt must not contain vocal keywords
        for forbidden in ["singer", "choir", "vocals", "lyrics"]:
            self.assertNotIn(forbidden, clean_prompt.lower())
        # Non-vocal elements preserved
        self.assertIn("heavy metal", clean_prompt)
        self.assertIn("driving drums", clean_prompt)


class TestDimension4ContractCompliance(unittest.TestCase):
    """【維度四】輸出合約與規格標準檢驗（Contract Compliance）"""

    def test_T4_2_sample_manifest_schema_compliance(self):
        """T4.2: 檢驗 sample_manifest.json 是否完整具備 v4 規範欄位。"""
        manifest_path = Path("examples/sample_manifest.json")
        if not manifest_path.exists():
            manifest_path = Path(".agents/skills/agent-game-music/examples/sample_manifest.json")
        self.assertTrue(manifest_path.exists(), f"sample_manifest.json not found at {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIsInstance(manifest, list)
        self.assertGreaterEqual(len(manifest), 4)

        for item in manifest:
            # 1. All items must have out_sr == 44100
            self.assertEqual(item.get("post", {}).get("out_sr"), 44100, f"Item {item.get('id')} out_sr must be 44100")

            # 2. Outro items must declare relative mid ratio gate
            if item.get("qa", {}).get("role") == "bgm_outro":
                self.assertIn(
                    "min_climax_relative_mid_ratio",
                    item["qa"],
                    f"Outro {item.get('id')} missing min_climax_relative_mid_ratio",
                )
                self.assertGreaterEqual(item["qa"]["min_climax_relative_mid_ratio"], 1.10)

            # 3. Stem layer items must have valid stem_weights
            if item.get("type") == "stem_layer":
                self.assertIn("stem_weights", item)
                self.assertIn("drums", item["stem_weights"])
                self.assertIn("bass", item["stem_weights"])


if __name__ == "__main__":
    unittest.main()
