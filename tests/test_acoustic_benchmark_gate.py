"""【維度五】聲學客觀指標回歸基準測試 (Acoustic Benchmark Gate: T5.1, T5.2, T5.3, T5.4).

Verifies real production audio assets generated in v4 against the Acoustic Benchmark Matrix:
- Loop Seam Jump Rate <= 0.02
- Intro Energy Drop <= 2.5 dB
- SFX Pocket Protection <= 8.0%
- Stem Waveform Correlation >= 0.90
- Stem Measured Onset Lag == 0.0 ms
- Native 44.1 kHz sampling across all assets
- Sidecar QA reports status == "passed"
"""

import json
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf

from tests.dsp_core import compute_diagnostics, measure_onset_lag_ms, to_cf


class TestDimension5AcousticBenchmarkGate(unittest.TestCase):
    """【維度五】聲學客觀指標回歸基準測試"""

    def setUp(self):
        self.results_dir = Path("outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results")
        if not self.results_dir.exists():
            self.skipTest(f"v4 results directory not found at {self.results_dir}")

    def test_T5_1_loop_acoustic_benchmark(self):
        """T5.1: 主幹循環 (Loop A & B) 聲學指標基準檢驗 (接縫 < 0.02, 前奏 < 2.5dB, SFX < 8%)。"""
        loop_files = ["02_bgm_main_loop_A.wav", "03_bgm_main_loop_B.wav"]
        for name in loop_files:
            path = self.results_dir / name
            self.assertTrue(path.exists(), f"File {name} missing")
            audio, sr = sf.read(path)
            self.assertEqual(sr, 44100, f"{name} sampling rate must be 44100 Hz")

            diag = compute_diagnostics(audio, sr, bar_sec=2.0)
            self.assertLessEqual(
                diag['seam_jump'], 0.02,
                f"{name} seam_jump {diag['seam_jump']} exceeds 0.02 threshold",
            )
            self.assertLessEqual(
                diag['intro_drop_db'], 2.5,
                f"{name} intro_drop_db {diag['intro_drop_db']} exceeds 2.5 dB threshold",
            )
            self.assertLessEqual(
                diag['sfx_zone_3to6k_pct'], 8.0,
                f"{name} sfx_zone_3to6k_pct {diag['sfx_zone_3to6k_pct']} exceeds 8.0% threshold",
            )

    def test_T5_2_stem_layer_synchronization_benchmark(self):
        """T5.2: HDemucs 動態分軌取樣鎖定基準檢驗 (點數吻合、延遲 0.0ms、波形相關 >= 0.90)。"""
        loop_path = self.results_dir / "02_bgm_main_loop_A.wav"
        stem_path = self.results_dir / "05_bgm_pause_swap_stem_layer.wav"

        loop_audio, loop_sr = sf.read(loop_path)
        stem_audio, stem_sr = sf.read(stem_path)

        self.assertEqual(loop_sr, 44100)
        self.assertEqual(stem_sr, 44100)

        # 1. Exact sample length match check (1,058,400 samples)
        self.assertEqual(
            len(loop_audio), len(stem_audio),
            f"Sample counts do not match: {len(loop_audio)} vs {len(stem_audio)}",
        )
        self.assertEqual(len(loop_audio), 1058400, "24s at 44.1k must equal 1,058,400 samples")

        # 2. Measured onset lag must strictly be 0.0 ms
        loop_mono = to_cf(loop_audio)[0].mean(axis=0)
        stem_mono = to_cf(stem_audio)[0].mean(axis=0)
        lag_ms = measure_onset_lag_ms(loop_mono, stem_mono, loop_sr)
        self.assertEqual(lag_ms, 0.0, f"Measured onset lag must be 0.0 ms, got {lag_ms} ms")

        # 3. Waveform correlation check
        corr = float(np.corrcoef(loop_mono, stem_mono)[0, 1])
        self.assertGreaterEqual(
            corr, 0.90,
            f"Stem layer waveform correlation {corr} should be >= 0.90 with master loop",
        )

    def test_T5_3_outro_acoustic_benchmark(self):
        """T5.3: 衝線尾奏聲學指標檢驗 (SFX頻段保護 <= 8.0% 且平滑淡出)。"""
        outro_path = self.results_dir / "06_bgm_finish_outro.wav"
        audio, sr = sf.read(outro_path)
        self.assertEqual(sr, 44100)

        diag = compute_diagnostics(audio, sr, bar_sec=2.0)
        self.assertLessEqual(
            diag['sfx_zone_3to6k_pct'], 8.0,
            f"Outro SFX zone {diag['sfx_zone_3to6k_pct']} exceeds 8.0% threshold",
        )

        # Check tail 100ms fade-out
        tail_samples = int(sr * 0.1)
        tail_rms = float(20.0 * np.log10(np.sqrt(np.mean(audio[-tail_samples:] ** 2)) + 1e-9))
        self.assertLessEqual(tail_rms, -25.0, f"Outro tail RMS {tail_rms} dBFS should fade cleanly")

    def test_T5_4_sidecar_qa_reports_all_passed(self):
        """T5.4: 驗證 v4 產出之全數 JSON Sidecar QA 報告狀態皆為 'passed'。"""
        json_files = list(self.results_dir.glob("*.wav.json"))
        self.assertGreaterEqual(len(json_files), 6, "Must have at least 6 sidecar reports")

        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            qa_report = data.get("qa_report", {})
            status = qa_report.get("status")
            self.assertEqual(
                status, "passed",
                f"Asset {jf.name} QA status is '{status}', expected 'passed'. Reasons: {qa_report.get('status_reasons')}",
            )
            self.assertTrue(qa_report.get("passed"), f"Asset {jf.name} passed boolean is False")


if __name__ == "__main__":
    unittest.main()
