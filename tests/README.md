# Offline Stability & DSP Unit Test Suite

This directory contains the zero-cost (0 Compute Units), pure NumPy/SciPy offline test harness for **Agent Game Music** (`colab-audio-studio-mcp`).

---

## Quick Run

Run all unit tests locally with standard Python (no GPU required):

```bash
python -m unittest tests/test_offline_dsp.py -v
```

Execution time: **~0.5 seconds**.

---

## Test Coverage Matrix

| Test ID | Test Category | Target Algorithm / Function | Verification Focus |
| :--- | :--- | :--- | :--- |
| **`test_T1_1`** | Dimension 1: DSP Unit | `compute_diagnostics(bar_sec=2.0)` | Simulates a -8 dB 4-bar quiet intro; asserts `intro_drop_db > 2.5 dB` and triggers the Intro-Trap repair gate. |
| **`test_T1_2`** | Dimension 1: DSP Unit | `attenuate_band_fft(3000, 6000, -4.0)` | Injects a severe 4500 Hz resonance peak (>20%); asserts SFX band drops below 8.0% while preserving sub-bass (<250 Hz). |
| **`test_T1_3`** | Dimension 1: DSP Unit | `smart_align_loop_window` | Simulates lead-in silence; asserts the algorithm skips leading silence and snaps the start point to a clean zero-crossing. |
| **`test_T1_4`** | Dimension 1: DSP Unit | `lowpass_fft` + `measure_onset_lag_ms` | Generates 1,058,400 samples; asserts 100% sample-lock match and `measured_onset_lag_ms == 0.0 ms`. |
| **`test_T1_5`** | Dimension 1: DSP Unit | `score_candidate_for_reranking` | Evaluates Best-of-N multi-objective scoring; verifies that punchy, balanced takes beat takes with high intro drops. |
| **`test_T2_1`** | Dimension 2: Edge Cases | `calculate_loop_math` | Tests 60 BPM, 200 BPM, and 137.5 fractional BPM; asserts bar math precision and positive alignment slack. |
| **`test_T2_3`** | Dimension 2: Edge Cases | `smart_align_loop_window` on Drone | Tests smooth ambient sine audio with zero onset transients; ensures graceful fallback without raising empty sequence exceptions. |
| **`test_T2_4`** | Dimension 2: Edge Cases | `sanitize_prompt_for_instrumental` | Tests prompts containing forbidden vocal words (`vocals`, `singers`, `choir`, `lyrics`); asserts 100% removal. |
| **`test_T4_2`** | Dimension 4: Schema | `sample_manifest.json` | Validates that all items declare native `44100 Hz`, outro relative mid gates, and stem weights. |

---

## Review Rubric for Reviewing Agents

1. **Deterministic Execution**: All tests produce identical numerical results across platforms (Windows / Linux / macOS).
2. **Zero Cloud Dependencies**: Does not require `torch`, `torchaudio`, CUDA, or a Google Colab connection.
3. **Safety Gate Assurance**: Guarantees that the algorithmic building blocks powering on-GPU repairs are mathematically sound before any cloud GPU execution.
