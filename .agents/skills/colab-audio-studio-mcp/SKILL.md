---
name: colab-audio-studio-mcp
description: >-
  End-to-end game audio production workflow combining scene-first game music
  design (`sora16bit/codex-game-music-skill`), A/B arrangement specs (`ARR-SPEC`
  from `jtydhr88/music-composition-skills`), prompt compilation
  (`yzfly/awesome-music-prompts`), and studio generation/editing using
  `kanaworksai-hue/audio-studio-mcp` (`Stability-AI/stable-audio-3`) on Google
  Colab NVIDIA L4 GPUs via the colab CLI — complete with pre-download bar-by-bar
  QA diagnostics, automatic on-GPU repair, local SHA-256 verification, and
  guaranteed VM shutdown.
---

# Colab L4 `audio-studio-mcp` End-to-End Game Audio Design, Generation & Auto-QA Skill (v4)

This skill provides a complete, closed-loop pipeline for designing, generating, diagnosing, repairing, and delivering **game background music (BGM)**, **bar-aligned seamless loops**, **adaptive state layers**, **outro stingers**, and **sound effects (SFX)** on a Google Colab **NVIDIA L4 (24GB VRAM)** instance.

---

## 1. Architecture: Four-Stage Game Audio Pipeline

Never jump straight into opening a Colab GPU session from a raw user prompt. Always execute the **4-Stage Pipeline** below, completing **Stages 1–3 locally first** and only launching Colab L4 in **Stage 4** after planning is complete:

| Stage | Open-Source Methodology | Deliverable in Batch Folder | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **Stage 1: Scene & State Machine Design** | `sora16bit/codex-game-music-skill` (`SKILL.md`, `scene-recipes.md`, `output-contracts.md`) | [examples/01_bgm_spec_card_template.md](./examples/01_bgm_spec_card_template.md) → `01_bgm_spec_card.md` | Translate gameplay scene & player cadence into a state machine (e.g., Countdown SFX → Main Loop → Pause/Low-Density Layer → Finish Outro), BPM/bar math, Point/Line/Plane roles, and `3–6 kHz` SFX negative space. |
| **Stage 2: A/B Arrangement Spec & QA Gates** | `jtydhr88/music-composition-skills` (`ARR-SPEC`, `backends.yaml`) + `SJY051/music-composition` | [examples/02_arr_spec_ab_template.yaml](./examples/02_arr_spec_ab_template.yaml) → `02_arr_spec_ab.yaml` | Lock shared invariants (BPM, loop bars, LUFS, SFX frequency pocket), define Version A vs. Version B instrument rosters (`Lead / Harmony / Bass / Drums`), and set quantitative `qa_gates`. |
| **Stage 3: Prompt Compilation & Manifest** | `yzfly/awesome-music-prompts` + `audio_studio_mcp.prompts` | [examples/sample_manifest.json](./examples/sample_manifest.json) → `manifest.json` | Compile `ARR-SPEC` into natural English sentences ($\le 45$ words, 100% instrumental, zero vocal words, explicit `"starting immediately at full intensity from beat one"` for loops) + 4-field structured SFX + `qa` & `post` configs. |
| **Stage 4: Colab L4 Generation + Pre-Download Auto-QA & Repair** | `kanaworksai-hue/audio-studio-mcp` + `Stability-AI/stable-audio-3` (`77e0398`) | `results/*.wav` + `verification.json` + `04_critique_and_delivery_report.md` | Run 8-step SDPA inference on Colab L4, execute **bar-by-bar QA diagnostics & on-GPU auto-repair BEFORE packaging/downloading**, verify SHA-256 + QA pass locally, and guarantee `colab stop`. |

> **When to use `colab-audio-studio-mcp` vs. `colab-minimax-music3`**:
> - Use **`colab-audio-studio-mcp`** for **game BGM loops (10–60s)**, **adaptive state variations (`edit_variation`)**, **outro extensions (`edit_continue`)**, **bar-level inpainting (`edit_inpaint`)**, and **sound effects (SFX, 1–15s)**.
> - Use **`colab-minimax-music3`** when you need **full-length 3–5 minute multi-section songs** (Intro → Verse → Chorus → Solo → Outro) or tracks with vocal singing.

---

## 2. Empirical Colab L4 Performance & Six Golden Rules

Tested on Google Colab **NVIDIA L4 (23,034 MiB VRAM)**, the `sa3-torch` (`medium`) backend achieves:
- **Git Clone (Both Repos)**: **5.6 seconds** (`audio-studio-mcp` + `Stability-AI/stable-audio-3` @ `77e0398c26f7d22d801ec1bb036307e8b231f741`).
- **Python 3.11 `uv venv` + PyTorch 2.7.1 (`cu126`) + Dependencies**: **32.0 seconds**.
- **Weight Download (`stabilityai/stable-audio-3-medium`, ~10.4 GB)**: **34.1 seconds** (~300 MB/s via `HF_XET_HIGH_PERFORMANCE=1` + `max_workers=8` + `allow_patterns`).
- **Model Cold-Start Load (`1.4B DiT` + `same-l` AE + `t5gemma-b-b-ul2`)**: **35.1 seconds** (Peak VRAM Allocated: **8.59 GiB** / Reserved: **8.62 GiB** — only 38% of L4 VRAM).
- **Inference + Pre-Download QA Speed (8 Distilled Steps, `cfg=1.0`, SDPA)**:
  - **3s Game SFX**: **0.9–2.0 seconds** (**4.93 GiB** VRAM)
  - **24s (12-Bar) Stereo BGM Loop**: **2.8 seconds** (11.9 it/s, **4.97 GiB** VRAM)
  - **On-GPU Auto-Repair (`edit_inpaint` / `edit_continue` retry when QA gate catches a defect)**: **1.5–4.5 seconds** inline before zipping
  - **Total Colab Spend for a 6-Item Game Audio Batch**: **`0.00–0.04 Compute Units`** (~2.5 minutes end-to-end)

### Six Mandatory Rules

1. **Plan First, Launch Colab Last**:
   - Always write `01_bgm_spec_card.md`, `02_arr_spec_ab.yaml`, and `manifest.json` locally and confirm with the user **before** starting the Colab L4 VM.
2. **Do NOT Run `install.sh` or `install.ps1` on Colab Linux**:
   - `audio-studio-mcp`'s `install.sh` hardcodes an Apple Silicon (`Darwin/arm64`) check and exits with an error on Linux. Always bootstrap via [scripts/job_remote_template.py](./scripts/job_remote_template.py).
3. **Disable `flex_attention_compiled` Autotune Delay**:
   - Always set `transformer.flex_attention_compiled = None` right after creating `SA3TorchBackend` so PyTorch 2.7's native C++ **SDPA** runs immediately at **12 it/s** without a 60-second Triton compile stall.
4. **Require `transformers>=5.8.0`**:
   - Stable Audio 3 uses `T5GemmaEncoderModel` (`t5gemma-b-b-ul2/*`).
5. **Gated Hugging Face Repo & Zero-Leak `.hf_token` Handling**:
   - Upload `~/.cache/huggingface/token` to `/content/colab_job/.hf_token`, load it into `env['HF_TOKEN']` in `job_remote.py`, and **immediately `unlink()` `.hf_token` from disk**.
6. **Pre-Download Bar-by-Bar QA & On-GPU Auto-Repair (`run_qa_and_auto_repair`)**:
   - Never rely on whole-track averages alone (which hide weak 4-bar intros or hollow outro endings). Always enable the `qa` gates in `manifest.json` so [scripts/generate_remote_template.py](./scripts/generate_remote_template.py) diagnoses and repairs defects on the remote GPU **before** packaging `.zip` files and shutting down the VM.

---

## 3. Game Audio Prompting & Seamless Loop Math (`audio_studio_mcp.prompts`)

### A. Game BGM (`audio_type: "music"`) & Loops/Stems (`audio_type: "instrument"`)
- **Prefix added automatically**: `TrackType: Music, VocalType: Instrumental, ... BPM: <bpm>.`
- **Prompt Formula ($\le 45$ words)**:
  `Genre/Style + "starting immediately at full intensity from beat one" → Rhythm/Harmony Instruments → Lead Motif → Bass & Drums → Mood/Energy`
- **Why `"starting immediately at full intensity"` matters**: Without this cue, generative models often write a quiet 4-bar intro (`Bars 1–4`), causing a `3–6 dB` energy drop when `Bar 12` loops back to `Bar 1` ("Intro Trap").
- **Avoid vocal trigger words**: Never include `vocals`, `singer`, `lyrics`, `choir`, `voice`, `scream`.

### B. Seamless Game Music Loops (Bar-Aligned Duration Math)
1. Choose a target `bpm` (`4/4` time signature = 4 beats per bar):
   $$\text{Seconds per Bar} = \frac{240}{\text{BPM}}$$
2. Because `post.make_loop(audio, sr, crossfade_ms)` folds the last `crossfade_ms` into the start of the clip, set `duration_seconds` to:
   $$\text{duration\_seconds} = \left(\text{Bars} \times \frac{240}{\text{BPM}}\right) + \frac{\text{loop\_crossfade\_ms}}{1000}$$
   - **120 BPM, 12 bars (`24.0s`) + `200ms` crossfade**: set `duration_seconds: 24.2` → outputs an exact **`24.00s`** (12-bar) seamless loop!
   - **120 BPM, 8 bars (`16.0s`) + `200ms` crossfade**: set `duration_seconds: 16.2` → outputs an exact **`16.00s`** (8-bar) seamless loop!
   - **100 BPM, 8 bars (`19.2s`) + `200ms` crossfade**: set `duration_seconds: 19.4` → outputs an exact **`19.20s`** (8-bar) seamless loop!
3. **Format for Loops**: Always export loops as **`wav`** or **`ogg`** (`48000 Hz`). Never use MP3 for loops.

### C. Sound Effects (`audio_type: "sfx"`)
- **4 Structured Fields (`sfx_fields`)**:
  1. `sound_source`: Object making the sound (`"electronic race start timer and metallic stadium horn"`).
  2. `sound_action`: How it triggers & evolves (`"three distinct rhythmic low-mid electronic countdown beeps once per second followed by a bright high-energy race start horn blast"`).
  3. `sound_material`: Texture & material (`"crisp digital timer synth and punchy metallic air horn"`).
  4. `sound_space`: Environment & mic distance (`"mountain race start gate, clean close studio mix, tight punchy transient"`).

---

## 4. Pre-Download Bar-by-Bar Auto-QA & On-GPU Repair Gates

Configured via the `"qa"` object on each item in `manifest.json` and executed automatically inside [scripts/generate_remote_template.py](./scripts/generate_remote_template.py) **before** creating each `.zip` pack:

| QA Role (`qa.role`) | Quantitative Checks Performed on Remote VM | Automatic On-GPU / DSP Repair Triggered Before Download |
| :--- | :--- | :--- |
| **`"bgm_loop"`** | 1. **Intro-Trap Check**: `intro_drop_db = mean_rms(last 4 bars) - mean_rms(first 2 bars) <= max_intro_drop_db` (`2.5 dB`) and `bar_last - bar_1 <= 2.5 dB`<br>2. **SFX Frequency Pocket**: `3000–6000 Hz` energy `<= max_sfx_zone_pct` (`8.0%`)<br>3. **Loop Seam Continuity**: `seam_jump <= 0.02` | 1. Runs GPU `edit_inpaint(0.0, 8.0)` with `inpaint_fix_prompt` (`seed + 777`) to fill in weak intro bars with full drums/bass.<br>2. If still `> 2.5 dB`, promotes locked-in full-energy `Bars 5–8` into `Bars 1–4` and re-folds `200ms` crossfade.<br>3. If `3–6 kHz > 8.0%`, applies surgical `-4.0 dB` `3–6 kHz` attenuation before EBU R128 LUFS normalization.<br>4. Saves `_unlooped_fixed.wav` so downstream `edit_variation` / `edit_continue` inherit the repaired master. |
| **`"bgm_outro"`** | 1. **Climax Energy**: `24–28s` RMS drop relative to main loop `<= max_climax_drop_db` (`3.5 dB`)<br>2. **Climax Mid Presence**: `24–28s` mid (`250–3000 Hz`) `>= min_climax_mid_pct` (`20.0%`) | 1. Automatically retries `edit_continue` on GPU with new seeds (`seed + 111`, `seed + 222`) and selects the strongest climax.<br>2. If still below threshold, blends a 4-bar summit climax cadence with an `1800ms` stadium reverb fade-out. |
| **`"countdown_sfx"`** | Splits countdown into 1-second windows (`0–1s`, `1–2s`, `2–3.5s`) and requires `>= min_active_windows` (`3`) with `RMS > -32 dBFS`. | Automatically retries generation on GPU with new seeds (`seed + 101`, `seed + 202`) until all countdown beats are present. |

---

## 4b. v4 Canonical Optimizations (44.1 kHz Native, Tri-State QA, Relative Climax, Full SFX Pocket)

| Optimization | Manifest Config | What Happens on the Remote VM |
| :--- | :--- | :--- |
| **A. 44.1 kHz Native Workflow** | `"post": {"out_sr": 44100}` across all items | Stable Audio 3 native SR is 44.1 kHz; TorchAudio HDemucs is trained at 44.1 kHz. Bypasses all 48k↔44.1k resampling passes, eliminating resampling phase artifacts and CPU cycles while guaranteeing exact sample count matching (e.g. 1,058,400 samples). |
| **B. Tri-State QA Architecture** | `"qa_report": {"status": "passed" / "warning" / "failed"}` | Replaces rigid boolean with a three-tier status model. Non-blocking degradations produce `"warning"` with actionable `status_reasons`, while fatal issues trigger `"failed"`. |
| **C. Relative Mid-Frequency Climax Ratio** | `"qa": {"min_climax_relative_mid_ratio": 1.15}` | Replaces arbitrary absolute mid-frequency percentage with a dynamic relative ratio (`climax_mid / body_mid >= 1.15`), correctly adapting to bass-heavy styles (heavy metal, synthwave, trap) where low frequencies dominate. |
| **D. Universal SFX Pocket Protection** | Auto-applied to all BGM roles (`bgm_loop`, `bgm_outro`, `bgm_stem_layer`) | Outro and stem layers often contain high-frequency resonance or long reverb. The `3000–6000 Hz` notch check is extended across all music tracks, dynamically compressing the SFX band to leave full acoustic space for game horns, timers, and bells. |
| **E. Best-of-N Reranking** | `"candidates": 3` on a `"generate"` item | Generates N seeds (`seed`, `seed+101`, `seed+202`), scores each candidate via `bar_chroma_similarity`, `intro_drop_db`, `seam_chroma_similarity`, and picks the winner. |
| **F. Smart Loop Alignment (±1 Beat Radius)** | Extended alignment search with Inpaint fallback | Searches ±1 beat around detected beat-1 onset for optimal chroma & RMS seam match; snaps to zero-crossing; falls back to seamless inpaint crossfade if boundary correlation remains weak. |
| **G. HDemucs Sample-Locked Stem Layer** | `"type": "stem_layer"`, `"source_id"`, `"stem_weights"`, `"other_lowpass_hz"` | Derives drums/bass/filtered-guitar stems from the master loop with zero lag (`measured_onset_lag_ms == 0.0`), exporting ready-to-mix vertical game audio layers. |

> [!NOTE]
> Prefer `stem_layer` over `edit_variation` for pause/low-intensity states that must cross-fade with the main loop at any moment: `edit_variation` regenerates audio and is not guaranteed to be beat-aligned, while `stem_layer` is derived from the same samples.

---

## 5. Step-by-Step Execution Workflow

1. **Create a Dedicated Local Output Directory**:
   e.g., `outputs/game-audio-<YYYY-MM-DD>-<scene-name>/`.
2. **Author Stage 1–3 Planning Files Locally (Do NOT Start Colab Yet)**:
   - Copy and customize [examples/01_bgm_spec_card_template.md](./examples/01_bgm_spec_card_template.md) → `01_bgm_spec_card.md`.
   - Copy and customize [examples/02_arr_spec_ab_template.yaml](./examples/02_arr_spec_ab_template.yaml) → `02_arr_spec_ab.yaml`.
   - Copy and customize [examples/sample_manifest.json](./examples/sample_manifest.json) → `manifest.json`.
   - Present the Stage 1–3 plan to the user for confirmation.
3. **Deploy the 3 Execution Scripts into the Batch Directory**:
   - [scripts/job_remote_template.py](./scripts/job_remote_template.py) → `job_remote.py`
   - [scripts/generate_remote_template.py](./scripts/generate_remote_template.py) → `generate_remote.py`
   - [scripts/supervise_template.py](./scripts/supervise_template.py) → `supervise.py`
4. **Verify Prerequisites & Run `python -u supervise.py`**:
   - Confirm `colab usage ; colab sessions` shows zero orphan sessions and `~/.cache/huggingface/token` exists.
   - Run `python -u supervise.py`. It provisions Colab L4, runs remote generation + pre-download auto-QA/repair, verifies `qa_report.passed == True` + SHA-256 + WAV headers locally, and **guarantees `colab stop`** in `finally:`.
5. **Write `04_critique_and_delivery_report.md`**:
   - Summarize the `initial_diagnostics` vs. `final_diagnostics` and `repairs_applied` from `verification.json` along with the 8-point `codex-game-music-skill` critique table.
