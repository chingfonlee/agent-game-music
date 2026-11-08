# Agent Game Music (AGM) 🎮🎵

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Stable-Audio-3](https://img.shields.io/badge/Model-Stable--Audio--3-purple.svg)](https://huggingface.co/stabilityai/stable-audio-3-medium)
[![Model: TorchAudio-HDemucs](https://img.shields.io/badge/Model-TorchAudio--HDemucs-green.svg)](https://pytorch.org/audio/stable/models.html)
[![Hardware: Google Colab L4](https://img.shields.io/badge/Hardware-Google%20Colab%20L4%20(24GB)-orange.svg)](https://colab.research.google.com/)
[![Agents: Claude Code & Codex](https://img.shields.io/badge/Agents-Claude%20Code%20%7C%20Codex%20%7C%20Antigravity-red.svg)](#multi-agent-ecosystem-support)

**Autonomous, Closed-Loop Game Audio Directing, Production, and Pre-Download QA Pipeline for AI Coding Agents.**

---

## 💡 Overview

Game audio development requires strict structural and acoustic constraints that raw generative music prompts fail to provide:
- **Intro Trap**: Generative music models typically write quiet 4-bar intros, causing a jarring $3\text{--}6\text{ dB}$ energy drop when the loop repeats.
- **Loop Seam Clicks**: Random cross-fades cause phase cancellation, transient distortion, or rhythmic stutters.
- **SFX Masking**: Heavy instruments drown out crucial gameplay sound effects, UI chimes, and character speech in the $3000\text{--}6000\text{ Hz}$ band.
- **Outro Cadence**: Generative endings often fade out prematurely or collapse into chaotic noise instead of delivering a satisfying musical resolution.
- **Interactive Stems**: Dynamic audio in game engines (e.g. FMOD, Wwise) requires sample-locked rhythm and melody stems with zero phase lag.

**Agent Game Music** solves these challenges by combining scene-first game music design, structured arrangement contracts, prompt compilation, and remote GPU generation on **Google Colab NVIDIA L4 GPUs** with **in-situ bar-by-bar diagnostics, self-healing repairs on the GPU before downloading, and guaranteed automatic VM shutdown**.

---

## 🏛️ Architecture: The 4-Stage Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Scene & State Machine Design (Local Agent)                    │
│ └─► 01_bgm_spec_card.md                                                │
│     Translates game scenes into state machines (BPM, Bars, 4-Field SFX)│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 2: A/B Arrangement Spec & Frequency Allocation (Local Agent)     │
│ └─► 02_arr_spec_ab.yaml                                                │
│     Lead, Harmony, Bass, Drums rosters + 3–6 kHz SFX negative space    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 3: Prompt Compilation & Manifest (Local Agent)                   │
│ └─► manifest.json                                                      │
│     Target LUFS, Loop math (+0.4s slack), HDemucs weights, Outro gates │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (Confirm with user before GPU launch)
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 4: Colab L4 Generation + Pre-Download Auto-QA & Repair (Cloud)   │
│ ├─► Stable Audio 3 (8-step SDPA distilled inference ~11 it/s)          │
│ ├─► Best-of-3 Candidate Reranking on GPU                               │
│ ├─► Smart ±1 Beat Loop Alignment with Zero-Crossing Snapping           │
│ ├─► TorchAudio HDemucs Sample-Locked Stem Separation (Zero Lag)        │
│ ├─► Bar-by-Bar QA Diagnostics (Intro, Seam, Climax, SFX Pocket)        │
│ ├─► Automated On-GPU Inpaint & Surgical Notch Repair BEFORE Download   │
│ ├─► Local SHA-256 Verification & Output Delivery                       │
│ └─► Guaranteed colab stop in finally block (CU Cost ~0.00–0.04 Units)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Core Technical Features (v4)

1. **Native 44.1 kHz End-to-End Pipeline**:
   Both `Stability-AI/stable-audio-3` and `TorchAudio HDemucs` operate natively at 44.1 kHz. Bypasses intermediate 48 kHz resampling passes, eliminating resampling phase artifacts and guaranteeing exact sample synchronization (e.g., $1,058,400$ samples).
2. **Tri-State QA Reporting Architecture**:
   Replaces rigid boolean checks with a three-tier model (`passed` / `warning` / `failed`). Non-fatal degradations surface warnings with actionable diagnostic context, while fatal defects trigger immediate on-GPU repairs.
3. **Relative Mid-Frequency Climax Gate**:
   Evaluates outro peaks using dynamic relative energy ratios ($\frac{\text{climax\_mid}}{\text{body\_mid}} \ge 1.15$), preventing false-positive rejections on bass-heavy genres (Heavy Metal, Synthwave, Trap).
4. **Universal SFX Frequency Pocket Protection**:
   Automatically monitors the $3000\text{--}6000\text{ Hz}$ band across all music assets (loops, outros, stems), applying surgical dynamic attenuation to preserve acoustic headroom for game UI, horns, and timers.
5. **Best-of-N Candidate Reranking**:
   Generates $N$ candidates per loop item on GPU (~1 second each), evaluating bar chroma consistency, intro drop, and onset clarity, and automatically selecting the best candidate before looping.
6. **Smart ±1 Beat Loop Alignment**:
   Scans $\pm 1$ beat around the detected beat-1 onset for optimal chroma and RMS continuity, snaps to clean zero-crossings, and falls back to inpaint cross-fading if boundary correlation is weak.
7. **HDemucs Sample-Locked Vertical Stems**:
   Derives drums, bass, and filtered melodic stems from the final master loop with measured zero onset lag (`measured_onset_lag_ms == 0.0`), ready for interactive game engine dynamic layering.

---

## 📦 Multi-Agent Ecosystem Support

Dedicated packages are provided under [`skills_export/`](./skills_export/) for immediate integration into your agent workflow:

### 1. Anthropic Claude Code (`claude`)
Located in [`skills_export/claude-code/colab-audio-studio-mcp/`](./skills_export/claude-code/colab-audio-studio-mcp/):
- **`SKILL.md`**: Claude Code YAML frontmatter with non-interactive CLI guidelines and error recovery.
- **`CLAUDE.md`**: Project rule card to drop into project root or `~/.claude/CLAUDE.md`.
```bash
# Install to workspace:
mkdir -p .claude/skills
cp -r skills_export/claude-code/colab-audio-studio-mcp .claude/skills/
```

### 2. OpenAI Codex / Cursor / Copilot Workspace
Located in [`skills_export/codex/colab-game-audio-skill/`](./skills_export/codex/colab-game-audio-skill/):
- **`SKILL.md`**: Agent prompt contract and role definition.
- **`output-contracts.md`**: Strict schema definitions for Stages 1–4 to prevent model hallucination.
- **`scene-recipes.md`**: Pre-calibrated recipes for Racing, Cyberpunk, Boss Battle, Retro Platformer, and RPG Village.

### 3. Google Antigravity / Gemini CLI
Located in [`.agents/skills/colab-audio-studio-mcp/`](./.agents/skills/colab-audio-studio-mcp/):
- Fully compliant with the Universal Agent Skills specification.

---

## 📊 Empirical Benchmarks (Google Colab L4)

Tested on **Google Colab NVIDIA L4 (24GB VRAM)**:

| Operation | Execution Time | VRAM Allocated | Notes |
| :--- | :--- | :--- | :--- |
| **Git Clone (Both Repos)** | 5.8 s | - | `audio-studio-mcp` + `stable-audio-3` |
| **uv venv + PyTorch 2.7.1 + Deps** | 30.8 s | - | Linux-optimized bootstrap |
| **Weight Download (SA3 + HDemucs)** | 33.7 s | - | ~300 MB/s via HF XET transfer |
| **Cold-Start Model Load** | 35.0 s | 8.59 GiB (38%) | Native SDPA (No Triton delay) |
| **24s Stereo BGM Loop (Best-of-3)** | ~10.1 s | 4.97 GiB | 11.9 it/s (8 distilled steps) |
| **HDemucs Stem Separation** | 3.5 s | 6.34 GiB | 44.1 kHz, 1,058,400 samples exact match |
| **32s Outro Extension + 4-bar Repair** | 15.5 s | 5.09 GiB | Inpaint + surgical EQ notch |
| **Total 6-Asset Batch Spend** | **~3.5 minutes** | Peak 8.6 GiB | **`0.00–0.04 Compute Units`** |

---

## 🎧 Production Demo: Cycling Mountain Metal Time Trial

A complete production batch is included in [`outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/):

| Asset ID | Role & Spec | QA Status | Diagnostics Highlights |
| :--- | :--- | :---: | :--- |
| [`01_sfx_countdown_3s.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/01_sfx_countdown_3s.wav) | 3s Start Countdown SFX (44.1k) | `passed` | 3 distinct 1s-spaced synth/horn pulses, 14.93 dB transient |
| [`02_bgm_main_loop_A.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/02_bgm_main_loop_A.wav) | Flat/Downhill Sprint Loop (24.0s) | `passed` | Best-of-3 Seed 2303, intro drop 0.26 dB, seam jump 0.0043 |
| [`03_bgm_main_loop_B.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/03_bgm_main_loop_B.wav) | Mountain Climb Loop (24.0s) | `passed` | Best-of-3 Seed 3303, seam jump 0.0013, SFX pocket 5.17% |
| [`04_sfx_pause_swap_bell.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/04_sfx_pause_swap_bell.wav) | Pause/Swap Bike Bell (1.5s) | `passed` | Transient 12.56 dB, 3–6 kHz clarity 70.18% |
| [`05_bgm_pause_swap_stem_layer.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer.wav) | Low-Density Pause State (24.0s) | `passed` | Zero onset lag (0.0 ms), 91.02% low-frequency rhythm pocket |
| ├─ [`stem_drums_bass.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_drums_bass.wav) | Isolated Rhythm Stem (44.1k) | `passed` | Dynamic interactive mixing |
| └─ [`stem_guitars_synths.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_guitars_synths.wav) | Isolated Lead & Harmony (44.1k) | `passed` | Dynamic interactive mixing |
| [`06_bgm_finish_outro.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/06_bgm_finish_outro.wav) | Triumphant Victory Outro (32.0s) | `passed` | On-GPU 4-bar summit cadence + surgical EQ notch to 3.45% |

Full diagnostics report: [04_v4_validation_report.md](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/04_v4_validation_report.md).

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- [Google Colab CLI](https://github.com/googlecolab/colab-cli) (`colab auth login`)
- Hugging Face Token with accepted access to [`stabilityai/stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium) stored in `~/.cache/huggingface/token`.

### 2. Run a Batch
1. Create a batch folder: `outputs/my-game-audio/`
2. Prepare `01_bgm_spec_card.md`, `02_arr_spec_ab.yaml`, and `manifest.json` using templates in `examples/`.
3. Copy `job_remote_template.py`, `generate_remote_template.py`, and `supervise_template.py` to your batch folder.
4. Run:
   ```bash
   cd outputs/my-game-audio
   python -u supervise.py
   ```
5. All `.wav` audio files and `.json` sidecars will be verified via SHA-256 and downloaded locally. The Colab VM is guaranteed to stop automatically upon completion.

---

## 🤝 Citations & Acknowledgments

This project builds upon the foundational research and open-source contributions of:
- **`kanaworksai-hue/audio-studio-mcp`**: MCP server implementation for Stable Audio 3.
- **`Stability-AI/stable-audio-3`**: High-fidelity latent diffusion audio model.
- **`sora16bit/codex-game-music-skill`**: Scene-first game music directing and critique contracts.
- **`jtydhr88/music-composition-skills`** & **`SJY051/music-composition`**: A/B arrangement specification and quantitative diagnostic tables.
- **`yzfly/awesome-music-prompts`**: Prompt compilation heuristics for generative audio.
- **`pytorch/audio`**: High-performance HDemucs music demixing models.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE) - see the LICENSE file for details.
