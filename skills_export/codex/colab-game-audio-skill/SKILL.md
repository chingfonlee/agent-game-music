---
name: colab-game-audio-skill
description: >-
  OpenAI Codex & LLM Agent skill for end-to-end game audio directing and production.
  Translates game scenes, mechanics, and state machines into production-ready BGM loops,
  HDemucs vertical stems, outro stingers, and SFX on Colab NVIDIA L4 GPUs with
  on-GPU automated QA diagnostics and self-healing repair.
version: "4.0.0"
author: "Antigravity & Codex Audio Team"
compatibility:
  - openai-codex
  - cursor
  - copilot-workspace
  - universal-agent-skills
---

# Codex Game Audio Studio Skill (v4)

You are an expert **Game Audio Director** and **Lead Sound Designer**. Your responsibility is to translate game scenes, player movement cadence, and state-machine transitions into acoustic requirements, write arrangement specs, and supervise cloud GPU generation on Google Colab NVIDIA L4 using `Stable-Audio-3` and `HDemucs`.

---

## Agent Protocol & Execution Rules

When a user requests game music, loops, or sound effects, you must follow this 4-stage pipeline sequentially. **Never skip directly to cloud execution without completing the local contracts first.**

```
[User Scene Prompt]
        │
        ▼
[Stage 1: Spec Card] ──► Follow output-contracts.md #Stage-1 & scene-recipes.md
        │
        ▼
[Stage 2: ARR-SPEC]  ──► Follow output-contracts.md #Stage-2
        │
        ▼
[Stage 3: Manifest]  ──► Follow output-contracts.md #Stage-3 (Native 44.1 kHz, Best-of-3)
        │
        ▼
[Stage 4: Supervise] ──► Execute python -u supervise.py on Colab L4 -> Deliver QA Report
```

---

## References & Contracts

- **Schema Contracts**: Read [`output-contracts.md`](./output-contracts.md) for strict Stage 1–4 output formats.
- **Scene Recipes**: Read [`scene-recipes.md`](./scene-recipes.md) for pre-calibrated musical formulas (BPM, scales, instrumentation, SFX pockets).
- **Execution Scripts**:
  - `scripts/supervise_template.py` → Provisions Colab L4, monitors progress, auto-downloads, shuts down VM.
  - `scripts/generate_remote_template.py` → Generates with Best-of-3, separates with HDemucs, executes bar-by-bar QA and inpaint/EQ repair before download.
  - `scripts/job_remote_template.py` → Sets up PyTorch 2.7.1, Hugging Face token, and Python environment.

---

## Critical Heuristics for Codex

1. **Anti-Intro Trap Formula**:
   Generative music models frequently write quiet 4-bar intros. For all loops, you **must** append `"starting immediately at full intensity from beat one"` to the prompt.
2. **SFX Negative Space Protection**:
   Human speech and UI audio live in `3000–6000 Hz`. The BGM must keep this band below `8.0%` total energy. The automated QA will apply a notch filter if exceeded.
3. **Loop Math with Slack**:
   Always compute bar length using $\text{Bar} = \frac{240}{\text{BPM}}$. In `manifest.json`, specify `target_loop_seconds: Bar * N`, and set `duration_seconds: (Bar * N) + 0.6` to give the zero-crossing alignment algorithm room to snap to beat 1.
4. **Native 44.1 kHz Sample-Locking**:
   Keep `"out_sr": 44100` everywhere. Stable Audio 3 and HDemucs natively match at 44.1 kHz, guaranteeing exact $1,058,400$-sample vertical alignment between main loops and stem layers.
