---
name: colab-audio-studio-mcp
description: >-
  Claude Code game audio skill for designing, generating, diagnosing, and delivering
  bar-aligned game music loops, stem layers, outros, and SFX on Google Colab NVIDIA L4 GPUs
  using Stability-AI/stable-audio-3 and TorchAudio HDemucs. Features native 44.1 kHz,
  tri-state QA (passed/warning/failed), relative climax mid-ratio, universal SFX notch filtering,
  and guaranteed VM shutdown.
tags:
  - claude-code
  - game-audio
  - stable-audio-3
  - colab-l4
  - hdemucs
  - audio-qa
---

# Colab L4 Game Audio Studio Skill for Claude Code (v4)

This skill enables Claude Code to act as an autonomous Game Audio Director and Sound Designer, producing production-ready game audio batches on a Google Colab **NVIDIA L4 (24GB VRAM)** GPU instance with pre-download QA and automatic repair.

---

## Claude Code Workflow Philosophy

1. **Local Planning First**: Always complete Stages 1–3 locally (Spec Card, Arrangement Spec, Manifest) and confirm with the user before touching Colab.
2. **Terminal Execution Safety**: Use non-interactive commands. Never run blocking interactive processes without timeouts.
3. **Guaranteed GPU Shutdown**: Ensure `colab stop` is called in the `finally` block of any script to prevent runaway Compute Unit billing.
4. **Native 44.1 kHz End-to-End**: Stable Audio 3 and HDemucs both operate at 44.1 kHz. Never resample to 48 kHz.

---

## 4-Stage Execution Pipeline

```
Stage 1: Scene & State Machine Design (Local)
   │  → 01_bgm_spec_card.md (BPM, Bars, 4-Field SFX, 3-6kHz Negative Space)
   ▼
Stage 2: A/B Arrangement Spec & QA Gates (Local)
   │  → 02_arr_spec_ab.yaml (Lead / Harmony / Bass / Drums, Target LUFS)
   ▼
Stage 3: Prompt Compilation & Manifest (Local)
   │  → manifest.json (Best-of-N, Loop Math, HDemucs Stems, Outro Continue)
   ▼
Stage 4: Colab L4 Generation & Auto-QA (Remote GPU + Download)
      → Run python -u supervise.py
      → Best-of-3 selection on GPU
      → Bar-by-bar QA + On-GPU inpaint/EQ repairs BEFORE download
      → SHA-256 download verification
      → Guaranteed automatic VM shutdown
```

---

## Stage Checklist for Claude Code

### Stage 1: Scene Design (`01_bgm_spec_card.md`)
- Copy `examples/01_bgm_spec_card_template.md` to target directory.
- Define game state transitions:
  - State 0: Countdown SFX (`01_sfx_...`)
  - State 1: Main Loop A / B (`02_bgm_main_loop_A`, `03_bgm_main_loop_B`)
  - State 2: Pause / Low-Intensity Stem Layer (`05_bgm_pause_swap_stem_layer`)
  - State 3: Finish / Outro (`06_bgm_finish_outro`)
- Lock BPM and bar duration: $\text{Bar Duration (s)} = \frac{240}{\text{BPM}}$.

### Stage 2: Arrangement Spec (`02_arr_spec_ab.yaml`)
- Copy `examples/02_arr_spec_ab_template.yaml` to target directory.
- Define shared invariants (BPM, LUFS, sample rate `44100`).
- Define Version A (fast/driving) and Version B (heavy/pressured) instrument rosters.
- Specify frequency allocation:
  - Low (<250 Hz): Kick, sub-bass
  - Mid (250–3000 Hz): Guitars, synths, leads
  - SFX Pocket (3000–6000 Hz): Maintained $\le 8.0\%$ energy to allow UI cues through

### Stage 3: Manifest Compilation (`manifest.json`)
- Copy `examples/sample_manifest.json` to target directory.
- For Loops: add `+0.4s` to `duration_seconds` for alignment slack:
  $$\text{duration\_seconds} = \left(\text{Bars} \times \frac{240}{\text{BPM}}\right) + \frac{\text{crossfade\_ms}}{1000} + 0.4$$
- Use `"candidates": 3` for Best-of-N reranking on main loops.
- Use `"type": "stem_layer"` for sample-locked pause/low-intensity layers.
- For Outro: use `"type": "edit_continue"` with `"min_climax_relative_mid_ratio": 1.15`.
- Present the planned assets to user and get confirmation.

### Stage 4: Run Supervisor & Verify
Deploy the 3 scripts to the batch directory:
```bash
cp scripts/job_remote_template.py <batch_dir>/job_remote.py
cp scripts/generate_remote_template.py <batch_dir>/generate_remote.py
cp scripts/supervise_template.py <batch_dir>/supervise.py
```

Run in terminal:
```bash
cd <batch_dir>
python -u supervise.py
```

When execution completes:
1. Confirm all assets reported `qa_report.status: "passed"` in `verification.json`.
2. Verify local SHA-256 integrity and audio playback.
3. Write `04_v4_validation_report.md` summarizing diagnostics.

---

## Troubleshooting Guide for Claude Code

| Issue | Cause | Resolution in Claude Code |
| :--- | :--- | :--- |
| `KeyError: 'loop_chroma_similarity'` | Old script template | Ensure `bar_chroma_similarity` and `seam_chroma_similarity` are used. |
| Outro QA fails relative mid check | High-frequency splash dominating | Check `repairs_applied` in sidecar JSON; 4-bar summit extension or surgical notch will automatically fire. |
| Colab session timeout or orphan VM | Interrupted execution | Run `colab sessions` followed by `colab stop <session_id>` immediately. |
| Resampling phase lag on stems | Using 48k for HDemucs | Set `"out_sr": 44100` on all items in `manifest.json`. |
