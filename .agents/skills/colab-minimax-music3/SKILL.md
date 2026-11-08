---
name: colab-minimax-music3
description: >-
  End-to-end workflow for generating full-length (~3-minute) music tracks using
  MiniMaxAI/MiniMax-Music3 on Google Colab NVIDIA L4 GPUs via the colab CLI.
  Use this skill whenever the user wants to generate background music (BGM),
  workout/cycling music, or game soundtracks with MiniMax-Music3 on Colab L4,
  export directly to loudness-normalized MP3, verify downloads locally, and
  automatically shut down the Colab VM within a strict Compute Unit budget.
---

# Colab L4 MiniMax-Music3 Music Generation Skill (v2 Optimized)

This skill provides a battle-tested, automated pipeline for generating full-length (~3-minute) instrumental or vocal music tracks with Hugging Face's [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3) on a Google Colab **NVIDIA L4 (24GB VRAM)** instance using the `colab` CLI.

---

## 1. Hardware, VRAM & Speed Optimizations (Empirical L4 Telemetry)

`MiniMaxAI/MiniMax-Music3` is a hybrid architecture combining:
- **8B Global LLM (`Qwen3ForCausalLM`)**: Predicts semantic music tokens at 25 frames/sec.
- **0.6B Local LLM (`MiniMaxMusic3RVQDepthDecoder`)**: Predicts 7 residual acoustic codebooks per frame.
- **2.4B Flow-Matching DiT (`MiniMaxMusic3Transformer1DModel`) + Condition Encoder**: Synthesizes continuous Flow-VAE latents in overlapping 200-frame windows (100-frame hop).
- **123M Flow-VAE Vocoder (`MiniMaxMusic3Vocoder`)**: Decodes latent windows into 44.1 kHz stereo audio.

When running on an NVIDIA L4 (`23,034 MiB` / `22.49 GiB` usable VRAM), enforce these 6 optimizations:

1. **Mandatory Stage-Level CPU Offload (`enable_auto_cpu_offload`)**:
   - **NEVER** disable CPU offload (`pipe.to("cuda")`) or set `batch_size > 1`. With `ComponentsManager.enable_auto_cpu_offload(device='cuda')` and `bfloat16`, peak allocated VRAM reaches **`19.34–19.54 GiB`** and reserved VRAM reaches **`21.79 GiB`**. Putting all 5 components on GPU simultaneously causes an immediate CUDA OOM crash.
   - `enable_auto_cpu_offload` swaps components **only 3 times per song** (Stage 1: 8B Global LLM + 0.6B RVQ stay resident on GPU together for all frames; Stage 2: Condition Encoder + 2.4B DiT Transformer; Stage 3: Vocoder).
2. **CUDA Allocator Fragmentation Prevention (`expandable_segments:True`)**:
   - Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the remote environment so variable-length KV caches in Stage 1 do not fragment VRAM before Stage 2 (DiT) and Stage 3 (Vocoder).
3. **Isolated Python 3.11 `uv venv` with Matched `cu126` Wheels + Fast Weight Download (`hf_transfer`)**:
   - **Never** use `--system-site-packages` on Colab: Colab's system environment mixes `torchaudio` (`cu130`) with newer PyPI `torch` (`cu132`), causing a `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions`. Always create an isolated `uv venv --python 3.11` and install matched `torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126` (which `uv` downloads and installs in ~15 seconds on Colab's 10 Gbps link).
   - Enabling `HF_HUB_ENABLE_HF_TRANSFER=1` (via `hf-transfer`) and `max_workers=8` in `snapshot_download` saturates Google Cloud's 10 Gbps link, cutting the ~22 GB checkpoint download to ~15–25 seconds.
4. **Enable Ada Lovelace `bfloat16` & `TF32`**:
   - Set `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True`.
   - Load pipeline components with `pipe.load_components(dtype=torch.bfloat16)`.
   - Rely on PyTorch 2.7's built-in `sdpa` FlashAttention-2 kernels (never compile `flash-attn` from source on Colab).
5. **Conditional Regional `torch.compile` for Large Batches ($\ge 3$ Tracks)**:
   - For 1–2 tracks, run in eager mode to avoid compilation warmup overhead.
   - For batches of **3 or more tracks**, optionally enable `torch.compile(pipe.transformer, mode="reduce-overhead")` to accelerate the 2,700+ fixed-shape DiT forward passes across songs.
6. **Tunable Flow-Matching Steps & Guidance (`ClassifierFreeGuidance`)**:
   - Default `num_inference_steps=30` (can be set per-track in `music_manifest.json`, e.g., `25` for ~17% faster Flow-Matching or `30` for maximum fidelity).
   - Default Flow-Matching CFG is `1.7` (`ClassifierFreeGuidance(guidance_scale=1.7)`), tunable per track via `flow_cfg` in `music_manifest.json`.
7. **Per-Song VRAM Garbage Collection**:
   - After encoding each song to `.mp3`, explicitly delete intermediate tensors and run `gc.collect()` + `torch.cuda.empty_cache()`.

---

## 2. Step 0: Pre-Flight Prompt Interview & Reference Sample Alignment (Mandatory Before Planning)

Because `MiniMax-Music3` uses an **8B `Qwen3` Global LLM** to interpret the structured prompt and compose 3–5 minutes of music (~15 minutes of L4 GPU time per track), vague prompts waste Compute Units on the wrong vibe.

**Before writing `music_manifest.json` or starting Colab L4**, evaluate what the user has specified and proactively ask about any missing dimensions below:

1. **Vocal vs. Instrumental (`Vocal Details`)**:
   - **100% Instrumental BGM** (no singing, no vocal chops, no choir), OR **Vocal Song** (specify male/female/duet, vocal timbre, delivery style, and lyrics)?
2. **Music Style, Scenario & Energy Contour (`Global Metadata`)**:
   - What is the exact scenario (e.g., flat-road cycling cruise, heavy resistance hill climb, final sprint, lobby/menu)?
   - What subgenre and rhythmic feel is desired (e.g., 90s Funky Disco House, House Piano & Synth Boogie, 128 BPM Tech House, High-Energy Drum & Bass, Lo-Fi Chill)?
3. **Reference Sample Alignment (YouTube Preview Links)**:
   - If the user has not provided a reference track or if the style description is broad (e.g., "upbeat rhythmic music"), **search the web for 2–3 distinct, verified YouTube sample links** matching candidate subgenres and let the user listen and pick one first. Only write the final `Qwen3` prompt after the user confirms the reference sample.
4. **Instrument Palette & Explicit Exclusions (`Arrangement`)**:
   - Which lead/rhythm instruments should drive the track (e.g., slap bass + brass stabs + wah guitar vs. percussive M1 house piano + analog synth bass)?
   - Are there elements to explicitly exclude (e.g., "no soft ambient pads", "no slow breakdowns/drops", "no brass/guitars")?
5. **BPM, Track Count & Duration**:
   - Target BPM (e.g., `124–126 BPM`), number of tracks, and target duration per track (default `~180s` / `minimax_max_seconds: 190`).
   - If generating $\ge 2$ tracks in the same style, confirm how they should be differentiated so they never sound like the same song.

---

## 3. Prompt Engineering Contract (Official `music-caption-rewriter` Rules)

### A. Three-Section Structured Caption (`250–450` English Words Recommended)
`MiniMax-Music3` expects its `prompt` parameter in a 3-section Markdown format (**never** include song titles, track IDs, or reasoning traces in `prompt`):
- `### Global Metadata`: Macro genre, subgenres, exact BPM, key/scale, meter (`steady four-on-the-floor`, etc.), target duration (~3 minutes), emotional progression, and sonic/production profile.
- `### Vocal Details`:
  - For **instrumental BGM**: Use **positive instrumental phrasing only**: `Purely instrumental composition throughout the entire piece. All lead melodies, rhythmic motifs, and harmonic layers are performed exclusively by <list exact lead, bass, harmony, and drum instruments>.` **Avoid negative vocal trigger words** (`singing`, `vocals`, `spoken words`, `vocal chops`, `choir`, `voice`), as mentioning vocal terms in positive LLM conditioning activates vocal token attention.
  - For **vocal songs**: Describe vocal gender, timbre, register, delivery, backing harmonies, and vocal effects.
- `### Arrangement`:
  - For **instrumental BGM**: **NEVER** use the words `Verse`, `Pre-Chorus`, `Chorus`, or `Bridge` in `### Arrangement`! Use purely instrumental section names (`Intro → Main Groove Theme A → Arpeggio Build → First Instrumental Hook → Main Groove Theme B → Filter Sweep Interlude → Solo → Peak Instrumental Theme → Outro`).
  - For **vocal songs**: Describe the song timeline (`Intro → First Verse → Pre-Chorus → First Chorus → Second Verse → Bridge / Solo → Final Chorus → Outro`).
  - For every section, state **what enters, exits, changes, or intensifies** to create a coherent 3-minute energy arc.

### B. Structural Tags (`lyrics` Parameter) & High-Energy Instrumental Strategy
- Each tag **must be on its own line** (any text on the same line as a leading bracketed tag is stripped by `_normalize_lyrics()`).
- **Important Trade-off Discovered on Colab L4 (Energy vs. Vocal Triggering)**:
  1. **Why `[Verse]` & `[Chorus]` Matter for High-Energy Rhythms**: In `MiniMax-Music3`'s training corpus, high-energy 120–130 BPM electronic, house, funk, and rock tracks all use `[Intro] → [Verse] → [Pre-Chorus] → [Chorus] → [Verse] → [Bridge] → [Solo] → [Chorus] → [Outro]`. If you remove `[Verse]` and `[Chorus]` and only pass `[Interlude]` + `[Solo]`, `Qwen3-8B` outputs 0% vocals natively (`-58.4 dBFS`), **but** shifts into a slow smooth-jazz/lounge tempo!
  2. **Mandatory Best Practice for High-Energy Instrumental BGM (Cycling / Workout / Action)**:
     - **Keep the full high-energy `[Intro] → [Verse] → [Pre-Chorus] → [Chorus] → [Verse] → [Bridge] → [Solo] → [Chorus] → [Outro]` structure** (with 8–16 lines of `[Instrumental]` underneath each tag) AND explicitly exclude slow genres (`Strictly exclude jazz, blues, saxophone, flute, woodwinds, or slow ballad rhythms`) so `MiniMax-Music3` generates its strongest 124+ BPM four-on-the-floor drums, bass, electric guitar, and synth groove.
     - **Run L4 GPU `HDEMUCS_HIGH_MUSDB_PLUS` (`remove_vocals_hdemucs`) Before MP3 Encoding**: Built directly into `torchaudio==2.7.1`, `HDemucs` runs in **~5 seconds** on L4, strips 100% of any vocal singing generated during `[Verse]`/`[Chorus]`, and keeps 100% of `drums + bass + other` (guitar/synths/percussion).
  3. **For Calm/Slow Instrumental BGM**: You may use `[Intro] → [Interlude] → [Solo] → [Outro]` with `[Instrumental]` lines.
- **For Vocal Songs**:
  - Use `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Solo]`, `[Outro]` with the actual lyrics lines underneath each vocal section header, and set `"remove_vocals": false` in `music_manifest.json`.

### C. Making Multiple Tracks Sound Distinct (5-Dimension Rule)
Never rely on changing `seed` alone! Differentiate tracks in the same genre across **5 dimensions**:
1. **BPM**: e.g., `124 BPM` vs `126 BPM`.
2. **Tonality/Key**: e.g., `E Dorian` (funky minor groove) vs `F Major` (bright uplifting major).
3. **Non-Overlapping Lead Instruments**: e.g., Track 1 uses *Brass Section Stabs + Thumb-Slapped Electric Bass + Wah-Wah Rhythm Guitar*; Track 2 uses *90s Percussive M1 House Piano + Octave-Jumping Analog Synth Bass + Sparkling Synth Lead* (explicitly excluding brass and guitar).
4. **Section Arrangement**: Vary the middle sections (e.g., `[Bridge]` with conga percussion workout vs `[Solo]` with analog synth filter sweep).
5. **Random Seed**: Use distant seeds (e.g., `510801` vs `884209`).

See [examples/sample_music_manifest.json](./examples/sample_music_manifest.json) for a complete two-track template.

---

## 4. Pure MP3 Direct Encoding Pipeline (Zero WAV Disk Footprint)

To save disk space and reduce download time by 95% (~4.36 MiB per track instead of 96 MiB WAV packs):
1. Trim leading/trailing digital silence (`threshold_db=-60.0`, `pad_ms=100.0`) and apply fade-in (`15 ms`) / fade-out (`1500 ms`) in memory on the `float32` stereo array.
2. Pipe the raw `f32le` PCM buffer directly to `ffmpeg` via `stdin` (`pipe:0`) for **two-pass EBU R128 loudness normalization** (`-16.0 LUFS`, `-1.5 dBFS` True Peak) and encode directly with `libmp3lame` (`-b:a 192k -ar 44100 -ac 2`) to `.mp3`.
3. Never write `_raw.wav`, `.wav`, or `.ogg` to disk.

---

## 5. Step-by-Step Execution Workflow

0. **Run Pre-Flight Interview & Sample Alignment (Section 2)**:
   Clarify vocal vs. instrumental, style/scenario, BPM/count/duration, instrument exclusions, and share YouTube reference samples for user confirmation before locking in prompts.
1. **Create a Dedicated Local Output Directory**:
   e.g., `output/audio-production/<YYYY-MM-DD>/<batch-name>`.
2. **Prepare `music_manifest.json`**:
   Adapt [examples/sample_music_manifest.json](./examples/sample_music_manifest.json) into the batch folder.
3. **Deploy the Remote & Local Scripts into the Batch Folder**:
   - [scripts/generate_remote_template.py](./scripts/generate_remote_template.py) → `generate_remote.py`
   - [scripts/job_remote_template.py](./scripts/job_remote_template.py) → `job_remote.py`
   - [scripts/supervise_template.py](./scripts/supervise_template.py) → `supervise.py` (configure `SESSION`, `EXPECTED_TRACKS`, and `MAX_BATCH_SPEND`).
4. **Verify Prerequisites Locally**:
   - Check `colab sessions` and `colab usage` to ensure no orphan VMs are running.
   - Verify local Python has `numpy` and `soundfile`, and `~/.cache/huggingface/token` exists.
5. **Run `supervise.py`**:
   Execute `python -u supervise.py`. It handles L4 creation with warmup retry, `.hf_token` upload & auto-shredding, incremental pack download, **4-Layer Local MP3 Verification** (ZIP CRC, SHA-256 hash match, `soundfile` full MP3 stream decode & RMS check), and **guaranteed `colab stop` shutdown** in its `finally` block.
