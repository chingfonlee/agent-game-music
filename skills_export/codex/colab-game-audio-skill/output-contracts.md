# Codex Game Audio Skill: Output Contracts & Schemas

Codex models and agents must strictly follow these schemas when emitting outputs for each stage.

---

## Stage 1: BGM & SFX Spec Card (`01_bgm_spec_card.md`)

```markdown
# Step 1: 電玩配樂設計卡 (BGM & SFX Spec Card)

## 1. 概念轉譯 (Concept Translation)
- **Cue 名稱**: `<identifier>`
- **場景角色 (Scene Role)**: `<Visual scene & tension>`
- **玩家動作與功能 (Gameplay Function)**:
  - **節拍與動作鎖定**: `<BPM> BPM` (`4/4` 拍，每小節 240/BPM 秒)
  - **遊戲狀態機 (State Machine)**:
    1. `[State 0: 開場/倒數 SFX]`: `<Description>`
    2. `[State 1: 主循環 BGM A/B]`: `<Bars> 小節無縫 Loop`
    3. `[State 2: 暫停/換人低密度分層]`: `HDemucs 抽離主奏旋律`
    4. `[State 3: 結束/衝線終止式]`: `edit_continue 終止高潮`

## 2. 聲學表面與負空間設計
- **Point (節奏點)**: `<Drums / percussion>`
- **Line (旋律線)**: `<Lead instruments>`
- **Plane (空間面)**: `<Pads / chords / reverb>`
- **Negative Space (SFX 保護區 3–6 kHz)**: 能量佔比 `<= 8.0%`
```

---

## Stage 2: Arrangement Spec (`02_arr_spec_ab.yaml`)

```yaml
version: "4.0"
cue_id: "<identifier>"
invariants:
  bpm: <number>
  time_signature: "4/4"
  bar_seconds: <240/BPM>
  sample_rate: 44100
  target_lufs: -16.0
  sfx_pocket_max_energy_pct: 8.0

version_a:
  label: "Main Loop A (Fast / Motivating)"
  lead: "<lead instrument>"
  harmony: "<rhythm chords>"
  bass: "<bassline>"
  drums: "<drum pattern>"

version_b:
  label: "Main Loop B (High Pressure / Intensity)"
  lead: "<modified lead>"
  harmony: "<denser harmony>"
  bass: "<heavier bass>"
  drums: "<double kick or faster drums>"

stems:
  pause_layer:
    source: "version_a"
    drums: 1.0
    bass: 1.0
    other: 0.25
    other_lowpass_hz: 2200.0
    vocals: 0.0
```

---

## Stage 3: Manifest Schema (`manifest.json`)

```json
[
  {
    "id": "01_sfx_countdown",
    "type": "generate",
    "audio_type": "sfx",
    "duration_seconds": 3.5,
    "seed": 1101,
    "steps": 8,
    "cfg": 1.0,
    "sfx_fields": ["source", "action", "material", "space"],
    "qa": { "role": "countdown_sfx", "min_active_windows": 3 },
    "post": { "out_sr": 44100, "trim_silence": true, "normalize": "peak", "peak_ceiling_db": -1.0, "format": "wav" }
  },
  {
    "id": "02_bgm_main_loop_A",
    "type": "generate",
    "audio_type": "music",
    "candidates": 3,
    "duration_seconds": 24.6,
    "bpm": 120,
    "seed": 2202,
    "steps": 8,
    "cfg": 1.0,
    "prompt": "<Genre> starting immediately at full intensity from beat one with <Instruments>, <Mood>",
    "qa": {
      "role": "bgm_loop",
      "bar_seconds": 2.0,
      "max_intro_drop_db": 2.5,
      "max_sfx_zone_pct": 8.0,
      "max_seam_jump": 0.02,
      "inpaint_fix_prompt": "<Inpaint prompt if bar 1 is quiet>"
    },
    "post": {
      "out_sr": 44100,
      "target_lufs": -16.0,
      "target_loop_seconds": 24.0,
      "loop_crossfade_ms": 200.0,
      "format": "wav"
    }
  },
  {
    "id": "05_bgm_pause_swap_stem_layer",
    "type": "stem_layer",
    "source_id": "02_bgm_main_loop_A",
    "stem_weights": { "drums": 1.0, "bass": 1.0, "other": 0.25, "vocals": 0.0 },
    "other_lowpass_hz": 2200.0,
    "export_all_stems": true,
    "qa": { "role": "bgm_stem_layer", "bar_seconds": 2.0 },
    "post": { "out_sr": 44100, "target_lufs": -18.0, "format": "wav" }
  },
  {
    "id": "06_bgm_finish_outro",
    "type": "edit_continue",
    "source_id": "02_bgm_main_loop_A",
    "audio_type": "music",
    "duration_seconds": 32.0,
    "bpm": 120,
    "seed": 6606,
    "steps": 8,
    "cfg": 1.0,
    "prompt": "<Triumphant ending with sustained chords and reverb>",
    "qa": {
      "role": "bgm_outro",
      "bar_seconds": 2.0,
      "climax_start_sec": 24.0,
      "climax_end_sec": 28.0,
      "max_climax_drop_db": 3.5,
      "min_climax_relative_mid_ratio": 1.15,
      "max_sfx_zone_pct": 8.0
    },
    "post": { "out_sr": 44100, "target_lufs": -16.0, "fade_out_ms": 600.0, "format": "wav" }
  }
]
```
