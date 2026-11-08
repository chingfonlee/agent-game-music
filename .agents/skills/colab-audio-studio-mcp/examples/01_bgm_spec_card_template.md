# Step 1: 電玩配樂設計卡 (BGM & SFX Spec Card Template)
> 依據 `sora16bit/codex-game-music-skill` (`SKILL.md`, `scene-recipes.md`, `output-contracts.md`) 與自動驗收規範產出

## 1. 概念轉譯 (Concept Translation)
- **Cue 名稱**: `<scene_cue_name>`
- **場景角色 (Scene Role)**: `<描述視覺場景、環境氛圍與運動/戰鬥張力>`
- **玩家動作與功能 (Gameplay Function)**:
  - **節拍與動作鎖定**: 設定 `<BPM> BPM` (`4/4` 拍，每小節 $\frac{240}{\text{BPM}}$ 秒)。
  - **遊戲狀態機切換 (State Machine)**:
    1. **`[State 0: 開場/倒數 SFX]` (`01_sfx_...`)**: `<描述開場或倒數音效>`
    2. **`[State 1: 主循環 BGM A/B]` (`02_bgm_main_loop_A` / `03_bgm_main_loop_B`)**: `<小節數>` 小節無縫 Loop（從第 1 拍即滿編進入，無慢速前奏）。
    3. **`[State 2: 暫停/低威脅/換人分層]` (`04_sfx_...` + `05_bgm_pause_swap_layer`)**: 透過 `edit_variation` 抽離主奏旋律，保留同 BPM 律動與低密度伴奏（`-18 LUFS`）。
    4. **`[State 3: 結束/衝線終止式]` (`06_bgm_finish_outro`)**: 透過 `edit_continue` 無縫延伸出終止高潮與殘響收尾。

## 2. 聲學表面與負空間設計 (Surface & Negative Space)
- **Point (節奏點)**: `<大鼓、小鼓、悶音切音或節奏撥弦>`
- **Line (旋律線)**: `<主奏樂器與動機音域>`
- **Plane (空間面)**: `<和聲襯底與空間感>`
- **Negative Space (SFX 保護區 `3–6 kHz`)**: 控制 `3–6 kHz` 能量佔比 `<= 8.0%`，確保 UI 與關鍵遊戲 SFX 清晰可讀。

## 3. 遠端下載前自動驗收與當場修復閘門 (Pre-Download Auto-QA Gates)
- **Gate 1 (Loop 首尾能量連續性)**: `intro_drop_db <= 2.5 dB`、`seam_jump <= 0.02`（未達標時當場執行 GPU `edit_inpaint` 與小節對齊律動置換）。
- **Gate 2 (SFX 頻段負空間保護)**: `3–6 kHz` 佔比 `<= 8.0%`（超標時自動執行 `-4.0 dB` 頻段微調）。
- **Gate 3 (Outro 終止高潮檢查)**: 延伸高潮段音量落差 `<= 3.5 dB`（未達標時當場換 Seed 重試 `edit_continue` 並強化終止段）。
- **Gate 4 (SFX 瞬態涵蓋率檢查)**: 確認倒數/多段式 SFX 每個時間窗均具明顯瞬態能量（`> -32 dBFS`）。
