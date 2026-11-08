# Step 1: 電玩配樂設計卡 (BGM & SFX Spec Card) — 丘陵山地重金屬篇
> 依據 `sora16bit/codex-game-music-skill` (`SKILL.md`, `scene-recipes.md`, `output-contracts.md`) 與自動診斷修復規範產出

## 1. 概念轉譯 (Concept Translation)
- **Cue 名稱**: `mountain_climb_metal_time_trial` (丘陵山地單車訓練台計時賽 — 重金屬節奏)
- **場景角色 (Scene Role)**: 險峻起伏的丘陵山地公路，視覺上具有坡度爬升與高速下坡的張力，需要高腎上腺素、高扭矩推進感的重金屬器樂 BGM。
- **玩家動作與功能 (Gameplay Function)**:
  - **高阻力爬坡與抽車踏頻鎖定**：設定 **`120 BPM`**（`4/4` 拍，每小節精確 `2.00 秒`）。四分音符大鼓與小鼓直接鎖定 **60 RPM（站姿重齒比抽車）** 與 **120 RPM（平路/下坡高轉速衝刺）**，並以十六分音符悶音電吉他（Palm-Muted Gallop）提供密集的推進感。
  - **四個遊戲狀態機切換**：
    1. **`[State 0: 3秒倒數開始]` (`01_sfx_countdown_3s`)**: 3 聲重裝金屬計時器敲擊聲 + 第 4 拍高亢出發氣笛與電吉他強力和弦刷奏（Power Chord Hit）。
    2. **`[State 1: 山地騎行主循環]` (`02_bgm_metal_loop_A` / `03_bgm_metal_loop_B`)**: 12 小節（`24.00 秒`）滿編重金屬無縫 Loop（**強制從第 1 拍就進入完整鼓組與貝斯，禁止出現弱前奏**）。
    3. **`[State 2: 暫停換人狀態]` (`04_sfx_pause_swap_bell` + `05_bgm_pause_swap_layer`)**: 清脆車鈴與棘輪咬合聲後，切換至 `edit_variation` 衍生的「Clean/Crunch 電吉他分解和弦 + 貝斯與閉合ハイハット（Closed Hi-Hat）低密度律動層」（`-18 LUFS`），退去破音主奏吉他，讓換人溝通清晰但維持戰意。
    4. **`[State 3: 登頂衝線結束]` (`06_bgm_finish_outro`)**: 從主循環第 `23.0s` 處無縫延伸至 `32.0s`，必須在第 `24–28s` 爆發雙吉他勝利終止和弦與雙踏過門，再於 `28–32s` 乾淨收尾。

---

## 2. 聲學表面設計與重金屬頻段防衝突策略 (Surface & Negative Space)
重金屬音樂最大的風險是**失真電吉他（Distorted Guitars）與銅鈸（Cymbals）佔滿 `3–6 kHz` 高中頻**，容易蓋過遊戲內的計時器與鈴聲 SFX。因此本設計採取以下聲學控制：
- **Point (節奏點)**: 緊湊有彈性的雙踏大鼓（Punchy Kick）、清脆小鼓與悶音電吉他切音（Palm-Muted Chugs），集中於 `80–250 Hz` 與 `500–1500 Hz`。
- **Line (旋律線)**: 雙吉他旋律 Riff（Twin-Lead Guitar Motif），控制在中頻 `800–2800 Hz`。
- **Negative Space (SFX 保護區 `3–6 kHz`)**:
  - 提示詞明確要求 `tight controlled mix, crisp drums without harsh cymbal wash`。
  - **遠端自動頻譜防禦（Auto-EQ Gate）**：若遠端檢測 `3–6 kHz` 能量佔比超過 `8.0%`，自動在 `4.2 kHz` 施加 `-3.5 dB` 動態凹槽濾波（Notch/Dip EQ），確保 `01_sfx_countdown_3s` 與 `04_sfx_pause_swap_bell` 100% 清晰可讀。

---

## 3. 遠端下載前自動驗收與自動修復閘門 (Pre-Download Auto-QA & Repair Gates)
在 Colab L4 遠端模型尚未卸載、尚未打包下載與關機前，`generate_remote.py` 會對每一首生成的音訊執行以下 **4 道自動化逐小節診斷與當場修復**：

| 閘門編號 | 檢查對象 | 量化通過標準 (Pass Criteria) | 未通過時的遠端 GPU 自動修復動作 (Auto-Repair before Download) |
| :--- | :--- | :--- | :--- |
| **Gate 1: Loop 首尾小節能量連續性** | `02_loop_A`<br>`03_loop_B`<br>`05_pause_layer` | • 尾段小節與第 1–2 小節 RMS 落差 `intro_drop_db <= 2.5 dB`（嚴防前奏陷阱）<br>• 首尾樣本跳變 `seam_jump <= 0.02` | 1. **當場 `edit_inpaint` 重繪前 4 小節 (`0.0–8.0s`)**：強制注入滿編重金屬鼓組與悶音吉他；<br>2. 若仍超過閾值，自動更換 Seed 重生或擷取最穩定連續小節段並重新執行 `make_loop(200ms)`。 |
| **Gate 2: SFX 頻段負空間保護** | 所有 BGM (`02`, `03`, `05`, `06`) | • `3000–6000 Hz` 能量佔比 `<= 8.0%` | 自動呼叫 `ffmpeg` 針對 `4200 Hz` 進行 `-3.5 dB` 溫和衰減，再重新執行 EBU R128 `-16/-18 LUFS` 標準化。 |
| **Gate 3: 衝線尾奏高潮與收尾檢查** | `06_bgm_finish_outro` | • 衝線高潮段 (`24.0–28.0s`) RMS 不得低於主循環超過 `3.5 dB`<br>• 高潮段中頻 (`250–3000 Hz`) 佔比 `>= 20%`（防止只有低頻空響） | 自動在 GPU 上更換 Seed（最多重試 3 組候選 `edit_continue`），挑選 `24–28s` 終止和弦最飽滿、且尾音乾淨收束的版本。 |
| **Gate 4: SFX 瞬態與非靜音分布** | `01_countdown`<br>`04_pause_bell` | • 有效發聲區間涵蓋率達標（3秒倒數在 0–1s, 1–2s, 2–3.5s 均需有明顯瞬態能量） | 自動更換 Seed 在 GPU 上重新生成（耗時僅 `1.5s`），直到三段倒數能量皆達標。 |
