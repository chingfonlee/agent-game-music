# Step 1: 電玩配樂設計卡 (BGM & SFX Spec Card)
> 依據 `sora16bit/codex-game-music-skill` (`SKILL.md`, `scene-recipes.md`, `output-contracts.md`) 規範產出

## 1. 概念轉譯 (Concept Translation)
- **Cue 名稱**: `countryside_cycling_time_trial` (鄉村田野單車訓練台計時賽)
- **場景角色 (Scene Role)**: 陽光開闊的鄉村田野公路，結合室內單車訓練台（Smart Trainer）的運動節奏感與計時挑戰。
- **玩家動作與功能 (Gameplay Function)**:
  - 維持穩定踩踏踏頻（Cadence）：**120 BPM** 對應單車黃金有氧踏頻 **60 RPM（每兩拍一圈）或 120 RPM（每拍一圈）**，讓音樂拍點直接成為踩踏節拍器。
  - 四個明確的遊戲狀態機切換：
    1. **`[State 0: 3秒倒數開始]` (`01_sfx_countdown_3s`)**: 3 秒精準倒數提示音（3 聲短促電子滴聲 + 高音出發號角）。
    2. **`[State 1: 計時騎行主循環]` (`02_bgm_countryside_loop_A` / `03_bgm_countryside_loop_B`)**: 輕快、推進力強、陽光開闊的 12 小節（24.0 秒）無縫循環 BGM。
    3. **`[State 2: 暫停換人狀態]` (`04_sfx_pause_swap_bell` + `05_bgm_pause_swap_layer`)**: 觸發清脆單車鈴聲提示後，音樂無縫切換至「同 BPM、抽離主旋律、降密度的輕鬆伴奏層」，保持現場熱度但留白給玩家交談換人。
    4. **`[State 3: 計時結束衝線]` (`06_bgm_finish_outro`)**: 從主循環無縫延伸出收尾終止式（Outro Cadence）與餘韻。

## 2. 作曲與聲學表面設計 (Composition & Surface Design)
- **BPM / 拍號**: `120 BPM`, `4/4` 拍（每小節 `2.0 秒`）
- **調性 / 調式**: `G Major` / `G Mixolydian`（陽光、開闊、正向推進，無壓迫感）
- **無縫循環長度 (Loop Math)**:
  - 12 小節 = $12 \times \frac{240}{120} = 24.00\text{ 秒}$
  - 加上 `loop_crossfade_ms: 200`（`0.2 秒` 等功率交疊），模型生成長度設定為 **`24.2 秒`**，折疊後精確輸出 **`24.00 秒`** 無縫 WAV Loop。
- **點 / 線 / 面角色分配 (Point / Line / Plane Roles)**:
  - **Point (節奏點)**: 清脆的原聲木吉他刷弦 / 輕快電子合成器拔奏 (Pluck) + 四四拍大鼓與沙鈴 (Shaker)，穩定鎖住 120 BPM 踏頻。
  - **Line (旋律線)**: 明亮的長笛感合成器 / 口哨風主旋律（短促好記的 2–4 小節動機，避免過長連音造成聽覺疲勞）。
  - **Plane (空間面)**: 溫暖薄透的暖調電鋼琴與弦樂襯底（控制在中低頻，不填滿高頻）。
- **負空間保留 (Negative Space for SFX)**:
  - 嚴格限制 `3–6 kHz` 的刺耳銅鈸（避免長音 Crash Cymbal 轟炸），把高頻空間留給**3秒倒數音、單車鈴聲、鏈條轉動聲與 UI 計時讀秒音效**。
