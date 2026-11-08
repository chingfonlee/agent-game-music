# 單車訓練台計時賽遊戲音效與重金屬配樂 v4 驗證成果報告

> **驗證批次 ID**：`cycling-mountain-metal-v4` (Task `task-218`)  
> **硬體環境**：Google Colab NVIDIA L4 GPU (24GB VRAM)  
> **核心架構**：Stability-AI/stable-audio-3 + TorchAudio HDemucs (MUSDB-HQ) 原生 44.1 kHz 全鏈路  
> **成果總結**：6 組音訊資產全數通過 QA（Status: `passed` 100%），雲端 GPU 實體自動關機，CU 消耗 0.00 Units。

---

## 1. 核心優化改動驗證 (v3 vs v4)

| 改善項目 | 原始設計 (v3) | v4 實裝與驗證成果 | 效益評估 |
| :--- | :--- | :--- | :--- |
| **取樣率架構** | 48 kHz 轉換模式 (44.1k → 48k → 44.1k HDemucs → 48k) | **全原生 44.1 kHz 模式** (Stable Audio 3 + HDemucs 原生匹配) | 消除 3 次重取樣相位耗損，取樣數精確匹配 1,058,400 samples (零相位位移) |
| **QA 狀態分級** | 二元布林值 (`passed: true/false`) | **三段式狀態** (`passed` / `warning` / `failed`) | 提升容錯與診斷細緻度，所有非阻斷性異常均記錄於 `status_reasons` |
| **尾奏高潮檢驗** | 絕對中頻能量門檻 (要求中頻 > 20%) | **相對中頻比** (`climax_mid / body_mid >= 1.15`) | 成功解決重金屬低頻主導（低頻佔 80%）誤判問題，尾奏修復順利達標 |
| **SFX 負空間保護** | 僅針對 Main Loop 進行 3–6 kHz 壓制 | **全 BGM 角色自動 3–6 kHz 凹槽濾波** (Loop, Outro, Stem Layer) | Outro 頻段由 8.23% 壓制至 3.45%，保證遊戲鈴聲/倒數穿透度 |
| **循環接縫對齊** | ±0.5 拍單側搜尋，弱和聲易落空 | **±1.0 拍雙向相關搜尋 + 自動 Inpaint 備援** | Loop A 接縫跳變低至 0.0043，Loop B 低至 0.0013，接縫聽感無縫 |
| **指標命名清晰化** | 混用 `loop_chroma_similarity` | 分離為 `bar_chroma_similarity` 與 `seam_chroma_similarity` | 語意明確，避免診斷報告日誌鍵名衝突 |

---

## 2. 生成資產清單與診斷數據

全數資產均保存在本地目錄：
`c:\Users\user\claudecode\serpentine\outputs\game-audio-2026-11-08-cycling-mountain-metal-v4\results`

### 1. `01_sfx_countdown_3s.wav` (3 秒發車倒數嗶聲)
- **類型**：SFX / Procedural Synth (3 個 1000Hz/1500Hz 階梯脈衝)
- **規格**：44.1 kHz, 24-bit PCM Stereo, 長度 3.0 秒
- **QA 指標**：
  - 瞬態清晰度：14.93 dB (門檻 >= 4.0 dB)
  - 瞬態事件數：3 個（精確落在 0.0s, 1.0s, 2.0s）
  - 狀態：`passed`
- **本地檔案**：[01_sfx_countdown_3s.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/01_sfx_countdown_3s.wav)

### 2. `02_bgm_main_loop_A.wav` (主幹騎行循環 A - 疾速平路/下坡)
- **類型**：BGM Loop (Best-of-3 評分機制，由種子 2202, 2303, 2404 競賽)
- **選勝種子**：Seed 2303（綜合評分 112.50 分獲勝）
- **規格**：44.1 kHz, 24-bit PCM Stereo, 長度 24.0 秒 (BPM 120, 12 小節)
- **QA 指標**：
  - 開頭衝擊力落差 (`intro_drop_db`)：0.26 dB（第 1 小節開頭無拖沓）
  - 小節色度一致性 (`bar_chroma_similarity`)：0.9590
  - 接縫能量突躍 (`seam_jump`)：0.0043
  - 接縫和聲相似度 (`seam_chroma_similarity`)：0.8004
  - 3–6 kHz SFX 避讓區間：7.60% (<= 8.0%)
  - 狀態：`passed`
- **本地檔案**：[02_bgm_main_loop_A.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/02_bgm_main_loop_A.wav)

### 3. `03_bgm_main_loop_B.wav` (爬坡加壓循環 B - 丘陵攻頂)
- **類型**：BGM Loop (Best-of-3 評分機制，由種子 3101, 3202, 3303 競賽)
- **選勝種子**：Seed 3303（綜合評分 109.91 分獲勝）
- **規格**：44.1 kHz, 24-bit PCM Stereo, 長度 24.0 秒 (BPM 120, 12 小節)
- **QA 指標**：
  - 開頭衝擊力落差 (`intro_drop_db`)：1.69 dB
  - 小節色度一致性 (`bar_chroma_similarity`)：0.9317
  - 接縫能量突躍 (`seam_jump`)：0.0013
  - 3–6 kHz SFX 避讓區間：5.17% (<= 8.0%)
  - 狀態：`passed`
- **本地檔案**：[03_bgm_main_loop_B.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/03_bgm_main_loop_B.wav)

### 4. `04_sfx_pause_swap_bell.wav` (暫停/換人提示金屬單車鈴聲)
- **類型**：SFX / Physical Acoustic Model + High-Q Resonance
- **規格**：44.1 kHz, 24-bit PCM Stereo, 長度 1.5 秒
- **QA 指標**：
  - 瞬態清晰度：12.56 dB (門檻 >= 4.0 dB)
  - 3–6 kHz 主頻能量比：70.18% (典型穿透鈴聲)
  - 狀態：`passed`
- **本地檔案**：[04_sfx_pause_swap_bell.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/04_sfx_pause_swap_bell.wav)

### 5. `05_bgm_pause_swap_stem_layer.wav` (暫停換人狀態動態背景層)
- **類型**：TorchAudio HDemucs 分軌重組層 (以 Loop A 為來源抽離高頻主奏，保留重低音與鼓組)
- **規格**：44.1 kHz, 24-bit PCM Stereo, 長度 24.0 秒
- **對齊精度與分軌指標**：
  - 取樣數匹配：1,058,400 / 1,058,400 (完美等長)
  - 主循環波形相關係數：0.9364
  - 量測起音延遲：0.0 ms (無任何相位延遲)
  - 低頻佔比：91.02% (高頻吉他成功抽離，留出對話與換人語音空間)
  - 狀態：`passed`
- **附帶導出獨立 Stem**：
  - 鼓與貝斯分軌：[05_bgm_pause_swap_stem_layer_stem_drums_bass.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_drums_bass.wav)
  - 吉他與合成器分軌：[05_bgm_pause_swap_stem_layer_stem_guitars_synths.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_guitars_synths.wav)
- **本地檔案**：[05_bgm_pause_swap_stem_layer.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer.wav)

### 6. `06_bgm_finish_outro.wav` (衝線完賽勝利尾奏)
- **類型**：Stable Audio 3 `edit_continue` 延展接續 + 4-bar Summit Cadence Extension
- **規格**：44.1 kHz, 24-bit PCM Stereo, 長度 32.0 秒 (前 24s 承接 Loop A，第 24~32s 為勝利高潮與殘響尾奏)
- **在線修復與診斷歷程**：
  - Initial Diagnositcs: 發現原始生成的種子在高潮段中頻提升不明顯 (0.91x~0.99x) 且 3–6 kHz 能量超標至 8.23%
  - 自動在線修復觸發：
    1. `retry_outro_seed_6717` (備選重試)
    2. `retry_outro_seed_6828` (備選重試)
    3. `climax_4bar_summit_cadence_extension_with_1800ms_fade` (注入終止式和弦延音)
    4. `surgical_sfx_pocket_eq_3to6k` (精準壓低 3–6 kHz 至 3.45%)
  - 最終結果：相對中頻能量達標，SFX 頻段壓至 3.45%，淡出正常無突兀截斷。
  - 狀態：`passed`
- **本地檔案**：[06_bgm_finish_outro.wav](file:///c:/Users/user/claudecode/serpentine/outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/06_bgm_finish_outro.wav)

---

## 3. 雲端執行效能與成本結算

- **Colab GPU 型號**：NVIDIA L4 (24GB VRAM)
- **執行耗時分析**：
  - 環境初始化 (uv venv + torch + soundfile)：30.8 秒
  - 模型權重載入 (Stable Audio 3 + HDemucs)：33.6 秒
  - 6 組資產生成 + Best-of-3 評分 + HDemucs 分軌 + 逐小節診斷修復總耗時：約 110 秒
  - 產物壓縮、本地 SHA-256 下載驗證：約 8 秒
  - **總連線時間**：約 3.5 分鐘
- **Compute Unit (CU) 消耗**：0.00 Units (在免費或基本配額寬限內完成，未產生超額點數扣除)
- **VM 狀態**：作業結束後觸發 `colab stop`，已確認完全關閉（`No active sessions found on server`）。
