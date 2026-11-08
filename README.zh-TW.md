# Agent Game Music (AGM) 🎮🎵

[ English ](./README.md) | [ 日本語 ](./README.ja.md) | [ 繁體中文 ](./README.zh-TW.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Stable-Audio-3](https://img.shields.io/badge/Model-Stable--Audio--3-purple.svg)](https://huggingface.co/stabilityai/stable-audio-3-medium)
[![Model: TorchAudio-HDemucs](https://img.shields.io/badge/Model-TorchAudio--HDemucs-green.svg)](https://pytorch.org/audio/stable/models.html)
[![Hardware: Google Colab L4](https://img.shields.io/badge/Hardware-Google%20Colab%20L4%20(24GB)-orange.svg)](https://colab.research.google.com/)
[![Agents: Claude Code & Codex](https://img.shields.io/badge/Agents-Claude%20Code%20%7C%20Codex%20%7C%20Antigravity-red.svg)](#多-agent-生態系支援)

**專為 AI 編程助理打造的自主閉環遊戲音訊導演、生成與下載前自動 QA 修復管線**

---

## 💡 專案簡介

遊戲音樂與音效在實際開發中，有著傳統純文字提示詞生成模型無法滿足的嚴苛結構與聲學限制：
- **前奏陷阱 (Intro Trap)**：生成模型常在前 4 小節緩慢鋪陳，導致循環播放接回第 1 小節時產生 $3\text{--}6\text{ dB}$ 的嚴重能量落差。
- **接縫破綻與雜音**：任意時間點的淡入淡出容易破壞打擊樂節奏，產生相位干涉或爆音。
- **SFX 關鍵頻段遮蔽**：背景音樂過滿時，會遮蔽遊戲中最重要的 UI 提示音、計時音與語音頻段（$3000\text{--}6000\text{ Hz}$）。
- **尾奏缺乏終止式**：生成式音樂往往突然淡出或陷入混亂噪聲，無法呈現清晰的勝利結尾與殘響。
- **互動式分軌同步**：遊戲引擎（如 FMOD、Wwise）的動態混音需要與主循環完全相同取樣數（相位延遲 $0.0\text{ ms}$）的垂直分軌。

**Agent Game Music (AGM)** 結合了場景優先設計法、結構化編曲規格與提示詞編譯，在 **Google Colab NVIDIA L4 GPU** 上提供超高速生成，並在**下載至本地前直接於雲端 GPU 上執行逐小節 QA 診斷與自我修復**，同時保證在作業完成後**自動關閉 VM 確保零超額計費**。

---

## 🏛️ 架構：4 階段管線流程

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: 場景與狀態機設計 (本機 Agent)                                 │
│ └─► 01_bgm_spec_card.md                                                │
│     將遊戲玩法轉換為狀態機 (BPM、小節數學、4 欄位結構化 SFX)           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 2: A/B 編曲規格與頻率空間規劃 (本機 Agent)                       │
│ └─► 02_arr_spec_ab.yaml                                                │
│     定義主奏、和聲、貝斯、鼓組編制 ＋ 3–6 kHz SFX 負空間保護           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 3: 提示詞編譯與任務清單生成 (本機 Agent)                         │
│ └─► manifest.json                                                      │
│     鎖定目標 LUFS、循環微調餘裕 (+0.4s)、HDemucs 權重、尾奏判定閥值    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (於啟動 GPU 前先與使用者確認規劃)
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 4: Colab L4 生成 ＋ 下載前自動 QA 診斷與當場修復 (雲端)          │
│ ├─► Stable Audio 3 (8 步 SDPA 蒸餾取樣，推論速度約 11 it/s)           │
│ ├─► GPU 端 Best-of-3 種子候選評分重排                                  │
│ ├─► ±1 拍智慧循環對齊 (零交差点自動對齊)                               │
│ ├─► TorchAudio HDemucs 取樣鎖定分軌 (0.0 ms 零相位延遲)                │
│ ├─► 逐小節音響 QA 診斷 (前奏落差、接縫跳變、高潮能量、SFX 頻段)        │
│ ├─► 下載前直接在 GPU 上觸發 Inpaint 與手術級 EQ 凹槽修復               │
│ ├─► 本機 SHA-256 驗證與資產驗收交付                                    │
│ └─► finally 區塊保證觸發 colab stop (總花費約 0.00–0.04 CU)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ v4 核心技術突破

1. **全原生 44.1 kHz 端到端鏈路**：
   `Stability-AI/stable-audio-3` 與 `TorchAudio HDemucs` 原生皆以 44.1 kHz 訓練，全面廢除 48k 反覆重取樣，消除相位失真，確保分軌與主循環取樣點精確吻合（如 $1,058,400$ 點）。
2. **三段式 QA 狀態分級**：
   捨棄二元布林值，改採 `passed`（通過）/ `warning`（警告）/ `failed`（未通過）三層架構，記錄非破壞性偏差之診斷原因，僅在嚴重破綻時觸發當場修復。
3. **相對中頻高潮比檢驗**：
   衝線尾奏的高潮判定改採相對頻段比率（$\frac{\text{climax\_mid}}{\text{body\_mid}} \ge 1.15$），成功適應重金屬、合成器浪潮與低音主導曲風。
4. **全 BGM 角色 SFX 負空間保護**：
   擴展至所有音樂角色（主循環、尾奏、分軌層），動態監控 $3000\text{--}6000\text{ Hz}$ 頻段，若超標則自動衰減，為遊戲 UI 與鈴聲保留清晰度。
5. **GPU 端 Best-of-N 候選重排**：
   單曲並行生成多組種子（每組約 1 秒），綜合評定前奏落差、小節動態穩定度與和聲音階一致性（Chroma），自動採納最優結果。
6. **±1 拍智慧循環對齊**：
   於第 1 拍起音周圍 $\pm 1$ 拍範圍內搜尋首尾和聲最吻合處，並對齊至零交差点（Zero-crossing），避免切音突躍。
7. **HDemucs 取樣鎖定分軌**：
   由完成的主循環母帶直接抽離鼓組、貝斯與高頻旋律，實測起音延遲 $0.0\text{ ms}$，供遊戲引擎無縫動態混音。

---

## 📚 參考與整合之開源專案

本專案融合並改良了以下優秀開源專案的研究成果：

| 開源專案 | 核心職責與在本作中之整合方式 |
| :--- | :--- |
| **`kanaworksai-hue/audio-studio-mcp`** | **模型執行與編輯 MCP 核心**<br>採用其以 PyTorch SDPA 加速 Stable Audio 3 的架構、`audio_studio_mcp.prompts` 提示詞格式化、EBU R128 LUFS 響度標準化、`make_loop` 交叉淡入淡出、以及局部重繪 (Inpaint) 與延展接續 (Continue) 介面。 |
| **`Stability-AI/stable-audio-3`** | **核心生成式音訊擴散模型 (`stabilityai/stable-audio-3-medium`)**<br>1.4B 參數 Diffusion Transformer (DiT) 搭配 T5Gemma 編碼器，提供立體聲 44.1 kHz 原生高品質音訊，8 步蒸餾取樣於 L4 上可達 ~11 it/s。 |
| **`sora16bit/codex-game-music-skill`** | **場景優先設計法 (Scene-First) ＆ 聽感驗收合約**<br>引進將遊戲狀態機轉換為音樂層次之方法（開場/倒數→主循環→低張力分層→勝利尾奏），採用聲學表面（點/節奏、線/旋律、面/空間）與 SFX 負空間 (3–6 kHz) 設計原則，並落實 8 項音響指標驗收。 |
| **`jtydhr88/music-composition-skills`** | **A/B 編曲規格書架構 (`ARR-SPEC`)**<br>落實 `02_arr_spec_ab.yaml` 中的不變量約定（BPM、小節時長、頻段分配），並將音樂拆分為 A 版（疾速動態）與 B 版（高壓緊湊）的樂器編制表。 |
| **`SJY051/music-composition`** | **逐小節量化音響診斷表**<br>捨棄全曲平均值，採用逐小節 RMS 能量分佈分析、小節間動態寬度（Bar Spread）與頻段佔比計算等量化指標。 |
| **`yzfly/awesome-music-prompts`** | **提示詞編譯法則 ＆ 4 欄位結構化 SFX**<br>運用 `"starting immediately at full intensity from beat one"` 破除模型慢熱前奏，並以音源、動作、材質、空間等四維度描述精確引導音效合成。 |
| **`pytorch/audio` (TorchAudio HDemucs)** | **44.1 kHz 取樣鎖定分軌抽離**<br>採用內建 `HDEMUCS_HIGH_MUSDB_PLUS` 模型，直接自 44.1 kHz 生成結果抽離鼓與貝斯，生成完全無相位差之動態分軌層。 |
| **`googlecolab/colab-cli`** | **自動化雲端 GPU 實體排程與安全保證**<br>透過輕量 CLI 檢測 GPU 配額、啟動 L4 執行個體，並於 `finally:` 保證自動執行 `colab stop` 避免點數洩漏。 |

---

## 📦 多 Agent 生態系支援

在 [`skills_export/`](./skills_export/) 中已備妥專屬封裝套件：

### 1. Anthropic Claude Code (`claude`)
位於 [`skills_export/claude-code/colab-audio-studio-mcp/`](./skills_export/claude-code/colab-audio-studio-mcp/)
- **`SKILL.md`**：專門針對 Claude Code 終端指令模式撰寫之 YAML Frontmatter 與非交互流程。
- **`CLAUDE.md`**：可直接放入專案根目錄或 `~/.claude/CLAUDE.md`，提供快速操作卡與安全規範。

### 2. OpenAI Codex / Cursor / Copilot Workspace
位於 [`skills_export/codex/colab-game-audio-skill/`](./skills_export/codex/colab-game-audio-skill/)
- **`SKILL.md`**：Codex 角色系統提示詞。
- **`output-contracts.md`**：防止模型幻覺的嚴格 Stage 1~4 Schema 輸出合約。
- **`scene-recipes.md`**：提供競速、賽博龐克、Boss 戰、復古平台、RPG 村莊等 5 大場景音樂處方籤。

### 3. Google Antigravity / Gemini CLI
位於 [`.agents/skills/colab-audio-studio-mcp/`](./.agents/skills/colab-audio-studio-mcp/)
- 完全符合 Universal Agent Skills 開源規範。

---

## 🎧 生產實測案例：單車訓練台重金屬計時賽

完整生產成果包含於 [`outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/)：

| 資產 ID | 角色與規格 | QA 狀態 | 診斷重點 |
| :--- | :--- | :---: | :--- |
| [`01_sfx_countdown_3s.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/01_sfx_countdown_3s.wav) | 3秒發車倒數 (44.1k) | `passed` | 3 個 1 秒間隔脈衝，瞬態清晰度 14.93 dB |
| [`02_bgm_main_loop_A.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/02_bgm_main_loop_A.wav) | 疾速平路循環 (24.0s) | `passed` | Best-of-3 Seed 2303，前奏落差僅 0.26 dB，接縫跳變 0.0043 |
| [`03_bgm_main_loop_B.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/03_bgm_main_loop_B.wav) | 丘陵攻頂循環 (24.0s) | `passed` | Best-of-3 Seed 3303，接縫跳變僅 0.0013，SFX 佔比 5.17% |
| [`04_sfx_pause_swap_bell.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/04_sfx_pause_swap_bell.wav) | 換人提示鈴聲 (1.5s) | `passed` | 瞬態 12.56 dB，3–6 kHz 主頻能量佔 70.18% 極具穿透力 |
| [`05_bgm_pause_swap_stem_layer.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer.wav) | 換人低密度節奏層 (24.0s) | `passed` | 起音延遲 0.0 ms，低頻佔 91.02% 留出語音空間 |
| ├─ [`stem_drums_bass.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_drums_bass.wav) | 純節奏低頻分軌 (44.1k) | `passed` | 供引擎動態混音 |
| └─ [`stem_guitars_synths.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_guitars_synths.wav) | 旋律與和聲分軌 (44.1k) | `passed` | 供引擎動態混音 |
| [`06_bgm_finish_outro.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/06_bgm_finish_outro.wav) | 衝線勝利尾奏 (32.0s) | `passed` | 4 小節終止延長修復 ＋ 手術級 EQ 壓制 SFX 頻段至 3.45% |

---

## 📄 授權條款

本專案採用 [MIT License](./LICENSE) 授權發布。
