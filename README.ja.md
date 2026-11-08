# Agent Game Music (AGM) 🎮🎵

[ English ](./README.md) | [ 日本語 ](./README.ja.md) | [ 繁體中文 ](./README.zh-TW.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Stable-Audio-3](https://img.shields.io/badge/Model-Stable--Audio--3-purple.svg)](https://huggingface.co/stabilityai/stable-audio-3-medium)
[![Model: TorchAudio-HDemucs](https://img.shields.io/badge/Model-TorchAudio--HDemucs-green.svg)](https://pytorch.org/audio/stable/models.html)
[![Hardware: Google Colab L4](https://img.shields.io/badge/Hardware-Google%20Colab%20L4%20(24GB)-orange.svg)](https://colab.research.google.com/)
[![Agents: Claude Code & Codex](https://img.shields.io/badge/Agents-Claude%20Code%20%7C%20Codex%20%7C%20Antigravity-red.svg)](#マルチエージェントエコシステム対応)

**AIコーディングエージェントのための、自律型・完全閉環ゲームオーディオ演出・生成・ダウンロード前自動QAパイプライン**

---

## 💡 プロジェクト概要

ゲームオーディオの制作現場では、単純なプロンプトによる音楽生成AIでは解決できない、ゲーム特有の構造的・音響的制約が存在します：
- **イントロトラップ (Intro Trap)**：生成モデルは冒頭4小節を静かに始める傾向があり、ループが先頭に戻った瞬間に $3\text{--}6\text{ dB}$ の急激な音量低下が発生します。
- **ループ接合部のクリックノイズ**：ランダムなクロスフェード処理は、位相干渉やリズムのヨレ、耳障りなノイズを引き起こします。
- **SFX帯域のマスキング**：BGMの厚みによって、ゲーム内で最も重要な効果音・UIチャイム・キャラクターボイス（$3000\text{--}6000\text{ Hz}$）が埋もれてしまいます。
- **アウトロの終止感欠如**：生成されるエンディングが唐突にフェードアウトしたり、不協和音に崩れてしまい、勝利やクリアの余韻を作れません。
- **インタラクティブなステム同期**：ゲームエンジン（FMOD、Wwise等）での動的レイヤリングには、メインループと完全同一サンプル数（位相遅延 $0.0\text{ ms}$）のステム分離が不可欠です。

**Agent Game Music (AGM)** は、シーン優先のゲーム音楽設計論、構造化編曲スペック、プロンプトコンパイル技術を統合し、**Google Colab NVIDIA L4 GPU** 上で高速生成・**ダウンロード前のGPU上での小節単位診断と自己修復**・**完全自動VMシャットダウン**を実現するソリューションです。

---

## 🏛️ アーキテクチャ：4段階パイプライン

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: シーン＆ステートマシン設計 (ローカル Agent)                   │
│ └─► 01_bgm_spec_card.md                                                │
│     ゲームシーンをステートマシンへ変換 (BPM、小節数、4フィールドSFX)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 2: A/B 編曲スペック＆周波数ポケット設計 (ローカル Agent)         │
│ └─► 02_arr_spec_ab.yaml                                                │
│     Lead / Harmony / Bass / Drums 配分 + 3–6 kHz SFX 負の空間確保      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 3: プロンプトコンパイル＆マニフェスト生成 (ローカル Agent)       │
│ └─► manifest.json                                                      │
│     目標LUFS、ループ余白(+0.4s)、HDemucs重み付け、アウトロ判定閾値     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ (GPU起動前にユーザーと要件確定)
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 4: Colab L4 生成 ＋ ダウンロード前自動QA＆自己修復 (クラウド)    │
│ ├─► Stable Audio 3 (8ステップ SDPA蒸留推論 約11 it/s)                  │
│ ├─► GPU上での Best-of-3 候補リランキング                              │
│ ├─► ±1拍 スマートループアライメント (ゼロクロス点スナップ)              │
│ ├─► TorchAudio HDemucs サンプル完全一致ステム分離 (遅延0.0ms)          │
│ ├─► 小節単位QA診断 (イントロ音量、接合部、クライマックス、SFX帯域)     │
│ ├─► ダウンロード前に GPU 上で Inpaint & ノッチEQ による自動修復        │
│ ├─► ローカル SHA-256 検証 ＆ アセット納品                              │
│ └─► finally 句による確実な colab stop (消費CU 約0.00–0.04 Units)       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ v4 コア技術ハイライト

1. **完全 44.1 kHz ネイティブパイプライン**：
   `Stability-AI/stable-audio-3` と `TorchAudio HDemucs` の両方が 44.1 kHz で学習されているため、不要な 48k リサンプリング処理を全廃。リサンプリングによる位相歪みを完全に防ぎ、正確なサンプル数（例：$1,058,400$ サンプル）で完全同期します。
2. **三段階 QA 判定ステータス**：
   単なる二値（True/False）ではなく、`passed`（合格）/ `warning`（警告）/ `failed`（不合格）の3段階モデルを採用。致命的でない偏差は理由とともにログ記録し、致命的な不具合のみ自動修復を発火させます。
3. **相対中域クライマックス比判定**：
   アウトロの盛り上がり判定において、曲全体の帯域バランスに応じた相対比（$\frac{\text{climax\_mid}}{\text{body\_mid}} \ge 1.15$）を適用。ヘヴィメタルやトラップなど低音優勢のジャンルでの誤判定を排除しました。
4. **全 BGM 共通の SFX 負の空間保護**：
   ループ曲だけでなくアウトロやステム層を含む全楽曲アセットで $3000\text{--}6000\text{ Hz}$ 帯域を監視。$8.0\%$ を超過した場合は即座に精密なノッチ減衰を適用し、効果音のためのスペースを確保します。
5. **Best-of-N 候補リランキング**：
   GPU上でループごとに複数のシード候補を並行生成（1候補約1秒）し、イントロドロップや和声一貫性（Chroma）を総合評価して最も安定したテイクを自動採用します。
6. **±1拍 スマートループアライメント**：
   1拍目のアタック周辺を探索し、先頭と末尾の和声・RMSが最も綺麗に繋がるゼロクロス点でスライス。接合部の違和感を解消します。
7. **HDemucs サンプルロック・ステム分離**：
   完成したマスター音源から直接リズム隊と上モノを分離。位相ズレ $0.0\text{ ms}$ の動的レイヤー用ファイルを即座に書き出します。

---

## 📚 参照・統合したオープンソースプロジェクト

本プロジェクトは、オープンソースコミュニティの以下の優れた研究・実装を組み合わせ、発展させたものです：

| 参照プロジェクト | 役割と当プロジェクトへの統合内容 |
| :--- | :--- |
| **`kanaworksai-hue/audio-studio-mcp`** | **モデル実行・編集MCP基盤**<br>Stable Audio 3 の SDPA 高速推論、プロンプト修飾（`TrackType: Music...`）、EBU R128 LUFS ラウドネス正規化、ループクロスフェード、インペイント・継続生成インターフェースを活用。 |
| **`Stability-AI/stable-audio-3`** | **コア音声生成モデル (`stabilityai/stable-audio-3-medium`)**<br>1.4B DiT + T5Gemma による高品質ステレオ 44.1 kHz 生成エンジン。8ステップの蒸留サンプラーにより L4 GPU で約 11 it/s を実現。 |
| **`sora16bit/codex-game-music-skill`** | **シーン優先ゲーム音楽設計論 ＆ 批評合約**<br>ゲームステートマシン（カウントダウン→主ループ→低密度層→勝利アウトロ）への分解、音響表面（点・線・面）と SFX 負の空間（3–6 kHz 保護）理論、8項目評価基準を採用。 |
| **`jtydhr88/music-composition-skills`** | **A/B 編曲スペック設計 (`ARR-SPEC`)**<br>`02_arr_spec_ab.yaml` における不変条件（BPM、小節長、周波数帯）の固定と、A面（快速/疾走）と B面（重圧/高密度）の楽器編成分離手法を採用。 |
| **`SJY051/music-composition`** | **小節単位の定量的音響診断テーブル**<br>曲全体の平均値ではなく、小節単位の RMS レベル推移、小節間ダイナミクス幅（Bar Spread）、帯域別エネルギー比率を定量化する手法を導入。 |
| **`yzfly/awesome-music-prompts`** | **音楽生成プロンプト工学 ＆ 4フィールドSFX**<br>冒頭の「イントロトラップ」を回避する `"starting immediately at full intensity from beat one"` 構文、および音源・動作・材質・空間の 4 要素で効果音を定義するプロンプト手法を採用。 |
| **`pytorch/audio` (TorchAudio HDemucs)** | **44.1 kHz サンプルロック・ステム分離**<br>`HDEMUCS_HIGH_MUSDB_PLUS` 事前学習モデルを用い、生成されたループと完全同サンプル数（遅延 0.0 ms）の垂直ステムレイヤーを分離・生成。 |
| **`googlecolab/colab-cli`** | **自律的クラウド GPU ライフサイクル管理**<br>セッション状態の検知、GPU インスタンスの起動、および終了時の確実な自動シャットダウン（`colab stop`）による課金保護。 |

---

## 📦 マルチエージェントエコシステム対応

[`skills_export/`](./skills_export/) 配下に各エージェント専用のパッケージを用意しています：

### 1. Anthropic Claude Code (`claude`)
[`skills_export/claude-code/colab-audio-studio-mcp/`](./skills_export/claude-code/colab-audio-studio-mcp/)
- **`SKILL.md`**：Claude Code 用 YAML Frontmatter と非対話型 CLI コマンド規約。
- **`CLAUDE.md`**：プロジェクト直下または `~/.claude/CLAUDE.md` に配置可能なクイックリファレンス。
```bash
# プロジェクトへの導入:
mkdir -p .claude/skills
cp -r skills_export/claude-code/colab-audio-studio-mcp .claude/skills/
```

### 2. OpenAI Codex / Cursor / Copilot Workspace
[`skills_export/codex/colab-game-audio-skill/`](./skills_export/codex/colab-game-audio-skill/)
- **`SKILL.md`**：エージェント向けシステムプロンプト。
- **`output-contracts.md`**：ハルシネーションを防止する厳格な入出力スキーマ。
- **`scene-recipes.md`**：5大ゲームシーン（レース、サイバーパンク、ボス戦、レトロ、RPG酒場）の即時利用可能なレシピ集。

### 3. Google Antigravity / Gemini CLI
[`.agents/skills/colab-audio-studio-mcp/`](./.agents/skills/colab-audio-studio-mcp/)
- Universal Agent Skills オープン規格に完全準拠。

---

## 🎧 実装検証サンプル：ロードバイク・ヒルクライム（ヘヴィメタル風）

[`outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/) にて実証済みの 6 つのアセット：

| アセット ID | 役割・スペック | QA ステータス | 診断ハイライト |
| :--- | :--- | :---: | :--- |
| [`01_sfx_countdown_3s.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/01_sfx_countdown_3s.wav) | 3秒スタートカウントダウン (44.1k) | `passed` | 1秒間隔の3パルス、トランジェント 14.93 dB |
| [`02_bgm_main_loop_A.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/02_bgm_main_loop_A.wav) | 平坦スプリントループ (24.0s) | `passed` | Best-of-3 Seed 2303、イントロ低下 0.26 dB、接合部跳変 0.0043 |
| [`03_bgm_main_loop_B.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/03_bgm_main_loop_B.wav) | 登坂アタックループ (24.0s) | `passed` | Best-of-3 Seed 3303、接合部跳変 0.0013、SFX帯域 5.17% |
| [`04_sfx_pause_swap_bell.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/04_sfx_pause_swap_bell.wav) | 交代・一時停止ベル (1.5s) | `passed` | トランジェント 12.56 dB、3–6 kHz 高明瞭度 70.18% |
| [`05_bgm_pause_swap_stem_layer.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer.wav) | 交代用低密度リズム層 (24.0s) | `passed` | 位相遅延 0.0 ms、低音域 91.02% のクリーンな対話空間 |
| ├─ [`stem_drums_bass.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_drums_bass.wav) | ドラム＆ベース独立ステム (44.1k) | `passed` | ゲームエンジン動的フェード用 |
| └─ [`stem_guitars_synths.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/05_bgm_pause_swap_stem_layer_stem_guitars_synths.wav) | ギター＆シンセ独立ステム (44.1k) | `passed` | ゲームエンジン動的フェード用 |
| [`06_bgm_finish_outro.wav`](./outputs/game-audio-2026-11-08-cycling-mountain-metal-v4/results/06_bgm_finish_outro.wav) | ゴール勝利アウトロ (32.0s) | `passed` | 4小節の終止延長修復 ＋ 3–6 kHz帯域を 3.45% に圧縮 |

---

## 📄 ライセンス

本プロジェクトは [MIT License](./LICENSE) のもとで公開されています。
