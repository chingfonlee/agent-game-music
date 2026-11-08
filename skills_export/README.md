# Game Audio Studio Skill: Multi-Platform Export Packages

This directory contains standalone export packages of the **Colab L4 Game Audio Studio Skill (v4)** tailored for **Claude Code** and **OpenAI Codex / LLM Agents**.

---

## Package Directory Structure

```
skills_export/
├── claude-code/
│   └── colab-audio-studio-mcp/
│       ├── SKILL.md            # Formatted with Claude Code frontmatter, non-interactive execution & checks
│       ├── CLAUDE.md           # Quick-reference card to drop into project root or ~/.claude/CLAUDE.md
│       ├── scripts/            # v4 Python templates (generate_remote, job_remote, supervise)
│       └── examples/           # Sample manifest, spec card, arrangement YAML
│
├── codex/
│   └── colab-game-audio-skill/
│       ├── SKILL.md            # Codex Agent system prompt & execution contract
│       ├── output-contracts.md # Strict schema definitions for Stages 1–4
│       ├── scene-recipes.md    # Pre-tested recipes (Racing, Cyberpunk, Boss, Retro, Cozy RPG)
│       ├── scripts/            # v4 Python templates (generate_remote, job_remote, supervise)
│       └── examples/           # Sample manifest, spec card, arrangement YAML
│
└── README.md                   # This installation & deployment guide
```

---

## 1. How to Use with Claude Code (`claude`)

### Option A: Install as Workspace Skill
Copy the skill into your project's `.claude/skills/` directory:
```bash
mkdir -p .claude/skills
cp -r skills_export/claude-code/colab-audio-studio-mcp .claude/skills/
```

### Option B: Install Globally for Claude Code
Copy to your user's global Claude skills folder:
- **macOS / Linux**: `~/.claude/skills/colab-audio-studio-mcp/`
- **Windows**: `%USERPROFILE%\.claude\skills\colab-audio-studio-mcp\`

### Option C: Quick Project Rules (`CLAUDE.md`)
Copy `skills_export/claude-code/colab-audio-studio-mcp/CLAUDE.md` to your project root. Claude Code will automatically ingest the guidelines, shortcuts, and safety rules whenever you launch `claude`.

**Triggering in Claude Code**:
Ask Claude Code:
> *"Design and generate a complete audio package for a dark cyberpunk motorcycle chase on Colab GPU."*

Claude Code will follow the 4-stage pipeline, write `01_bgm_spec_card.md`, `02_arr_spec_ab.yaml`, and `manifest.json`, ask for your confirmation, and execute `supervise.py`.

---

## 2. How to Use with OpenAI Codex / LLM Agents

### Installation
Copy the `colab-game-audio-skill` folder into your agent's skill path:
- **Cursor / Copilot**: Place in `.agent/skills/` or reference `SKILL.md` in `.cursorrules`.
- **Codex / Universal Agent**: Point your agent runner to `skills_export/codex/colab-game-audio-skill/SKILL.md`.

**Key Features for Codex**:
1. **Zero Hallucination with `output-contracts.md`**: Enforces strict JSON/YAML contracts for each stage.
2. **Instant Genre Mapping with `scene-recipes.md`**: Provides exact instrumentation, BPM, bar math, and frequency pockets for 5 major game scene archetypes.
3. **Autonomous Cloud Execution**: Uses `scripts/supervise_template.py` with automatic error handling, Best-of-3 candidate scoring, on-GPU self-healing repair, and guaranteed shutdown.

---

## 3. Core Architectural Highlights (v4)

Both packages include all v4 enhancements verified on Colab L4:
- **44.1 kHz Native Workflow**: Bypasses multiple 48k/44.1k resampling passes; preserves phase and guarantees exact sample synchronization (`1,058,400 samples`).
- **Tri-State QA System**: `passed` / `warning` / `failed` reporting.
- **Relative Mid-Ratio Climax Gate**: $\frac{\text{climax\_mid}}{\text{body\_mid}} \ge 1.15$ adapts to heavy metal, trap, and bass-heavy genres.
- **Universal SFX Frequency Pocket**: Dynamic 3–6 kHz notch protection across loops, outros, and stem layers.
- **Smart Alignment with ±1 Beat Radius**: Zero-crossing snapping with automatic seamless Inpaint fallback.
