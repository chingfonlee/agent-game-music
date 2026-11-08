# Claude Code Quick Reference: Game Audio Studio

Add this file to your project root or `.claude/CLAUDE.md` to equip Claude Code with game audio generation shortcuts.

## Commands Cheat Sheet
- **Check Colab Quota & Active VMs**:
  ```bash
  colab usage
  colab sessions
  ```
- **Stop Any Orphan Sessions Immediately**:
  ```bash
  colab stop $(colab sessions --json | jq -r '.[0].id // empty')
  ```
- **Run Supervised Audio Batch**:
  ```bash
  python -u supervise.py
  ```

## Mandatory Rules for Claude Code
1. **Never start Colab without confirmation**: Always write `01_bgm_spec_card.md`, `02_arr_spec_ab.yaml`, and `manifest.json` first, and show them to the user.
2. **Native 44.1 kHz**: Keep `"out_sr": 44100` across all items in `manifest.json` (avoids resampling losses between Stable Audio 3 and HDemucs).
3. **Loop Slack Math**: Always add `+0.4s` to `duration_seconds` for loops so smart zero-crossing alignment has room to search.
4. **Guaranteed Stop**: `supervise.py` has an automatic `finally: colab stop` block. Do not cancel the process abruptly without checking `colab sessions`.
