"""【維度四】多 Agent 生態系呼叫相容性測試 (Claude Code / Codex: T4.1, T4.2).

Verifies:
- T4.1: Non-interactive execution guarantee (zero `input()` calls, `flush=True` logging).
- T4.2: Contract compliance and structural completeness across all export packages.
"""

import json
from pathlib import Path
import re
import unittest


class TestDimension4AgentCompatibility(unittest.TestCase):
    """【維度四】多 Agent 生態系呼叫相容性測試"""

    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent

    def test_T4_1_claude_code_non_interactive_compliance(self):
        """T4.1: 檢驗所有執行腳本皆無任何阻塞式 input() 呼叫，且進度日誌均帶有 flush=True。"""
        scripts_to_check = [
            self.root / ".agents/skills/colab-audio-studio-mcp/scripts/supervise_template.py",
            self.root / ".agents/skills/colab-audio-studio-mcp/scripts/generate_remote_template.py",
            self.root / ".agents/skills/colab-audio-studio-mcp/scripts/job_remote_template.py",
        ]
        for script in scripts_to_check:
            self.assertTrue(script.exists(), f"Script {script} not found")
            content = script.read_text(encoding="utf-8")

            # Check that there is NO interactive prompt: `input(`
            interactive_calls = re.findall(r'\binput\s*\(', content)
            self.assertEqual(
                len(interactive_calls), 0,
                f"Found interactive input() call in {script.name}, which will hang Claude Code!",
            )

            # Check that print statements in supervisor have flush=True
            if script.name == "supervise_template.py":
                self.assertIn("flush=True", content, "Supervisor must use flush=True for real-time terminal progress")

    def test_T4_2_claude_code_export_structure(self):
        """T4.2a: 檢驗 skills_export/claude-code/ 目錄結構與文件完整性。"""
        claude_pkg = self.root / "skills_export/claude-code/colab-audio-studio-mcp"
        self.assertTrue(claude_pkg.exists(), "Claude Code export package folder missing")
        self.assertTrue((claude_pkg / "SKILL.md").exists(), "Claude SKILL.md missing")
        self.assertTrue((claude_pkg / "CLAUDE.md").exists(), "Claude CLAUDE.md cheat sheet missing")
        self.assertTrue((claude_pkg / "scripts/supervise_template.py").exists(), "Claude scripts missing")
        self.assertTrue((claude_pkg / "examples/sample_manifest.json").exists(), "Claude examples missing")

        skill_text = (claude_pkg / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(skill_text.startswith("---"), "SKILL.md must have YAML frontmatter")
        self.assertIn("name: colab-audio-studio-mcp", skill_text)

    def test_T4_2_codex_export_structure(self):
        """T4.2b: 檢驗 skills_export/codex/ 目錄結構與合約文件完整性。"""
        codex_pkg = self.root / "skills_export/codex/colab-game-audio-skill"
        self.assertTrue(codex_pkg.exists(), "Codex export package folder missing")
        self.assertTrue((codex_pkg / "SKILL.md").exists(), "Codex SKILL.md missing")
        self.assertTrue((codex_pkg / "output-contracts.md").exists(), "Codex output-contracts.md missing")
        self.assertTrue((codex_pkg / "scene-recipes.md").exists(), "Codex scene-recipes.md missing")
        self.assertTrue((codex_pkg / "scripts/supervise_template.py").exists(), "Codex scripts missing")

        contracts_text = (codex_pkg / "output-contracts.md").read_text(encoding="utf-8")
        self.assertIn("min_climax_relative_mid_ratio", contracts_text)
        self.assertIn("44100", contracts_text)

        recipes_text = (codex_pkg / "scene-recipes.md").read_text(encoding="utf-8")
        self.assertIn("Racing & Cycling Time Trial", recipes_text)
        self.assertIn("Cyberpunk Infiltration", recipes_text)


if __name__ == "__main__":
    unittest.main()
