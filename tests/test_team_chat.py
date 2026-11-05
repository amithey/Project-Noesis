import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/team_chat.py"


def module():
    spec = importlib.util.spec_from_file_location("chat_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_parallel_writers_preserve_every_message(tmp_path):
    code = """
import importlib.util,sys
spec=importlib.util.spec_from_file_location('chat',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from pathlib import Path
for i in range(8):
    m.post(Path(sys.argv[2]),sys.argv[3],'all','status',f'שלום {sys.argv[3]} {i}')
"""
    workers = [subprocess.Popen([sys.executable, "-c", code, str(SCRIPT), str(tmp_path), sender])
               for sender in ("codex", "claude", "gemini")]
    assert all(worker.wait(timeout=20) == 0 for worker in workers)
    text = (tmp_path / "TEAM_CHAT.md").read_text(encoding="utf-8")
    assert text.count("## ") == 24
    assert text.count("# Team chat — Noesis") == 1
    for sender in ("codex", "claude", "gemini"):
        for i in range(8):
            assert text.count(f"> שלום {sender} {i}\n") == 1


def test_replies_and_quoted_multiline_content(tmp_path):
    chat = module()
    identifier = chat.post(tmp_path, "codex", "claude", "question", "first")
    chat.post(tmp_path, "claude", "codex", "answer", "שלום\n## quoted content", identifier)
    text = (tmp_path / "TEAM_CHAT.md").read_text(encoding="utf-8")
    assert f"Reply to: `{identifier}`" in text
    assert "\n> ## quoted content\n" in text


def test_worktree_helper_resolves_the_primary_checkout():
    chat = module()
    assert chat.shared_root() != SCRIPT.parents[1]
    assert (chat.shared_root() / ".git").is_dir()
