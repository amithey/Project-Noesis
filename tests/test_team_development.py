import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import team_development as d


def test_ownership_rejects_extra_file(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'changed_paths', lambda _: set(d.OWNED['claude']) | {'gpu-pilot/src/noesis_gpu/train.py'})
    with pytest.raises(ValueError, match='outside assignment'):
        d.validate_changes(tmp_path, 'claude')


def test_missing_or_empty_deliverable_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'changed_paths', lambda _: set())
    with pytest.raises(ValueError, match='Missing'):
        d.validate_changes(tmp_path, 'claude')
    monkeypatch.setattr(d, 'changed_paths', lambda _: set(d.OWNED['claude']))
    for name in d.OWNED['claude']:
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('')
    with pytest.raises(ValueError, match='Invalid'):
        d.validate_changes(tmp_path, 'claude')


def test_hashes_detect_modified_output(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'changed_paths', lambda _: set(d.OWNED['claude']))
    for name in d.OWNED['claude']:
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('first')
    before = d.validate_changes(tmp_path, 'claude')
    (tmp_path/d.OWNED['claude'][0]).write_text('second')
    assert d.validate_changes(tmp_path, 'claude') != before


def test_worker_commands_have_no_permission_bypass_or_shell_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(d.review, 'cli', lambda provider: [provider])
    claude = d.commands('claude', tmp_path)
    assert 'Read,Write,Edit,Glob,Grep' in claude
    assert 'Bash' not in ' '.join(claude)
    gemini = d.commands('antigravity', tmp_path)
    assert 'gemini-3.8-flash-medium' in gemini
    assert 'command(*)' in ' '.join(gemini)
    assert '--dangerously-skip-permissions' not in claude + gemini
