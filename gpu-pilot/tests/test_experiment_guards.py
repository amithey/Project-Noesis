import importlib.util
from pathlib import Path

import pytest


def runner():
    script = Path(__file__).resolve().parents[2] / "scripts/compare_exploration.py"
    spec = importlib.util.spec_from_file_location("experiment_guard_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_guard_rejects_missing_code_and_detects_edits(tmp_path):
    module = runner()
    with pytest.raises(ValueError, match="No Python source"):
        module.tree_digest(tmp_path)
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    before = module.tree_digest(tmp_path)
    source.write_text("VALUE = 2\n")
    assert module.tree_digest(tmp_path) != before


def test_existing_experiment_is_never_modified_on_rejected_restart(tmp_path, monkeypatch):
    from noesis_gpu import run

    module = runner()
    root = tmp_path / "shared"
    (root / "gpu-pilot/outputs").mkdir(parents=True)
    output = tmp_path / "existing"
    output.mkdir()
    original = output / "worker-exit.json"
    original.write_text('{"historical": true}')
    monkeypatch.setattr(module.sys, "argv", ["runner", "--shared-root", str(root),
                        "--core-source", str(tmp_path / "src"), "--output", str(output)])
    monkeypatch.setattr(module, "digest", lambda _: "626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1")
    monkeypatch.setattr(run, "source_digest", lambda _: "test-core-digest")

    def output_for(command, **kwargs):
        if command[0] == "powershell":
            return "[]"
        return "41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953" if "-C" in command else "test-commit"

    monkeypatch.setattr(module.subprocess, "check_output", output_for)
    with pytest.raises(FileExistsError):
        module.main()
    assert original.read_text() == '{"historical": true}'
    assert not (root / "gpu-pilot/outputs/codex-training-host.lock").exists()
