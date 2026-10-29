import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("workflow", Path(__file__).parents[1] / "scripts/team_workflow.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def seed_queue(state):
    state.mkdir()
    w.dump(state / "evidence.json", {"test": True})
    w.dump(state / "manifest.json", {"evidence_sha256": w.hashlib.sha256((state / "evidence.json").read_bytes()).hexdigest()})
    with w.database(state) as db:
        for phase in (1, 2):
            for provider in ("claude", "gemini"):
                db.execute("INSERT INTO tasks(id,provider,phase,status) VALUES(?,?,?,'queued')", (f"{provider}-{phase}", provider, phase))


def test_lock_excludes_second_coordinator(tmp_path):
    with w.coordinator_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with w.coordinator_lock(tmp_path):
                pass


def test_provider_error_cannot_be_marked_complete():
    assert w.parse_response("claude", '{"result":"ok"}') == "ok"
    assert w.parse_response("gemini", '{"response":"ok"}') == "ok"
    for data in ({"is_error": True, "result": "bad"}, {"result": ""}):
        with pytest.raises(ValueError):
            w.parse_response("claude", json.dumps(data))


def test_cross_review_waits_and_completed_tasks_are_not_repeated(tmp_path, monkeypatch):
    state = tmp_path / "state"
    seed_queue(state)
    calls = []
    def fake(state, task):
        prompt = w.prompt_for(state, task)
        if task["phase"] == 2:
            assert "claude first" in prompt and "gemini first" in prompt
        calls.append(task["id"])
        with w.database(state) as db:
            db.execute("UPDATE tasks SET status='complete',result=? WHERE id=?", (task["provider"] + " first", task["id"]))
    monkeypatch.setattr(w, "execute", fake)
    w.run(state)
    w.run(state)
    assert len(calls) == 4
    assert max(calls.index(x + "-1") for x in ("claude", "gemini")) < min(calls.index(x + "-2") for x in ("claude", "gemini"))


def test_crashed_request_not_retried_and_cross_review_blocked(tmp_path, monkeypatch):
    state = tmp_path / "state"
    seed_queue(state)
    with w.database(state) as db:
        db.execute("UPDATE tasks SET status='running' WHERE phase=1")
    monkeypatch.setattr(w, "execute", lambda *args: pytest.fail("must not launch"))
    w.run(state)
    with w.database(state) as db:
        assert [r[0] for r in db.execute("SELECT status FROM tasks WHERE phase=1")] == ["interrupted"] * 2


def test_changed_evidence_is_rejected(tmp_path):
    state = tmp_path / "state"
    seed_queue(state)
    (state / "evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        w.run(state)


def test_html_escapes_provider_content(tmp_path):
    with w.database(tmp_path) as db:
        db.execute("INSERT INTO tasks(id,provider,phase,status,result) VALUES('c','claude',1,'complete','<script>bad</script>')")
    w.render(tmp_path)
    assert "<script>" not in (tmp_path / "status.html").read_text(encoding="utf-8")


def test_payload_excludes_environment_and_personal_paths():
    data = {"seed": 1001, "import_checkpoint": "C:/Users/private/path", "token": "secret", "packages": {"credentials": "secret"}}
    assert w.compact_evidence(data) == {"seed": 1001}


def test_auth_errors_are_actionable_without_exposing_credentials():
    assert "not logged in" in w.authentication_problem("Not logged in · Please run /login")
    assert "unsupported" in w.authentication_problem("IneligibleTierError")
    assert w.authentication_problem("ordinary error") is None


def test_antigravity_only_accepts_success():
    assert w.parse_response("antigravity", '{"status":"SUCCESS","response":"ok"}') == "ok"
    for status in ("ERROR", "CANCELLED", "RUNNING"):
        with pytest.raises(ValueError):
            w.parse_response("antigravity", json.dumps({"status": status, "response": "partial"}))


def test_antigravity_is_bounded_and_tools_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "cli", lambda provider: ["agy"])
    work = w.prepare_workspace(tmp_path, "antigravity")
    cmd = w.command("antigravity", work)
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--mode") + 1] == "plan"
    assert cmd[cmd.index("--print-timeout") + 1] == "210s"
    assert "tools: []" in (work / ".agents/agents/noesis-reviewer/agent.md").read_text()
    assert "--disable-slash-commands" not in cmd  # installed agy says it disables plan mode


def test_exhausted_quota_is_not_an_authentication_failure():
    assert w.quota_problem("RESOURCE_EXHAUSTED Individual quota reached")
    assert not w.authentication_problem("RESOURCE_EXHAUSTED Individual quota reached")
    assert w.quota_problem("ok") is None
