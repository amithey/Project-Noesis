"""Review exact committed sensory controls without running a simulation."""

import ast
import hashlib
import json
import subprocess

CLAUDE = "d2160a9714dfe279d928c28aa323e78dbe700635"
GEMINI = "2c3dca97af1b0e64f112728ff551548557927995"
SENSED = "gpu-pilot/src/noesis_gpu/sensed_brain.py"
MANIFEST = "docs/workstreams/gemini/sensory_evidence_manifest.json"


def blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_manifest():
    manifest = json.loads(blob(GEMINI, MANIFEST))
    rows = {}
    for path, entry in manifest["files"].items():
        data = blob(GEMINI, path)
        crlf = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        rows[path] = {
            "git_bytes": len(data),
            "git_sha256": sha(data),
            "git_match": sha(data) == entry["git_lf_sha256"],
            "crlf_match": sha(crlf) == entry["crlf_sha256"],
            "working_tree_hash_matches_variant": (
                entry["working_tree_sha256"] in {sha(data), sha(crlf)}
            ),
            "working_tree_size_matches_declared": (
                len(crlf if entry["working_tree_sha256"] == sha(crlf) else data)
                == entry["bytes"]
            ),
        }
    return rows


def probe_modes(source):
    """Execute the exact method body with spies for field and drive adapter."""
    method = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_extra_drive_parts"
    )
    namespace = {"thermo_signals": lambda u: u}
    exec(compile(ast.Module(body=[method], type_ignores=[]), SENSED, "exec"), namespace)
    step = namespace[method.name]
    rows = {}
    for mode in ("off", "zero", "constant", "active"):
        for missing in (False, True):
            calls = []

            def field(obs):
                calls.append("field")
                return obs["position"]

            class Adapter:
                def as_drive_part(self, signal):
                    calls.append("adapter")
                    return [], [], []

            class State:
                pass

            state = State()
            state.mode, state.field, state.level = mode, field, 0.5
            state.thermo = Adapter()
            try:
                step(state, {} if missing else {"position": 0.25})
                error = None
            except KeyError as exc:
                error = str(exc)
            rows[f"{mode}_{'missing' if missing else 'present'}"] = {
                "calls": calls,
                "last_u": getattr(state, "last_u", None),
                "error": error,
            }
    return rows


if __name__ == "__main__":
    source = blob(CLAUDE, SENSED)
    print(json.dumps({
        "claude_commit": CLAUDE,
        "sensed_source_sha256": sha(source),
        "control_probe": probe_modes(source),
        "gemini_commit": GEMINI,
        "gemini_manifest_verification": verify_manifest(),
    }, indent=2))
