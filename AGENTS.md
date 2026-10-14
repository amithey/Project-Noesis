# Instructions for AI agents working on Project Noesis

Read first: `docs/PROJECT_BRIEF.md` (goal, status, workstreams), `docs/HANDOFF.md` (technical state),
`docs/findings.md` (every measured result, dated). Several AI tools work on this repo at the same time.

## Collaboration rules
- **Never edit or switch branches in the OneDrive checkout**
  (`C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis`) or in another agent's worktree.
  Create your own: `git worktree add C:\dev\noesis-<you>-<topic> -b <type>/<topic> <base branch>`.
- Pick one workstream from `docs/PROJECT_BRIEF.md` section 7; touch only its files. Coordinate (ask the user)
  before changing a shared module (`lif.py`, `connectome.py`, `body.py`, `motor.py`).
- One topic per branch; stacked PRs on the latest relevant branch; nothing goes to `main` directly.
- Write results to `docs/findings-<workstream>.md`, not `docs/findings.md` (avoids merge conflicts).
- `git add` only your own files; never `git add -A` in a shared tree.

## Environment
- Python: `C:\dev\venvs\noesis\Scripts\python.exe`, always with `PYTHONPATH=src`
  (`pip install -e .` fails on the non-ASCII path; pytest already has `pythonpath = ["src"]`).
- Windows PowerShell 5.1: no `&&` (use `;`); scripts containing Hebrew need a UTF-8 BOM.
- Data is read-only and never committed (size, CC-BY-NC licence): `<checkout>\data\raw` (BANC),
  `<checkout>\data\fafb`, `<checkout>\MANC`; pass `--data "<checkout>\data"` from a worktree.
- The corrected brain: `load_cached(data/"raw", data/"processed", region_scale=corrected_scale(data))`.
- Physics runs ~50x slower than real time: run long simulations in the background.

## Scientific standards
- Fix predictions and controls before running; report negative results as they are.
- Repeat over seeds (>= 3); state single-run results as such.
- Every modelling assumption is labelled in code and in the findings; never present one as data.
- Validate against known biology (Shiu et al. 2024 and the predictions in `experiments/response_battery.py`).

## Quality gate
`ruff check src tests` and `pytest` pass; new pure functions get unit tests; experiments are reproducible
scripts under `src/noesis/experiments/`.
