"""One exploratory training birth with a selectable reward profile (Claude, issue #19).

train.py (Codex-owned) is fixed to curriculum-standing-v7. This wrapper leaves it untouched,
swaps only the objective passed to score_transition, and records the real objective in the
report. It does not promote weights or change any threshold.
"""
import functools
import json
import sys

from noesis_gpu import learning, train
from noesis_gpu.reward_profiles import resolve_objective

args = train.parser()
args.add_argument("--profile", default="gripped-steps")
args.add_argument("--exploration-mv-ms", type=float, default=learning.MAX_EXPLORATION_MV_MS)
args = args.parse_args()
objective = resolve_objective(args.profile)
if not 0 < args.exploration_mv_ms <= learning.MAX_EXPLORATION_MV_MS:
    raise SystemExit("--exploration-mv-ms must be in (0, 1.5]")
learning.score_transition = functools.partial(learning.score_transition, objective=objective)
learning.LearningBrain = functools.partial(learning.LearningBrain,
                                           exploration_mv_ms=args.exploration_mv_ms)
code = 0
try:
    train.run(args)
except BaseException:
    code = 1
    raise
finally:
    for path in args.output.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(data, dict) and "reward_objective" in data:
            data["reward_objective_requested_by_wrapper"] = objective
            data["reward_objective_note"] = ("train.py label may read standing-v7; actual "
                                             "objective is the wrapper value")
            data["reward_objective"] = objective
            data["motor_exploration_mv_ms"] = args.exploration_mv_ms
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
sys.exit(code)
