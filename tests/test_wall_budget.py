import os
import subprocess
import sys
from pathlib import Path


def test_hard_deadline_stops_native_or_python_work_without_false_completion():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    result = subprocess.run([sys.executable, '-c',
        ('import time; from noesis.wall_budget import WallBudget; '
        'guard=WallBudget(time.monotonic()+.1); guard.__enter__(); time.sleep(5)')],
        check=False, env=env, timeout=3, capture_output=True)
    assert result.returncode == 124
