"""Real subprocess containment tests; no brain/body simulation."""
import json
import os
from pathlib import Path
import sys

import psutil
import pytest

from noesis_gpu.process_watchdog import run_windows_job

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows job object')


def test_successful_worker_exits_and_entire_job_is_empty(tmp_path):
    with (tmp_path / 'log').open('w') as log:
        receipt = run_windows_job([sys.executable, '-c', 'print("complete")'], timeout=10, stdout=log)
    assert receipt['exit_code'] == 0 and receipt['tree_empty'] and not receipt['timed_out']
    assert 'complete' in (tmp_path / 'log').read_text()


@pytest.mark.parametrize('parent_exits', [False, True])
def test_timeout_kills_descendant_even_if_parent_has_exited(tmp_path, parent_exits):
    pidfile = tmp_path / 'child.json'
    child = 'import os,time,json; from pathlib import Path; Path(' + repr(str(pidfile)) + ').write_text(json.dumps(os.getpid())); time.sleep(120)'
    parent = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child) + ']); '
    if not parent_exits:
        parent += 'time.sleep(120)'
    with (tmp_path / 'log').open('w') as log:
        receipt = run_windows_job([sys.executable, '-c', parent], timeout=2, stdout=log)
    assert receipt['timed_out'] and receipt['tree_empty']
    assert pidfile.exists(), (tmp_path / 'log').read_text()
    assert not psutil.pid_exists(json.loads(pidfile.read_text()))
