"""R4 process-guard tests; fake children are not physical evidence."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_resource_envelope import preflight, stop_reason, monitored_run
from training_resources import assess


def test_diagnostic_envelope_does_not_change_training_default():
    samples = [dict(available_gib=2.1, host_cpu_percent=40.) for _ in range(10)]
    assert preflight(samples)['passed']
    assert not assess(samples)['passed']
    for key, value in [('available_gib', 1.99), ('host_cpu_percent', 60.01)]:
        changed = [dict(r) for r in samples]; changed[-1][key] = value
        assert not preflight(changed)['passed']


def test_immediate_memory_floor_and_deadline():
    assert stop_reason(329.9, 1.) is None
    assert stop_reason(330, 2.) == 'process_deadline'
    assert stop_reason(1, .999) == 'host_memory_below_1GiB'
    for elapsed, memory in [(float('nan'), 2.), (1, float('inf')), (-1, 2.), (1, -1)]:
        with pytest.raises(ValueError): stop_reason(elapsed, memory)


@pytest.mark.parametrize('failure', ['memory', 'deadline', 'telemetry', 'interrupt', None])
def test_owned_child_reaped_and_telemetry_saved_on_all_terminal_paths(tmp_path, failure):
    now = [0.]
    class Child:
        returncode = None
        killed = False
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            if timeout:
                if failure is None: self.returncode = 0; now[0] += 2; return 0
                raise subprocess.TimeoutExpired('fake', timeout)
            return self.returncode
        def kill(self): self.killed = True; self.returncode = -9
    child = Child()
    def spawn(*a, **k): return child
    def sample(_):
        if failure == 'telemetry': raise ValueError('unreadable')
        if failure == 'interrupt': raise KeyboardInterrupt()
        if failure == 'deadline': now[0] = 331.
        return dict(host_available_gib=.8 if failure == 'memory' else 2.)
    # Deadline is checked on the next sample after the clock advances.
    kwargs = dict(spawn=spawn, clock=lambda: now[0], sample=sample)
    path = tmp_path / 'owned.log'
    if failure:
        expected = KeyboardInterrupt if failure == 'interrupt' else ValueError if failure == 'telemetry' else RuntimeError
        with pytest.raises(expected): monitored_run(['fake'], path, **kwargs)
        assert child.killed and child.returncode == -9
    else:
        assert monitored_run(['fake'], path, **kwargs)['exit_code'] == 0
        assert not child.killed
    recorded = json.loads(path.with_suffix('.telemetry.json').read_text())
    assert recorded['final_exit_code'] == child.returncode
