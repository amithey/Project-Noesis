import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from training_power import CONTINUOUS, SYSTEM_REQUIRED, keep_system_awake


def test_request_is_scoped_and_released_after_failure():
    calls = []
    def request(flags):
        calls.append(flags)
        return CONTINUOUS
    with pytest.raises(ValueError):
        with keep_system_awake(request):
            assert calls == [CONTINUOUS | SYSTEM_REQUIRED]
            raise ValueError('interrupted training')
    assert calls == [CONTINUOUS | SYSTEM_REQUIRED, CONTINUOUS]


def test_failed_request_does_not_start_training():
    with pytest.raises(RuntimeError, match='Could not'):
        with keep_system_awake(lambda _: 0):
            pytest.fail('body must not execute')
