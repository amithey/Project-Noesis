import pytest
from awake_guard import AwakeRequest,CONTINUOUS,SYSTEM_REQUIRED


@pytest.mark.parametrize('failure',[False,True])
def test_request_is_scoped_and_restored_even_on_failure(failure):
    calls=[]
    def api(flags):calls.append(flags);return CONTINUOUS
    guard=AwakeRequest(api)
    try:
        with guard:
            assert calls==[CONTINUOUS|SYSTEM_REQUIRED]
            if failure:raise ValueError('worker failure fixture')
    except ValueError:assert failure
    assert calls==[CONTINUOUS|SYSTEM_REQUIRED,CONTINUOUS] and guard.released


def test_failed_request_prevents_entry():
    with pytest.raises(OSError):
        with AwakeRequest(lambda flags:0):raise AssertionError('must not enter')


def test_previous_thread_flags_are_preserved():
    calls=[]
    def api(flags):calls.append(flags);return CONTINUOUS|2
    with AwakeRequest(api):pass
    assert calls[-1]==CONTINUOUS|2
