"""Scoped Windows idle-sleep prevention; never changes a global power plan."""
from contextlib import contextmanager
import os

CONTINUOUS = 0x80000000
SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_system_awake(request=None):
    if request is None:
        if os.name != 'nt':
            yield
            return
        import ctypes
        request = ctypes.windll.kernel32.SetThreadExecutionState
    if not request(CONTINUOUS | SYSTEM_REQUIRED):
        raise RuntimeError('Could not request scoped idle-sleep prevention')
    try:
        yield
    finally:
        request(CONTINUOUS)
