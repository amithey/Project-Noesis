"""Absolute process deadlines and scoped Windows sleep inhibition for active jobs."""
import os
import threading
import time


def _hold_system_awake():
    """Request CPU availability for this thread, returning its restoration callback.

    No persistent power policy, display requirement or away mode is changed.
    Microsoft: https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
    """
    if os.name != 'nt':
        return lambda: None
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    set_state = kernel.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_uint
    previous = set_state(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: set_state(previous)


class WallBudget:
    """Stop at an absolute deadline; ledger is persisted before entering work.

    Polling the absolute clock handles a jump after suspension, unlike a single
    long relative wait on Windows. No code runs while the OS itself is suspended;
    an overdue process exits at the watchdog's first scheduled check on resume.
    Incomplete episodes remain uncompleted and restart from their saved start.
    """
    def __init__(self, deadline):
        self.deadline = deadline
        self.cancel = threading.Event()
        self.worker = threading.Thread(target=self._watch, daemon=True)
        self.release_awake = None

    def _watch(self):
        while not self.cancel.is_set():
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                os._exit(124)
                return  # Also terminates the watchdog when exit is mocked in a test.
            self.cancel.wait(min(.25, remaining))

    def __enter__(self):
        self.release_awake = _hold_system_awake()
        try:
            self.worker.start()
        except Exception:
            self.release_awake()
            raise
        return self

    def __exit__(self, *args):
        self.cancel.set()
        try:
            self.worker.join(timeout=.5)
        finally:
            if self.release_awake is not None:
                self.release_awake()
