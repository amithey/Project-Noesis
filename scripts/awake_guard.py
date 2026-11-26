"""Thread-scoped Windows idle-sleep request, cleared on context exit.

https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
No display/away-mode or persistent power-plan changes. Explicit user sleep and
critical-power overrides remain possible; a failed worker is never auto-retried.
"""
import ctypes
import os
import threading

CONTINUOUS=0x80000000
SYSTEM_REQUIRED=0x00000001


class AwakeRequest:
    def __init__(self,set_state=None):
        if set_state is None:
            if os.name!='nt':raise RuntimeError('Windows execution-state request required')
            self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            set_state=self.kernel.SetThreadExecutionState
            set_state.argtypes=[ctypes.c_uint32];set_state.restype=ctypes.c_uint32
        self.set_state=set_state;self.previous=None;self.released=False

    def __enter__(self):
        if self.previous is not None:raise RuntimeError('Execution-state request cannot be reentered')
        self.thread=threading.get_ident()
        previous=int(self.set_state(CONTINUOUS|SYSTEM_REQUIRED))
        if not previous:raise OSError('SetThreadExecutionState failed; do not allocate worker')
        self.previous=previous
        return self

    def __exit__(self,exc_type,exc,tb):
        if threading.get_ident()!=self.thread:raise RuntimeError('Restore execution state on requesting thread')
        if not self.set_state(self.previous|CONTINUOUS):raise OSError('Execution-state restoration failed')
        self.released=True
