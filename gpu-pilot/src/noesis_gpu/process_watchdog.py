"""Bound a Windows worker and all descendants with a kill-on-close Job Object."""
from __future__ import annotations

import ctypes
from ctypes import wintypes as wt
import os
import subprocess
import time

import psutil


class WatchdogError(RuntimeError):
    def __init__(self, message, cleanup_verified):
        super().__init__(message)
        self.cleanup_verified = cleanup_verified


def run_windows_job(command, *, timeout, stdout, cwd=None, env=None):
    """Start suspended, assign before any child can spawn, resume, verify empty job.

    Returns an exit/timeout receipt. An unverified cleanup raises WatchdogError;
    callers must retain their host lease in that case. No shell is involved.
    """
    if os.name != 'nt':
        raise RuntimeError('This watchdog requires Windows')
    if timeout <= 0:
        raise ValueError('Positive timeout required')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)

    class BasicLimit(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                    ('flags', wt.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                    ('active_limit', wt.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', wt.DWORD), ('scheduling', wt.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [('basic', BasicLimit), ('io', IoCounters), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]

    class Accounting(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in ('user', 'kernel', 'period_user', 'period_kernel')]
        _fields_ += [(name, wt.DWORD) for name in ('faults', 'total', 'active', 'terminated')]

    signatures = {
        'CreateJobObjectW': ([ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE),
        'SetInformationJobObject': ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL),
        'QueryInformationJobObject': ([wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p], wt.BOOL),
        'AssignProcessToJobObject': ([wt.HANDLE, wt.HANDLE], wt.BOOL),
        'TerminateJobObject': ([wt.HANDLE, wt.UINT], wt.BOOL),
        'OpenThread': ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
        'ResumeThread': ([wt.HANDLE], wt.DWORD),
        'CloseHandle': ([wt.HANDLE], wt.BOOL),
    }
    for name, (args, result) in signatures.items():
        getattr(kernel, name).argtypes = args
        getattr(kernel, name).restype = result

    def checked(ok):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    job = kernel.CreateJobObjectW(None, None)
    checked(job)
    process = None
    assigned = False
    started = time.monotonic()

    def active():
        info = Accounting()
        checked(kernel.QueryInformationJobObject(job, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return int(info.active)

    def cleanup():
        if assigned:
            checked(kernel.TerminateJobObject(job, 124))
        elif process is not None and process.poll() is None:
            process.kill()  # suspended process has not executed and cannot have descendants
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if (process is None or process.poll() is not None) and active() == 0:
                return True
            time.sleep(.02)
        return False

    try:
        info = ExtendedLimit()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway flag
        checked(kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)))
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                                   creationflags=0x4 | 0x08000000)  # CREATE_SUSPENDED | CREATE_NO_WINDOW
        checked(kernel.AssignProcessToJobObject(job, wt.HANDLE(int(process._handle))))
        assigned = True
        threads = psutil.Process(process.pid).threads()
        if len(threads) != 1:
            raise RuntimeError('Suspended worker must have exactly its initial thread')
        thread = kernel.OpenThread(0x2, False, threads[0].id)
        checked(thread)
        try:
            previous = kernel.ResumeThread(thread)
            if previous != 1:
                raise RuntimeError(f'Unexpected initial suspend count: {previous}')
        finally:
            kernel.CloseHandle(thread)
        while process.poll() is None or active() != 0:
            if time.monotonic() - started >= timeout:
                empty = cleanup()
                if not empty:
                    raise WatchdogError('Worker tree termination not verified', False)
                return dict(exit_code=process.returncode, timed_out=True, tree_empty=True,
                            seconds=time.monotonic() - started, containment='windows_job_before_resume')
            time.sleep(.02)
        return dict(exit_code=process.returncode, timed_out=False, tree_empty=True,
                    seconds=time.monotonic() - started, containment='windows_job_before_resume')
    except BaseException as error:
        try:
            empty = cleanup()
        except BaseException:
            empty = False
        raise WatchdogError(str(error), empty) from error
    finally:
        kernel.CloseHandle(job)
