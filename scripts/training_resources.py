"""Measured host-resource preflight; no process termination or settings changes."""
import math


def assess(samples, minimum_gib=3., maximum_cpu_percent=30., count=10):
    if len(samples)!=count:
        raise ValueError('Require the complete resource sampling window')
    for row in samples:
        available,cpu=row['available_gib'],row['host_cpu_percent']
        if not math.isfinite(available) or available<0 or not math.isfinite(cpu) or not 0<=cpu<=100:
            raise ValueError('Invalid resource measurement')
    minimum=min(row['available_gib'] for row in samples)
    maximum=max(row['host_cpu_percent'] for row in samples)
    return dict(passed=minimum>=minimum_gib and maximum<=maximum_cpu_percent,
        sample_count=len(samples),minimum_available_gib=minimum,maximum_host_cpu_percent=maximum,
        required_available_gib=minimum_gib,allowed_host_cpu_percent=maximum_cpu_percent,
        note='Observed preflight window only; does not guarantee future resources or isolate historical slowdown cause')


def sample_quiescence():
    import psutil
    import time
    samples=[]
    for _ in range(10):
        cpu=psutil.cpu_percent(interval=1)
        samples.append(dict(unix_time=time.time(),available_gib=psutil.virtual_memory().available/2**30,
                            host_cpu_percent=cpu))
    return samples
