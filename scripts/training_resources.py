"""Measured host-resource preflight; no process termination or settings changes."""
import math


def window_due(elapsed,last_started,interval=60.,budget=900.,window=10.):
    """Never start a sampling window that cannot fit before the fixed deadline."""
    return elapsed+window<=budget and (last_started is None or elapsed-last_started>=interval)


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


def main():
    import argparse
    import json
    from pathlib import Path
    import time
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--watch-seconds',type=float,default=900.)
    args=ap.parse_args()
    if not math.isfinite(args.watch_seconds) or not 10<=args.watch_seconds<=900:
        raise ValueError('Resource-only watch budget must be10..900seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();last=None
    result=dict(status='watching_resources_only',budget_seconds=args.watch_seconds,windows=[],simulation_started=False)
    def save():
        from compare_exploration import write
        result['elapsed_seconds']=time.monotonic()-started
        write(args.output/'result.json',result)
    try:
        while time.monotonic()-started<args.watch_seconds:
            elapsed=time.monotonic()-started
            if window_due(elapsed,last,budget=args.watch_seconds):
                last=elapsed;samples=sample_quiescence();decision=assess(samples)
                result['windows'].append(dict(start_elapsed=elapsed,decision=decision,samples=samples))
                save();print(json.dumps(dict(window=len(result['windows']),**decision)),flush=True)
                if decision['passed']:
                    result['status']='quiescent_window_observed_no_simulation_started';break
            else:
                # Short interruptible waits; never waits beyond the fixed resource budget.
                time.sleep(min(1.,max(0.,args.watch_seconds-(time.monotonic()-started))))
        else:
            result['status']='resource_watch_deadline_no_quiescent_window'
    except BaseException as error:
        result.update(status='resource_watch_stopped',error_type=type(error).__name__,error=str(error));raise
    finally:
        save()
    return 0 if result['status']=='quiescent_window_observed_no_simulation_started' else 3


if __name__=='__main__':
    raise SystemExit(main())
