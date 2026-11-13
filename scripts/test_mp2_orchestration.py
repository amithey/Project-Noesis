import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_training_context_mp2 as run


class Awake:
    previous=None
    released=False
    def __enter__(self):self.previous=1;return self
    def __exit__(self,*args):self.released=True


class Tests(unittest.TestCase):
    def test_exact_registered_allocation(self):
        jobs=run.jobs();self.assertEqual(len(jobs),37)
        self.assertEqual(jobs[0]['phase'],'baseline')
        for s in (1012,1013,1014,1015):self.assertEqual(sum(j['seed']==s for j in jobs),9)
        self.assertFalse(any(j['seed'] in (10,11,200,201,202,10006,11008) for j in jobs))

    def test_remaining_budget_requires_full_worker(self):
        with patch.object(run.time,'monotonic',return_value=run.LIMIT-299):
            with self.assertRaises(TimeoutError):run.remaining(0)
        with patch.object(run.time,'monotonic',return_value=run.LIMIT-300):run.remaining(0)

    def test_first_failed_worker_has_no_retry_and_releases_lease_awake(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'runtime';out=base/'new'
            lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';lock.parent.mkdir(parents=True)
            prepared=base/'prepared.json';cp=root/'parent.npz';cp.write_bytes(b'unchanged')
            run.write(prepared,dict(runtime_root=str(root),output=str(out),old_hashes={},data_hashes={},
                source_hashes={},parent_checkpoint='parent.npz'))
            awake=Awake();calls=[]
            def worker(*args,**kwargs):
                calls.append(args);return dict(exit_code=124,timed_out=True,tree_empty=True,seconds=300.)
            old=SimpleNamespace(native=SimpleNamespace(run_windows_job=worker,WatchdogError=RuntimeError))
            with patch.object(run,'legacy',return_value=(None,old,lambda:awake)),patch.object(run,'resources'),\
                    patch.object(run,'inventory'),patch.object(run,'reused_records',return_value=[]):
                with self.assertRaisesRegex(RuntimeError,'Worker failed'):run.execute(prepared)
            self.assertEqual(len(calls),1);self.assertFalse(lock.exists());self.assertTrue(awake.released)
            result=run.read(out/'result.json');self.assertEqual(result['status'],'failed_MP2')
            self.assertTrue(result['lease_released']);self.assertTrue(result['process_tree_empty_verified'])
            self.assertTrue(cp.exists())

    def test_occupied_canonical_lease_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'runtime';out=base/'new'
            lock=root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';lock.parent.mkdir(parents=True)
            lock.write_text('other owner');p=base/'prepared.json'
            run.write(p,dict(runtime_root=str(root),output=str(out),old_hashes={},data_hashes={},source_hashes={}))
            with patch.object(run,'legacy',return_value=(None,None,Awake)),patch.object(run,'resources'),patch.object(run,'inventory'):
                with self.assertRaises(FileExistsError):run.execute(p)
            self.assertEqual(lock.read_text(),'other owner');self.assertFalse(out.exists())

    def test_changed_source_stops_before_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source.py';p.write_text('before');pins={str(p):run.sha(p)};p.write_text('after')
            with self.assertRaises(ValueError):run.check_pins(pins)


if __name__=='__main__':unittest.main()
