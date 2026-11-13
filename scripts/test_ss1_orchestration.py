from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_scale_ss1 as run


class Awake:
    previous=None;released=False
    def __enter__(self):self.previous=1;return self
    def __exit__(self,*args):self.released=True


class Tests(unittest.TestCase):
    def test_registered_25_jobs(self):
        jobs=run.jobs();self.assertEqual(len(jobs),25);self.assertEqual(jobs[0]['phase'],'baseline')
        self.assertEqual({j['direction'] for j in jobs[1:]},{0})
        for s in run.TRAIN:self.assertEqual(sum(j['seed']==s for j in jobs[1:]),4)
        self.assertEqual({j['amplitude'] for j in jobs[1:]},{.005,.0025})

    def test_remaining_budget(self):
        with patch.object(run.time,'monotonic',return_value=run.LIMIT-299):
            with self.assertRaises(TimeoutError):run.remaining(0)

    def test_failed_first_worker_no_retry_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runtime=root/'runtime';out=root/'new';lock=runtime/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';lock.parent.mkdir(parents=True)
            prepared=root/'prepared.json';cp=runtime/'parent.npz';cp.write_bytes(b'immutable')
            run.write(prepared,dict(runtime_root=str(runtime),output=str(out),parent_checkpoint='parent.npz',prior_hashes={},source_hashes={},input_hashes={}))
            awake=Awake();calls=[]
            def worker(*args,**kwargs):calls.append(args);return dict(exit_code=124,timed_out=True,tree_empty=True,seconds=300.)
            old=SimpleNamespace(native=SimpleNamespace(run_windows_job=worker,WatchdogError=RuntimeError))
            with patch.object(run.base,'legacy',return_value=(None,old,lambda:awake)),patch.object(run.base,'resources'),patch.object(run,'reused',return_value=[]):
                with self.assertRaisesRegex(RuntimeError,'Worker failed'):run.execute(prepared)
            self.assertEqual(len(calls),1);self.assertFalse(lock.exists());self.assertTrue(awake.released)
            self.assertEqual(run.read(out/'result.json')['status'],'failed_SS1');self.assertEqual(cp.read_bytes(),b'immutable')

    def test_existing_lease_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runtime=root/'runtime';lock=runtime/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock';lock.parent.mkdir(parents=True);lock.write_text('other owner')
            p=root/'prepared.json';run.write(p,dict(runtime_root=str(runtime),output=str(root/'new'),prior_hashes={},source_hashes={},input_hashes={}))
            with patch.object(run.base,'legacy',return_value=(None,None,Awake)),patch.object(run.base,'resources'):
                with self.assertRaises(FileExistsError):run.execute(p)
            self.assertEqual(lock.read_text(),'other owner')


if __name__=='__main__':unittest.main()
