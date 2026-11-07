"""Experimental eligibility-delay subclass; no native source or motor-policy edits.

Zero delay delegates directly to native update. The sole positive-delay option
is10control chunks (50ms), an unvalidated engineering hypothesis. It delays the
full presynaptic/innovation eligibility trace, not just the innovation vector.
Only fresh weights-only episodes are supported; this adds no exact-resume format.
"""
from collections import deque
import hashlib
import numpy as np


def delayed_learner(native,delay_chunks):
    if isinstance(delay_chunks,bool) or delay_chunks not in (0,10):raise ValueError('Only registered0/10chunk conditions')
    class Delayed(native):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.rule!='perturbation' or not self.normalize_reward_error or not self.bias_correct_reward_error:raise ValueError('ED1 requires the fixed normalized corrected perturbation rule')
            self.ed1_queue=deque();self.ed1_records=[]

        def load(self,path,restore_traces=False,*,weights_only=False):
            if delay_chunks and restore_traces:raise ValueError('Delayed eligibility queue is not checkpointed; exact trace resume unsupported')
            super().load(path,restore_traces,weights_only=weights_only)
            self.ed1_queue.clear();self.ed1_records.clear()

        def update(self,spike_idx,dt_ms,reward,*,perturbation=None):
            if delay_chunks==0:return super().update(spike_idx,dt_ms,reward,perturbation=perturbation)
            if dt_ms!=5:raise ValueError('ED1 delay is defined on fixed5ms updates')
            rate=self.learning_rate;old=self.net.data[self.edge_indices].copy()
            # Let native code validate inputs and advance exactly its activity,
            # eligibility and reward moments, while preventing its weight step.
            try:
                self.learning_rate=0.
                result=super().update(spike_idx,dt_ms,reward,perturbation=perturbation)
            finally:self.learning_rate=rate
            current=self.eligibility.copy();self.ed1_queue.append(current)
            used=self.ed1_queue.popleft() if len(self.ed1_queue)>delay_chunks else np.zeros_like(current)
            error=result['learning_signal']
            if error==0 or rate==0:delta=np.zeros_like(used)
            else:
                with np.errstate(over='ignore',invalid='ignore'):delta=rate*(dt_ms/1000)*error*used
                delta=np.nan_to_num(delta,nan=0.,posinf=2.,neginf=-2.);delta=np.clip(delta,-2*self.max_change,2*self.max_change)
            initial=np.abs(self.initial_weights);relative=np.divide(np.abs(old),initial,out=np.ones_like(initial),where=initial>0)
            new=self.initial_weights*np.clip(relative+delta,1-self.max_change,1+self.max_change)
            self.net.data[self.edge_indices]=new
            update=len(self.ed1_records)
            digest=lambda x:hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()
            self.ed1_records.append(dict(update_index=update,used_source_update_index=update-delay_chunks if update>=delay_chunks else -1,
                current_eligibility_sha256=digest(current),used_eligibility_sha256=digest(used),
                used_eligibility_L2=float(np.linalg.norm(used)),delay_chunks=delay_chunks))
            return {**result,'changed_edges':int(np.count_nonzero(new!=old)),**self.metrics()}
    return Delayed
