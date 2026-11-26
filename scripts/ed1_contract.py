"""Independent online queue/weight-step checks for ED1; no motor commands."""
from collections import deque
import hashlib
import numpy as np
from ed1_eligibility import delayed_learner


def digest(x):return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def expected_weights(old,initial,eligibility,signal,rate,dt_ms,bound):
    if signal==0 or rate==0:delta=np.zeros_like(eligibility)
    else:
        with np.errstate(over='ignore',invalid='ignore'):delta=rate*(dt_ms/1000.)*signal*eligibility
        delta=np.clip(np.nan_to_num(delta,nan=0.,posinf=2.,neginf=-2.),-2*bound,2*bound)
    relative=np.divide(np.abs(old),np.abs(initial),out=np.ones_like(initial),where=np.abs(initial)>0)
    return initial*np.clip(relative+delta,1-bound,1+bound)


def audited_learner(native,delay,instances):
    base=delayed_learner(native,delay)
    class Audited(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.ed1_independent_queue=deque();self.ed1_audit=[];instances.append(self)

        def load(self,path,restore_traces=False,*,weights_only=False):
            super().load(path,restore_traces,weights_only=weights_only)
            self.ed1_independent_queue.clear();self.ed1_audit.clear()

        def update(self,spike_idx,dt_ms,reward,*,perturbation=None):
            assert dt_ms==5
            old=self.net.data[self.edge_indices].copy();result=super().update(spike_idx,dt_ms,reward,perturbation=perturbation)
            current=self.eligibility.copy();self.ed1_independent_queue.append(current)
            used=self.ed1_independent_queue.popleft() if len(self.ed1_independent_queue)>delay else np.zeros_like(current)
            expected=expected_weights(old,self.initial_weights,used,result['learning_signal'],self.learning_rate,dt_ms,self.max_change)
            np.testing.assert_array_equal(self.net.data[self.edge_indices],expected)
            index=len(self.ed1_audit);source=index-delay if index>=delay else -1
            row=dict(update_index=index,control_chunk_index=index+10,used_source_update_index=source,
                current_eligibility_sha256=digest(current),used_eligibility_sha256=digest(used),
                learning_signal=result['learning_signal'],prediction_error=result['prediction_error'],reward_baseline=result['reward_baseline'],
                reward_variance=result['reward_variance'],reward_normalization_mass=self.reward_normalization_mass,
                used_eligibility_L2=float(np.sqrt(np.sum(used*used))),independent_weight_step_exact=True,delay_chunks=delay)
            if delay:
                actual=self.ed1_records[-1]
                for k in ('update_index','used_source_update_index','current_eligibility_sha256','used_eligibility_sha256','used_eligibility_L2','delay_chunks'):assert row[k]==actual[k]
            self.ed1_audit.append(row);return result
    return Audited
