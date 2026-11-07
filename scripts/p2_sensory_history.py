"""Diagnostic sensory-history intervention. All motor outputs come from the brain."""
import numpy as np

FIELDS={'angles':(1,42),'velocity':(1,42),'load':(1,6),'up_body':(1,3),'omega':(1,3)}


def history_population(original,history,instances):
    """Replace only native sensory inputs, with correctly timed own-baseline samples.

    Native body observation and recorded physical trajectory are never replaced.
    Input for chunk0 is the live undisturbed initial observation; chunkk>0 uses
    baseline post-step rowk-1. No output command or actuator value is replayed.
    """
    reference={}
    for k,shape in FIELDS.items():
        value=np.asarray(history[k])
        if value.shape!=(1000,*shape) or not np.isfinite(value).all():raise ValueError('Invalid sensory history: '+k)
        reference[k]=value.copy();reference[k].flags.writeable=False

    class HistoryPopulation(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.p2_records=[];instances.append(self)

        def step(self,observation,dt_ms):
            chunk=len(self.p2_records)
            if chunk>=1000 or dt_ms!=5:raise ValueError('Wrong P2 control schedule')
            actual={k:np.asarray(v).copy() for k,v in observation.items()}
            for k,shape in FIELDS.items():
                if actual[k].shape!=shape or not np.isfinite(actual[k]).all():raise ValueError('Invalid live sensory input: '+k)
            effective={k:v.copy() for k,v in actual.items()}
            for k in FIELDS:
                if chunk>0:effective[k]=reference[k][chunk-1].copy()
                # Until pulse physics begins, both streams must be exactly matched.
                if 0<chunk<=393:np.testing.assert_array_equal(actual[k],effective[k],err_msg=k)
            before={k:v.copy() for k,v in effective.items()}
            commands,pads=super().step(effective,dt_ms)
            for k in actual:np.testing.assert_array_equal(observation[k],actual[k],err_msg=k)
            for k in before:np.testing.assert_array_equal(effective[k],before[k],err_msg=k)
            row={'chunk':np.array(chunk),'commands':commands.copy(),'pads':pads.copy()}
            for k in FIELDS:row['live_'+k]=actual[k];row['input_'+k]=effective[k]
            self.p2_records.append(row)
            return commands,pads
    return HistoryPopulation
