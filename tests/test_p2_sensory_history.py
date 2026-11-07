import numpy as np
import pytest
from p2_sensory_history import history_population,FIELDS


def fixture():
    history={k:np.broadcast_to(np.arange(1000)[:,None,None],(1000,*shape)).astype(float).copy() for k,shape in FIELDS.items()}
    initial={k:np.full(shape,-1.) for k,shape in FIELDS.items()};initial['position']=np.ones((1,3))*17
    return history,initial


class NativeSentinel:
    def step(self,observation,dt_ms):
        self.received=observation
        # Sentinel stands for native neural delegation, never a fly-ability result.
        return observation['angles']+7,observation['load']+3


def test_exact_history_alignment_and_output_delegation_without_motor_replay():
    history,initial=fixture();instances=[];population=history_population(NativeSentinel,history,instances)()
    frozen={k:v.copy() for k,v in history.items()}
    for chunk in range(1000):
        current={k:(initial[k].copy() if chunk==0 else history[k][chunk-1].copy()) for k in FIELDS}
        current['position']=initial['position'].copy()
        if chunk>=394:
            for k in FIELDS:current[k]+=30000
        saved={k:v.copy() for k,v in current.items()};command,pads=population.step(current,5)
        expected=initial['angles'] if chunk==0 else history['angles'][chunk-1]
        np.testing.assert_array_equal(command,expected+7)
        np.testing.assert_array_equal(pads,(initial['load'] if chunk==0 else history['load'][chunk-1])+3)
        np.testing.assert_array_equal(population.received['position'],current['position'])
        for k in saved:np.testing.assert_array_equal(current[k],saved[k])
    for k in history:np.testing.assert_array_equal(history[k],frozen[k])
    assert len(population.p2_records)==1000 and instances==[population]
    with pytest.raises(ValueError):population.step(initial,5)


def test_reject_wrong_time_axis_and_early_divergence():
    history,initial=fixture()
    for field in FIELDS:
        bad=dict(history);bad[field]=history[field][1:]
        with pytest.raises(ValueError):history_population(NativeSentinel,bad,[])
    population=history_population(NativeSentinel,history,[])();population.step(initial,5)
    # Would be an off-by-one error if the next control call received row1.
    wrong={k:v[1].copy() for k,v in history.items()}
    with pytest.raises(AssertionError):population.step(wrong,5)
    assert len(population.p2_records)==1


def test_private_history_copy_and_reject_input_mutation():
    history,initial=fixture();population=history_population(NativeSentinel,history,[])()
    population.step(initial,5)
    correct={k:v[0].copy() for k,v in history.items()}
    for k in history:history[k][:]=-999
    command,_=population.step(correct,5);np.testing.assert_array_equal(command,np.full((1,42),7.))
    class Mutator:
        def step(self,observation,dt_ms):
            observation['omega'][:]=5
            return np.zeros((1,42)),np.zeros((1,6))
    population=history_population(Mutator,fixture()[0],[])()
    before={k:v.copy() for k,v in initial.items()}
    with pytest.raises(AssertionError):population.step(initial,5)
    for k in before:np.testing.assert_array_equal(initial[k],before[k])
