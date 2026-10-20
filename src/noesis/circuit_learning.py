"""Opt-in bounded circuit exploration and checked existing-subset weight import."""
from pathlib import Path

import numpy as np

from noesis.embodiment import MotorExploration
from noesis.plasticity import MotorPlasticity

LEARNING_SCOPES=('motor','motor-circuit-v1')


class CircuitExploration(MotorExploration):
    """Independent nonperiodic currents for selected existing CNS targets.

    The inherited motor-pool generator/stream stays exact. A separate seeded
    generator explores each selected CNS neuron, with the same bounded current
    and normalized innovation-credit assumption. No phase or joint command exists.
    """
    def __init__(self,neurons,circuit_indices,amplitude_mv_ms=0.,tau_ms=100.,seed=0):
        ids=np.asarray(circuit_indices)
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0)
                or np.any(ids>=len(neurons))):
            raise ValueError('circuit indices must be valid integers')
        ids=np.unique(ids).astype(np.int64)
        if not neurons.iloc[ids]['Class'].isin(('single_leg_neuromere','sequential_leg_neuromeres')).all():
            raise ValueError('circuit targets must be annotated leg-neuromere cells')
        if amplitude_mv_ms>1.5:
            raise ValueError('circuit exploration is bounded at1.5mV/ms')
        super().__init__(neurons,amplitude_mv_ms,tau_ms,seed)
        self.circuit_idx=ids
        self.circuit_state=np.zeros(len(ids))
        self.circuit_innovation=np.zeros(len(ids))
        self.circuit_rng=np.random.default_rng(np.random.SeedSequence([seed,0x4E4F4553]))

    def step(self,dt_ms):
        base=super().step(dt_ms)
        if self.amplitude==0 or not len(self.circuit_idx):
            return base
        decay=np.exp(-dt_ms/self.tau)
        noise=self.circuit_rng.normal(size=len(self.circuit_idx))
        self.circuit_innovation=np.tanh(noise)
        self.circuit_state=decay*self.circuit_state+np.sqrt(1-decay**2)*noise
        current=self.amplitude*np.tanh(self.circuit_state)
        if not base:
            return {'current_idx':self.circuit_idx,'current_mv_per_ms':current}
        return {'current_idx':np.concatenate((base['current_idx'],self.circuit_idx)),
                'current_mv_per_ms':np.concatenate((base['current_mv_per_ms'],current))}

    def credit(self,motor_indices):
        result=super().credit(motor_indices)
        if len(self.circuit_idx):
            slots=np.searchsorted(motor_indices,self.circuit_idx)
            valid=slots<len(motor_indices)
            valid[valid]&=np.asarray(motor_indices)[slots[valid]]==self.circuit_idx[valid]
            result[slots[valid]]=self.circuit_innovation[valid]
        return result


def import_subset_weights(learner,path):
    """Explicitly transfer a compatible old target subset into a fresh learner.

    Expanded original references must be captured before importing learned motor
    weights. Existing load validation checks topology, original references, signs
    and original bounds before any mutation. New target edges keep original values.
    """
    if (not np.array_equal(learner.net.data[learner.edge_indices],learner.initial_weights)
            or learner.activity.any() or learner.eligibility.any()
            or learner.reward_baseline or learner.reward_variance or learner.reward_normalization_mass):
        raise ValueError('subset import requires a fresh original-reference learner')
    try:
        with np.load(Path(path),allow_pickle=False) as z:
            old_indices=z['motor_indices'].copy()
    except (KeyError,TypeError) as error:
        raise ValueError('invalid subset checkpoint') from error
    if not np.isin(old_indices,learner.motor_indices).all():
        raise ValueError('checkpoint targets are not a subset of current targets')
    old=MotorPlasticity(learner.net,old_indices,learning_rate=learner.learning_rate,
                        tau_eligibility_ms=learner.tau_eligibility_ms,max_change=learner.max_change,
                        rule=learner.rule,normalize_reward_error=learner.normalize_reward_error,
                        bias_correct_reward_error=learner.bias_correct_reward_error)
    slots=np.searchsorted(learner.edge_indices,old.edge_indices)
    if (np.any(slots>=len(learner.edge_indices))
            or not np.array_equal(learner.edge_indices[slots],old.edge_indices)
            or not np.array_equal(learner.initial_weights[slots],old.initial_weights)):
        raise ValueError('subset original edge references differ')
    old.load(path,weights_only=True)
    return {'imported_targets':len(old.motor_indices),'imported_edges':len(old.edge_indices)}
