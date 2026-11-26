"""No-plasticity exposure control; independent wrapper, never an alternative learner."""
import numpy as np


def control_brain(base):
    class NoPlasticityBrain(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.control_steps=0
            self.net.data.flags.writeable=False

        def learn(self,reward,dt_ms):
            if self.learning_active:
                credit=self.exploration.credit(self.learner.motor_indices)
                if self._log is not None:
                    self._log['credit'].append(np.asarray(credit,np.float32).copy())
                self.control_steps+=1
            # No update(), trace update or reward-baseline mutation. Retain drive
            # generation and neural/body execution through the inherited step().
            return self.learner.metrics()
    return NoPlasticityBrain


def validate_control_native(report):
    expected=dict(status='completed_training_birth',requested_seconds=25,simulated_seconds=25,
                  completed_chunks=5000,learning_updates=0,weights_changed=False,
                  reward_objective='curriculum-standing-v7',learning_rate=.2,eligibility_ms=100,
                  motor_exploration_mv_ms=1.5,original_weight_bound=.5,
                  control_ms=5,physics_dt=.0001,birth_ms=50,worlds=1)
    if any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError('Incomplete or weight-mutating no-plasticity exposure')
    log=report.get('exploration_log',{})
    if log.get('steps')!=4990 or log.get('first_control_chunk')!=10 or report.get('exploration_log_error'):
        raise ValueError('Incomplete exploration exposure log')
