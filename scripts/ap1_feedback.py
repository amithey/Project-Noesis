"""Opt-in scalar curriculum correction; does not modify neural/physical acceptance rules."""
import numpy as np
from s2_contract import validate_training_report

OBJECTIVE='curriculum-standing-v7-ap1-nontarsal-0.5'


def non_tarsal_fraction(region_reaction):
    a=np.asarray(region_reaction,float)
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('Require C0nonnegative4region reactions')
    total=float(a.sum())
    return 0. if total<=1e-5 else float(np.clip((a[0]+a[3])/total,0,1))


def corrected_reward(native_reward,fraction,strength):
    if not np.isfinite(native_reward) or not -1<=native_reward<=1:raise ValueError('Require bounded native reward')
    if not np.isfinite(fraction) or not 0<=fraction<=1 or strength not in (0.,.5):raise ValueError('Invalid declared correction')
    if strength==0:return native_reward
    return float(np.clip(native_reward-.5*fraction,-1,1))


def apply_feedback(native_result,fraction,strength):
    reward=corrected_reward(native_result['reward'],fraction,strength)
    return native_result if strength==0 else {**native_result,'reward':reward}


def validate_ap1_training(report,strength):
    if report.get('base_reward_objective')!='curriculum-standing-v7':raise ValueError('Wrong base objective')
    if report.get('reward_objective')!=(OBJECTIVE if strength else 'curriculum-standing-v7'):raise ValueError('Wrong actual objective identity')
    if report.get('ap1_context',{}).get('effective_training_penalty')!=strength or report.get('ap1_worker_completed') is not True:raise ValueError('Missing actual correction/completion context')
    # Validate unchanged native horizon/rule parameters without falsifying the stored objective.
    validate_training_report({**report,'reward_objective':report['base_reward_objective']})
