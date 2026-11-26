"""Read the pinned Numba0.68.1 MT19937 state without consuming random values.

Uses the same private helper API as that installed release's test_random.py.
The core uses uniform/Poisson/integer draws, with no cached Gaussian variate.
Fail closed on an unsupported version or state schema.
"""
import numba as nb
from numba import _helperlib
import numpy as np


def rng_state():
    if nb.__version__ != '0.68.1':
        raise RuntimeError('RNG state observation requires pinned Numba0.68.1')
    index, words = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
    if not 0 <= index <= 624 or len(words) != 624:
        raise RuntimeError('Unexpected MT19937 state layout')
    return np.asarray([index, *words], dtype=np.uint32)
