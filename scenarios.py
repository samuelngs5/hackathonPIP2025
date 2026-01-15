# scenarios.py
from __future__ import annotations
import numpy as np


def cadence_low(T: int) -> np.ndarray:
    return 0.1 * np.ones(T)


def cadence_high(T: int) -> np.ndarray:
    return 0.5 * np.ones(T)


def cadence_spikes(T: int, base: float = 0.1, spike: float = 0.3, period: int = 500, width: int = 100) -> np.ndarray:
    x = base * np.ones(T)
    for t in range(T):
        if (t % period) < width:
            x[t] = spike
    return x


def cadence_ramp(T: int, start: float = 0, end: float = 0.5) -> np.ndarray:
    x = start * np.ones(T)
    for t in range(T):
       
        x[t] = start + (end - start) * (t /  T )
        
    return x


def all_scenarios(T: int):
    return {
        "low_const": cadence_low(T),
        "high_const": cadence_high(T),
        "spikes": cadence_spikes(T),
        "ramp": cadence_ramp(T),
    }