# viz.py
from __future__ import annotations

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def _stack(list_of_arr):
    return np.vstack(list_of_arr) if len(list_of_arr) else np.empty((0,))


def plot_run(run: Dict, title: str = "") -> None:
    t = np.arange(len(run["reward"]))
    y = _stack(run["y"])           # (T,3)
    h = _stack(run["h_true"])      # (T,3)
    c = _stack(run["c_true"])      # (T,3)
    a = np.array(run["action"], dtype=int)
    demand = np.array(run["demand"], dtype=float)
    eff = np.array(run["eff_cadence"], dtype=float)
    r_cum = np.cumsum(run["reward"])

    plt.figure()
    plt.plot(t, demand, label="demand cadence")
    plt.plot(t, eff, label="effective cadence")
    plt.xlabel("t")
    plt.ylabel("cadence")
    plt.legend()
    plt.title(title + " — Cadence")
    plt.tight_layout()

    plt.figure()
    for i in range(h.shape[1]):
        plt.plot(t, h[:, i], label=f"health M{i+1}")
    plt.xlabel("t")
    plt.ylabel("health (latent true)")
    plt.legend()
    plt.title(title + " — Health")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, np.sum(c, axis=1), label="WIP true (sum congestion)")
    plt.plot(t, y[:, 1], label="WIP observed")
    plt.xlabel("t")
    plt.ylabel("WIP")
    plt.legend()
    plt.title(title + " — Congestion/WIP")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, y[:, 0], label="throughput observed")
    plt.plot(t, y[:, 2], label="downtime observed")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.legend()
    plt.title(title + " — Observations")
    plt.tight_layout()

    plt.figure()
    plt.step(t, a, where="post")
    plt.yticks([0, 1, 2, 3], ["decrease", "hold", "increase", "maintenance"])
    plt.xlabel("t")
    plt.ylabel("action")
    plt.title(title + " — Actions")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, r_cum, label="cumulative reward")
    plt.xlabel("t")
    plt.ylabel("cum reward")
    plt.legend()
    plt.title(title + " — Cumulative reward")
    plt.tight_layout()


def plot_comparison(runs: Dict[str, Dict], scenario_name: str) -> None:
    # runs: policy_name -> run dict
    plt.figure()
    for name, run in runs.items():
        r_cum = np.cumsum(run["reward"])

        if name == "nonlin_TS":
            n = "Jumeau numérique"
            col = "#2ca02c"
            l = '-'
            lw = 2.5
            a = 1

        if name == "obs_bandit":
            n = "Approche standard"
            col = "#7f7f7f"
            l = '--'
            lw = 1.8
            a = 0.9

        
        plt.plot(np.arange(len(r_cum)), r_cum, label=n, c = col, linestyle = l, linewidth = lw, alpha = a )
    plt.xlabel("Temps (en jours)")
    plt.ylabel("Fonction de coût")
    plt.legend()
    if scenario_name=='ramp':
        scenario='Montée en cadence'

    if scenario_name=='low_const':
        scenario='Cadence faible constante'

    if scenario_name=='high_const':
        scenario='Cadence forte constante'

    if scenario_name=='spikes':
        scenario='Pics de cadence'


    plt.title(f"{scenario} ")
    plt.tight_layout()