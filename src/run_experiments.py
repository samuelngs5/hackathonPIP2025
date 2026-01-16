# run_experiments.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from env_munitions import MunitionsLineEnv, EnvParams
from twin_kalman import build_linear_twin, KalmanFilter, KalmanState
from policies import HeuristicPolicy, ObsOnlyBanditPolicy, LatentTwinBanditPolicy, HybridTwinBanditPolicy, NonLinearTwinBanditPolicy, LinUCBTwinBanditPolicy, ObsNonlinTS
from scenarios import all_scenarios
from viz import plot_run, plot_comparison
from economic_evaluation import evaluate_economic_metrics


def run_episode(env: MunitionsLineEnv, cadence: np.ndarray, policy_name: str, seed: int = 0) -> Dict:
    T = len(cadence)
    y, info = env.reset(seed=seed)

    # Initialize policies
    heuristic = HeuristicPolicy(wip_th=4.0, down_th=0.35)
    
    # Digital twin
    A, B, C, Q, R = build_linear_twin(n_machines=env.p.n_machines)
    kf = KalmanFilter(A, B, C, Q, R)

    n = A.shape[0]
    mu0 = np.concatenate([np.ones(env.p.n_machines), 0.3*np.ones(env.p.n_machines)])
    Sigma0 = np.diag([0.20**2]*env.p.n_machines + [0.80**2]*env.p.n_machines)
    twin_state0 = KalmanState(mu=mu0, Sigma=Sigma0)



    obs_bandit = ObsOnlyBanditPolicy(seed=seed)
    obs_bandit_nonlinTS = ObsNonlinTS(seed=seed)
    linTS_bandit = LatentTwinBanditPolicy(kf=kf, init_state=twin_state0, seed=seed)

    hybrid_twin_bandit = HybridTwinBanditPolicy(
    kf=kf,
    init_state=twin_state0,
    seed=seed
)
    
    nonlin_TS = NonLinearTwinBanditPolicy(
    kf=kf,
    init_state=twin_state0,
    seed=seed
)
    lin_UCB = LinUCBTwinBanditPolicy(
    kf=kf,
    init_state=twin_state0
)
    

    run = {
        "y": [], "reward": [], "action": [], "action_name": [],
        "h_true": [], "c_true": [], "wip_true": [],
        "demand": [], "eff_cadence": [],

        # --- éco ---
    "throughput": [],
    "downtime": [],
    "maintenance": [],
    }

    # prev control u = [effective_cadence, maintenance_flag]
    u_prev = np.array([0.0, 0.0], dtype=float)

    for t in range(T):
        demand = float(cadence[t])

        # choose action
        if policy_name == "heuristic":
            out = heuristic.act(y)
        elif policy_name == "obs_bandit":
            out = obs_bandit.act(y, demand=demand)
        elif policy_name == "obs_bandit_nonlinTS":
            out = obs_bandit_nonlinTS.act(y, demand=demand)
        elif policy_name == "linTS_bandit":
            out = linTS_bandit.act(y=y, u_prev=u_prev, demand=demand)
        elif policy_name == "hybrid_twin_bandit":
            out = hybrid_twin_bandit.act(y=y, u_prev=u_prev, demand=demand)

        elif policy_name == "nonlin_TS":
            out = nonlin_TS.act(y=y, u_prev=u_prev, demand=demand)

        elif policy_name == "lin_UCB":
            out = lin_UCB.act(y=y, u_prev=u_prev, demand=demand)
        else:
            raise ValueError("policy_name must be in {'heuristic','obs_bandit','twin_bandit'}")

        action = int(out.action)

        # environment step
        y_next, r, info = env.step(action=action, demand_cadence=demand)

        # learning update
        if policy_name == "obs_bandit":
            obs_bandit.learn(action, out.aux, r)
        if policy_name == "obs_bandit_nonlinTS":
            obs_bandit_nonlinTS.learn(action, out.aux, r)
        if policy_name == "linTS_bandit":
            linTS_bandit.learn(action, out.aux, r)
        if policy_name == "hybrid_twin_bandit":
            hybrid_twin_bandit.learn(action, out.aux, r)
        if policy_name == "nonlin_TS":
            nonlin_TS.learn(action, out.aux, r)

        if policy_name == "lin_UCB":
            lin_UCB.learn(action, out.aux, r)

        # log
        run["y"].append(y_next.copy())
        run["reward"].append(float(r))
        run["action"].append(action)
        run["action_name"].append(info["action_name"])
        run["h_true"].append(info["h_true"])
        run["c_true"].append(info["c_true"])
        run["wip_true"].append(info["wip_true"])
        run["demand"].append(info["demand_cadence"])
        run["eff_cadence"].append(info["effective_cadence"])

        #eco 
        # --- logging économique ---
        run["throughput"].append(info["throughput_true"])
        run["downtime"].append(info["downtime_true"])
        run["maintenance"].append(1.0 if action == 3 else 0.0)


        # update prev control for twin
        maint_flag = 1.0 if action == 3 else 0.0
        u_prev = np.array([info["effective_cadence"], maint_flag], dtype=float)

        y = y_next
        # --- évaluation économique a posteriori ---
        run["economic_metrics"] = evaluate_economic_metrics(run)
    return run

import pandas as pd

def build_economic_table(all_runs: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """
    all_runs[scenario][policy] = run dict
    """
    rows = []

    for scen_name, runs in all_runs.items():
        for pol_name, run in runs.items():
            econ = run["economic_metrics"]

            rows.append({
                "Scenario": scen_name,
                "Policy": pol_name,
                "Units produced": econ["produced_units"],
                "Revenue (€)": econ["revenue_eur"],
                "Downtime cost (€)": econ["downtime_cost_eur"],
                "Maintenance cost (€)": econ["maintenance_cost_eur"],
                "Congestion cost (€)": econ["congestion_cost_eur"],
                "Total cost (€)": econ["total_cost_eur"],
                "Net balance (€)": econ["net_balance_eur"],
            })

    df = pd.DataFrame(rows)

    # Optionnel : arrondir pour lisibilité
    money_cols = [c for c in df.columns if "€" in c]
    df[money_cols] = df[money_cols].round(0)
    df["Units produced"] = df["Units produced"].round(0)

    return df

def main():
    params = EnvParams(total_steps=2000)
    env = MunitionsLineEnv(params=params, seed=0, reward_mode="classic")

    scenarios = all_scenarios(params.total_steps)
    policies = [ "nonlin_TS", "obs_bandit"]
    #"obs_bandit", "obs_bandit_nonlinTS"

    all_runs = {}

    for scen_name, cadence in scenarios.items():
        runs = {}
        for pol in policies:
            run = run_episode(env, cadence=cadence, policy_name=pol, seed=0)
            runs[pol] = run

        all_runs[scen_name] = runs
        # Comparison plot
        plot_comparison(runs, scenario_name=scen_name)

        # Detailed plots for twin_bandit (can switch)
        #plot_run(runs["twin_bandit"], title=f"{scen_name} — twin_bandit")

    import matplotlib.pyplot as plt
    plt.show()

    for pol, run in runs.items():
        econ = run["economic_metrics"]
        print(scen_name, pol, econ)

    # --- TABLE ECONOMIQUE ---
    econ_table = build_economic_table(all_runs)

    print("\n=== ECONOMIC SUMMARY ===\n")
    print(econ_table.to_string(index=False))

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()