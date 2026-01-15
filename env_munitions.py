# env_munitions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class EnvParams:
    n_machines: int = 3
    total_steps: int = 300

    # "Cadence demand" profile is externally provided each step.
    # We model how much demand can be satisfied given health/congestion.

    # Health dynamics
    health_init: float = 1.0
    health_floor: float = 0.0
    health_ceil: float = 1.5
    wear_base: float = 0.0           # baseline wear per step
    wear_cadence_mult: float = 0.0040   # additional wear proportional to cadence
    wear_cong_mult: float = 0.0010      # wear increases with congestion

    # Congestion dynamics (queue-like)
    cong_init: float = 0.2
    cong_floor: float = 0.0
    cong_ceil: float = 5.0
    cong_decay: float = 0.06            # how fast congestion dissipates when capacity exceeds flow

    # Machine capacity
    base_capacity: float = 1.0          # nominal capacity unit
    health_capacity_exp: float = 1.3    # capacity scales like health^exp
    failure_health_threshold: float = 0.25

    # Maintenance action
    maint_health_gain: float = 0.35
    maint_cong_gain: float = 0.20       # maintenance can also reduce congestion (cleanup)
    maint_downtime_steps: int = 1       # immediate downtime penalty
    maint_cost: float = 0.35

    # Random shocks
    process_noise_h: float = 0.003
    process_noise_c: float = 0.02
    shock_prob: float = 0.02            # sporadic disruption increases congestion
    shock_cong_add: float = 0.8

    # Observation noise
    obs_noise_thr: float = 0.05
    obs_noise_wip: float = 0.10
    obs_noise_down: float = 0.02

    # Reward weights
    lam_wip: float = 0.20
    lam_down: float = 0.20
    lam_maint: float = 0.10

    #reward viability
    wip_max       = 5.0
    down_max      = 0.2
    h_min         = 0.8
    r_safe        = 1.0
    r_violation   = 3.0
    epsilon_thr  = 0.1
    lam_maint    = 0.1

    #reward differential 
    alpha_thr  = 1.0
    beta_wip   = 0.5
    gamma_down = 1.0
    lam_maint  = 0.1


class MunitionsLineEnv:
    """
    3-machine serial line environment for munitions production.
    Latent per-machine:
      h_i(t) health
      c_i(t) congestion
    Observations:
      throughput (scalar)
      WIP (scalar, sum congestion)
      downtime (scalar in [0,1])
    Actions:
      0: decrease cadence factor
      1: keep cadence factor
      2: increase cadence factor
      3: maintenance (affects worst machine)
    Note: Actual "external demand cadence" comes from scenario at each step, and action modifies it.
    """
    ACTIONS = {0: "decrease", 1: "hold", 2: "increase", 3: "maintenance"}

    def __init__(self, params: EnvParams = EnvParams(), seed: Optional[int] = None, reward_mode="classic") -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.reward_mode = reward_mode
        self.h = np.zeros(self.p.n_machines, dtype=float)
        self.c = np.zeros(self.p.n_machines, dtype=float)

        # internal: ongoing maintenance downtime counter
        self.maint_timer = 0

        self.prev_thr = 0.0
        self.prev_wip = 0.0
        self.prev_down = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.maint_timer = 0
        self.h[:] = self.p.health_init + 0.05 * self.rng.standard_normal(self.p.n_machines)
        self.c[:] = self.p.cong_init + 0.05 * self.rng.standard_normal(self.p.n_machines)
        self._clip_state()

        self.logs = {
        "produced_units": [],
        "wip": [],
        "maintenance": [],
        "downtime": []
    }
     
        thr0 = 0.0
        wip0 = float(self.c.sum())
        down0 = 0.0

        self.prev_thr = thr0
        self.prev_wip = wip0
        self.prev_down = down0

        y = self._observe(throughput=0.0, downtime=0.0)
        info = self._info_dict(throughput=0.0, downtime=0.0, demand=0.0, eff_cadence=0.0, action=1)
        return y, info


    def _reward_viability(self, thr, wip, down, action):
        ok = (
            wip  <= self.p.wip_max
            and down <= self.p.down_max
            and self.h.mean() >= self.p.h_min
        )

        if ok:
            r = self.p.r_safe
        else:
            r = -self.p.r_violation

        # petit bonus de production (facultatif)
        r += self.p.epsilon_thr * thr

        # coût maintenance
        if action == 3:
            r -= self.p.lam_maint

        return r
    
    def _reward_differential(self, thr, wip, down, action):
        # variations
        d_thr  = thr  - self.prev_thr
        d_wip  = self.prev_wip - wip      # ↓ wip = bien
        d_down = self.prev_down - down    # ↓ downtime = bien

        r = (
            self.p.alpha_thr  * d_thr
            + self.p.beta_wip * d_wip
            + self.p.gamma_down * d_down
            - self.p.lam_maint * (1.0 if action == 3 else 0.0)
        )

        # mise à jour mémoire
        self.prev_thr  = thr
        self.prev_wip  = wip
        self.prev_down = down

        return r



    def step(self, action: int, demand_cadence: float) -> Tuple[np.ndarray, float, Dict]:
        """
        demand_cadence: exogenous requested cadence at time t (scenario).
        action modifies effective cadence.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action={action}. Must be in {list(self.ACTIONS.keys())}.")

        # action modifies effective cadence (except maintenance which is separate)
        cadence_factor = {0: 0.85, 1: 1.0, 2: 1.15}.get(action, 1.0)
        eff_cadence = float(np.clip(demand_cadence * cadence_factor, 0.0, 2.0))

        # Apply maintenance if action==3: target worst health machine
        if action == 3:
            idx = int(np.argmin(self.h))
            self.h[idx] += self.p.maint_health_gain
            self.c[idx] -= self.p.maint_cong_gain
            self.maint_timer = max(self.maint_timer, self.p.maint_downtime_steps)

        # Compute per-machine capacity from health (if failed, capacity ~0)
        cap = self._capacity_from_health(self.h)

        # Effective flow limited by bottleneck and congestion
        # Simple serial line: throughput limited by min capacity and upstream congestion
        bottleneck = float(np.min(cap))
        # Congestion increases when demand exceeds bottleneck; decreases otherwise
        flow = min(eff_cadence, bottleneck)
        # downtime if maintenance active or if any machine "fails"
        failure = float(np.any(self.h < self.p.failure_health_threshold))
        downtime = 0.0
        if self.maint_timer > 0:
            downtime = 1.0
            self.maint_timer -= 1
            flow = 0.0
        elif failure > 0:
            # failure causes partial downtime and sharp throughput loss
            downtime = 1.0
            flow = 0.0

        # Update congestion (queue proxy) across machines
        # If demand higher than flow => queues build; also propagate along line
        excess = max(0.0, eff_cadence - flow)

        # Add sporadic shock
        if self.rng.random() < self.p.shock_prob:
            self.c += self.p.shock_cong_add

        # Congestion update: build where capacity insufficient, decay otherwise
        for i in range(self.p.n_machines):
            self.c[i] += 0.6 * excess
            self.c[i] -= self.p.cong_decay * max(0.0, cap[i] - eff_cadence)
            # small coupling: downstream congestion influences upstream slightly
            if i < self.p.n_machines - 1:
                self.c[i] += 0.05 * self.c[i + 1]

        # Update health: degrades with cadence + congestion
        wear = self.p.wear_base + self.p.wear_cadence_mult * eff_cadence + self.p.wear_cong_mult * float(np.mean(self.c))
        self.h -= wear
        # process noise
        self.h += self.p.process_noise_h * self.rng.standard_normal(self.p.n_machines)
        self.c += self.p.process_noise_c * self.rng.standard_normal(self.p.n_machines)

        self._clip_state()

        
        

        # Observations
        y = self._observe(throughput=flow, downtime=downtime)

        # Reward: encourage throughput, penalize WIP, downtime, maintenance
        thr, wip, down = float(y[0]), float(y[1]), float(y[2])

        if self.reward_mode == "classic":
            r = thr \
                - self.p.lam_wip * wip \
                - self.p.lam_down * down \
                - self.p.lam_maint * (1.0 if action == 3 else 0.0)

        elif self.reward_mode == "differential":
            r = self._reward_differential(thr, wip, down, action)
            

        elif self.reward_mode == "viability":
            r = self._reward_viability(thr, wip, down, action)

        else:
            raise ValueError(f"Unknown reward_mode {self.reward_mode}")
        
        self.prev_thr = thr
        self.prev_wip = wip
        self.prev_down = down




        self.t += 1
        done = self.t >= self.p.total_steps
        info = self._info_dict(throughput=flow, downtime=downtime, demand=demand_cadence, eff_cadence=eff_cadence, action=action)
        info["done"] = done

        # --- Logging économique ---
        
        '''
        self.logs["produced_units"].append(float(flow))
        self.logs["effective_cadence"].append(float(eff_cadence))
        self.logs["demand_cadence"].append(float(demand_cadence))
        self.logs["wip"].append(float(wip))
        self.logs["downtime"].append(float(downtime))
        self.logs["maintenance_action"].append(1.0 if action == 3 else 0.0)

        '''
        
      
       
        
        return y, r, info

    # ---------- helpers ----------

    def _capacity_from_health(self, h: np.ndarray) -> np.ndarray:
        h_eff = np.clip(h, self.p.health_floor, self.p.health_ceil)
        cap = self.p.base_capacity * np.power(h_eff, self.p.health_capacity_exp)
        cap = np.where(h_eff < self.p.failure_health_threshold, 0.0, cap)
        return cap

    def _observe(self, throughput: float, downtime: float) -> np.ndarray:
        wip = float(np.sum(self.c))
        thr_obs = max(0.0, throughput + self.p.obs_noise_thr * self.rng.standard_normal())
        wip_obs = max(0.0, wip + self.p.obs_noise_wip * self.rng.standard_normal())
        down_obs = float(np.clip(downtime + self.p.obs_noise_down * self.rng.standard_normal(), 0.0, 1.0))
        return np.array([thr_obs, wip_obs, down_obs], dtype=float)

    def _clip_state(self) -> None:
        self.h[:] = np.clip(self.h, self.p.health_floor, self.p.health_ceil)
        self.c[:] = np.clip(self.c, self.p.cong_floor, self.p.cong_ceil)

    def _info_dict(self, throughput: float, downtime: float, demand: float, eff_cadence: float, action: int) -> Dict:
        return {
            "t": self.t,
            "action": action,
            "action_name": self.ACTIONS[action],
            "demand_cadence": float(demand),
            "effective_cadence": float(eff_cadence),
            "throughput_true": float(throughput),
            "downtime_true": float(downtime),
            "h_true": self.h.copy(),
            "c_true": self.c.copy(),
            "wip_true": float(np.sum(self.c)),
        }