# policies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from bandit import LinearThompsonSampling, KernelThompsonSampling, RandomFourierFeatures, LinUCB
from twin_kalman import KalmanFilter, KalmanState


@dataclass
class PolicyOutput:
    action: int
    aux: Dict


class HeuristicPolicy:
    """
    Simple, intentionally brittle policy:
      - if downtime high -> maintenance
      - elif WIP high -> decrease cadence
      - else -> increase cadence
    """
    def __init__(self, wip_th: float = 4.0, down_th: float = 0.35) -> None:
        self.wip_th = wip_th
        self.down_th = down_th

    def act(self, y: np.ndarray, **kwargs) -> PolicyOutput:
        thr, wip, down = float(y[0]), float(y[1]), float(y[2])
        if down > self.down_th:
            return PolicyOutput(3, {"rule": "downtime>th -> maintenance"})
        if wip > self.wip_th:
            return PolicyOutput(0, {"rule": "wip>th -> decrease"})
        return PolicyOutput(2, {"rule": "else -> increase"})


class ObsNonlinTS:
    """
    Bandit that uses ONLY observations y (throughput, WIP, downtime) and demand cadence as context.
    """
    def __init__(

        self,
       
        phi_dim: int = 64,
        sigma_rbf: float = 1.0,
        seed: int = 0,
    ):
        
        self.context_dim = 4  # mu + log_sigma

        self.rff = RandomFourierFeatures(
            in_dim=self.context_dim,
            out_dim=phi_dim,
            sigma=sigma_rbf,
            seed=seed,
        )

        self.bandit = KernelThompsonSampling(
            n_actions=4,
            phi_dim=phi_dim,
            seed=seed,
        )

    def act(self, y: np.ndarray, demand: float, reward_prev: Optional[float] = None, last_aux: Optional[Dict] = None) -> PolicyOutput:
        x = np.array([float(y[0]), float(y[1]), float(y[2]), float(demand)], dtype=float)
        phi = self.rff.transform(x)

        a, aux = self.bandit.choose(phi)
        aux["phi"] = phi
        return PolicyOutput(a, aux)

    def learn(self, action: int, aux: Dict, reward: float) -> None:
        self.bandit.update(action, aux["phi"], reward)

class ObsOnlyBanditPolicy:
    """
    Bandit that uses ONLY observations y (throughput, WIP, downtime) and demand cadence as context.
    """
    def __init__(self, seed: int = 0) -> None:
        self.bandit = LinearThompsonSampling(n_actions=4, context_dim=4, seed=seed)



    def act(self, y: np.ndarray,  demand: float) -> PolicyOutput:
        x = np.array([float(y[0]), float(y[1]), float(y[2]), float(demand)], dtype=float)
        a, aux = self.bandit.choose(x)
        aux["context"] = x
        return PolicyOutput(a, aux)

    def learn(self, action: int, aux: dict, reward: float):
        self.bandit.update(action, aux["z"], reward)

   


class LatentTwinBanditPolicy:
    """
    Bandit that uses the DIGITAL TWIN belief (mu + diag(Sigma)) as context.
    Context includes demand cadence too.
    """
    def __init__(self, kf: KalmanFilter, init_state: KalmanState, seed: int = 0) -> None:
        self.kf = kf
        self.state = init_state
        # context: mu (n) + diag(Sigma) (n) + demand (1) + last down (1) = 2n+2
        n = init_state.mu.shape[0]
        self.bandit = LinearThompsonSampling(n_actions=4, context_dim=2*n, seed=seed)

    def act(self, y: np.ndarray, u_prev: np.ndarray, demand: float) -> PolicyOutput:
        # Update twin with last observed y using prev control u_prev
        pred = self.kf.predict(self.state, u_prev)
        self.state = self.kf.update(pred, y)

        throughput = float(y[0])
        wip = float(y[1])
        downtime = float(y[2])

        mu = self.state.mu
        sig = np.diag(self.state.Sigma)
        x = np.concatenate([mu, np.log(sig)], axis=0)

        a, aux = self.bandit.choose(x)
        aux["context"] = x
        return PolicyOutput(a, aux)

    def learn(self, action: int, aux: Dict, reward: float) -> None:
        self.bandit.update(action, aux["z"], reward)





class HybridTwinBanditPolicy:
    """
    Hybrid bandit:
    - uses DIGITAL TWIN belief (mu + diag(Sigma))
    - PLUS selected real observations (throughput, downtime)
    - PLUS demand cadence

    This allows robustness to twin mismatch.
    """

    def __init__(
        self,
        kf: KalmanFilter,
        init_state: KalmanState,
        seed: int = 0,
    ) -> None:
        self.kf = kf
        self.state = init_state

        n = init_state.mu.shape[0]

        # context = mu (n) + diag(Sigma) (n) + demand (1) + throughput (1) + downtime (1)
        context_dim =  2*n

        self.bandit = LinearThompsonSampling(
            n_actions=4,
            context_dim=context_dim,
            seed=seed
        )

    def act(
        self,
        y: np.ndarray,
        u_prev: np.ndarray,
        demand: float,
    ) -> PolicyOutput:

        # --- Digital twin update ---
        pred = self.kf.predict(self.state, u_prev)
        self.state = self.kf.update(pred, y)

        mu = self.state.mu
        sig = np.diag(self.state.Sigma)

        throughput = float(y[0])
        downtime = float(y[2])


        x = np.concatenate(
            [
                mu,
                sig
                #np.array([ demand, downtime ])
               
            ],
            axis=0,
        )
        # np.array([demand], dtype=float),

        a, aux = self.bandit.choose(x)
        aux["context"] = x

        return PolicyOutput(a, aux)

    def learn(self, action: int, aux: Dict, reward: float) -> None:
        self.bandit.update(action, aux["z"], reward)


class NonLinearTwinBanditPolicy:
    def __init__(
        self,
        kf: KalmanFilter,
        init_state: KalmanState,
        phi_dim: int = 64,
        sigma_rbf: float = 1.0,
        seed: int = 0,
    ):
        self.kf = kf
        self.state = init_state

        n = init_state.mu.shape[0]
        self.context_dim = 2 * n  # mu + log_sigma

        self.rff = RandomFourierFeatures(
            in_dim=self.context_dim,
            out_dim=phi_dim,
            sigma=sigma_rbf,
            seed=seed,
        )

        self.bandit = KernelThompsonSampling(
            n_actions=4,
            phi_dim=phi_dim,
            seed=seed,
        )

    def act(self, y: np.ndarray, u_prev: np.ndarray, demand: float) -> PolicyOutput:
        pred = self.kf.predict(self.state, u_prev)
        self.state = self.kf.update(pred, y)

        mu = self.state.mu
        log_sig = np.log(np.diag(self.state.Sigma) + 1e-6)

        x = np.concatenate([mu, log_sig])
        phi = self.rff.transform(x)

        a, aux = self.bandit.choose(phi)
        aux["phi"] = phi
        return PolicyOutput(a, aux)

    def learn(self, action: int, aux: dict, reward: float):
        self.bandit.update(action, aux["phi"], reward)

class LinUCBTwinBanditPolicy:
    def __init__(
        self,
        kf: KalmanFilter,
        init_state: KalmanState,
        alpha: float = 1.0,
    ):
        self.kf = kf
        self.state = init_state

        n = init_state.mu.shape[0]
        self.context_dim = 2 * n

        self.bandit = LinUCB(
            n_actions=4,
            d=self.context_dim,
            alpha=alpha,
        )

    def act(self, y: np.ndarray, u_prev: np.ndarray, demand: float) -> PolicyOutput:
        pred = self.kf.predict(self.state, u_prev)
        self.state = self.kf.update(pred, y)

        mu = self.state.mu
        log_sig = np.log(np.diag(self.state.Sigma) + 1e-6)

        x = np.concatenate([mu, log_sig])
        a, aux = self.bandit.choose(x)
        aux["x"] = x
        return PolicyOutput(a, aux)

    def learn(self, action: int, aux: dict, reward: float):
        self.bandit.update(action, aux["x"], reward)