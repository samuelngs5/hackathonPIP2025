# twin_kalman.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class KalmanState:
    mu: np.ndarray      # (n,)
    Sigma: np.ndarray   # (n,n)


class KalmanFilter:
    """
    x_{t+1} = A x_t + B u_t + w_t,  w_t~N(0,Q)
    y_t     = C x_t + v_t,          v_t~N(0,R)
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.n = self.A.shape[0]

    def predict(self, st: KalmanState, u: np.ndarray) -> KalmanState:
        u = np.asarray(u, dtype=float).reshape(-1)
        mu = self.A @ st.mu + self.B @ u
        Sigma = self.A @ st.Sigma @ self.A.T + self.Q
        return KalmanState(mu, Sigma)

    def update(self, pred: KalmanState, y: np.ndarray) -> KalmanState:
        y = np.asarray(y, dtype=float).reshape(-1)
        S = self.C @ pred.Sigma @ self.C.T + self.R
        K = pred.Sigma @ self.C.T @ np.linalg.inv(S)
        innov = y - (self.C @ pred.mu)
        mu = pred.mu + K @ innov
        Sigma = (np.eye(self.n) - K @ self.C) @ pred.Sigma
        Sigma = 0.5 * (Sigma + Sigma.T)
        return KalmanState(mu, Sigma)


def build_linear_twin(n_machines: int = 3):
    """
    State x = [h1..hN, c1..cN] (2N)
    Observation y = [throughput, WIP, downtime] (3)

    linear-ish twin:
      health drifts down with cadence, congestion drifts up with cadence and down with maintenance.
    Control u = [effective_cadence, maintenance_flag]
    """
    N = n_machines
    n = 2 * N

    # A: mild persistence
    Ah = 0.995 * np.eye(N)
    Ac = 0.92 * np.eye(N)
    # small coupling (downstream congestion influences upstream)
    for i in range(N - 1):
        Ac[i, i + 1] = 0.03
    A = np.block([
        [Ah, np.zeros((N, N))],
        [np.zeros((N, N)), Ac],
    ])

    # B: (2N x 2)
    # cadence reduces health, increases congestion; maintenance increases health, decreases congestion
    B = np.zeros((n, 2), dtype=float)
    B[0:N, 0] = -0.015     # cadence -> health down
    B[N:2*N, 0] = +0.18    # cadence -> congestion up
    B[0:N, 1] = +0.25      # maintenance -> health up
    B[N:2*N, 1] = -0.30    # maintenance -> congestion down

    # C: map x -> [throughput, WIP, downtime]
    # throughput increases with average health, decreases with average congestion
    c_thr = np.concatenate([np.ones(N)/N, -0.6*np.ones(N)/N])
    # WIP approx sum congestion
    c_wip = np.concatenate([np.zeros(N), np.ones(N)])
    # downtime proxy: higher when health low and congestion high
    c_down = np.concatenate([-0.7*np.ones(N)/N, +0.4*np.ones(N)/N])

    C = np.vstack([c_thr, c_wip, c_down])

    # Q, R
    Q = np.diag([0.02**2]*N + [0.10**2]*N)
    R = np.diag([0.06**2, 0.12**2, 0.06**2])

    return A, B, C, Q, R
