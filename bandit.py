# bandit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


class OnlineStandardizer:
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.dim = dim
        self.eps = eps
        self.n = 0
        self.mean = np.zeros(dim, dtype=float)
        self.M2 = np.zeros(dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.n < 2:
            return x.copy()
        var = self.M2 / (self.n - 1)
        return (x - self.mean) / np.sqrt(var + self.eps)


@dataclass
class LinearTSParams:
    lam: float = 1.0
    sigma_r: float = 0.6

class RandomFourierFeatures:
    def __init__(self, in_dim: int, out_dim: int, sigma: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / sigma, size=(out_dim, in_dim))
        self.b = rng.uniform(0, 2*np.pi, size=out_dim)
        self.scale = np.sqrt(2.0 / out_dim)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        return self.scale * np.cos(self.W @ x + self.b)
    

class KernelThompsonSampling:
    def __init__(self, n_actions: int, phi_dim: int, lam: float = 1.0, sigma_r: float = 0.5, seed: int = 0):
        self.k = n_actions
        self.d = phi_dim
        self.lam = lam
        self.sigma_r = sigma_r
        self.rng = np.random.default_rng(seed)

        self.V = [lam * np.eye(self.d) for _ in range(self.k)]
        self.b = [np.zeros(self.d) for _ in range(self.k)]

    def choose(self, phi: np.ndarray):
        scores = np.zeros(self.k)
        for a in range(self.k):
            V_inv = np.linalg.inv(self.V[a])
            mu = V_inv @ self.b[a]
            Sigma = (self.sigma_r ** 2) * V_inv
            theta = self.rng.multivariate_normal(mu, Sigma)
            scores[a] = phi @ theta
        return int(np.argmax(scores)), {"phi": phi}

    def update(self, action: int, phi: np.ndarray, reward: float):
        self.V[action] += np.outer(phi, phi)
        self.b[action] += phi * reward
    



class LinearThompsonSampling:
    def __init__(self, n_actions: int, context_dim: int, params: LinearTSParams = LinearTSParams(), seed: int = 0) -> None:
        self.k = n_actions
        self.d = context_dim
        self.p = params
        self.rng = np.random.default_rng(seed)

        self.V = [self.p.lam * np.eye(self.d) for _ in range(self.k)]
        self.b = [np.zeros(self.d) for _ in range(self.k)]
        self.std = OnlineStandardizer(self.d)

    def choose(self, x: np.ndarray) -> Tuple[int, Dict]:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.std.update(x)
        z = self.std.transform(x)

        scores = np.zeros(self.k, dtype=float)
        z_list = []
        for a in range(self.k):
            V_inv = np.linalg.inv(self.V[a])
            mu = V_inv @ self.b[a]
            Sigma = (self.p.sigma_r**2) * V_inv
            theta = self.rng.multivariate_normal(mu, Sigma)
            scores[a] = float(z @ theta)
            z_list.append(z)
        a_star = int(np.argmax(scores))
        return a_star, {"z": z, "scores": scores}

    def update(self, action: int, z: np.ndarray, reward: float) -> None:
        z = np.asarray(z, dtype=float).reshape(-1)
        r = float(reward)
        self.V[action] = self.V[action] + np.outer(z, z)
        self.b[action] = self.b[action] + z * r


class LinUCB:
    def __init__(self, n_actions: int, d: int, alpha: float = 1.0):
        self.k = n_actions
        self.d = d
        self.alpha = alpha

        self.V = [np.eye(d) for _ in range(self.k)]
        self.b = [np.zeros(d) for _ in range(self.k)]

    def choose(self, x: np.ndarray):
        scores = np.zeros(self.k)
        for a in range(self.k):
            V_inv = np.linalg.inv(self.V[a])
            theta = V_inv @ self.b[a]
            bonus = self.alpha * np.sqrt(x @ V_inv @ x)
            scores[a] = x @ theta + bonus
        return int(np.argmax(scores)), {"x": x}

    def update(self, action: int, x: np.ndarray, reward: float):
        self.V[action] += np.outer(x, x)
        self.b[action] += x * reward