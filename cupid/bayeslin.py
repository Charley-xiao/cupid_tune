# cupid/bayeslin.py
from __future__ import annotations
import torch
from typing import Tuple


class BayesLinReg:
    """
    Bayesian linear regression:
      y = w^T x + eps,  eps~N(0, sigma^2)
    Posterior:
      Sigma = (Sigma0^{-1} + X^T X / sigma^2)^{-1}
      mu    = Sigma (Sigma0^{-1} mu0 + X^T y / sigma^2)

    We use it for uncertainty-aware acquisition (UCB/TS).
    """

    def __init__(self, dim: int, prior_var: float = 10.0, noise_var: float = 0.1):
        self.dim = dim
        self.noise_var = float(noise_var)
        self.mu0 = torch.zeros(dim)
        self.Sigma0 = prior_var * torch.eye(dim)
        self.mu = self.mu0.clone()
        self.Sigma = self.Sigma0.clone()
        self._X = []
        self._y = []

    def add(self, x: torch.Tensor, y: float):
        self._X.append(x.detach().float().view(-1))
        self._y.append(float(y))

    def fit(self):
        if len(self._X) == 0:
            return
        X = torch.stack(self._X, dim=0)  # [N, D]
        y = torch.tensor(self._y, dtype=torch.float32).view(-1, 1)  # [N, 1]
        Sigma0_inv = torch.linalg.inv(self.Sigma0)
        A = Sigma0_inv + (X.T @ X) / self.noise_var
        self.Sigma = torch.linalg.inv(A)
        b = Sigma0_inv @ self.mu0.view(-1, 1) + (X.T @ y) / self.noise_var
        self.mu = (self.Sigma @ b).view(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        """
        Returns: (mean, std)
        """
        x = x.view(-1).float()
        mean = float(torch.dot(self.mu, x))
        var = float(x @ self.Sigma @ x + self.noise_var)
        std = var**0.5
        return mean, std
