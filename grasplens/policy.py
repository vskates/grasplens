from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only when torch is installed.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - the local execution environment has no torch.
    torch = None
    nn = None


@dataclass
class TrainingHistory:
    backend: str
    losses: list[float]


if torch is not None:

    class _TorchMLP(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> Any:
            h = self.trunk(x)
            return self.head(h).squeeze(-1), h

else:
    _TorchMLP = None


class GraspScorer:
    """MLP-style learned scorer with hidden activations.

    It uses a PyTorch MLP when torch is available. Otherwise it uses the NumPy
    implementation while preserving the same API.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 48, seed: int = 0) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.backend = "torch" if torch is not None else "numpy"
        self.rng = np.random.default_rng(seed)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.model: Any | None = None
        self.params: dict[str, np.ndarray] = {}

    def _standardize_x(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit or self.x_mean is None or self.x_std is None:
            self.x_mean = x.mean(axis=0, keepdims=True)
            self.x_std = x.std(axis=0, keepdims=True) + 1e-6
        return (x - self.x_mean) / self.x_std

    def _standardize_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.y_mean = float(y.mean())
            self.y_std = float(y.std() + 1e-6)
        return (y - self.y_mean) / self.y_std

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 16,
        batch_size: int = 512,
        lr: float = 2e-3,
    ) -> TrainingHistory:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        xz = self._standardize_x(x, fit=True).astype(np.float32)
        yz = self._standardize_y(y, fit=True).astype(np.float32)
        if self.backend == "torch":
            return self._fit_torch(xz, yz, epochs, batch_size, lr)
        return self._fit_numpy(xz, yz, epochs, batch_size, lr)

    def _fit_torch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> TrainingHistory:
        assert torch is not None
        torch.manual_seed(self.seed)
        self.model = _TorchMLP(self.input_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        losses: list[float] = []
        n = x.shape[0]
        for _ in range(epochs):
            order = self.rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                pred, _ = self.model(x_t[idx])
                loss = torch.mean((pred - y_t[idx]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
            losses.append(epoch_loss / n)
        return TrainingHistory(backend="torch", losses=losses)

    def _fit_numpy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> TrainingHistory:
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        self.params = {
            "w1": self.rng.normal(0.0, scale1, size=(self.input_dim, self.hidden_dim)).astype(np.float32),
            "b1": np.zeros(self.hidden_dim, dtype=np.float32),
            "w2": self.rng.normal(0.0, scale2, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32),
            "b2": np.zeros(self.hidden_dim, dtype=np.float32),
            "w3": self.rng.normal(0.0, scale2, size=(self.hidden_dim, 1)).astype(np.float32),
            "b3": np.zeros(1, dtype=np.float32),
        }
        losses: list[float] = []
        n = x.shape[0]
        beta1, beta2 = 0.9, 0.999
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        step = 0
        for _ in range(epochs):
            order = self.rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb = x[idx]
                yb = y[idx, None]
                z1 = xb @ self.params["w1"] + self.params["b1"]
                h1 = np.maximum(z1, 0.0)
                z2 = h1 @ self.params["w2"] + self.params["b2"]
                h2 = np.maximum(z2, 0.0)
                pred = h2 @ self.params["w3"] + self.params["b3"]
                diff = pred - yb
                loss = float(np.mean(diff**2))
                epoch_loss += loss * len(idx)

                grad_pred = 2.0 * diff / len(idx)
                grads: dict[str, np.ndarray] = {}
                grads["w3"] = h2.T @ grad_pred
                grads["b3"] = grad_pred.sum(axis=0)
                grad_h2 = grad_pred @ self.params["w3"].T
                grad_z2 = grad_h2 * (z2 > 0.0)
                grads["w2"] = h1.T @ grad_z2
                grads["b2"] = grad_z2.sum(axis=0)
                grad_h1 = grad_z2 @ self.params["w2"].T
                grad_z1 = grad_h1 * (z1 > 0.0)
                grads["w1"] = xb.T @ grad_z1
                grads["b1"] = grad_z1.sum(axis=0)

                step += 1
                for key, grad in grads.items():
                    m[key] = beta1 * m[key] + (1 - beta1) * grad
                    v[key] = beta2 * v[key] + (1 - beta2) * (grad**2)
                    m_hat = m[key] / (1 - beta1**step)
                    v_hat = v[key] / (1 - beta2**step)
                    self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            losses.append(epoch_loss / n)
        return TrainingHistory(backend="numpy", losses=losses)

    def predict(self, x: np.ndarray) -> np.ndarray:
        score, _ = self.predict_with_hidden(x)
        return score

    def predict_with_hidden(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        xz = self._standardize_x(x, fit=False).astype(np.float32)
        if self.backend == "torch":
            assert torch is not None and self.model is not None
            with torch.no_grad():
                pred, hidden = self.model(torch.tensor(xz, dtype=torch.float32))
            score = pred.numpy() * self.y_std + self.y_mean
            return score.astype(np.float32), hidden.numpy().astype(np.float32)
        hidden = self._hidden_numpy(xz)
        score_z = self.score_from_hidden(hidden)
        score = score_z * self.y_std + self.y_mean
        return score.astype(np.float32), hidden.astype(np.float32)

    def _hidden_numpy(self, xz: np.ndarray) -> np.ndarray:
        z1 = xz @ self.params["w1"] + self.params["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.params["w2"] + self.params["b2"]
        return np.maximum(z2, 0.0)

    def score_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        if self.backend == "torch":
            assert torch is not None and self.model is not None
            with torch.no_grad():
                h_t = torch.tensor(hidden, dtype=torch.float32)
                pred = self.model.head(h_t).squeeze(-1)
            return pred.numpy().astype(np.float32)
        return (hidden @ self.params["w3"] + self.params["b3"]).squeeze(-1).astype(np.float32)

    def predict_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        score_z = self.score_from_hidden(hidden)
        return (score_z * self.y_std + self.y_mean).astype(np.float32)
