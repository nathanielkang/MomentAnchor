#!/usr/bin/env python
"""Post-2021 deep tabular generative baselines for the LOWEM benchmark harness.

TabDDPM (Kotelnikov et al., ICML 2023)
    Simplified *Gaussian* DDPM baseline: all joint columns (label-encoded
    categoricals + standardized numerics + standardized response) are treated
    as continuous, matching the CTGAN/TVAE integration pattern in
    ``lowem_bench.py``.  This is a faithful CPU-viable DDPM sampler (MLP noise
    predictor + linear beta schedule) rather than the full official TabDDPM
    multinomial/Gaussian hybrid implementation (no pip package; upstream repo
    is heavy).  Meta tag: ``simplified_gaussian_tabddpm``.

CTAB-GAN+ (Zhao et al., Frontiers in Big Data 2023)
    Lightweight WGAN-GP tabular GAN with an auxiliary regression head
    (downstream-task loss for regression), inspired by the official CTAB-GAN+
    design but without vendoring the image-convolution encoder (old sklearn pin,
    no license file).  Meta tag: ``ctab_gan_plus_wgan_gp_lite``.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

# Smoke defaults: abalone CPU run target <15 min for BOTH methods combined.
SMOKE_EPOCHS_TABDDPM = 35
SMOKE_EPOCHS_CTAB = 35
SMOKE_STEPS_TABDDPM = 50
FULL_EPOCHS_TABDDPM = 150
FULL_EPOCHS_CTAB = 120
FULL_STEPS_TABDDPM = 200


def _torch_seed_from_rng(rng: np.random.Generator) -> int:
    seed = int(rng.integers(0, 2 ** 31 - 1))
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    return seed


def _as_frame(z: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(z, columns=[f"c{i}" for i in range(z.shape[1])])


def _linear_beta_schedule(n_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    import torch

    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)


class _GaussianTabDDPMCore:
    """Minimal Gaussian DDPM core (continuous columns only)."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_timesteps: int = 100,
        device: str = "cpu",
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.n_timesteps = n_timesteps

        betas = _linear_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).to(device)

        time_dim = min(64, hidden_dim)

        class _SinusoidalEmbedding(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.dim = dim

            def forward(self, t):
                t = t.float().view(-1)
                half = self.dim // 2
                freqs = torch.exp(
                    -math.log(10_000) * torch.arange(half, device=t.device).float() / half
                )
                args = t[:, None] * freqs[None, :]
                emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
                if self.dim % 2 == 1:
                    emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
                return emb

        class _DenoisingMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embed = _SinusoidalEmbedding(time_dim)
                layers: list[nn.Module] = []
                in_features = input_dim + time_dim
                for _ in range(n_layers):
                    layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(0.05)])
                    in_features = hidden_dim
                layers.append(nn.Linear(hidden_dim, input_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x_t, t):
                t_emb = self.time_embed(t)
                return self.net(torch.cat([x_t, t_emb], dim=-1))

        self.model = _DenoisingMLP().to(device)
        self.input_dim = input_dim

    def fit(self, x_train: np.ndarray, *, epochs: int, batch_size: int, lr: float) -> list[float]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.model.train()
        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                opt.zero_grad()
                t = torch.randint(0, self.n_timesteps, (batch.shape[0],), device=self.device)
                noise = torch.randn_like(batch)
                sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
                sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
                x_t = sqrt_ab * batch + sqrt_omab * noise
                pred = self.model(x_t, t)
                loss = (noise - pred).pow(2).mean()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return losses

    def sample(self, n_samples: int) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.input_dim, device=self.device)
            for t_val in range(self.n_timesteps - 1, -1, -1):
                t = torch.full((n_samples,), t_val, dtype=torch.long, device=self.device)
                pred_noise = self.model(x, t)
                alpha = self.alphas[t_val]
                alpha_b = self.alpha_bar[t_val]
                beta = self.betas[t_val]
                coef1 = 1.0 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1.0 - alpha_b)
                mean = coef1 * (x - coef2 * pred_noise)
                if t_val > 0:
                    x = mean + torch.sqrt(beta) * torch.randn_like(x)
                else:
                    x = mean
        return x.cpu().numpy()


class _CtabGanPlusLite:
    """WGAN-GP + auxiliary regression head (regression downstream loss)."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        import torch.nn as nn

        self.device = device
        self.input_dim = input_dim

        def _mlp(in_dim: int, out_dim: int, *, n_layers: int = 3) -> nn.Sequential:
            layers: list[nn.Module] = []
            d = in_dim
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(d, hidden_dim), nn.LeakyReLU(0.2)])
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        self.generator = _mlp(input_dim, input_dim).to(device)
        self.critic = _mlp(input_dim, 1).to(device)
        self.reg_head = _mlp(input_dim, 1).to(device)

    @staticmethod
    def _gradient_penalty(critic, real, fake, device: str):
        import torch

        alpha = torch.rand(real.size(0), 1, device=device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        scores = critic(interp)
        grads = torch.autograd.grad(
            outputs=scores,
            inputs=interp,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    def fit(
        self,
        x_train: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        n_critic: int = 3,
        gp_lambda: float = 10.0,
        reg_lambda: float = 0.25,
    ) -> dict:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        y_col = x_train[:, -1:]
        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_col, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        c_opt = torch.optim.Adam(list(self.critic.parameters()) + list(self.reg_head.parameters()), lr=lr, betas=(0.5, 0.9))
        last: dict = {}
        for _ in range(epochs):
            for real, y_real in loader:
                for _ in range(n_critic):
                    c_opt.zero_grad()
                    z = torch.randn(real.size(0), self.input_dim, device=self.device)
                    fake = self.generator(z).detach()
                    c_real = self.critic(real)
                    c_fake = self.critic(fake)
                    gp = self._gradient_penalty(self.critic, real, fake, self.device)
                    reg_loss = (self.reg_head(real) - y_real).pow(2).mean()
                    c_loss = c_fake.mean() - c_real.mean() + gp_lambda * gp + reg_lambda * reg_loss
                    c_loss.backward()
                    c_opt.step()
                g_opt.zero_grad()
                z = torch.randn(real.size(0), self.input_dim, device=self.device)
                fake = self.generator(z)
                g_loss = -self.critic(fake).mean() + reg_lambda * (self.reg_head(fake) - y_real).pow(2).mean()
                g_loss.backward()
                g_opt.step()
                last = {
                    "c_loss": float(c_loss.item()),
                    "g_loss": float(g_loss.item()),
                }
        return last

    def sample(self, n_samples: int) -> np.ndarray:
        import torch

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.input_dim, device=self.device)
            out = self.generator(z)
        return out.cpu().numpy()


def gen_tabddpm(
    z: np.ndarray,
    n_syn: int,
    rng: np.random.Generator,
    *,
    epochs: int = FULL_EPOCHS_TABDDPM,
    n_timesteps: int = FULL_STEPS_TABDDPM,
    hidden_dim: int = 128,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """TabDDPM-style Gaussian diffusion baseline \\citep{kotelnikov2023tabddpm}."""
    try:
        _torch_seed_from_rng(rng)
        n, d = z.shape
        # CPU smoke: shrink model + steps for abalone-scale runs.
        if epochs <= SMOKE_EPOCHS_TABDDPM + 5:
            hidden_dim = min(hidden_dim, 64)
            batch_size = min(batch_size, 128)
        core = _GaussianTabDDPMCore(
            d,
            hidden_dim=hidden_dim,
            n_layers=3,
            n_timesteps=n_timesteps,
            device="cpu",
        )
        losses = core.fit(z, epochs=epochs, batch_size=min(batch_size, max(n // 4, 32)), lr=lr)
        out = core.sample(n_syn)
        lo = np.quantile(z, 0.001, axis=0)
        hi = np.quantile(z, 0.999, axis=0)
        out = np.clip(out, lo, hi)
        return out.astype(np.float64), {
            "backend": "simplified_gaussian_tabddpm",
            "epochs": epochs,
            "n_timesteps": n_timesteps,
            "hidden_dim": hidden_dim,
            "final_loss": float(losses[-1]) if losses else float("nan"),
            "note": "continuous-only DDPM; official TabDDPM uses mixed Gaussian/multinomial diffusion",
        }
    except Exception as exc:
        return z[rng.choice(len(z), size=n_syn, replace=True)].copy(), {
            "backend": "simplified_gaussian_tabddpm",
            "error": str(exc),
            "fallback": "bootstrap_rows",
        }


def gen_ctab_gan_plus(
    z: np.ndarray,
    n_syn: int,
    rng: np.random.Generator,
    *,
    epochs: int = FULL_EPOCHS_CTAB,
    hidden_dim: int = 128,
    batch_size: int = 256,
    lr: float = 2e-4,
) -> tuple[np.ndarray, dict]:
    """CTAB-GAN+ inspired WGAN-GP + regression head \\citep{zhao2022ctabganplus}."""
    try:
        _torch_seed_from_rng(rng)
        n, d = z.shape
        if epochs <= SMOKE_EPOCHS_CTAB + 5:
            hidden_dim = min(hidden_dim, 64)
            batch_size = min(batch_size, 128)
        model = _CtabGanPlusLite(d, hidden_dim=hidden_dim, device="cpu")
        stats = model.fit(
            z,
            epochs=epochs,
            batch_size=min(batch_size, max(n // 4, 32)),
            lr=lr,
        )
        out = model.sample(n_syn)
        lo = np.quantile(z, 0.001, axis=0)
        hi = np.quantile(z, 0.999, axis=0)
        out = np.clip(out, lo, hi)
        return out.astype(np.float64), {
            "backend": "ctab_gan_plus_wgan_gp_lite",
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "train_stats": stats,
            "note": "WGAN-GP + regression auxiliary; not the official Conv encoder CTAB-GAN+ repo",
        }
    except Exception as exc:
        return z[rng.choice(len(z), size=n_syn, replace=True)].copy(), {
            "backend": "ctab_gan_plus_wgan_gp_lite",
            "error": str(exc),
            "fallback": "bootstrap_rows",
        }


def smoke_tabddpm_partial(epochs: int = SMOKE_EPOCHS_TABDDPM, n_timesteps: int = SMOKE_STEPS_TABDDPM):
    return partial(gen_tabddpm, epochs=epochs, n_timesteps=n_timesteps)


def smoke_ctab_partial(epochs: int = SMOKE_EPOCHS_CTAB):
    return partial(gen_ctab_gan_plus, epochs=epochs)


GENERATOR_EXPORTS: dict[str, Callable] = {
    "TabDDPM": gen_tabddpm,
    "CTAB-GAN+": gen_ctab_gan_plus,
}
