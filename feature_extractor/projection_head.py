# projection_head.py

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP per projectar les característiques del backbone a un espai contrastiu X'.
    Ex: 2048 -> 512 -> 128
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NormalizedProjectionHead(nn.Module):
    """
    Igual que ProjectionHead, però normalitza l'output a la esfera unitat.
    Útil per algunes pèrdues contrastives.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, eps: float = 1e-8):
        super().__init__()
        self.proj = ProjectionHead(in_dim, hidden_dim, out_dim)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = z / (z.norm(dim=1, keepdim=True) + self.eps)
        return z
