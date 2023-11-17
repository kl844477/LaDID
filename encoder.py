import os
import torch
import torch.nn as nn

from einops import rearrange

Tensor = torch.Tensor
Module = nn.Module

class CNNEncoder(Module):
    """Mapping from R^{NxD} to R^K."""
    def __init__(self, K: int, N: int, D: int, n_channels: int) -> None:
        super().__init__()

        self.K = K
        self.N = N
        self.D = D

        self.n_channels = n_channels
        self.img_size = int(N**0.5)
        self.n_feat = (self.img_size//16)**2 * (8 * n_channels)

        self.f = nn.Sequential(
            nn.Conv2d(D, n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(n_channels),  # img_size/2

            nn.Conv2d(n_channels, 2*n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(2*n_channels),  # img_size/4

            nn.Conv2d(2*n_channels, 4*n_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(4*n_channels),  # img_size/8

            nn.Conv2d(4*n_channels, 8*n_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8*n_channels),  # img_size/16

            nn.Flatten(),
            nn.Linear(self.n_feat, K),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape (S, M, N, D)
        S, M, _, _ = x.shape
        x = rearrange(x, "s m (h w) d -> (s m) d h w", h=self.img_size, w=self.img_size)
        x = self.f(x)
        x = rearrange(x, "(s m) k -> s m k", s=S, m=M)
        return x