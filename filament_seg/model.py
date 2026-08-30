"""A clean, dependency-free U-Net baseline for filament segmentation, with
an optional Attention U-Net-style gating mechanism.

This started as a deliberately plain U-Net (no pretrained backbone needed,
works on single-channel H-alpha images out of the box) -- the same class of
model used as the baseline in the MAGFiLO/EdgeAttNet paper
(https://arxiv.org/abs/2509.02964) before their (undisclosed-implementation)
edge-attention addition. ``UNet(use_attention=True)`` adds attention gates on
the skip connections (Oktay et al. 2018, "Attention U-Net",
https://arxiv.org/abs/1804.03999): each gate learns to weight encoder
features by relevance to the decoder's current context before concatenating
them, which tends to help focus on small/thin structures against a
cluttered background -- a reasonable, well-documented stand-in for
EdgeAttNet's own (never-published-in-full) attention mechanism.
``use_attention`` defaults to False so existing checkpoints keep loading.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate (Oktay et al. 2018): learns a per-pixel weight in
    [0, 1] for the encoder's skip features, conditioned on the decoder's
    (coarser, more semantic) gating signal, so the network can suppress
    irrelevant background before it reaches the decoder's convolutions."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.theta_x = nn.Conv2d(skip_ch, inter_ch, kernel_size=1)
        self.phi_g = nn.Conv2d(gate_ch, inter_ch, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta_x(skip)
        phi_g = self.phi_g(gate)
        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = nn.functional.interpolate(phi_g, size=theta_x.shape[2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(theta_x + phi_g))
        return skip * attention


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(in_ch // 2, skip_ch, skip_ch // 2) if use_attention else None
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd input sizes
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h or diff_w:
            x = nn.functional.pad(
                x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
            )
        if self.attention is not None:
            skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_ch: int = 32, use_attention: bool = False):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4, use_attention=use_attention)
        self.up2 = Up(c4, c3, c3, use_attention=use_attention)
        self.up3 = Up(c3, c2, c2, use_attention=use_attention)
        self.up4 = Up(c2, c1, c1, use_attention=use_attention)

        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)  # logits, shape (B, num_classes, H, W)
