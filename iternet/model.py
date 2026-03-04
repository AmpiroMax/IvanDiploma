from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    # BatchNorm is unstable for batch_size=1; GroupNorm is robust.
    groups = 8
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


def _interpolate2d(x: torch.Tensor, *, size: tuple[int, int]) -> torch.Tensor:
    # Prefer anti-aliasing to reduce "checkerboard"/ringing on upscales when available.
    try:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    except TypeError:  # older torch without antialias kwarg
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _pad_to_multiple(x: torch.Tensor, *, multiple: int) -> tuple[torch.Tensor, int, int]:
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    b, c, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, pad_h, pad_w


def _crop_pad(x: torch.Tensor, *, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :, :-pad_w]
    return x


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = _group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResidualDWBlock(nn.Module):
    """
    Swin/ConvNeXt-like conv processing without attention:
    depthwise 3x3 -> GN -> SiLU -> pointwise 1x1 -> GN -> SiLU, with residual.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.gn1 = _group_norm(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gn2 = _group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.dw(x)))
        y = self.act(self.gn2(self.pw(y)))
        return x + y


class PatchEmbed(nn.Module):
    """Patchify via stride-conv (Swin/ViT-style), but keep everything convolutional."""

    def __init__(self, in_channels: int, embed_dim: int, *, patch_size: int) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=False,
        )
        self.norm = _group_norm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Strided conv is smoother than MaxPool for small inputs.
        self.conv = ConvGNAct(in_ch, out_ch, k=3, s=2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, blocks: int) -> None:
        super().__init__()
        self.reduce = ConvGNAct(in_ch + skip_ch, out_ch, k=1, s=1, p=0)
        self.blocks = nn.Sequential(*[ResidualDWBlock(out_ch) for _ in range(int(blocks))])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _interpolate2d(x, size=(int(skip.shape[-2]), int(skip.shape[-1])))
        x = torch.cat([skip, x], dim=1)
        x = self.reduce(x)
        return self.blocks(x)


class NKDProjection(nn.Module):
    """Deprecated placeholder (kept to avoid stale imports in old notebooks)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError("NKDProjection is deprecated. Use PolynomialProjection (A,B,C,D).")


class PolynomialProjection(nn.Module):
    """
    Project per-pixel coefficients (A,B,C,D) into the final value:

        y = A*1000 + B*100 + C*10 + D
    """

    def __init__(self) -> None:
        super().__init__()
        w: torch.Tensor = torch.tensor([1000.0, 100.0, 10.0, 1.0], dtype=torch.float32)
        self._w: torch.Tensor
        self.register_buffer("_w", w, persistent=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # a,b,c,d: (B,1,H,W)
        w = self._w.reshape(1, 4, 1, 1)
        abcd = torch.cat([a, b, c, d], dim=1)  # (B,4,H,W)
        return (abcd * w).sum(dim=1, keepdim=True)  # (B,1,H,W)


class IternetUNet(nn.Module):
    """
    Deeper U-Net with Swin/ViT-style patch embedding (fully convolutional).

    Key goals:
    - Make the model deeper (more residual conv blocks per stage).
    - Reduce "checkerboard" artifacts by avoiding transpose-conv and using resize-conv with anti-aliasing.
    - Use patchify + conv processing before the U-Net encoder/decoder.

    Input:  (B, 2, H, W) where H=29, W=47 for current processed data.
    Output: (B, 1, Z, X) where (Z,X)=grid_shape (e.g. 300x600)
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        patch_size: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        blocks_per_stage: int = 3,
        stem_blocks: int = 2,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError(f"Expected 2-channel input, got in_channels={in_channels}")
        if int(out_channels) != 1:
            raise ValueError(f"Polynomial projection outputs 1 channel, got out_channels={out_channels}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if blocks_per_stage <= 0:
            raise ValueError(f"blocks_per_stage must be > 0, got {blocks_per_stage}")
        if stem_blocks < 0:
            raise ValueError(f"stem_blocks must be >= 0, got {stem_blocks}")

        self.in_channels = in_channels
        self.patch_size = int(patch_size)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.blocks_per_stage = int(blocks_per_stage)
        self.stem_blocks = int(stem_blocks)
        self.out_channels = int(out_channels)

        # Patchify + conv stem (Swin-like entry).
        self.patch_embed = PatchEmbed(in_channels, self.base_channels, patch_size=self.patch_size)
        self.stem = nn.Sequential(*[ResidualDWBlock(self.base_channels) for _ in range(self.stem_blocks)])

        # Encoder stages (in patch space)
        enc_blocks = []
        downs = []
        ch = self.base_channels
        for _ in range(self.depth):
            enc_blocks.append(nn.Sequential(*[ResidualDWBlock(ch) for _ in range(self.blocks_per_stage)]))
            downs.append(Downsample(ch, ch * 2))
            ch *= 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResidualDWBlock(ch) for _ in range(self.blocks_per_stage + 1)])

        # Decoder stages (mirror)
        ups = []
        for i in range(self.depth - 1, -1, -1):
            skip_ch = self.base_channels * (2**i)
            out_ch = skip_ch
            in_ch = self.base_channels * (2 ** (i + 1))
            ups.append(UpBlock(in_ch, skip_ch, out_ch, blocks=self.blocks_per_stage))
            # After UpBlock, channel is out_ch; next stage expects in_ch=out_ch.
        self.ups = nn.ModuleList(ups)

        # Unpatch back to (H,W) and predict coefficients at input resolution.
        self.unpatch_refine = nn.Sequential(
            ConvGNAct(self.base_channels, self.base_channels, k=3, s=1, p=1),
            ResidualDWBlock(self.base_channels),
        )
        self.abcd_head = nn.Conv2d(self.base_channels, 4, kernel_size=1)

        self.projection = PolynomialProjection()

    def forward(
        self,
        x: torch.Tensor,
        *,
        grid_shape: tuple[int, int] = (300, 600),
        return_abcd: bool = False,
    ):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input (B,{self.in_channels},H,W), got {tuple(x.shape)}")

        # We pad so that patchify + depth downsamples stay integer.
        mult = self.patch_size * (2**self.depth)
        x_pad, pad_h, pad_w = _pad_to_multiple(x, multiple=mult)

        # Patch embedding (H,W) -> (H/ps, W/ps)
        x_p = self.patch_embed(x_pad)
        x_p = self.stem(x_p)

        # Encoder
        skips: list[torch.Tensor] = []
        x_e = x_p
        for enc, down in zip(self.enc_blocks, self.downs):
            x_e = enc(x_e)
            skips.append(x_e)
            x_e = down(x_e)

        # Bottleneck
        x_d = self.bottleneck(x_e)

        # Decoder (reverse over skips)
        for up, skip in zip(self.ups, reversed(skips)):
            x_d = up(x_d, skip)

        # Unpatch: upscale back to padded input resolution and refine
        x_u = _interpolate2d(x_d, size=(int(x_pad.shape[-2]), int(x_pad.shape[-1])))
        x_u = self.unpatch_refine(x_u)

        abcd_hw = self.abcd_head(x_u)  # (B,4,H_pad,W_pad)
        abcd_hw = _crop_pad(abcd_hw, pad_h=pad_h, pad_w=pad_w)  # (B,4,H,W)

        a_hw = abcd_hw[:, 0:1]
        b_hw = abcd_hw[:, 1:2]
        c_hw = abcd_hw[:, 2:3]
        d_hw = abcd_hw[:, 3:4]

        z, x_size = grid_shape
        a_zx = _interpolate2d(a_hw, size=(z, x_size))
        b_zx = _interpolate2d(b_hw, size=(z, x_size))
        c_zx = _interpolate2d(c_hw, size=(z, x_size))
        d_zx = _interpolate2d(d_hw, size=(z, x_size))

        pred = self.projection(a_zx, b_zx, c_zx, d_zx)  # (B,1,Z,X) raw scale

        if return_abcd:
            return pred, {"a": a_zx, "b": b_zx, "c": c_zx, "d": d_zx, "a_hw": a_hw, "b_hw": b_hw, "c_hw": c_hw, "d_hw": d_hw}
        return pred

