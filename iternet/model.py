from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


def _interpolate2d(x: torch.Tensor, *, size: tuple[int, int]) -> torch.Tensor:
    try:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    except TypeError:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _pad_to_multiple(x: torch.Tensor, *, multiple: int) -> tuple[torch.Tensor, int, int]:
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), pad_h, pad_w


def _crop_pad(x: torch.Tensor, *, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :, :-pad_w]
    return x


def _resolve_heads(channels: int, preferred: int) -> int:
    heads = max(1, int(preferred))
    while heads > 1 and (channels % heads) != 0:
        heads -= 1
    return heads


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = _group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm1 = _group_norm(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = _group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.dw(x)))
        y = self.act(self.norm2(self.pw(y)))
        return x + y


class MlpBlock(nn.Module):
    def __init__(self, dim: int, *, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinWindowAttention2D(nn.Module):
    """
    Lightweight Swin-style local attention over non-overlapping windows.

    We keep windowed self-attention, but do not use shifted-window masking here.
    For this task it already adds the missing attention pathway while staying simple.
    """

    def __init__(self, dim: int, *, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window_size = max(1, int(window_size))
        self.num_heads = _resolve_heads(self.dim, num_heads)
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MlpBlock(self.dim)

    def _window_partition(self, x: torch.Tensor, ws: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = h + pad_h, w + pad_w
        xw = x.view(b, c, hp // ws, ws, wp // ws, ws)
        xw = xw.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        return xw, (h, w, hp, wp)

    def _window_reverse(self, windows: torch.Tensor, meta: tuple[int, int, int, int], ws: int, channels: int, batch: int) -> torch.Tensor:
        h, w, hp, wp = meta
        x = windows.view(batch, hp // ws, wp // ws, ws, ws, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = max(1, min(self.window_size, h, w))
        windows, meta = self._window_partition(x, ws)
        x1 = self.norm1(windows)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))
        return self._window_reverse(windows, meta, ws, c, b)


class HybridBlock(nn.Module):
    def __init__(self, channels: int, *, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.conv = ResidualDWBlock(channels)
        self.attn = SwinWindowAttention2D(channels, num_heads=num_heads, window_size=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x + self.attn(x)


class PatchEmbed(nn.Module):
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


class StageBlocks(nn.Module):
    def __init__(self, channels: int, *, blocks: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[HybridBlock(channels, num_heads=num_heads, window_size=window_size) for _ in range(int(blocks))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvGNAct(in_ch, out_ch, k=3, s=2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, blocks: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.reduce = ConvGNAct(in_ch + skip_ch, out_ch, k=1, s=1, p=0)
        self.blocks = StageBlocks(out_ch, blocks=blocks, num_heads=num_heads, window_size=window_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _interpolate2d(x, size=(int(skip.shape[-2]), int(skip.shape[-1])))
        x = torch.cat([skip, x], dim=1)
        x = self.reduce(x)
        return self.blocks(x)


class IterativeUpscaler(nn.Module):
    def __init__(self, channels: int, *, stages: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.stages = max(1, int(stages))
        self.refiners = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(channels, channels, k=3, s=1, p=1),
                    HybridBlock(channels, num_heads=num_heads, window_size=window_size),
                )
                for _ in range(self.stages)
            ]
        )

    def forward(self, x: torch.Tensor, *, out_size: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = int(out_size[0]), int(out_size[1])
        cur_h, cur_w = int(x.shape[-2]), int(x.shape[-1])
        for idx, refiner in enumerate(self.refiners):
            if idx == self.stages - 1:
                next_h, next_w = target_h, target_w
            else:
                next_h = min(target_h, max(cur_h + 1, cur_h * 2))
                next_w = min(target_w, max(cur_w + 1, cur_w * 2))
            x = _interpolate2d(x, size=(next_h, next_w))
            x = refiner(x)
            cur_h, cur_w = next_h, next_w
        return x


class NKDProjection(nn.Module):
    """Deprecated placeholder (kept to avoid stale imports in old notebooks)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError("NKDProjection is deprecated. Use DigitPolynomialProjection (A,B,C logits + D).")


class DigitPolynomialProjection(nn.Module):
    """
    A/B/C are predicted as digit logits over the range [-9, 9], D stays free and continuous.

        y = A*1000 + B*100 + C*10 + D
    """

    def __init__(self) -> None:
        super().__init__()
        weights = torch.tensor([1000.0, 100.0, 10.0], dtype=torch.float32)
        digits = torch.arange(-9.0, 10.0, dtype=torch.float32)
        self._weights: torch.Tensor
        self._digits: torch.Tensor
        self.register_buffer("_weights", weights, persistent=False)
        self.register_buffer("_digits", digits, persistent=False)

    def _digit_expectation(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=1)
        digit_table = self._digits.reshape(1, 19, 1, 1)
        value = (probs * digit_table).sum(dim=1, keepdim=True)
        return value, probs

    def forward(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        c_logits: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, a_probs = self._digit_expectation(a_logits)
        b, b_probs = self._digit_expectation(b_logits)
        c, c_probs = self._digit_expectation(c_logits)
        weights = self._weights.reshape(1, 3, 1, 1)
        abc = torch.cat([a, b, c], dim=1)
        pred = (abc * weights).sum(dim=1, keepdim=True) + d
        return pred, {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "a_probs": a_probs,
            "b_probs": b_probs,
            "c_probs": c_probs,
        }


class IternetUNet(nn.Module):
    """
    Patch-UNet with Swin-style window attention and iterative output upscaling.
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
        num_heads: int = 4,
        window_size: int = 4,
        output_upsample_stages: int = 3,
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

        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.blocks_per_stage = int(blocks_per_stage)
        self.stem_blocks = int(stem_blocks)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.output_upsample_stages = int(output_upsample_stages)
        self.out_channels = int(out_channels)

        self.patch_embed = PatchEmbed(self.in_channels, self.base_channels, patch_size=self.patch_size)
        self.stem = StageBlocks(
            self.base_channels,
            blocks=max(1, self.stem_blocks),
            num_heads=self.num_heads,
            window_size=self.window_size,
        )

        enc_blocks: list[nn.Module] = []
        downs: list[nn.Module] = []
        ch = self.base_channels
        for _ in range(self.depth):
            enc_blocks.append(StageBlocks(ch, blocks=self.blocks_per_stage, num_heads=self.num_heads, window_size=self.window_size))
            downs.append(Downsample(ch, ch * 2))
            ch *= 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)

        self.bottleneck = StageBlocks(ch, blocks=self.blocks_per_stage + 1, num_heads=self.num_heads, window_size=self.window_size)

        ups: list[nn.Module] = []
        for i in range(self.depth - 1, -1, -1):
            skip_ch = self.base_channels * (2**i)
            in_ch = self.base_channels * (2 ** (i + 1))
            ups.append(
                UpBlock(
                    in_ch,
                    skip_ch,
                    skip_ch,
                    blocks=self.blocks_per_stage,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                )
            )
        self.ups = nn.ModuleList(ups)

        self.unpatch_refine = nn.Sequential(
            ConvGNAct(self.base_channels, self.base_channels, k=3, s=1, p=1),
            HybridBlock(self.base_channels, num_heads=self.num_heads, window_size=self.window_size),
        )
        self.output_upscaler = IterativeUpscaler(
            self.base_channels,
            stages=self.output_upsample_stages,
            num_heads=self.num_heads,
            window_size=self.window_size,
        )

        self.a_head = nn.Conv2d(self.base_channels, 19, kernel_size=1)
        self.b_head = nn.Conv2d(self.base_channels, 19, kernel_size=1)
        self.c_head = nn.Conv2d(self.base_channels, 19, kernel_size=1)
        self.d_head = nn.Conv2d(self.base_channels, 1, kernel_size=1)

        self.projection = DigitPolynomialProjection()

    def forward(
        self,
        x: torch.Tensor,
        *,
        grid_shape: tuple[int, int] = (300, 600),
        return_abcd: bool = False,
    ):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input (B,{self.in_channels},H,W), got {tuple(x.shape)}")

        mult = self.patch_size * (2**self.depth)
        x_pad, pad_h, pad_w = _pad_to_multiple(x, multiple=mult)

        x_p = self.patch_embed(x_pad)
        x_p = self.stem(x_p)

        skips: list[torch.Tensor] = []
        x_e = x_p
        for enc, down in zip(self.enc_blocks, self.downs):
            x_e = enc(x_e)
            skips.append(x_e)
            x_e = down(x_e)

        x_d = self.bottleneck(x_e)
        for up, skip in zip(self.ups, reversed(skips)):
            x_d = up(x_d, skip)

        x_u = _interpolate2d(x_d, size=(int(x_pad.shape[-2]), int(x_pad.shape[-1])))
        x_u = self.unpatch_refine(x_u)
        x_u = _crop_pad(x_u, pad_h=pad_h, pad_w=pad_w)

        z, x_size = int(grid_shape[0]), int(grid_shape[1])
        x_out = self.output_upscaler(x_u, out_size=(z, x_size))

        a_logits = self.a_head(x_out)
        b_logits = self.b_head(x_out)
        c_logits = self.c_head(x_out)
        d = self.d_head(x_out)

        pred, parts = self.projection(a_logits, b_logits, c_logits, d)

        if return_abcd:
            return pred, {
                "a": parts["a"],
                "b": parts["b"],
                "c": parts["c"],
                "d": parts["d"],
                "a_logits": a_logits,
                "b_logits": b_logits,
                "c_logits": c_logits,
            }
        return pred

