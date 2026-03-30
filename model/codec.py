"""APU-Codec: Neural audio codec for AMD APU tri-processor inference.

Architecture:
  Encoder (-> NPU): Conv1d stack, downsample 44.1kHz -> latent tokens
  Quantizer (-> CPU): Residual Vector Quantization with large codebooks
  Decoder (-> GPU): ConvTranspose1d stack, upsample tokens -> audio

Design choices:
  - 44.1kHz native sample rate (higher fidelity than EnCodec's 32kHz)
  - 8 codebooks x 2048 entries (vs EnCodec's 4 x 1024 — less quantization noise)
  - Encoder designed for INT8 quantization (NPU-friendly)
  - Decoder designed for Vulkan compute (parallel convolutions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    """Residual convolution block with dilation."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad = 3 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7,
                               dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x + residual


class EncoderBlock(nn.Module):
    """Downsampling block: 3 residual units + strided conv."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.res1 = ResidualUnit(in_ch, dilation=1)
        self.res2 = ResidualUnit(in_ch, dilation=3)
        self.res3 = ResidualUnit(in_ch, dilation=9)
        self.downsample = nn.Conv1d(in_ch, out_ch,
                                    kernel_size=2 * stride,
                                    stride=stride,
                                    padding=stride // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = F.gelu(self.downsample(x))
        return x


class DecoderBlock(nn.Module):
    """Upsampling block: transposed conv + 3 residual units."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch,
                                           kernel_size=2 * stride,
                                           stride=stride,
                                           padding=stride // 2)
        self.res1 = ResidualUnit(out_ch, dilation=1)
        self.res2 = ResidualUnit(out_ch, dilation=3)
        self.res3 = ResidualUnit(out_ch, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.upsample(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class Encoder(nn.Module):
    """Audio encoder: 44.1kHz mono -> latent vectors.

    Downsamples by 512x total: 44100 Hz -> ~86 tokens/sec.
    Designed for NPU deployment (small, INT8-friendly).
    """

    def __init__(self, latent_dim: int = 128, channels: int = 64):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, kernel_size=7, padding=3)

        # Progressive downsampling: 2x, 4x, 8x, 8x = 512x total
        self.blocks = nn.ModuleList([
            EncoderBlock(channels, channels * 2, stride=2),
            EncoderBlock(channels * 2, channels * 4, stride=4),
            EncoderBlock(channels * 4, channels * 8, stride=8),
            EncoderBlock(channels * 8, channels * 8, stride=8),
        ])

        self.output_conv = nn.Conv1d(channels * 8, latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, samples) -> (batch, latent_dim, tokens)"""
        x = F.gelu(self.input_conv(x))
        for block in self.blocks:
            x = block(x)
        return self.output_conv(x)


class ResidualVectorQuantizer(nn.Module):
    """RVQ: 8 codebooks x 2048 entries x latent_dim dims.

    Larger codebooks = less quantization noise per token.
    """

    def __init__(self, dim: int = 128, n_codebooks: int = 8, codebook_size: int = 2048):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim) / dim ** 0.5)
            for _ in range(n_codebooks)
        ])

    def _quantize_one(self, x: torch.Tensor, codebook: torch.Tensor):
        """Nearest neighbor lookup.

        Args:
            x: (batch, dim, tokens)
            codebook: (codebook_size, dim)

        Returns:
            quantized: (batch, dim, tokens)
            indices: (batch, tokens)
        """
        b, d, t = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, d)  # (b*t, d)

        # Efficient distance: ||x - c||^2 = ||x||^2 - 2<x,c> + ||c||^2
        x_sq = (flat ** 2).sum(dim=-1, keepdim=True)       # (b*t, 1)
        c_sq = (codebook ** 2).sum(dim=-1, keepdim=True).T  # (1, codebook_size)
        dist = x_sq - 2 * flat @ codebook.T + c_sq         # (b*t, codebook_size)

        indices = dist.argmin(dim=-1)  # (b*t,)
        quantized = codebook[indices].reshape(b, t, d).permute(0, 2, 1)

        return quantized, indices.reshape(b, t)

    def forward(self, x: torch.Tensor):
        """x: (batch, dim, tokens) -> quantized, codes, commitment_loss"""
        residual = x.clone()
        all_codes = []
        total_quantized = torch.zeros_like(x)
        commitment_loss = torch.tensor(0.0, device=x.device)

        for codebook in self.codebooks:
            quantized, codes = self._quantize_one(residual, codebook)

            # Straight-through estimator
            quantized_st = residual + (quantized - residual).detach()

            total_quantized = total_quantized + quantized_st
            residual = residual - quantized.detach()
            all_codes.append(codes)

            # Commitment loss: push encoder output toward codebook entries
            commitment_loss = commitment_loss + F.mse_loss(
                residual.detach(), torch.zeros_like(residual)
            )

        codes = torch.stack(all_codes, dim=1)  # (batch, n_codebooks, tokens)
        return total_quantized, codes, commitment_loss


class Decoder(nn.Module):
    """Audio decoder: latent vectors -> 44.1kHz audio.

    Mirror of encoder with ConvTranspose.
    Designed for GPU (Vulkan) deployment (parallel convolutions).
    """

    def __init__(self, latent_dim: int = 128, channels: int = 64):
        super().__init__()
        self.input_conv = nn.Conv1d(latent_dim, channels * 8, kernel_size=3, padding=1)

        # Progressive upsampling: 8x, 8x, 4x, 2x = 512x total
        self.blocks = nn.ModuleList([
            DecoderBlock(channels * 8, channels * 8, stride=8),
            DecoderBlock(channels * 8, channels * 4, stride=8),
            DecoderBlock(channels * 4, channels * 2, stride=4),
            DecoderBlock(channels * 2, channels, stride=2),
        ])

        self.output_conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, latent_dim, tokens) -> (batch, 1, samples)"""
        x = F.gelu(self.input_conv(x))
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.output_conv(x))


class APUCodec(nn.Module):
    """Complete neural audio codec: encode -> quantize -> decode.

    Routing through R.A.G-Race-Router:
      Encoder  -> NPU  (small, INT8-friendly)
      Quantizer -> CPU  (vector lookups, sequential)
      Decoder  -> GPU via Vulkan (parallel convolutions)
    """

    def __init__(self, latent_dim: int = 128, channels: int = 64,
                 n_codebooks: int = 8, codebook_size: int = 2048):
        super().__init__()
        self.encoder = Encoder(latent_dim, channels)
        self.quantizer = ResidualVectorQuantizer(latent_dim, n_codebooks, codebook_size)
        self.decoder = Decoder(latent_dim, channels)
        self.sample_rate = 44100
        self.latent_dim = latent_dim

    def forward(self, audio: torch.Tensor):
        """audio: (batch, 1, samples) -> reconstructed, codes, commitment_loss"""
        latent = self.encoder(audio)
        quantized, codes, commitment_loss = self.quantizer(latent)
        reconstructed = self.decoder(quantized)
        return reconstructed, codes, commitment_loss

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete codes. Returns (batch, n_codebooks, tokens)."""
        latent = self.encoder(audio)
        _, codes, _ = self.quantizer(latent)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes back to audio. Returns (batch, 1, samples)."""
        quantized = torch.zeros(
            codes.shape[0], self.latent_dim, codes.shape[2],
            device=codes.device
        )
        for i, codebook in enumerate(self.quantizer.codebooks):
            quantized = quantized + codebook[codes[:, i]].permute(0, 2, 1)
        return self.decoder(quantized)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def compression_ratio(self) -> float:
        """Samples per token (how much we compress)."""
        return 512.0  # 44100 / ~86 tokens/sec
