"""Pure shape/length arithmetic for the APU-Codec conv stack.

These helpers reproduce the output-length formulas used by PyTorch's
``nn.Conv1d`` and ``nn.ConvTranspose1d`` *without* importing torch, so the
architecture's compression invariants can be reasoned about (and tested) on
any machine — no GPU, NPU, or model weights required.

The layer configurations mirror :mod:`model.codec`:

  Encoder downsample strides: 2, 4, 8, 8  (512x total)
  Decoder upsample   strides: 8, 8, 4, 2  (512x total)

with each downsample/upsample conv using ``kernel_size = 2 * stride`` and
``padding = stride // 2``.
"""

from __future__ import annotations

# Stride schedule shared by encoder (downsample) and decoder (upsample).
ENCODER_STRIDES = (2, 4, 8, 8)
DECODER_STRIDES = (8, 8, 4, 2)

# Total downsampling factor claimed throughout the codebase / README.
TOTAL_DOWNSAMPLE = 512


def conv1d_out_length(
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Output length of ``nn.Conv1d`` for a 1-D input of ``length`` samples."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv_transpose1d_out_length(
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
) -> int:
    """Output length of ``nn.ConvTranspose1d`` for a 1-D input of ``length``."""
    return (
        (length - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


def downsample_out_length(length: int, stride: int) -> int:
    """Length after one encoder downsample conv (kernel=2*stride, pad=stride//2)."""
    return conv1d_out_length(
        length,
        kernel_size=2 * stride,
        stride=stride,
        padding=stride // 2,
    )


def upsample_out_length(length: int, stride: int) -> int:
    """Length after one decoder upsample conv (kernel=2*stride, pad=stride//2)."""
    return conv_transpose1d_out_length(
        length,
        kernel_size=2 * stride,
        stride=stride,
        padding=stride // 2,
    )


def encoder_token_count(samples: int) -> int:
    """Number of latent tokens produced by the encoder for ``samples`` inputs.

    The leading kernel-7/pad-3 input conv preserves length, so only the four
    strided downsample convs change it.
    """
    length = samples
    for stride in ENCODER_STRIDES:
        length = downsample_out_length(length, stride)
    return length


def decoder_sample_count(tokens: int) -> int:
    """Number of audio samples reconstructed by the decoder from ``tokens``."""
    length = tokens
    for stride in DECODER_STRIDES:
        length = upsample_out_length(length, stride)
    return length
