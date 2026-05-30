"""Unit tests for the pure conv-arithmetic helpers in ``model/shapes.py``.

These exercise *only* integer length arithmetic. They deliberately avoid
importing :mod:`model.codec` (and therefore torch / torchaudio / soundfile),
so the suite runs on any machine with no GPU, NPU, or model weights.

``model/shapes.py`` is loaded directly by path to bypass the package
``__init__`` (which imports the torch-backed codec).
"""

import importlib.util
import os

_SHAPES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "shapes.py",
)
_spec = importlib.util.spec_from_file_location("apu_codec_shapes", _SHAPES_PATH)
shapes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shapes)


# --- conv1d output-length formula -----------------------------------------

def test_conv1d_identity_kernel():
    # kernel=1, stride=1, no padding -> length unchanged.
    assert shapes.conv1d_out_length(100, kernel_size=1) == 100


def test_conv1d_input_conv_preserves_length():
    # The encoder's input conv: kernel=7, padding=3, stride=1 -> same length.
    assert shapes.conv1d_out_length(44100, kernel_size=7, padding=3) == 44100


def test_conv1d_stride2_halves_roughly():
    # kernel=4, stride=2, padding=1 (the stride-2 downsample block).
    assert shapes.conv1d_out_length(88200, kernel_size=4, stride=2, padding=1) == 44100


# --- conv-transpose output-length formula ---------------------------------

def test_conv_transpose_identity():
    # kernel=1, stride=1, no padding -> length unchanged.
    assert shapes.conv_transpose1d_out_length(50, kernel_size=1) == 50


def test_conv_transpose_doubles_roughly():
    # Inverse of the stride-2 downsample: kernel=4, stride=2, padding=1.
    assert shapes.conv_transpose1d_out_length(44100, kernel_size=4, stride=2, padding=1) == 88200


# --- per-block down/upsample helpers --------------------------------------

def test_downsample_then_upsample_same_stride_roundtrips():
    # For these symmetric kernel/padding choices a single down/up pair is exact.
    for stride in (2, 4, 8):
        length = 4096
        down = shapes.downsample_out_length(length, stride)
        up = shapes.upsample_out_length(down, stride)
        assert up == length, f"stride {stride}: {length} -> {down} -> {up}"


# --- end-to-end architecture invariants -----------------------------------

def test_one_second_gives_86_tokens():
    # README / docstrings claim ~86 tokens per 44.1 kHz second.
    assert shapes.encoder_token_count(44100) == 86


def test_two_second_segment_token_count():
    # README inline example claims codes shape (1, 8, ~172) for a 2s clip.
    assert shapes.encoder_token_count(44100 * 2) == 172


def test_compression_ratio_is_about_512x():
    samples = 44100 * 2
    tokens = shapes.encoder_token_count(samples)
    ratio = samples / tokens
    # The advertised compression ratio is 512x; arithmetic lands within 1%.
    assert abs(ratio - shapes.TOTAL_DOWNSAMPLE) / shapes.TOTAL_DOWNSAMPLE < 0.01


def test_decode_length_close_to_input():
    # Decoder output differs slightly from the input length due to stride
    # arithmetic — train.py compensates by trimming to min_len. Confirm the
    # discrepancy is small (well under one downsample stride of samples).
    samples = 44100 * 2
    tokens = shapes.encoder_token_count(samples)
    decoded = shapes.decoder_sample_count(tokens)
    assert abs(decoded - samples) < 512


def test_stride_schedules_are_mirrored():
    # Encoder and decoder must apply the same total factor in reverse order.
    assert tuple(reversed(shapes.ENCODER_STRIDES)) == shapes.DECODER_STRIDES
    prod = 1
    for s in shapes.ENCODER_STRIDES:
        prod *= s
    assert prod == shapes.TOTAL_DOWNSAMPLE
