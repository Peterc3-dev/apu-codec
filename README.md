# APU-Codec

**Neural audio codec built from scratch, designed for AMD APU tri-processor inference.**

The encoder runs on the NPU (INT8-friendly convolutions), the quantizer runs on the CPU (sequential codebook lookups), and the decoder runs on the GPU via Vulkan (parallel transposed convolutions). Routed by [R.A.G-Race-Router](https://github.com/Peterc3-dev/rag-race-router).

---

## Architecture

| Component | Target processor | Design rationale |
|-----------|-----------------|-----------------|
| Encoder | NPU (XDNA 2) | Small Conv1d stack, INT8-quantizable, low power |
| Quantizer (RVQ) | CPU | Sequential codebook lookups, not parallelizable |
| Decoder | GPU (Vulkan) | Parallel ConvTranspose1d, benefits from 16 CUs |

```
Audio (44.1kHz mono)
  -> Encoder: Conv1d stack, 512x downsample -> ~86 tokens/sec
  -> RVQ: 8 codebooks x 2048 entries, residual quantization
  -> Decoder: ConvTranspose1d stack, 512x upsample -> 44.1kHz audio
```

### Key specs

| Spec | Value |
|------|-------|
| Parameters | 32.4M |
| Sample rate | 44.1 kHz (native, no resampling from 32kHz) |
| Codebooks | 8 x 2048 entries |
| Latent dimension | 128 |
| Compression ratio | 512x (44,100 samples/sec -> ~86 tokens/sec) |
| Loss function | L1 + multi-resolution STFT + commitment (no adversarial) |

### Design choices vs EnCodec

| | APU-Codec | EnCodec |
|---|---|---|
| Sample rate | 44.1 kHz | 24/32 kHz |
| Codebooks | 8 x 2048 | 4 x 1024 |
| Processor split | NPU / CPU / GPU | Single GPU |
| Quantization noise | Lower (larger codebooks) | Higher |

---

## Status

**Architecture complete, training started (~1 epoch).**

- Model definition: done
- Training loop: done (L1 + spectral + commitment loss, AdamW, cosine LR)
- First training run: started on synthetic data
- Inference test harness: done
- Real audio training: pending (needs audio dataset)

---

## Quick start

### Train

```bash
# Set this for MIOpen kernel search on gfx1150
export MIOPEN_FIND_MODE=3

# Activate a PyTorch environment with ROCm support
source ~/pytorch-gfx1150-env/bin/activate

# Train (uses ~/Music/ or generates synthetic data)
python training/train.py
```

> **Note:** `MIOPEN_FIND_MODE=3` is required for training on gfx1150 (RDNA 3.5). Without it, MIOpen's default kernel search hangs or crashes on unsupported architectures.

### Test inference

```python
from model.codec import APUCodec
import torch

codec = APUCodec()
print(f"Parameters: {codec.param_count:,}")  # 32,400,xxx

audio = torch.randn(1, 1, 44100 * 2)  # 2 seconds
reconstructed, codes, loss = codec(audio)
print(f"Codes shape: {codes.shape}")  # (1, 8, ~172)
print(f"Compression: {codec.compression_ratio}x")  # 512x
```

---

## Project structure

```
apu-codec/
├── model/
│   └── codec.py          # APUCodec, Encoder, Decoder, RVQ
├── training/
│   └── train.py          # Training loop (L1 + spectral + commitment)
├── inference/
│   └── test_codec.py     # Inference test harness
└── README.md
```

---

## Related projects

- [R.A.G-Race-Router](https://github.com/Peterc3-dev/rag-race-router) — Tri-processor inference runtime that routes this codec across NPU/CPU/GPU
- [amdxdna-strix-fix](https://github.com/Peterc3-dev/amdxdna-strix-fix) — NPU driver fix required for encoder deployment
- [unified-ml](https://github.com/Peterc3-dev/unified-ml) — Vulkan + HIP kernel benchmarks for the GPU decoder path
- [pytorch-gfx1150](https://github.com/Peterc3-dev/pytorch-gfx1150) — PyTorch with native gfx1150 GPU support (used for training)

## Author

**Peter Clemente** ([@Peterc3-dev](https://github.com/Peterc3-dev))

## License

MIT
