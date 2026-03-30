"""Train APU-Codec on audio data.

Loss function:
  1. Reconstruction loss (L1 time-domain)
  2. Multi-scale spectral loss (STFT at multiple resolutions)
  3. Commitment loss (VQ training stability)

No adversarial loss in v1 — simpler, faster convergence.
"""

import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.codec import APUCodec


def spectral_loss(pred: torch.Tensor, target: torch.Tensor,
                  n_fft_list=(512, 1024, 2048)) -> torch.Tensor:
    """Multi-resolution STFT loss — catches frequency-domain artifacts."""
    loss = torch.tensor(0.0, device=pred.device)

    for n_fft in n_fft_list:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)

        pred_stft = torch.stft(pred.squeeze(1), n_fft, hop, window=window,
                               return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft, hop, window=window,
                                 return_complex=True)

        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        # Spectral convergence
        loss = loss + torch.norm(target_mag - pred_mag) / (torch.norm(target_mag) + 1e-8)

        # Log magnitude loss
        loss = loss + F.l1_loss(
            torch.log(pred_mag + 1e-8),
            torch.log(target_mag + 1e-8)
        )

    return loss / len(n_fft_list)


def find_audio_files() -> list:
    """Find all audio files for training."""
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(glob.glob(os.path.expanduser(f"~/Music/**/{ext}"), recursive=True))
        audio_files.extend(glob.glob(f"/tmp/**/{ext}", recursive=True))
    return audio_files


def generate_synthetic_data(n: int = 30) -> list:
    """Generate synthetic training data when no audio files are available."""
    out_dir = "/tmp/codec-training-data"
    os.makedirs(out_dir, exist_ok=True)

    sr = 44100
    duration = 3  # seconds
    files = []

    for i in range(n):
        t = np.linspace(0, duration, sr * duration, dtype=np.float32)

        # Mix of tones, harmonics, and noise at random frequencies
        freq = np.random.uniform(80, 6000)
        audio = np.sin(2 * np.pi * freq * t) * 0.3
        audio += np.sin(2 * np.pi * freq * 2 * t) * 0.15
        audio += np.sin(2 * np.pi * freq * 3 * t) * 0.08

        # Add some modulation for variety
        mod_freq = np.random.uniform(0.5, 8.0)
        audio *= (1.0 + 0.3 * np.sin(2 * np.pi * mod_freq * t))

        # Add noise
        audio += np.random.randn(len(t)).astype(np.float32) * 0.02

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

        path = os.path.join(out_dir, f"synth_{i:03d}.wav")
        sf.write(path, audio, sr)
        files.append(path)

    return files


def load_segment(path: str, segment_length: int, device: str) -> torch.Tensor:
    """Load a random audio segment from a file using soundfile."""
    data, sr = sf.read(path, dtype="float32")

    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    elif waveform.dim() == 2:
        waveform = waveform.T  # (channels, samples)

    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] < segment_length:
        return None

    start = np.random.randint(0, waveform.shape[1] - segment_length)
    segment = waveform[:, start:start + segment_length].unsqueeze(0)
    return segment.to(device)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    codec = APUCodec(latent_dim=128, channels=64, n_codebooks=8, codebook_size=2048)
    codec = codec.to(device)
    print(f"Model params: {codec.param_count:,}")

    optimizer = torch.optim.AdamW(codec.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Find training data
    audio_files = find_audio_files()
    if not audio_files:
        print("No audio files found. Generating synthetic training data...")
        audio_files = generate_synthetic_data(30)

    print(f"Training files: {len(audio_files)}")

    segment_length = 44100 * 2  # 2 seconds
    n_epochs = 50
    max_files_per_epoch = 50
    best_loss = float("inf")

    save_dir = os.path.expanduser("~/projects/apu-codec/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_spec = 0.0
        epoch_commit = 0.0
        n_batches = 0
        t0 = time.time()

        np.random.shuffle(audio_files)

        for audio_path in audio_files[:max_files_per_epoch]:
            try:
                segment = load_segment(audio_path, segment_length, device)
                if segment is None:
                    continue

                # Forward pass
                reconstructed, codes, commitment_loss = codec(segment)

                # Match lengths (decoder output may differ slightly due to stride arithmetic)
                min_len = min(reconstructed.shape[-1], segment.shape[-1])
                reconstructed = reconstructed[..., :min_len]
                target = segment[..., :min_len]

                # Losses
                l1_loss = F.l1_loss(reconstructed, target)
                spec_loss = spectral_loss(reconstructed, target)
                total_loss = l1_loss + spec_loss + 0.1 * commitment_loss

                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_l1 += l1_loss.item()
                epoch_spec += spec_loss.item()
                epoch_commit += commitment_loss.item()
                n_batches += 1

            except Exception:
                continue

        scheduler.step()

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_l1 = epoch_l1 / n_batches
            avg_spec = epoch_spec / n_batches
            avg_commit = epoch_commit / n_batches
            elapsed = time.time() - t0

            print(f"Epoch {epoch + 1:3d}/{n_epochs}: "
                  f"loss={avg_loss:.4f} (l1={avg_l1:.4f} spec={avg_spec:.4f} "
                  f"commit={avg_commit:.4f}) [{elapsed:.1f}s, {n_batches} batches]")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(codec.state_dict(), os.path.join(save_dir, "best.pt"))
                print(f"  -> Saved best model (loss={best_loss:.4f})")

    # Save final
    torch.save(codec.state_dict(), os.path.join(save_dir, "final.pt"))
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {save_dir}/")


if __name__ == "__main__":
    train()
