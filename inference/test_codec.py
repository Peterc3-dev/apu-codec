"""Test APU-Codec — encode and decode an audio file, compare quality."""

import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.codec import APUCodec


def test(audio_path: str, checkpoint_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    codec = APUCodec()
    if checkpoint_path and os.path.exists(checkpoint_path):
        codec.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("WARNING: No checkpoint — using random weights (will sound like noise)")

    codec = codec.to(device).eval()
    print(f"Model params: {codec.param_count:,}")
    print(f"Device: {device}")

    # Load audio
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.T
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.unsqueeze(0).to(device)

    # Encode -> decode
    with torch.no_grad():
        codes = codec.encode(waveform)
        reconstructed = codec.decode(codes)

    # Match lengths
    min_len = min(reconstructed.shape[-1], waveform.shape[-1])
    orig = waveform[0, 0, :min_len].cpu()
    recon = reconstructed[0, 0, :min_len].cpu()

    # Save reconstructed
    base, ext = os.path.splitext(audio_path)
    output_path = f"{base}_apu_codec.wav"
    sf.write(output_path, recon.numpy(), 44100)

    # Quality metrics
    snr = 10 * torch.log10(
        torch.sum(orig ** 2) / (torch.sum((orig - recon) ** 2) + 1e-8)
    )

    # Spectral similarity
    orig_stft = torch.stft(orig, 2048, 512, return_complex=True).abs()
    recon_stft = torch.stft(recon, 2048, 512, return_complex=True).abs()
    spectral_sim = F.cosine_similarity(
        orig_stft.flatten().unsqueeze(0),
        recon_stft.flatten().unsqueeze(0)
    ).item()

    duration_s = min_len / 44100

    print(f"\nResults:")
    print(f"  Original:      {audio_path}")
    print(f"  Reconstructed: {output_path}")
    print(f"  Duration:      {duration_s:.1f}s")
    print(f"  Codes shape:   {codes.shape} ({codes.shape[1]} codebooks x {codes.shape[2]} tokens)")
    print(f"  Compression:   {waveform.shape[-1] / codes.shape[-1]:.0f}x")
    print(f"  Bitrate:       {codes.shape[1] * codes.shape[2] * 11 / duration_s / 1000:.1f} kbps "
          f"(at 11 bits/code)")
    print(f"  SNR:           {snr:.1f} dB")
    print(f"  Spectral sim:  {spectral_sim:.4f}")

    return {
        "snr_db": snr.item(),
        "spectral_similarity": spectral_sim,
        "compression_ratio": waveform.shape[-1] / codes.shape[-1],
        "codes_shape": list(codes.shape),
        "output_path": output_path,
    }


# Need F for cosine_similarity
import torch.nn.functional as F


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Music/rag-playlist/01_midnight_highway.mp3"
    )
    ckpt = os.path.expanduser("~/projects/apu-codec/checkpoints/best.pt")
    if not os.path.exists(ckpt):
        ckpt = None
    test(audio, ckpt)
