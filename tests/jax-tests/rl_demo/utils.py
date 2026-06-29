"""
Shared utilities for RL demos.
"""

import sys
from pathlib import Path
import subprocess
import json
import jax
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import load_module
from jax import numpy as jnp
import librosax
from librosax.feature import (
    melspectrogram,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    spectral_contrast,
    rms,
    zero_crossing_rate,
    mfcc,
)


def compile_synth():
    """
    Compile synth.dsp to Python module.

    Returns:
        The compiled mydsp class, or None if compilation failed
    """
    print("Compiling synth.dsp...")

    dsp_file = Path(__file__).parent / "synth.dsp"
    output_file = Path(__file__).parent / "synth.py"

    # Try to use Faust from source tree first, fallback to installed version
    faust_bin = Path(__file__).parent.parent.parent.parent / "build" / "bin" / "faust"

    if faust_bin.exists():
        # Development: use local build with local architecture and libraries
        arch_file = Path(__file__).parent.parent.parent.parent / "architecture" / "jax" / "minimal.py"
        lib_dir = Path(__file__).parent.parent.parent.parent / "libraries"
        cmd = [
            str(faust_bin), "-lang", "nnx",
            "-a", str(arch_file),
            "-I", str(lib_dir),
            str(dsp_file),
            "-o", str(output_file)
        ]
    else:
        # Production: use installed Faust with installed architecture
        cmd = [
            "faust", "-lang", "nnx",
            "-a", "jax/minimal.py",
            str(dsp_file),
            "-o", str(output_file)
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return None

    print("✓ Compilation successful")
    return load_module(output_file, "synth").mydsp


def spectral_distance(audio1: jnp.ndarray, audio2: jnp.ndarray, sample_rate: int, n_fft: int = 2048) -> jnp.ndarray:
    """
    Compute spectral distance between two audio signals using mel-spectrogram.

    Args:
        audio1: Audio [channels, samples] or [batch, channels, samples]
        audio2: Audio [channels, samples] or [batch, channels, samples]
        sample_rate: Sample rate
        n_fft: FFT size

    Returns:
        Scalar distance (lower is better)
    """
    hop_length = n_fft // 4

    # Compute mel-spectrograms (96 bins for better frequency resolution)
    mel1 = melspectrogram(y=audio1, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=96)
    mel2 = melspectrogram(y=audio2, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=96)

    # L1 distance on log-magnitude
    log_mel1 = jnp.log(mel1 + 1e-5)
    log_mel2 = jnp.log(mel2 + 1e-5)

    return jnp.mean(jnp.abs(log_mel1 - log_mel2))


def batched_spectral_distance(audio1_batch: jnp.ndarray, audio2_batch: jnp.ndarray, sample_rate: int, n_fft: int = 2048) -> jnp.ndarray:
    """
    Compute spectral distance for batches by concatenating before melspectrogram.

    This is ~2x faster than vmapping spectral_distance because it calls melspectrogram once.

    Args:
        audio1_batch: Audio [batch, channels, samples]
        audio2_batch: Audio [batch, channels, samples]
        sample_rate: Sample rate
        n_fft: FFT size

    Returns:
        distances: [batch] array of distances
    """
    batch_size = audio1_batch.shape[0]
    hop_length = n_fft // 4

    # Concatenate both batches along batch axis
    combined = jnp.concatenate([audio1_batch, audio2_batch], axis=0)  # [2*batch, channels, samples]

    # Compute mel spectrogram once for all audio (96 bins for better frequency resolution)
    mel_combined = melspectrogram(y=combined, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=96)
    log_mel_combined = jnp.log(mel_combined + 1e-5)

    # Split back into two batches
    log_mel1 = log_mel_combined[:batch_size]  # [batch, channels, n_mels, time]
    log_mel2 = log_mel_combined[batch_size:]  # [batch, channels, n_mels, time]

    # Compute L1 distance per sample in batch
    distances = jnp.mean(jnp.abs(log_mel1 - log_mel2), axis=(1, 2, 3))  # [batch]

    return distances


def compute_synthrl_reward(
    audio1_batch: jnp.ndarray,
    audio2_batch: jnp.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    sc_coef: float = 0.7,
    mfcc_coef: float = 0.03,
) -> jnp.ndarray:
    """
    Compute reward following SynthRL paper (IJCAI 2025).

    Uses three components:
    1. Spectral Convergence (SC): Normalized Frobenius norm distance
    2. Log Magnitude Error: MAE on log-scale spectrograms
    3. MFCC MAE: Mean absolute error on MFCCs

    Reward = 1 / clamp(0.7*SC + 0.27*log_mae + 0.03*mfcc_mae, 0.1, 5.0)

    Args:
        audio1_batch: Generated audio [batch, channels, samples]
        audio2_batch: Target audio [batch, channels, samples]
        sample_rate: Sample rate
        n_fft: FFT size for STFT
        sc_coef: Weight for spectral convergence (default: 0.7)
        mfcc_coef: Weight for MFCC error (default: 0.03)

    Returns:
        rewards: [batch] array of rewards (higher is better)
    """
    batch_size = audio1_batch.shape[0]
    hop_length = n_fft // 4

    # Concatenate both batches for efficient processing
    combined = jnp.concatenate([audio1_batch, audio2_batch], axis=0)  # [2*batch, channels, samples]

    # 1. Compute magnitude spectrograms
    # STFT returns complex values, take magnitude
    stft_combined = librosax.stft(waveform=combined, n_fft=n_fft, hop_length=hop_length)  # [2*batch, channels, freq, time]
    spec_combined = jnp.abs(stft_combined)  # Magnitude spectrogram

    # Split into generated and target
    spec_gen = spec_combined[:batch_size]  # [batch, channels, freq, time]
    spec_target = spec_combined[batch_size:]  # [batch, channels, freq, time]

    # 2. Spectral Convergence (normalized Frobenius distance)
    # SC = ||spec_gen - spec_target||_F / ||spec_target||_F
    fro_diff = jnp.sqrt(jnp.sum((spec_gen - spec_target) ** 2, axis=(1, 2, 3)))  # [batch]
    fro_target = jnp.sqrt(jnp.sum(spec_target ** 2, axis=(1, 2, 3)))  # [batch]
    sc = jnp.clip(fro_diff / (fro_target + 1e-8), max=5.0)  # [batch]

    # 3. Log Magnitude Error
    log_spec_gen = jnp.log10(spec_gen + 1e-8)
    log_spec_target = jnp.log10(spec_target + 1e-8)
    log_mae = jnp.mean(jnp.abs(log_spec_gen - log_spec_target), axis=(1, 2, 3))  # [batch]

    # 4. MFCC MAE (compute MFCCs for both)
    # Use 13 MFCCs as in SynthRL
    mfcc_gen = mfcc(y=audio1_batch, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    mfcc_target = mfcc(y=audio2_batch, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    mfcc_mae = jnp.mean(jnp.abs(mfcc_gen - mfcc_target), axis=(1, 2, 3))  # [batch]

    # 5. Weighted combination
    # log_mae coefficient = (1 - sc_coef - mfcc_coef) = 1 - 0.7 - 0.03 = 0.27
    log_mae_coef = 1.0 - sc_coef - mfcc_coef
    combined_error = sc_coef * sc + log_mae_coef * log_mae + mfcc_coef * mfcc_mae

    # 6. Inverse reward with clamping
    rewards = 1.0 / jnp.clip(combined_error, min=0.1, max=5.0)

    return rewards


def extract_features(audio: jnp.ndarray, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512,
                    use_stats_normalization: bool = False) -> jnp.ndarray:
    """Extract audio features using standard librosax functions.

    Uses expanded feature set with MFCCs and spectral contrast for better
    representation of audio characteristics.

    Args:
        audio: Audio with shape [batch, channels, samples]
        sample_rate: Sample rate in Hz
        n_fft: FFT size for spectral features
        hop_length: Hop length for STFT
        use_stats_normalization: If True, use feature_stats.json for normalization

    Returns:
        features: [batch, n_features] where n_features = 20 (conservative approach)
            - RMS mean/std (2)
            - Zero crossing rate mean/std (2)
            - Spectral centroid mean/std (2)
            - Spectral bandwidth mean/std (2)
            - Spectral rolloff mean/std (2)
            - Spectral flatness mean/std (2)
            - MFCCs 0-4 mean (5 coefficients, expanded from 1)
            - Spectral contrast mean (3 bands)
    """
    # Compute spectral features using librosax
    # These functions expect audio with shape (batch, time) or (time,)
    # But our audio is (batch, channels, samples) - take first channel
    audio_mono = audio[:, 0, :]  # [batch, samples]

    # Compute features - some support batch natively, others need vmap
    # Native batch support: rms, zero_crossing_rate, spectral_flatness, mfcc
    rms_vals = rms(y=audio_mono, frame_length=n_fft, hop_length=hop_length)
    zcr_vals = zero_crossing_rate(audio_mono, frame_length=n_fft, hop_length=hop_length)
    flatness = spectral_flatness(y=audio_mono, n_fft=n_fft, hop_length=hop_length)

    # Expanded MFCCs: 5 coefficients instead of 1
    mfcc_vals = mfcc(y=audio_mono, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=5)

    # Spectral contrast: 3 bands (conservative)
    contrast_vals = jax.vmap(lambda y: spectral_contrast(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_bands=3))(audio_mono)

    # Need vmap for these (librosax batch handling bug)
    centroid = jax.vmap(lambda y: spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))(audio_mono)
    bandwidth = jax.vmap(lambda y: spectral_bandwidth(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))(audio_mono)
    rolloff = jax.vmap(lambda y: spectral_rolloff(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length))(audio_mono)

    # Aggregate over time: compute mean and std for each feature
    # Shape: (batch, 1, n_frames) -> squeeze to (batch, n_frames)
    rms_squeezed = rms_vals[:, 0, :]
    zcr_squeezed = zcr_vals[:, 0, :]
    centroid_squeezed = centroid[:, 0, :]
    bandwidth_squeezed = bandwidth[:, 0, :]
    rolloff_squeezed = rolloff[:, 0, :]
    flatness_squeezed = flatness[:, 0, :]

    if use_stats_normalization:
        # Load normalization statistics
        import json
        stats_path = Path(__file__).parent / 'feature_stats.json'
        with open(stats_path) as f:
            stats = json.load(f)

        # Raw features (unnormalized)
        raw_features = []

        # Original features (12)
        raw_features.extend([
            jnp.mean(rms_squeezed, axis=1),
            jnp.std(rms_squeezed, axis=1),
            jnp.mean(zcr_squeezed, axis=1),
            jnp.std(zcr_squeezed, axis=1),
            jnp.mean(centroid_squeezed, axis=1),
            jnp.std(centroid_squeezed, axis=1),
            jnp.mean(bandwidth_squeezed, axis=1),
            jnp.std(bandwidth_squeezed, axis=1),
            jnp.mean(rolloff_squeezed, axis=1),
            jnp.std(rolloff_squeezed, axis=1),
            jnp.mean(flatness_squeezed, axis=1),
            jnp.std(flatness_squeezed, axis=1),
        ])

        # Expanded MFCCs: mean of 5 coefficients
        for i in range(5):
            mfcc_coeff = mfcc_vals[:, i, :]  # [batch, n_frames]
            raw_features.append(jnp.mean(mfcc_coeff, axis=1))

        # Spectral contrast: mean across 3 bands
        for i in range(3):
            contrast_band = contrast_vals[:, i, :]  # [batch, n_frames]
            raw_features.append(jnp.mean(contrast_band, axis=1))

        # Stack and normalize using statistics
        features = jnp.stack(raw_features, axis=1)  # [batch, 20]
        mean = jnp.array(stats['mean'])
        std = jnp.array(stats['std'])
        normalized = (features - mean) / (std + 1e-8)
        features = jnp.clip(normalized, -3.0, 3.0)  # Clip outliers

    else:
        # Heuristic normalization (for generating stats)
        features = jnp.stack([
            # RMS: typically 0-0.3 for normalized audio
            jnp.mean(rms_squeezed, axis=1) * 3.0,
            jnp.std(rms_squeezed, axis=1) * 10.0,

            # Zero crossing rate: 0-0.5 typically
            jnp.mean(zcr_squeezed, axis=1) * 2.0,
            jnp.std(zcr_squeezed, axis=1) * 10.0,

            # Spectral centroid: 0-sr/2, normalize by Nyquist
            jnp.mean(centroid_squeezed, axis=1) / (sample_rate / 2),
            jnp.std(centroid_squeezed, axis=1) / (sample_rate / 4),

            # Spectral bandwidth: 0-sr/2, normalize by Nyquist
            jnp.mean(bandwidth_squeezed, axis=1) / (sample_rate / 2),
            jnp.std(bandwidth_squeezed, axis=1) / (sample_rate / 4),

            # Spectral rolloff: 0-sr/2, normalize by Nyquist
            jnp.mean(rolloff_squeezed, axis=1) / (sample_rate / 2),
            jnp.std(rolloff_squeezed, axis=1) / (sample_rate / 4),

            # Spectral flatness: 0-1, already normalized
            jnp.mean(flatness_squeezed, axis=1),
            jnp.std(flatness_squeezed, axis=1) * 5.0,

            # MFCCs 0-4: typically -50 to 50, rescale
            jnp.mean(mfcc_vals[:, 0, :], axis=1) / 50.0,
            jnp.mean(mfcc_vals[:, 1, :], axis=1) / 50.0,
            jnp.mean(mfcc_vals[:, 2, :], axis=1) / 50.0,
            jnp.mean(mfcc_vals[:, 3, :], axis=1) / 50.0,
            jnp.mean(mfcc_vals[:, 4, :], axis=1) / 50.0,

            # Spectral contrast (3 bands): typically 0-40, rescale
            jnp.mean(contrast_vals[:, 0, :], axis=1) / 40.0,
            jnp.mean(contrast_vals[:, 1, :], axis=1) / 40.0,
            jnp.mean(contrast_vals[:, 2, :], axis=1) / 40.0,
        ], axis=1)  # Stack along feature dimension -> [batch, 20]

    return features


def multi_scale_spectral_loss(predicted: jnp.ndarray, target: jnp.ndarray, sample_rate: int = 44100) -> jnp.ndarray:
    """
    Multi-scale spectral loss using mel-spectrograms.

    Computes loss at multiple FFT sizes for better perceptual similarity.

    Args:
        predicted: Predicted audio [batch, channels, samples]
        target: Target audio [batch, channels, samples]
        sample_rate: Sample rate in Hz

    Returns:
        Combined spectral loss
    """
    fft_sizes = [2048, 1024, 512]
    losses = [spectral_distance(predicted, target, sample_rate, fft_size) for fft_size in fft_sizes]
    return jnp.mean(jnp.array(losses))


def _flatten_params(tree, prefix=""):
    """Flatten a nested pure-dict of arrays into dot-separated keys."""
    out = {}
    for k, v in tree.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        else:
            out[key] = np.asarray(v)
    return out


def _unflatten_params(flat):
    """Rebuild a nested dict from dot-separated keys (inverse of _flatten_params).

    All-digit path components are converted back to ``int`` so that list/Sequential
    indices in the nnx state (e.g. ``backbone.layers.0.kernel``) round-trip — nnx
    uses integer keys for sequences, and module attribute names are never all-digit.
    """
    def _key(part):
        return int(part) if part.isdigit() else part

    out = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = out
        for part in parts[:-1]:
            d = d.setdefault(_key(part), {})
        d[_key(parts[-1])] = value
    return out


def save_nnx_params(module: nnx.Module, path) -> None:
    """Save an nnx.Module's learnable parameters to a portable .safetensors file.

    Mirrors the ``save_params`` helper that Faust now emits in its NNX
    architecture file, so the RL demo can checkpoint the policy without Orbax.
    0-dim scalars (which safetensors cannot store) are reshaped to (1,) and their
    keys recorded under the ``__zero_dim__`` metadata key for exact restoration.

    Args:
        module: The nnx.Module whose ``nnx.Param`` leaves are saved.
        path: Destination ``.safetensors`` file path.
    """
    from safetensors.numpy import save_file

    pure = nnx.to_pure_dict(nnx.state(module, nnx.Param))
    flat = _flatten_params(pure)
    zero_dim = []
    out = {}
    for k, v in flat.items():
        arr = np.asarray(v)
        if arr.ndim == 0:
            # np.ascontiguousarray promotes 0-dim to (1,), so detect/reshape first.
            zero_dim.append(k)
            arr = arr.reshape(1)
        out[k] = np.ascontiguousarray(arr)
    save_file(out, str(path), metadata={"__zero_dim__": json.dumps(zero_dim)})


def load_nnx_params(module: nnx.Module, path) -> None:
    """Load learnable parameters in-place into an nnx.Module from a .safetensors file.

    Args:
        module: The nnx.Module to update in place (same structure as when saved).
        path: Source ``.safetensors`` file written by :func:`save_nnx_params`.
    """
    from safetensors import safe_open

    flat = {}
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            flat[key] = f.get_tensor(key)
    zero_dim = json.loads(meta["__zero_dim__"]) if "__zero_dim__" in meta else []
    for key in zero_dim:
        flat[key] = flat[key].reshape(())
    nnx.update(module, _unflatten_params(flat))
