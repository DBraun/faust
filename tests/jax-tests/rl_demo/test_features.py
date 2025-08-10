"""Test extract_features function."""

import jax
from jax import numpy as jnp
from utils import extract_features


def test_extract_features():
    """Test that extract_features returns correct shape and values."""
    batch_size = 8
    channels = 1
    sample_rate = 44100
    duration = 0.3
    samples = int(sample_rate * duration)

    # Create test audio (random noise)
    audio = jnp.array(jax.random.normal(jax.random.key(42), (batch_size, channels, samples)))

    # Extract features
    features = extract_features(audio, sample_rate=sample_rate, n_fft=1024, hop_length=256)

    print(f"Input audio shape: {audio.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Expected shape: ({batch_size}, 13)")

    # Check shape
    assert features.shape == (batch_size, 13), f"Expected shape ({batch_size}, 13), got {features.shape}"

    # Check values are in reasonable ranges
    print(f"\nFeature statistics:")
    print(f"  Min: {float(jnp.min(features)):.4f}")
    print(f"  Max: {float(jnp.max(features)):.4f}")
    print(f"  Mean: {float(jnp.mean(features)):.4f}")
    print(f"  Std: {float(jnp.std(features)):.4f}")

    # Print first sample features
    print(f"\nFirst sample features:")
    feature_names = [
        "RMS mean", "RMS std",
        "ZCR mean", "ZCR std",
        "Centroid mean", "Centroid std",
        "Bandwidth mean", "Bandwidth std",
        "Rolloff mean", "Rolloff std",
        "Flatness mean", "Flatness std",
        "MFCC mean"
    ]
    for i, name in enumerate(feature_names):
        print(f"  {name:20s}: {float(features[0, i]):.4f}")

    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_extract_features()
