"""Generate normalization statistics for audio features.

This script generates random synthesizer sounds and computes feature statistics
(mean, std, min, max) for proper normalization in the RL training pipeline.
"""

import jax
from jax import numpy as jnp, random
from flax import nnx
import json
from pathlib import Path
from utils import compile_synth, extract_features


def generate_feature_stats(num_samples: int = 1000):
    """Generate random synth sounds and compute feature statistics.

    Args:
        num_samples: Number of random audio samples to generate
    """
    print(f"\nGenerating feature statistics from {num_samples} random synth sounds...")

    # Compile synthesizer
    synth_class = compile_synth()
    if synth_class is None:
        print("Failed to compile synthesizer")
        return

    sample_rate = 44100
    duration = 0.3

    # Initialize synthesizer
    rngs = nnx.Rngs(0, params=0, rng_stream=0, gumbel=42)
    synth = synth_class(sample_rate=sample_rate, faust_float=jnp.float32, rngs=rngs)
    synth.eval()

    # Get parameter info
    continuous_names = list(synth.get_continuous_params().keys())
    categorical_info = {
        name: len(info['logits'])
        for name, info in synth.get_categorical_params().items()
    }

    # C3 note input
    num_samples_audio = int(sample_rate * duration)
    c3_freq = 130.81
    c3_input = jnp.full((synth.num_inputs, num_samples_audio), c3_freq)

    print(f"Synthesizer: {synth.__class__.__name__}")
    print(f"Continuous parameters: {continuous_names}")
    if categorical_info:
        print(f"Categorical parameters: {list(categorical_info.keys())}")

    # Generate audio samples and extract features
    all_features = []

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")

        # Generate random parameters
        rng = random.key(i)
        key_cont, key_cat, key_synth = random.split(rng, 3)

        random_continuous = {}
        keys_params = random.split(key_cont, len(continuous_names))
        for name, k in zip(continuous_names, keys_params):
            random_continuous[name] = random.uniform(k)

        random_categorical = {}
        if categorical_info:
            for name, num_opts in categorical_info.items():
                random_categorical[name] = random.randint(key_cat, (), 0, num_opts).astype(jnp.float32)

        params = {**random_continuous, **random_categorical}

        # Render audio
        with jax.default_device(jax.devices('cpu')[0]):
            audio = synth(c3_input, normalized_params=params, rngs=key_synth)

        # Extract features (without stats normalization)
        # Add batch dimension for extract_features
        audio_batch = audio[None, ...]  # [1, channels, samples]
        features = extract_features(audio_batch, sample_rate=sample_rate,
                                   n_fft=2048, hop_length=512,
                                   use_stats_normalization=False)
        all_features.append(features[0])  # Remove batch dimension

    # Stack all features and compute statistics
    stacked = jnp.stack(all_features, axis=0)  # [num_samples, n_features]

    stats = {
        'mean': jnp.mean(stacked, axis=0).tolist(),
        'std': jnp.std(stacked, axis=0).tolist(),
        'min': jnp.min(stacked, axis=0).tolist(),
        'max': jnp.max(stacked, axis=0).tolist(),
    }

    # Save to JSON
    output_path = Path(__file__).parent / 'feature_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Saved statistics for {len(stats['mean'])} features to: {output_path}")
    print(f"\nFeature statistics:")
    print(f"  Mean range: [{min(stats['mean']):.3f}, {max(stats['mean']):.3f}]")
    print(f"  Std range:  [{min(stats['std']):.3f}, {max(stats['std']):.3f}]")
    print(f"  Min range:  [{min(stats['min']):.3f}, {max(stats['min']):.3f}]")
    print(f"  Max range:  [{min(stats['max']):.3f}, {max(stats['max']):.3f}]")


if __name__ == "__main__":
    generate_feature_stats(num_samples=1000)
