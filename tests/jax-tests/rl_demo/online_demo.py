"""
Online Processing Demo: Streaming audio with dynamic parameter updates.

This demonstrates:
1. Block-by-block processing using process_block (stateful carry)
2. Updating parameters between blocks (real-time parameter modulation)
3. Policy adapting over time based on recent audio
4. Streaming use case (like VST plugin or real-time effects)

Key difference from reinforce_demo.py:
- Uses process_block() instead of __call__()
- Maintains carry state across blocks
- Parameters can change between blocks
- Models real-time/streaming scenarios

Run: python3 online_demo.py

By default the policy is randomly initialized (it demonstrates the streaming
API, not learned behaviour). To stream with a policy trained by
reinforce_demo.py, pass its checkpoint:

    python3 online_demo.py --restore-checkpoint checkpoints_stage2/best.safetensors
"""

import argparse

import jax
from jax import numpy as jnp, random
from flax import nnx
from pathlib import Path
import matplotlib.pyplot as plt
from utils import compile_synth, extract_features as extract_window_features, load_nnx_params
from librosax.feature import rms, spectral_centroid, zero_crossing_rate, spectral_rolloff


class OnlinePolicy(nnx.Module):
    """
    Online policy that adapts parameters based on recent audio features.

    Extracts features (RMS, spectral centroid, ZCR, rolloff) from recent audio
    and predicts adapted synthesizer parameters for the next block.

    Like an adaptive effects processor or intelligent audio plugin.
    """

    def __init__(self, feature_dim: int, continuous_names: list, categorical_info: dict, rngs: nnx.Rngs):
        self.continuous_names = continuous_names
        self.categorical_info = categorical_info
        self.num_continuous = len(continuous_names)

        # Feature extractor (processes audio features → parameter adjustments)
        self.dense1 = nnx.Linear(feature_dim, 32, rngs=rngs)
        self.dense2 = nnx.Linear(32, 16, rngs=rngs)

        # Continuous head (outputs mode directly - no sampling for online use)
        self.continuous_head = nnx.Linear(16, self.num_continuous, rngs=rngs)

        # Categorical heads
        for name, num_options in categorical_info.items():
            setattr(self, f'categorical_head_{name}', nnx.Linear(16, num_options, rngs=rngs))

    def __call__(self, audio_features: jnp.ndarray):
        """
        Predict parameters based on audio features.

        Args:
            audio_features: Features extracted from recent audio [batch, feature_dim]

        Returns:
            params_dict: {param_name: values} (normalized [0,1] for continuous)
        """
        x = nnx.relu(self.dense1(audio_features))
        x = nnx.relu(self.dense2(x))

        # Continuous: directly predict mode (deterministic for online)
        continuous_out = self.continuous_head(x)
        continuous_params = nnx.sigmoid(continuous_out)  # [0, 1]

        # Build params dict
        params_dict = {}
        for i, name in enumerate(self.continuous_names):
            params_dict[name] = continuous_params[:, i]

        # Categorical: argmax (deterministic for online)
        for name in self.categorical_info.keys():
            head = getattr(self, f'categorical_head_{name}')
            logits = head(x)
            params_dict[name] = jnp.argmax(logits, axis=-1).astype(jnp.float32)

        return params_dict


def extract_features(audio_block: jnp.ndarray, sample_rate: int = 44100) -> jnp.ndarray:
    """
    Extract audio features from block for policy input.

    Uses librosax for standard audio features with short FFT for small blocks.

    Args:
        audio_block: Audio [channels, samples]
        sample_rate: Sample rate

    Returns:
        Feature vector [4]: [rms, spectral_centroid, zcr, spectral_rolloff]
    """
    # Use short FFT for small blocks (512 samples)
    n_fft = min(512, audio_block.shape[-1])

    # RMS: Energy of the signal
    rms_val = rms(y=audio_block, frame_length=n_fft, hop_length=n_fft)
    rms_val = jnp.mean(rms_val)  # Scalar

    # Spectral centroid: Brightness (center of mass of spectrum)
    spec_centroid = spectral_centroid(y=audio_block, sr=sample_rate, n_fft=n_fft, hop_length=n_fft)
    spec_centroid = jnp.mean(spec_centroid) / 1000.0  # Normalize to kHz

    # Zero crossing rate: Noisiness/harmonicity
    zcr = zero_crossing_rate(audio_block, frame_length=n_fft, hop_length=n_fft)
    zcr = jnp.mean(zcr)  # Scalar

    # Spectral rolloff: High frequency content
    rolloff = spectral_rolloff(y=audio_block, sr=sample_rate, n_fft=n_fft, hop_length=n_fft)
    rolloff = jnp.mean(rolloff) / 1000.0  # Normalize to kHz

    return jnp.array([rms_val, spec_centroid, zcr, rolloff])


def online_demo(args):
    """Demonstrate online/streaming processing with adaptive parameters."""
    print("\n" + "="*70)
    print("Online Demo: Streaming Processing with Adaptive Parameters")
    print("="*70)

    # Compile synthesizer
    synth_class = compile_synth()
    if synth_class is None:
        return

    sample_rate = 44100
    block_size = 512  # Process in small blocks (like real-time audio)
    num_blocks = 100  # Total blocks to process

    # Initialize synthesizer
    rngs = nnx.Rngs(0, params=0, rng_stream=0, nentry=42)
    synth = synth_class(sample_rate=sample_rate, faust_float=jnp.float32, rngs=rngs)
    # Use eval mode (deterministic) since online policy provides discrete values
    synth.eval()

    # Discover parameters
    continuous_names = list(synth.get_continuous_params().keys())
    categorical_info = {
        name: len(info['logits'])
        for name, info in synth.get_categorical_params().items()
    }

    print(f"\nSynthesizer: {synth.__class__.__name__}")
    print(f"Continuous parameters: {continuous_names}")
    print(f"Categorical parameters: {list(categorical_info.keys())}")
    print(f"\nProcessing:")
    print(f"  Block size: {block_size} samples")
    print(f"  Total blocks: {num_blocks}")
    print(f"  Duration: {num_blocks * block_size / sample_rate:.2f} seconds")

    # Initialize policy.
    # Trained mode: reuse the actor trained by reinforce_demo.py. It expects the
    # same 20-dim normalized features it saw in training, computed over a rolling
    # window matching the training clip length (0.3 s).
    use_trained = args.restore_checkpoint is not None
    if use_trained:
        from reinforce_demo import PolicyWithBaseline

        feature_dim = 20
        policy = PolicyWithBaseline(feature_dim, continuous_names, categorical_info,
                                    rngs=nnx.Rngs(42))
        ckpt = Path(args.restore_checkpoint).resolve()
        load_nnx_params(policy, ckpt)
        policy.eval()  # deterministic: Beta mode / argmax
        window_samples = int(0.3 * sample_rate)
        audio_window = jnp.zeros((1, synth.num_outputs, window_samples))
        print(f"\nLoaded trained policy from: {ckpt}")
    else:
        feature_dim = 4  # RMS, spectral centroid, ZCR, rolloff
        policy = OnlinePolicy(feature_dim, continuous_names, categorical_info, rngs=nnx.Rngs(42))
        print("\nNo --restore-checkpoint given: using a randomly initialized policy.")
        print("(This demonstrates the streaming API; train one with reinforce_demo.py.)")

    print(f"\nOnline Policy:")
    if use_trained:
        print(f"  Input: Audio features [{feature_dim}] over a rolling 0.3 s window "
              f"(same features as reinforce_demo.py training)")
    else:
        print(f"  Input: Audio features [{feature_dim}] (RMS, spectral centroid, ZCR, rolloff)")
    print(f"  Output: Adapted parameters for next block")
    print(f"  Updates: Every block (real-time adaptation)")

    # Initialize carry state
    carry = synth.initialize_carry()

    # Create sweeping frequency input (changes over time)
    total_samples = num_blocks * block_size
    freq_sweep = jnp.linspace(220, 880, total_samples)

    # Split the (inference-only) policy once, then run a plain jax.jit step that
    # merges inside — the outer-jax.jit convention. policy_state is constant across
    # blocks (the policy is never updated); `carry` is the real loop-carried state.
    policy_graphdef, policy_state = nnx.split(policy)

    @jax.jit
    def process_one_block(policy_state, carry_in, input_block, features, rng):
        """Process one block with policy-predicted parameters."""
        policy_model = nnx.merge(policy_graphdef, policy_state)

        # Policy predicts parameters based on features. The trained actor
        # (PolicyWithBaseline, in eval mode) returns (actions, log_prob, entropy);
        # the simple OnlinePolicy returns the params dict directly.
        if use_trained:
            params_normalized_batch, _, _ = policy_model(features[None, :], None)
        else:
            params_normalized_batch = policy_model(features[None, :])

        # Extract scalars for process_block
        params_normalized = {name: value[0] for name, value in params_normalized_batch.items()}

        # Process block (synth is a frozen closure: its params are constants here)
        audio_block, carry_out = synth.process_block(
            carry_in,
            input_block,
            unroll=1,
            normalized_params=params_normalized,
            rngs=rng
        )

        return carry_out, audio_block, params_normalized

    # Storage for results
    all_audio_blocks = []
    all_params_physical = []

    # Process blocks sequentially (like real-time streaming)
    print(f"\nProcessing blocks...")

    for block_idx in range(num_blocks):
        # Get input for this block
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        freq_block = freq_sweep[start_idx:end_idx]
        input_block = freq_block.reshape(synth.num_inputs, -1)

        # Extract features from previous output (or zeros for first block)
        if use_trained:
            # Rolling 0.3 s window of recent output, analyzed with the same
            # 20-dim normalized features the policy was trained on.
            if block_idx > 0:
                audio_window = jnp.concatenate(
                    [audio_window[..., block_size:], all_audio_blocks[-1][None]], axis=-1)
            features = extract_window_features(
                audio_window, sample_rate=sample_rate, use_stats_normalization=True)[0]
        elif block_idx == 0:
            features = jnp.zeros(feature_dim)
        else:
            features = extract_features(all_audio_blocks[-1], sample_rate)

        # Process this block (JIT-compiled)
        rng = random.key(block_idx)
        carry, audio_block, params_normalized = process_one_block(
            policy_state, carry, input_block, features, rng)

        # Store results
        all_audio_blocks.append(audio_block)

        # Unnormalize params for tracking
        params_norm_for_unnorm = {name: jnp.array([value]) for name, value in params_normalized.items()}
        params_physical = synth.unnormalize_params(params_norm_for_unnorm)
        all_params_physical.append(params_physical)

        if block_idx % 20 == 0:
            cutoff = float(params_physical[continuous_names[0]][0])
            gain = float(params_physical[continuous_names[1]][0])
            print(f"  Block {block_idx:3d}: cutoff={cutoff:.0f} Hz, gain={gain:.2f}")

    print(f"\n✓ Processed {num_blocks} blocks")

    # Concatenate all blocks
    full_audio = jnp.concatenate(all_audio_blocks, axis=-1)
    print(f"Full audio shape: {full_audio.shape}")

    # Plot parameter evolution over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot cutoff over time
    cutoff_history = [float(params[continuous_names[0]][0]) for params in all_params_physical]
    ax1.plot(cutoff_history, linewidth=2)
    ax1.set_ylabel('Cutoff Frequency (Hz)')
    ax1.set_title('Parameter Evolution Over Time (Online Processing)')
    ax1.grid(True, alpha=0.3)

    # Plot gain over time
    gain_history = [float(params[continuous_names[1]][0]) for params in all_params_physical]
    ax2.plot(gain_history, linewidth=2, color='orange')
    ax2.set_xlabel('Block Number')
    ax2.set_ylabel('Gain')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(__file__).parent / "online_params.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved parameter evolution plot to: {plot_path}")

    print("\n" + "="*70)
    print("ONLINE DEMO COMPLETE!")
    print("="*70)
    print()
    print("Key Concepts:")
    print("  1. process_block() maintains carry state across blocks")
    print("  2. Parameters can change between blocks (real-time adaptation)")
    print("  3. Policy adjusts params based on audio features (feedback loop)")
    print("  4. Models streaming/real-time scenarios (VST, effects, live audio)")
    print()
    print("Differences from batch processing:")
    print("  - Sequential: blocks processed one at a time (not parallel)")
    print("  - Stateful: carry persists across blocks")
    print("  - Adaptive: parameters update based on recent audio")
    print("  - Real-time: like audio plugin or live processing")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming demo: process_block with per-block parameter updates")
    parser.add_argument(
        "--restore-checkpoint", type=str, default=None,
        help="Load a trained policy (.safetensors from reinforce_demo.py, e.g. "
             "checkpoints_stage2/best.safetensors). Without it, a randomly "
             "initialized policy demonstrates the streaming API.")
    online_demo(parser.parse_args())
