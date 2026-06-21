"""Analyze learned weights from trained REINFORCE policy.

This script loads a checkpoint and analyzes:
- Weight distributions and statistics
- Feature importance (which audio features matter most)
- Parameter prediction patterns
- Concentration dynamics
"""

import jax
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import compile_synth
import argparse


def analyze_checkpoint(checkpoint_path: str):
    """Analyze weights from a saved checkpoint."""
    print("\n" + "=" * 70)
    print("REINFORCE Policy Weight Analysis")
    print("=" * 70)

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Compile synth to get parameter info
    synth_class = compile_synth()
    if synth_class is None:
        return

    sample_rate = 44100
    synth = synth_class(sample_rate=sample_rate, faust_float=jnp.float32, rngs=nnx.Rngs(0))
    synth.eval()

    continuous_names = list(synth.get_continuous_params().keys())
    categorical_info = {
        name: len(info['logits'])
        for name, info in synth.get_categorical_params().items()
    }

    obs_dim = 20

    # Create abstract model and restore
    from reinforce_demo import PolicyWithBaseline
    default_params_normalized = {name: 0.5 for name in continuous_names}

    abstract_policy = nnx.eval_shape(lambda: PolicyWithBaseline(
        obs_dim, continuous_names, categorical_info, rngs=nnx.Rngs(0),
        concentration_base=2.0, concentration_scale=5.0,
        default_mode_values=default_params_normalized
    ))
    graphdef, abstract_state = nnx.split(abstract_policy)

    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(checkpoint_path, abstract_state)
    policy = nnx.merge(graphdef, state_restored)

    print("✓ Checkpoint loaded successfully")

    # Extract weights
    print("\n" + "=" * 70)
    print("Network Architecture")
    print("=" * 70)

    print(f"\nBackbone:")
    print(f"  Layer 1: {policy.backbone.layers[0].kernel.value.shape} (input → 256)")
    print(f"  Layer 2: {policy.backbone.layers[3].kernel.value.shape} (256 → 128)")

    print(f"\nActor heads:")
    print(f"  Continuous: {policy.head.actor_continuous.kernel.value.shape} (128 → {len(continuous_names) * 2})")
    for name in categorical_info.keys():
        head = getattr(policy.head, f'actor_categorical_{name}')
        print(f"  Categorical ({name}): {head.kernel.value.shape} (128 → {categorical_info[name]})")

    print(f"\nCritic head:")
    print(f"  Value: {policy.critic.kernel.value.shape} (128 → 1)")

    # Analyze backbone weights
    print("\n" + "=" * 70)
    print("Backbone Weight Statistics")
    print("=" * 70)

    layer1_weights = policy.backbone.layers[0].kernel.value
    layer2_weights = policy.backbone.layers[3].kernel.value

    print(f"\nLayer 1 (input → 256):")
    print(f"  Mean: {float(jnp.mean(layer1_weights)):.4f}")
    print(f"  Std:  {float(jnp.std(layer1_weights)):.4f}")
    print(f"  Min:  {float(jnp.min(layer1_weights)):.4f}")
    print(f"  Max:  {float(jnp.max(layer1_weights)):.4f}")

    print(f"\nLayer 2 (256 → 128):")
    print(f"  Mean: {float(jnp.mean(layer2_weights)):.4f}")
    print(f"  Std:  {float(jnp.std(layer2_weights)):.4f}")
    print(f"  Min:  {float(jnp.min(layer2_weights)):.4f}")
    print(f"  Max:  {float(jnp.max(layer2_weights)):.4f}")

    # Analyze feature importance (L1 norm of input layer weights)
    print("\n" + "=" * 70)
    print("Feature Importance (L1 norm of input weights)")
    print("=" * 70)

    feature_names = [
        'RMS_mean', 'RMS_std',
        'ZCR_mean', 'ZCR_std',
        'Centroid_mean', 'Centroid_std',
        'Bandwidth_mean', 'Bandwidth_std',
        'Rolloff_mean', 'Rolloff_std',
        'Flatness_mean', 'Flatness_std',
        'MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4',
        'Contrast_0', 'Contrast_1', 'Contrast_2',
    ]

    # Compute L1 norm for each input feature (sum across all output neurons)
    feature_importance = jnp.sum(jnp.abs(layer1_weights), axis=1)  # [20]

    # Sort by importance
    sorted_indices = jnp.argsort(feature_importance)[::-1]

    print("\nTop 10 most important features:")
    for i, idx in enumerate(sorted_indices[:10]):
        idx_int = int(idx)
        importance = float(feature_importance[idx])
        feature_name = feature_names[idx_int]
        print(f"  {i+1}. {feature_name:20s}: {importance:.4f}")

    print("\nBottom 5 least important features:")
    for i, idx in enumerate(sorted_indices[-5:]):
        idx_int = int(idx)
        importance = float(feature_importance[idx])
        feature_name = feature_names[idx_int]
        print(f"  {i+1}. {feature_name:20s}: {importance:.4f}")

    # Analyze actor continuous head
    print("\n" + "=" * 70)
    print("Continuous Actor Head Analysis")
    print("=" * 70)

    actor_cont_weights = policy.head.actor_continuous.kernel.value
    actor_cont_bias = policy.head.actor_continuous.bias.value

    num_continuous = len(continuous_names)

    print(f"\nMode outputs (first {num_continuous} neurons):")
    for i, name in enumerate(continuous_names):
        bias_val = float(actor_cont_bias[i])
        weight_norm = float(jnp.linalg.norm(actor_cont_weights[:, i]))
        print(f"  {name}: bias={bias_val:.4f}, weight_norm={weight_norm:.4f}")

    print(f"\nConcentration outputs (last {num_continuous} neurons):")
    for i, name in enumerate(continuous_names):
        idx = num_continuous + i
        bias_val = float(actor_cont_bias[idx])
        weight_norm = float(jnp.linalg.norm(actor_cont_weights[:, idx]))
        # After softplus and scaling: softplus(x) * 5.0 + 2.0
        # With bias = -2.0 initially, what did it learn?
        print(f"  {name}: bias={bias_val:.4f}, weight_norm={weight_norm:.4f}")

    # Test concentration range
    print("\n" + "=" * 70)
    print("Learned Concentration Behavior")
    print("=" * 70)

    # Generate test observations
    test_obs = jax.random.normal(jax.random.key(42), (100, obs_dim))
    mode, concentration, _ = policy.get_distribution_params(test_obs)

    print(f"\nConcentration statistics (over 100 random observations):")
    print(f"  Mean: {float(jnp.mean(concentration)):.4f}")
    print(f"  Std:  {float(jnp.std(concentration)):.4f}")
    print(f"  Min:  {float(jnp.min(concentration)):.4f}")
    print(f"  Max:  {float(jnp.max(concentration)):.4f}")

    print(f"\nMode statistics:")
    print(f"  Mean: {float(jnp.mean(mode)):.4f}")
    print(f"  Std:  {float(jnp.std(mode)):.4f}")
    print(f"  Min:  {float(jnp.min(mode)):.4f}")
    print(f"  Max:  {float(jnp.max(mode)):.4f}")

    # Visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    # Figure 1: Feature importance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Feature importance
    ax = axes[0, 0]
    feature_imp_sorted = feature_importance[sorted_indices]
    feature_names_sorted = [feature_names[int(i)] for i in sorted_indices]
    ax.barh(range(20), feature_imp_sorted[::-1])
    ax.set_yticks(range(20))
    ax.set_yticklabels(feature_names_sorted[::-1], fontsize=8)
    ax.set_xlabel('L1 Norm')
    ax.set_title('Feature Importance (Input Layer Weights)')
    ax.grid(True, alpha=0.3)

    # Concentration distribution
    ax = axes[0, 1]
    ax.hist(np.array(concentration).flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Frequency')
    ax.set_title('Learned Concentration Distribution')
    ax.grid(True, alpha=0.3)

    # Mode distribution per parameter
    ax = axes[1, 0]
    for i, name in enumerate(continuous_names):
        ax.hist(np.array(mode[:, i]), bins=30, alpha=0.5, label=name)
    ax.set_xlabel('Mode Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Mode Distribution by Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Weight magnitude by layer
    ax = axes[1, 1]
    layer_norms = [
        float(jnp.linalg.norm(layer1_weights)),
        float(jnp.linalg.norm(layer2_weights)),
        float(jnp.linalg.norm(actor_cont_weights)),
        float(jnp.linalg.norm(policy.critic.kernel.value)),
    ]
    layer_names = ['Backbone L1', 'Backbone L2', 'Actor Cont', 'Critic']
    ax.bar(layer_names, layer_norms)
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Weight Magnitude by Layer')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(__file__).parent / 'weight_analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n✓ Saved weight analysis plot to: {plot_path}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze REINFORCE policy weights")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best",
                        help="Path to checkpoint to analyze (default: checkpoints/best)")

    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint)
