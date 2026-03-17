"""
REINFORCE Policy Gradient Demo with Baseline

Task: Parameter estimation from audio (inverse synthesis)

Environment:
- Each episode: Black-box synthesizer plays C3 with random unknown parameters
- State: Audio features extracted from the black-box output
- Action: Policy predicts synthesizer parameters
- Reward: -spectral_distance(generated_audio, black_box_audio)

Improvements over vanilla REINFORCE:
- Learned value baseline (critic) for variance reduction
- Entropy bonus to encourage exploration
- Close-match bonus for final convergence push
- Advantage normalization per batch
- Comprehensive metrics tracking

Algorithm: REINFORCE with baseline
Update: ∇_θ J = E[∇_θ log π_θ(a|s) * (R - V(s))]
"""

import jax
from jax import numpy as jnp, random
from jax.lax import stop_gradient
from jax.tree_util import tree_map_with_path, DictKey, GetAttrKey
from flax import nnx, struct
import optax
import distrax
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import argparse
from clu import metric_writers, periodic_actions
from clu.metrics import Collection, Average, LastValue
from flax.training.early_stopping import EarlyStopping
from einops import rearrange
import orbax.checkpoint as ocp
from utils import compile_synth, spectral_distance, batched_spectral_distance, extract_features


def stack_forest(forest):
    """Stack the leaves of a sequence of pytrees.

    Args:
        forest: A sequence of pytrees (e.g., list) of matching structure
            whose leaves are arrays with individually matching shapes.

    Returns:
        A single pytree with the same structure where each leaf is a stacked array.
    """
    def stack_args(*args):
        return jnp.stack(args)

    return jax.tree.map(stack_args, *forest)


def compute_global_grad_norm(grads):
    """Compute the global L2 norm of gradients.

    Args:
        grads: Gradient tree structure containing gradient arrays.

    Returns:
        Global L2 norm of all gradients as a scalar.
    """
    squared_sums = jax.tree_util.tree_map(lambda g: jnp.sum(jnp.square(g)), grads)
    global_norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(squared_sums)))
    return global_norm


def compute_gradient_norm_by_prefix(grads, attr_prefix: str) -> jnp.ndarray:
    """Compute gradient norm for parameters matching an attribute prefix.

    Args:
        grads: Gradient pytree
        attr_prefix: Prefix to match (e.g., "actor_continuous" or "actor_categorical")

    Returns:
        L2 norm of gradients for matching parameters
    """
    matched_sq_norms = []

    def collect_matched_sq_norms(path, grad):
        if grad is None:
            return
        # Check if any part of the path matches the prefix
        for key in path:
            if isinstance(key, GetAttrKey) and key.name.startswith(attr_prefix):
                matched_sq_norms.append(jnp.sum(jnp.square(grad)))
                return

    tree_map_with_path(collect_matched_sq_norms, grads)

    if matched_sq_norms:
        return jnp.sqrt(jnp.sum(jnp.stack(matched_sq_norms)))
    else:
        return jnp.array(0.0)


@jax.jit
def reduce_metrics(metrics: list[Collection]):
    """JIT-compiled metric reduction for efficient aggregation."""
    computed = stack_forest(metrics).reduce().compute()
    return {f"train/{k}": v for k, v in computed.items()}


@struct.dataclass
class TrainMetrics(Collection):
    """Training metrics for REINFORCE."""
    reward: Average.from_output("reward")
    policy_loss: Average.from_output("policy_loss")
    value_loss: Average.from_output("value_loss")
    entropy: Average.from_output("entropy")
    entropy_coeff: LastValue.from_output("entropy_coeff")
    concentration_scale: LastValue.from_output("concentration_scale")
    mean_value: Average.from_output("mean_value")
    mean_advantage: Average.from_output("mean_advantage")
    learning_rate: LastValue.from_output("learning_rate")
    grad_norm_total: Average.from_output("grad_norm_total")
    grad_norm_actor_continuous: Average.from_output("grad_norm_actor_continuous")
    grad_norm_actor_categorical: Average.from_output("grad_norm_actor_categorical")


@struct.dataclass
class EvalMetrics(Collection):
    """Evaluation metrics for parameter estimation quality.

    Note: continuous_mae is computed in normalized space (0-1 range)
    to make it meaningful across parameters with different scales.
    """
    continuous_mae: Average.from_output("continuous_mae")
    categorical_accuracy: Average.from_output("categorical_accuracy")
    spectral_distance: Average.from_output("spectral_distance")
    concentration_mean: Average.from_output("concentration_mean")
    concentration_std: Average.from_output("concentration_std")


@jax.custom_vjp
def magic_clamp(x, min_val, max_val):
    """Clamp with conditional straight-through estimator (STE) gradient.

    Forward: hard clamp to [min_val, max_val]
    Backward: pass gradient if moving toward valid region, else zero
    """
    return jnp.clip(x, min_val, max_val)


def magic_clamp_fwd(x, min_val, max_val):
    return jnp.clip(x, min_val, max_val), (x, min_val, max_val)


def magic_clamp_bwd(res, g):
    x, min_val, max_val = res
    # Pass gradient if:
    # - x is within bounds, OR
    # - x is below min and gradient is positive (pushing up), OR
    # - x is above max and gradient is negative (pushing down)
    in_bounds = (x >= min_val) & (x <= max_val)
    below_min_pushing_up = (x < min_val) & (g > 0)
    above_max_pushing_down = (x > max_val) & (g < 0)
    mask = in_bounds | below_min_pushing_up | above_max_pushing_down
    return (g * mask.astype(g.dtype), None, None)


magic_clamp.defvjp(magic_clamp_fwd, magic_clamp_bwd)


class PolicyWithBaseline(nnx.Module):
    """
    Actor-Critic policy for REINFORCE with learned baseline.

    Actor: Predicts action distributions (Beta for continuous, Categorical for discrete)
    Critic: Predicts state value V(s) for variance reduction
    """

    def __init__(self, in_features: int, continuous_names: list, categorical_info: dict, rngs: nnx.Rngs,
                 concentration_base: float = 2.0, concentration_scale: float = 5.0,
                 default_mode_values: dict = None):
        self.continuous_names = continuous_names
        self.categorical_info = categorical_info
        self.num_continuous = len(continuous_names)
        self.deterministic = False
        self.concentration_base = concentration_base
        self.concentration_scale = concentration_scale
        self.default_mode_values = default_mode_values

        # Shared backbone - larger network performs better (tested: small+dropout was 53% worse)
        self.backbone = nnx.Sequential(
            nnx.Linear(in_features, 256, rngs=rngs),
            nnx.LayerNorm(256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 128, rngs=rngs),
            nnx.LayerNorm(128, rngs=rngs),
            nnx.relu
        )

        # Actor head: continuous params (mode + concentration for Beta)
        # Custom bias initialization for better starting point
        def custom_bias_init(key, shape, dtype=jnp.float32):
            """Initialize bias: mode outputs to logit(defaults), concentration outputs to -2.0."""
            bias = jnp.zeros(shape, dtype=dtype)
            # Mode outputs: initialize to logit of default values (or 0 for sigmoid(0)=0.5)
            if default_mode_values is not None:
                for i, name in enumerate(continuous_names):
                    default_val = default_mode_values.get(name, 0.5)
                    # logit(p) = log(p / (1-p))
                    logit_val = jnp.log(default_val / (1.0 - default_val + 1e-8))
                    bias = bias.at[i].set(logit_val)
            # Concentration outputs: start at -2.0 for low initial concentration
            bias = bias.at[self.num_continuous:].set(-2.0)
            return bias

        self.actor_continuous = nnx.Linear(128, self.num_continuous * 2, rngs=rngs, bias_init=custom_bias_init)

        # Actor head: categorical params
        for name, num_options in categorical_info.items():
            setattr(self, f'actor_categorical_{name}', nnx.Linear(128, num_options, rngs=rngs))

        # Critic head: value function V(s)
        self.critic = nnx.Linear(128, 1, rngs=rngs)

    def get_value(self, observations: jnp.ndarray):
        """Critic: estimate state value V(s)."""
        x = self.backbone(observations)
        return self.critic(x).squeeze(-1)  # [batch]

    def get_distribution_params(self, observations: jnp.ndarray, concentration_scale_override: float = None):
        """Get distribution parameters for actor.

        Args:
            observations: Input observations [batch, obs_dim]
            concentration_scale_override: Optional override for concentration scale (for scheduling)
        """
        x = self.backbone(observations)

        # Continuous: mode-concentration parameterization
        # Higher concentration gives sharper distributions for better convergence
        # Configurable scale and base for concentration tuning
        continuous_out = self.actor_continuous(x)
        mode = nnx.sigmoid(continuous_out[..., :self.num_continuous])
        scale = concentration_scale_override if concentration_scale_override is not None else self.concentration_scale
        concentration = nnx.softplus(continuous_out[..., self.num_continuous:]) * scale + self.concentration_base

        # Categorical: logits
        categorical_logits = {}
        for name in self.categorical_info.keys():
            head = getattr(self, f'actor_categorical_{name}')
            categorical_logits[name] = head(x)

        return mode, concentration, categorical_logits

    def __call__(self, observations: jnp.ndarray, rng: jax.Array = None):
        """
        Sample actions (training) or return mode (eval).

        Returns:
            actions_dict: {param_name: sampled_values}
            log_prob: Total log probability
            entropy: Total entropy (for exploration bonus)
        """
        mode, concentration, categorical_logits = self.get_distribution_params(observations)

        # Convert mode-concentration to alpha-beta
        alpha = mode * (concentration - 2.0) + 1.0
        beta = (1.0 - mode) * (concentration - 2.0) + 1.0

        actions_dict = {}

        B = observations.shape[0]

        if self.deterministic:
            # Eval mode: return mode directly
            for i, name in enumerate(self.continuous_names):
                actions_dict[name] = mode[:, i]

            for name in self.categorical_info.keys():
                logits = categorical_logits[name]
                actions_dict[name] = jnp.argmax(logits, axis=-1).astype(jnp.float32)

            return actions_dict, jnp.zeros(B), jnp.zeros(B)

        else:
            # Training mode: sample and compute log_prob + entropy
            rng_cont, rng_cat = random.split(rng, 2)

            # Continuous: Beta distribution
            beta_dist = distrax.Beta(alpha, beta)
            continuous_samples = beta_dist.sample(seed=rng_cont)
            continuous_log_probs = beta_dist.log_prob(continuous_samples)
            continuous_entropy = beta_dist.entropy()

            total_log_prob = jnp.sum(continuous_log_probs, axis=-1)
            total_entropy = jnp.sum(continuous_entropy, axis=-1)

            for i, name in enumerate(self.continuous_names):
                actions_dict[name] = continuous_samples[:, i]

            # Categorical
            if self.categorical_info:
                keys_cat = random.split(rng_cat, len(self.categorical_info))
                for (name, _), key_c in zip(self.categorical_info.items(), keys_cat):
                    logits = categorical_logits[name]
                    cat_dist = distrax.Categorical(logits=logits)
                    cat_samples = cat_dist.sample(seed=key_c)
                    cat_log_probs = cat_dist.log_prob(cat_samples)
                    cat_entropy = cat_dist.entropy()

                    actions_dict[name] = cat_samples.astype(jnp.float32)
                    total_log_prob = total_log_prob + cat_log_probs
                    total_entropy = total_entropy + cat_entropy

            return actions_dict, total_log_prob, total_entropy


def reinforce_demo(args):
    """REINFORCE with baseline for synthesizer parameter estimation."""
    print("\n" + "=" * 70)
    print("REINFORCE Demo: Policy Gradient with Baseline")
    print("=" * 70)

    # Compile synthesizer
    synth_class = compile_synth()
    if synth_class is None:
        return

    sample_rate = 44100
    duration = 0.3
    batch_size = args.batch_size

    # Initialize synthesizer
    rngs = nnx.Rngs(0, params=0, rng_stream=0, nentry=42)
    synth = synth_class(sample_rate=sample_rate, faust_float=jnp.float32, rngs=rngs)
    synth.eval()

    # Discover parameters
    continuous_names = list(synth.get_continuous_params().keys())
    categorical_info = {
        name: len(info['logits'])
        for name, info in synth.get_categorical_params().items()
    }

    print(f"\nSynthesizer: {synth.__class__.__name__}")
    print(f"Continuous parameters: {continuous_names}")
    if categorical_info:
        print(f"Categorical parameters: {list(categorical_info.keys())}")

    # C3 note
    num_samples = int(sample_rate * duration)
    c3_freq = 130.81
    c3_input = jnp.full((synth.num_inputs, num_samples), c3_freq)

    print(f"\nInput: C3 note ({c3_freq} Hz) for {duration}s")

    def generate_black_box_batch(rng):
        """Generate batch of black-box audio with random parameters."""
        keys = random.split(rng, batch_size)

        def generate_one(key):
            random_continuous = {}
            key_cont, key_cat = random.split(key, 2)
            keys_params = random.split(key_cont, len(continuous_names))

            for name, k in zip(continuous_names, keys_params):
                random_continuous[name] = random.uniform(k)

            random_categorical = {}
            if categorical_info:
                for name, num_opts in categorical_info.items():
                    random_categorical[name] = random.randint(key_cat, (), 0, num_opts).astype(jnp.float32)

            black_box_params = {**random_continuous, **random_categorical}
            # Render audio on CPU for faster synthesis
            with jax.default_device(jax.devices('cpu')[0]):
                audio = synth(c3_input, normalized_params=black_box_params, rngs=key)

            return audio, black_box_params

        # Vmap only audio generation, not feature extraction
        batch_audio, batch_params = jax.vmap(generate_one)(keys)

        # Extract features from entire batch using librosax functions
        batch_features = extract_features(batch_audio, sample_rate=sample_rate, n_fft=2048, hop_length=512,
                                         use_stats_normalization=True)

        return batch_audio, batch_features, batch_params

    # Get synth defaults for smart initialization
    default_params_normalized = {}
    for name in continuous_names:
        # Use synth's default value (normalized to 0-1)
        default_params_normalized[name] = 0.5  # Synth defaults to mid-range

    # Initialize policy with baseline
    obs_dim = 20  # Expanded features: RMS, ZCR, spectral (centroid/bandwidth/rolloff/flatness), 5 MFCCs, 3 spectral contrast bands
    policy = PolicyWithBaseline(obs_dim, continuous_names, categorical_info, rngs=nnx.Rngs(42),
                                concentration_base=args.concentration_base,
                                concentration_scale=args.concentration_scale,
                                default_mode_values=default_params_normalized)

    # Hyperparameters (from command line args)
    num_updates = args.num_updates
    learning_rate = args.learning_rate
    entropy_coeff = args.entropy_coeff
    value_loss_coeff = 0.5
    param_loss_coeff = args.param_loss_coeff
    waveform_bonus_coeff = args.waveform_bonus_coeff

    # Schedules - use warmup + cosine decay for LR
    warmup_steps = min(100, num_updates // 5)  # Adaptive warmup
    if warmup_steps > 0 and num_updates > warmup_steps:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.1,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=num_updates - warmup_steps,
            end_value=learning_rate * 0.01
        )
    else:
        # No warmup for very short runs
        lr_schedule = optax.constant_schedule(learning_rate)

    # Entropy decay schedule (exploration -> exploitation)
    if args.entropy_decay:
        entropy_schedule = optax.linear_schedule(
            init_value=entropy_coeff,
            end_value=entropy_coeff * 0.1,
            transition_steps=num_updates
        )
    else:
        entropy_schedule = optax.constant_schedule(entropy_coeff)

    # Concentration schedule (broader -> sharper distributions)
    if args.concentration_schedule:
        concentration_scale_schedule = optax.linear_schedule(
            init_value=args.concentration_scale,
            end_value=args.concentration_scale * 3.0,  # 3x sharper at end
            transition_steps=num_updates
        )
    else:
        concentration_scale_schedule = optax.constant_schedule(args.concentration_scale)

    # Curriculum schedule using optax.join_schedules for composability
    # This creates a single schedule that handles all three stages:
    # Stage 1: alpha=0 (supervised only)
    # Stage 2: alpha transitions 0→1 (linear blend)
    # Stage 3: alpha=1 (RL only)

    if args.curriculum_stage == 1:
        # Stage 1 only: constant supervised
        rl_weight_schedule = optax.constant_schedule(0.0)
    elif args.curriculum_stage == 2:
        # Stage 2: transition from supervised to RL
        transition_duration = args.stage2_transition_end - args.stage2_transition_start
        rl_weight_schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(0.0),  # Start at 0 (supervised)
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=1.0,
                    transition_steps=transition_duration
                ),
            ],
            boundaries=[args.stage2_transition_start]
        )
    else:  # Stage 3
        # Stage 3: pure RL
        rl_weight_schedule = optax.constant_schedule(1.0)

    optimizer = nnx.Optimizer(
        policy,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)),
        wrt=nnx.Param
    )

    # Setup checkpointing with Orbax
    ckpt_dir = Path(__file__).parent / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    # Restore from checkpoint if requested
    start_update = 0
    if args.restore_checkpoint is not None:
        restore_path = Path(args.restore_checkpoint).resolve()  # Convert to absolute path
        if restore_path.exists():
            print(f"\nRestoring checkpoint from: {restore_path}")
            # Create abstract model for structure
            abstract_policy = nnx.eval_shape(lambda: PolicyWithBaseline(
                obs_dim, continuous_names, categorical_info, rngs=nnx.Rngs(0),
                concentration_base=args.concentration_base,
                concentration_scale=args.concentration_scale,
                default_mode_values=default_params_normalized
            ))
            graphdef, abstract_state = nnx.split(abstract_policy)
            # Restore state
            state_restored = checkpointer.restore(restore_path, abstract_state)
            # Merge into policy
            policy = nnx.merge(graphdef, state_restored)
            # Re-create optimizer with restored policy
            optimizer = nnx.Optimizer(
                policy,
                optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)),
                wrt=nnx.Param
            )
            print("✓ Checkpoint restored successfully")
        else:
            print(f"⚠️  Checkpoint not found: {restore_path}, starting from scratch")

    # Setup TensorBoard logging
    log_dir = Path(__file__).parent / "logs" / f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = metric_writers.create_default_writer(logdir=str(log_dir))

    # Setup progress reporting with timing
    report_every_steps = args.report_every_steps
    log_every_steps = args.log_every_steps
    eval_every_steps = args.eval_every_steps
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=num_updates,
        every_secs=None,
        every_steps=report_every_steps,
        writer=writer,
    )

    print(f"\nTensorBoard logs: {log_dir}")
    print(f"  View with: tensorboard --logdir {log_dir}")
    print(f"\nCheckpoint directory: {ckpt_dir}")
    print(f"  Best model will be saved to: {ckpt_dir / 'best'}")

    print(f"\nTask: Parameter estimation from audio")
    print(f"  - Black-box synth plays C3 with unknown params")
    print(f"  - Policy observes audio features, predicts params")
    print(f"  - Reward = SynthRL-style (SC + log_mae + MFCC)")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {learning_rate} (warmup + cosine decay)")
    print(f"  Entropy coeff: {entropy_coeff} {'(decay)' if args.entropy_decay else '(constant)'}")
    print(f"  Concentration scale: {args.concentration_scale} {'(scheduled 3x)' if args.concentration_schedule else '(constant)'}")
    print(f"  Batch size: {batch_size}")
    print(f"  Updates: {num_updates}")

    # Curriculum stage description
    if args.curriculum_stage == 1:
        stage_desc = "Stage 1: Supervised only (param_loss)"
    elif args.curriculum_stage == 2:
        stage_desc = f"Stage 2: Transition (steps {args.stage2_transition_start}→{args.stage2_transition_end})"
    else:
        stage_desc = "Stage 3: RL only (policy gradient)"
    print(f"  Curriculum: {stage_desc}")

    print(f"\nAdvanced features:")
    print(f"  Supervised param loss coeff: {param_loss_coeff}")
    print(f"  Waveform bonus: {waveform_bonus_coeff}")
    print(f"  Multi-scale spectral loss: {args.use_multiscale_loss}")
    print(f"  Feature count: {obs_dim} (expanded MFCCs + spectral contrast)")

    # JIT-compiled update step
    @nnx.jit
    def train_step(policy_model, opt, rng, step):
        """Single REINFORCE update with baseline."""
        rng_blackbox, rng_policy, rng_synth = random.split(rng, 3)
        black_box_audio, observations, black_box_params = generate_black_box_batch(rng_blackbox)

        # Sample actions
        sampled_params, log_probs, entropy = policy_model(observations, rng_policy)
        sampled_params_sg = jax.tree.map(stop_gradient, sampled_params)

        # Render audio on CPU for faster synthesis
        synth_rngs = random.split(rng_synth, batch_size)
        c3_inputs_batch = jnp.tile(c3_input[None, :, :], (batch_size, 1, 1))

        @jax.vmap
        def render(params, inputs, rng_key):
            with jax.default_device(jax.devices('cpu')[0]):
                return synth(inputs, normalized_params=params, rngs=rng_key)

        generated_audio = render(sampled_params_sg, c3_inputs_batch, synth_rngs)

        # Compute rewards using SynthRL-style multi-component reward
        # Combines spectral convergence, log magnitude error, and MFCC distance
        from utils import compute_synthrl_reward
        base_rewards = compute_synthrl_reward(generated_audio, black_box_audio, sample_rate)

        # Waveform bonus: extra reward for correct categorical prediction
        if categorical_info and waveform_bonus_coeff > 0:
            waveform_correct = jnp.ones(batch_size)
            for name in categorical_info.keys():
                matches = (sampled_params_sg[name] == black_box_params[name]).astype(jnp.float32)
                waveform_correct = waveform_correct * matches
            waveform_bonus = waveform_bonus_coeff * waveform_correct
            rewards = stop_gradient(base_rewards + waveform_bonus)
        else:
            rewards = stop_gradient(base_rewards)

        # Compute loss with baseline
        @nnx.value_and_grad(has_aux=True)
        def grad_loss_fn(model):
            # Get scheduled parameters
            current_entropy_coeff = entropy_schedule(step)
            current_concentration_scale = concentration_scale_schedule(step)
            current_rl_weight = rl_weight_schedule(step)  # Curriculum: 0→1 for param→RL transition

            # Recompute for gradients (with scheduled concentration)
            mode, concentration, cat_logits = model.get_distribution_params(
                observations, concentration_scale_override=current_concentration_scale
            )
            values = model.get_value(observations)

            # Recompute log probs
            alpha = mode * (concentration - 2.0) + 1.0
            beta = (1.0 - mode) * (concentration - 2.0) + 1.0

            continuous_samples = jnp.stack(
                [sampled_params_sg[name] for name in model.continuous_names],
                axis=-1
            )
            beta_dist = distrax.Beta(alpha, beta)
            continuous_log_probs = beta_dist.log_prob(continuous_samples)
            continuous_entropy = beta_dist.entropy()

            total_log_prob = jnp.sum(continuous_log_probs, axis=-1)
            total_entropy = jnp.sum(continuous_entropy, axis=-1)

            for name in model.categorical_info.keys():
                logits = cat_logits[name]
                cat_dist = distrax.Categorical(logits=logits)
                cat_log_probs = cat_dist.log_prob(sampled_params_sg[name].astype(jnp.int32))
                cat_entropy = cat_dist.entropy()
                total_log_prob = total_log_prob + cat_log_probs
                total_entropy = total_entropy + cat_entropy

            # Advantages: reward - baseline (stop gradient before normalization)
            # stop_gradient is necessary because we use supervised param_loss alongside RL
            # Prevents normalization from creating spurious gradients interfering with supervision
            advantages = stop_gradient(rewards - values)
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

            # Policy loss: -E[log π(a|s) * A(s,a)]
            policy_loss = -jnp.mean(total_log_prob * advantages)

            # Value loss: MSE between value predictions and actual rewards
            value_loss = jnp.mean((values - rewards) ** 2)

            # Supervised parameter loss: direct MSE on mode vs target params
            # Weighted by how good the audio match is (only enforce when audio is close)
            if param_loss_coeff > 0:
                # Extract target continuous params
                black_box_continuous = jnp.stack(
                    [black_box_params[name] for name in model.continuous_names],
                    axis=-1
                )
                # MSE between mode and target
                param_mse = jnp.mean((mode - black_box_continuous) ** 2, axis=-1)
                # Weight by audio quality: high reward = low spectral distance = more weight
                # Normalize rewards to [0, 1] range for weighting
                reward_weights = nnx.sigmoid(rewards)  # High reward -> weight ~1, low reward -> weight ~0
                weighted_param_loss = jnp.mean(param_mse * reward_weights)
                param_loss = param_loss_coeff * weighted_param_loss
            else:
                param_loss = 0.0
                weighted_param_loss = 0.0

            # Entropy bonus (scheduled: high initially for exploration, low later for exploitation)
            entropy_loss = -current_entropy_coeff * jnp.mean(total_entropy)

            # Three-stage curriculum learning (SynthRL-style)
            # RL components: policy_loss, value_loss, entropy_loss
            # Supervised component: param_loss
            # Blend with: total = alpha * RL + (1 - alpha) * supervised
            rl_loss = policy_loss + value_loss_coeff * value_loss + entropy_loss
            supervised_loss = param_loss if param_loss_coeff > 0 else 0.0

            total_loss = current_rl_weight * rl_loss + (1.0 - current_rl_weight) * supervised_loss

            return total_loss, {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy': jnp.mean(total_entropy),
                'entropy_coeff': current_entropy_coeff,
                'concentration_scale': current_concentration_scale,
                'rl_weight': current_rl_weight,
                'mean_value': jnp.mean(values),
                'mean_advantage': jnp.mean(advantages),
            }

        (total_loss, metrics), grads = grad_loss_fn(policy_model)

        # Compute gradient norms before optimizer clips them
        grad_norm_total = compute_global_grad_norm(grads)
        grad_norm_actor_continuous = compute_gradient_norm_by_prefix(grads, "actor_continuous")
        grad_norm_actor_categorical = compute_gradient_norm_by_prefix(grads, "actor_categorical")

        opt.update(model=policy_model, grads=grads)

        # Add learning rate, reward, and gradient norms to metrics
        current_lr = lr_schedule(step)
        metrics['learning_rate'] = current_lr
        metrics['reward'] = jnp.mean(rewards)
        metrics['grad_norm_total'] = grad_norm_total
        metrics['grad_norm_actor_continuous'] = grad_norm_actor_continuous
        metrics['grad_norm_actor_categorical'] = grad_norm_actor_categorical

        # Convert to TrainMetrics Collection
        train_metrics = TrainMetrics.single_from_model_output(**metrics)
        return train_metrics

    # JIT-compiled evaluation step
    @nnx.jit
    def eval_step(policy_model, observations, target_params, target_audio, rng):
        """Single evaluation step to compute parameter estimation metrics.

        Args:
            policy_model: The policy network
            observations: Batch of audio features [batch, obs_dim]
            target_params: Target parameters (normalized) dict with [batch] arrays
            target_audio: Target audio [batch, channels, samples]
            rng: Random key for synthesis

        Returns:
            Dictionary with eval metrics (continuous_mae, categorical_accuracy, spectral_distance)
        """
        # Get predictions (deterministic mode) and distribution params
        predicted_params, _, _ = policy_model(observations, rng=None)
        mode, concentration, _ = policy_model.get_distribution_params(observations)

        # Render audio for predicted params
        batch_size_eval = observations.shape[0]
        synth_rngs = random.split(rng, batch_size_eval)
        c3_inputs_batch = jnp.tile(c3_input[None, :, :], (batch_size_eval, 1, 1))

        @jax.vmap
        def render(params, inputs, rng_key):
            with jax.default_device(jax.devices('cpu')[0]):
                return synth(inputs, normalized_params=params, rngs=rng_key)

        generated_audio = render(predicted_params, c3_inputs_batch, synth_rngs)

        # Compute spectral distances
        distances = batched_spectral_distance(generated_audio, target_audio, sample_rate)
        mean_spectral_dist = jnp.mean(distances)

        # Compute parameter metrics IN NORMALIZED SPACE (0-1 range)
        # This makes MAE meaningful across different parameter scales

        # Continuous MAE (in normalized space)
        if continuous_names:
            maes = []
            for name in continuous_names:
                pred = predicted_params[name]  # Already normalized
                act = target_params[name]      # Already normalized
                mae = jnp.mean(jnp.abs(pred - act))
                maes.append(mae)
            continuous_mae = jnp.mean(jnp.array(maes))
        else:
            continuous_mae = jnp.array(0.0)

        # Categorical accuracy (no unnormalization needed - already indices)
        if categorical_info:
            accuracies = []
            for name in categorical_info.keys():
                # Predicted params in eval mode are already argmax indices (0, 1, 2, 3 as floats)
                # Target params are also generated as indices
                pred = predicted_params[name].astype(jnp.int32)
                act = target_params[name].astype(jnp.int32)
                acc = jnp.mean(pred == act)
                accuracies.append(acc)
            categorical_accuracy = jnp.mean(jnp.array(accuracies))
        else:
            categorical_accuracy = jnp.array(1.0)

        # Concentration statistics (sharpness of distributions)
        concentration_mean = jnp.mean(concentration)
        concentration_std = jnp.std(concentration)

        return {
            'continuous_mae': continuous_mae,
            'categorical_accuracy': categorical_accuracy,
            'spectral_distance': mean_spectral_dist,
            'concentration_mean': concentration_mean,
            'concentration_std': concentration_std,
            'generated_audio': generated_audio,
            'target_audio': target_audio,
        }

    # Print model architecture if requested
    if args.print_model:
        print("\n" + "=" * 70)
        print("Model Architecture")
        print("=" * 70)
        test_obs = random.normal(random.key(8888), (1, obs_dim))
        policy.eval()
        print(nnx.tabulate(policy, test_obs, depth=3))
        policy.train()

    # Training loop with metrics tracking
    print(f"\nTraining...")
    loss_history = []
    reward_history = []
    entropy_history = []

    # Initial mode
    test_obs = random.normal(random.key(8888), (1, obs_dim))
    mode_init, _, _ = policy.get_distribution_params(test_obs)
    print(f"Initial mode: {mode_init[0]}")

    train_metrics_all = []

    # Early stopping setup
    best_mae = float('inf')
    if args.early_stopping_patience > 0:
        early_stop = EarlyStopping(min_delta=args.early_stopping_min_delta, patience=args.early_stopping_patience)
        print(f"\nEarly stopping enabled: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
    else:
        early_stop = None

    with metric_writers.ensure_flushes(writer):
        for update in range(num_updates):
            # Generate data
            with report_progress.timed("data"):
                rng = random.key(update)

            # Train step
            with report_progress.timed("train"):
                train_metrics = train_step(policy, optimizer, rng, jnp.array(update))

            # Accumulate metrics for TensorBoard
            train_metrics_all.append(train_metrics)

            # Logging phase
            with report_progress.timed("logging"):
                if update % log_every_steps == 0 or update == num_updates - 1:
                    # Reduce accumulated metrics using JIT-compiled reduction
                    if train_metrics_all:
                        summary = reduce_metrics(train_metrics_all)
                        writer.write_scalars(update, summary)
                        train_metrics_all = []  # Reset

            # Progress hooks
            with report_progress.timed("hooks"):
                report_progress(update)

            # Periodic evaluation
            if eval_every_steps > 0 and (update % eval_every_steps == 0 or update == num_updates - 1):
                with report_progress.timed("eval"):
                    # Generate eval batch
                    eval_audio, eval_features, eval_params = generate_black_box_batch(random.key(10000 + update))
                    num_eval = 20

                    # Run eval step
                    policy.eval()
                    eval_params_slice = {k: v[:num_eval] for k, v in eval_params.items()}
                    eval_metrics_dict = eval_step(
                        policy,
                        eval_features[:num_eval],
                        eval_params_slice,
                        eval_audio[:num_eval],
                        random.key(20000 + update),
                    )
                    policy.train()

                    # Extract audio for logging (not part of metrics)
                    generated_audio = eval_metrics_dict.pop('generated_audio')
                    target_audio = eval_metrics_dict.pop('target_audio')

                    # Convert to Collection and log to TensorBoard
                    eval_metrics_result = EvalMetrics.single_from_model_output(**eval_metrics_dict)
                    eval_computed = eval_metrics_result.compute()
                    eval_summary = {f"eval/{k}": float(v) for k, v in eval_computed.items()}
                    writer.write_scalars(update, eval_summary)

                    # Log audio samples (first 3 examples)
                    num_audio_samples = min(3, generated_audio.shape[0])
                    writer.write_audios(
                        step=update,
                        audios={
                            'generated': rearrange(generated_audio[:num_audio_samples], "b c t -> b t c"),
                            'target': rearrange(target_audio[:num_audio_samples], "b c t -> b t c"),
                        },
                        sample_rate=sample_rate,
                    )

                    # Checkpoint management and early stopping
                    current_mae = float(eval_computed['continuous_mae'])

                    # Save best checkpoint
                    if current_mae < best_mae:
                        best_mae = current_mae
                        _, state = nnx.split(policy)
                        best_ckpt_path = ckpt_dir / 'best'
                        checkpointer.save(best_ckpt_path, state, force=True)
                        print(f"  ✓ Saved best checkpoint (MAE={best_mae:.4f}) to: {best_ckpt_path}")

                    # Save periodic checkpoint
                    if args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
                        _, state = nnx.split(policy)
                        periodic_ckpt_path = ckpt_dir / f'checkpoint_{update:06d}'
                        checkpointer.save(periodic_ckpt_path, state, force=True)
                        print(f"  ✓ Saved periodic checkpoint to: {periodic_ckpt_path}")

                    # Early stopping check
                    if early_stop is not None:
                        early_stop = early_stop.update(current_mae)
                        if early_stop.should_stop:
                            print(f"\n⚠️  Early stopping triggered at update {update}")
                            print(f"    Best MAE: {early_stop.best_metric:.4f}, Current MAE: {current_mae:.4f}")
                            print(f"    No improvement for {args.early_stopping_patience} evaluation checks")
                            break

    # Final mode
    mode_final, _, _ = policy.get_distribution_params(test_obs)
    print(f"\nFinal mode: {mode_final[0]}")

    # Checkpoint summary
    if best_mae < float('inf'):
        print(f"\n📦 Best checkpoint saved with MAE: {best_mae:.4f}")
        print(f"   Location: {ckpt_dir / 'best'}")
        print(f"   Restore with: --restore-checkpoint {ckpt_dir / 'best'}")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(reward_history, linewidth=1.5)
    axes[0].set_xlabel('Update')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Reward Progress')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss_history, linewidth=1.5)
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Negative Reward')
    axes[1].set_title('Loss (Spectral Distance)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(entropy_history, linewidth=1.5)
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Policy Entropy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(__file__).parent / "reinforce_loss.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to: {plot_path}")

    # Test on unseen examples
    with report_progress.timed("test"):
        num_test = 20
        test_audio, test_features, test_params = generate_black_box_batch(random.key(9999))

        # Run eval step
        policy.eval()
        test_params_slice = {k: v[:num_test] for k, v in test_params.items()}
        eval_metrics_dict = eval_step(
            policy,
            test_features[:num_test],
            test_params_slice,
            test_audio[:num_test],
            random.key(12345),
        )

        # Convert to Collection
        eval_metrics = EvalMetrics.single_from_model_output(**eval_metrics_dict)

        # Log to TensorBoard
        eval_summary = {f"eval/{k}": float(v) for k, v in eval_metrics.compute().items()}
        writer.write_scalars(num_updates, eval_summary)

        # Print summary metrics
        eval_computed = eval_metrics.compute()
        print(f"\nTest on {num_test} unseen examples:")
        print(f"  Continuous MAE (normalized): {float(eval_computed['continuous_mae']):.4f}")
        print(f"  Categorical Accuracy: {float(eval_computed['categorical_accuracy']) * 100:.1f}%")
        print(f"  Spectral Distance: {float(eval_computed['spectral_distance']):.4f}")

        # Detailed per-parameter breakdown (need to re-predict for this)
        predicted_params, _, _ = policy(test_features[:num_test])
        predicted_physical = synth.unnormalize_params(predicted_params)
        actual_physical = synth.unnormalize_params(test_params_slice)

        print(f"\nPer-parameter breakdown:")
        for name in continuous_names:
            predicted = predicted_physical[name]
            actual = actual_physical[name]
            mae = float(jnp.mean(jnp.abs(predicted - actual)))
            param_range = float(jnp.max(actual) - jnp.min(actual))
            range_error = mae / (param_range + 1e-6) * 100

            meta = synth.get_parameter_metadata()[name]
            scale = meta.get('scale_mode', 'linear')
            print(f"  {name} ({scale}):")
            print(f"    MAE: {mae:.1f}, Range-normalized error: {range_error:.1f}%")
            print(f"    Predicted: [{float(jnp.min(predicted)):.1f}, {float(jnp.max(predicted)):.1f}]")
            print(f"    Actual:    [{float(jnp.min(actual)):.1f}, {float(jnp.max(actual)):.1f}]")

        if categorical_info:
            for name in categorical_info.keys():
                # Categorical params are already indices, don't unnormalize
                predicted = predicted_params[name].astype(jnp.int32)
                actual = test_params_slice[name].astype(jnp.int32)
                accuracy = float(jnp.mean(predicted == actual)) * 100
                print(f"  {name}: Accuracy={accuracy:.0f}%")

    print("\n" + "=" * 70)
    print("REINFORCE DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE Policy Gradient Demo")
    parser.add_argument("--num-updates", type=int, default=500,
                        help="Number of training updates (default: 500)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--entropy-coeff", type=float, default=0.05,
                        help="Entropy coefficient for exploration (default: 0.05)")
    parser.add_argument("--concentration-base", type=float, default=2.0,
                        help="Base concentration for Beta distribution (default: 2.0)")
    parser.add_argument("--concentration-scale", type=float, default=5.0,
                        help="Scale factor for concentration (default: 5.0)")
    parser.add_argument("--report-every-steps", type=int, default=50,
                        help="Report progress every N steps (default: 50)")
    parser.add_argument("--log-every-steps", type=int, default=10,
                        help="Log metrics to TensorBoard every N steps (default: 10)")
    parser.add_argument("--eval-every-steps", type=int, default=0,
                        help="Run evaluation every N steps (0 = only at end, default: 0)")
    parser.add_argument("--param-loss-coeff", type=float, default=0.1,
                        help="Coefficient for direct parameter supervision loss (default: 0.1)")
    parser.add_argument("--waveform-bonus-coeff", type=float, default=0.0,
                        help="Bonus reward for correct waveform category (default: 0.0, disabled for black-box)")
    parser.add_argument("--use-multiscale-loss", action="store_true",
                        help="Use multi-scale spectral loss instead of single-scale")
    parser.add_argument("--entropy-decay", action="store_true",
                        help="Decay entropy coefficient from high to low over training")
    parser.add_argument("--concentration-schedule", action="store_true",
                        help="Increase concentration (sharper distributions) over training")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training if eval MAE doesn't improve for N checks (0=disabled, default: 0)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01,
                        help="Minimum improvement to reset patience counter (default: 0.01)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints (default: checkpoints)")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save checkpoint every N updates (0=only best, default: 0)")
    parser.add_argument("--restore-checkpoint", type=str, default=None,
                        help="Path to checkpoint to restore from (default: None)")
    parser.add_argument("--print-model", action="store_true",
                        help="Print model architecture table before training")

    # Three-stage curriculum learning (SynthRL-style)
    parser.add_argument("--curriculum-stage", type=int, default=2, choices=[1, 2, 3],
                        help="Training stage: 1=supervised only, 2=transition, 3=RL only (default: 2)")
    parser.add_argument("--stage2-transition-start", type=int, default=0,
                        help="Update when Stage 2 transition begins (default: 0)")
    parser.add_argument("--stage2-transition-end", type=int, default=500,
                        help="Update when Stage 2 transition ends (default: 500)")

    args = parser.parse_args()
    reinforce_demo(args)
