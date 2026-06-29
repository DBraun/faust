# RL Integration Demo

This directory demonstrates reinforcement learning for synthesizer parameter estimation using Faust's NNX backend and the `normalized_params` API.

## Overview

The REINFORCE demo solves inverse synthesis: given target audio, predict synthesizer parameters that reproduce it. Uses policy gradient RL with a learned baseline (actor-critic) and achieves **MAE < 0.07** on normalized continuous parameters.

**Task**: Black-box synth plays C3 with unknown params → Policy observes audio features → Predicts params → Renders audio → Reward based on spectral similarity

## Best Results

| Method | MAE (normalized) | Categorical Acc | Training |
|--------|------------------|-----------------|----------|
| Stage 1 (Supervised) | **0.062** | 25% | 1000 updates, param loss only |
| Stage 2 (Curriculum) | 0.068 | **65%** | +1000 updates, supervised→RL transition |
| Direct RL | 0.081 | 65% | 1500 updates, hybrid loss |

**Per-parameter (Stage 2)**: fHslider0 (log-scale): 6.9% error, fHslider1 (linear): 11.4% error

## Key Features

### Architecture
- **Policy**: Beta distributions (mode-concentration) for continuous params, Categorical for discrete
- **Baseline**: Learned value function for variance reduction
- **Network**: 256→128→64 MLP with LayerNorm (40K params)
- **Features**: 20-dim (5 MFCCs, 3 spectral contrast bands, 12 traditional features)

### Training Techniques
- **SynthRL Reward**: Multi-component (70% spectral convergence + 27% log-magnitude + 3% MFCC)
- **Curriculum Learning**: Three-stage (supervised → transition → RL)
- **Schedules**: Entropy decay, concentration sharpening, LR warmup + cosine decay
- **Supervised Loss**: Parameter MSE weighted by audio quality
- **Normalization**: Statistics-based feature scaling from 1000 random samples

## Usage

### Basic Training
```bash
python reinforce_demo.py --num-updates 1500 --batch-size 256
```

### Three-Stage Curriculum (Recommended)

**Stage 1** (supervised foundation):
```bash
python reinforce_demo.py \
  --num-updates 1000 \
  --curriculum-stage 1 \
  --param-loss-coeff 1.0 \
  --checkpoint-dir checkpoints_stage1
```

**Stage 2** (RL transition):
```bash
python reinforce_demo.py \
  --num-updates 1000 \
  --curriculum-stage 2 \
  --stage2-transition-start 0 \
  --stage2-transition-end 1000 \
  --restore-checkpoint checkpoints_stage1/best.safetensors \
  --checkpoint-dir checkpoints_stage2
```

### Key Arguments

**Training**:
- `--num-updates`: Training iterations (default: 500)
- `--batch-size`: Samples per update (default: 128, use 256 for best results)
- `--learning-rate`: Adam LR (default: 1e-3)

**Curriculum**:
- `--curriculum-stage {1,2,3}`: 1=supervised, 2=transition, 3=RL only
- `--stage2-transition-start/end`: When to blend supervised→RL in Stage 2
- `--param-loss-coeff`: Supervised loss weight (1.0 for Stage 1, 0.3 for Stage 2)

**Schedules**:
- `--entropy-decay`: Decay exploration over training
- `--concentration-schedule`: Sharpen Beta distributions over time

**Checkpointing**:
- `--restore-checkpoint PATH`: Resume from a `.safetensors` checkpoint
- `--checkpoint-dir DIR`: Save location (default: checkpoints)
- `--checkpoint-every N`: Periodic saves (0=best only)
- `--early-stopping-patience N`: Stop if no improvement for N evals

### Monitoring
```bash
tensorboard --logdir logs/
```

Metrics logged: policy/value loss, reward, MAE, categorical accuracy, gradient norms, concentration stats, audio samples.

## Files

- **`synth.dsp`** - Example synth (2 continuous + 1 categorical param)
- **`reinforce_demo.py`** - Main REINFORCE implementation
- **`utils.py`** - Compilation, feature extraction, SynthRL reward
- **`compute_feature_stats.py`** - Generate normalization statistics
- **`feature_stats.json`** - Normalization statistics (mean/std/min/max for 20 features)
  - Pre-computed from 1000 random synthesizer sounds
  - Required for training (features normalized to zero-mean, unit-variance)
  - **Regenerate if you modify the synthesizer or feature extraction**:
    ```bash
    python compute_feature_stats.py  # Takes ~2 minutes
    ```

## Implementation Details

### SynthRL Reward Function
Combines three metrics with proven weights from [Shin & Lee, IJCAI 2025](https://github.com/argaaw/SynthRL):
```python
SC = ||pred_spec - target_spec||_F / ||target_spec||_F  # Normalized Frobenius
log_mae = mean(|log10(pred_spec) - log10(target_spec)|)
mfcc_mae = mean(|mfcc(pred) - mfcc(target)|)
reward = 1 / clamp(0.7*SC + 0.27*log_mae + 0.03*mfcc_mae, 0.1, 5.0)
```

### Curriculum Learning
Optax schedules blend supervised and RL losses:
- **Stage 1**: `loss = param_loss` (foundation)
- **Stage 2**: `loss = α*rl_loss + (1-α)*param_loss` where α transitions 0→1
- **Stage 3**: `loss = rl_loss` (cross-domain generalization)

### Feature Extraction
Extracts 20 features from audio (via `librosax`):
- 5 MFCCs (spectral envelope)
- 3 spectral contrast bands (timbral texture)
- RMS, zero-crossing rate, spectral centroid/bandwidth/rolloff/flatness

Features are **normalized** using `feature_stats.json`:
```python
normalized = (features - mean) / (std + 1e-8)
normalized = clip(normalized, -3.0, 3.0)  # Remove outliers
```

This ensures:
- Zero-mean, unit-variance inputs for stable neural network training
- Consistent scale across different feature types (e.g., RMS vs MFCC)
- Better gradient flow and faster convergence

The statistics are computed from 1000 random synthesizer sounds to capture the typical range of each feature when parameters vary across their full ranges.

## Prerequisites

- Faust compiler with NNX backend
- Python 3.11+
- JAX, Flax NNX, optax, distrax, safetensors
- librosax (audio feature extraction)
- matplotlib, einops (utilities)

```bash
pip install jax jax-ai-stack librosax matplotlib einops
```

## References

- **SynthRL Paper**: Shin & Lee, "Cross-domain Synthesizer Sound Matching via Reinforcement Learning", IJCAI 2025 ([code](https://github.com/argaaw/SynthRL))
- **Faust NNX Backend**: `../../architecture/jax/README.md`

## Notes

- **Gradient flow**: Uses `stop_gradient` on advantages before normalization when combining supervised + RL losses
- **Evaluation**: Deterministic (returns mode/argmax), training samples from distributions
- **Audio rendering**: Vmapped on CPU for faster synthesis (vs GPU overhead for small DSP)
- **Checkpointing**: portable `.safetensors` files, store the policy's parameters
- **JIT**: outer `jax.jit` over `nnx.split`/`merge` (Trainer-style), not `nnx.jit`
