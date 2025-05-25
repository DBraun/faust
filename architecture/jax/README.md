# Faust JAX Backend Documentation

The JAX backend allows Faust to generate Python code that uses JAX and Flax for efficient numerical computation with automatic differentiation support.

## Installation

### Prerequisites

1. Python 3.9 or later
2. Faust compiler built with JAX backend support (included in regular builds)

### Installing JAX Dependencies

JAX and its ecosystem can be installed in two ways:

**Recommended - [JAX AI Stack](https://github.com/jax-ml/jax-ai-stack):**
```bash
pip install jax-ai-stack
```
This installs JAX, Flax, Optax, and other useful libraries for machine learning applications.

**Minimal Installation:**
```bash
pip install jax jaxlib flax
```

For GPU support, see the [JAX installation guide](https://github.com/google/jax#installation).

## Usage

Generate JAX code from a Faust DSP file:

```bash
faust -lang jax mydsp.dsp -cn MyDSP -o mydsp.py
```

Options:
- `-cn MyDSP`: Sets the class name (default is `mydsp`)
- `-o mydsp.py`: Specifies the output file

The generated code creates a Flax `nn.Module` that can be used in JAX programs.

### Basic Example

```python
import jax
import jax.numpy as jnp
from jax import random
from mydsp import MyDSP

# Initialize the model
model = MyDSP(sample_rate=48000)
key = random.PRNGKey(0)

# Create input (channels x samples)
n_samples = 48000  # 1 second
input_audio = jnp.zeros((model.getNumInputs(), n_samples))
input_audio = input_audio.at[:, 0].set(1.0)  # impulse

# Initialize and run the model
variables = model.init({'params': key}, input_audio, n_samples)
output_audio, mod_vars = model.apply(variables, input_audio, n_samples, mutable='intermediates')

# output_audio shape: (num_outputs, n_samples)
```

### Real-time Processing Example

todo(DBraun)

## Architecture

The JAX backend generates:

1. A Flax `nn.Module` class with:
   - `initialize()` method for state initialization
   - Static `tick()` method for DSP computation
   - `getNumInputs()` and `getNumOutputs()` methods
   - `getJSON()` method returning DSP metadata

2. Efficient JAX operations using `jnp` (JAX numpy)

3. Support for both single and double precision (controlled by `-single` or `-double` flags)

## Features

- **JIT Compilation**: Generated code is compatible with `jax.jit` and `nn.jit` for performance
- **Automatic Differentiation**: Can be used with `jax.grad` and other JAX transformations
- **Vectorization**: Compatible with `jax.vmap` for batch processing
- **State Management**: Proper handling of delays and stateful operations

## Testing

Run the JAX backend tests:

```bash
cd tests/impulse-tests
make jax
```

## Polyphony Support

The JAX backend supports polyphonic DSPs. Polyphony in JAX is naturally handled using `vmap` for efficient vectorized processing across multiple voices.

## Troubleshooting

### Performance

For optimal performance:
- Use `jax.jit` or `nn.jit`
- Consider using GPU acceleration with large batch sizes.

## Integration with Machine Learning

The JAX backend is particularly useful for:
- Differentiable audio effects
- Neural audio synthesis
- Audio processing in ML pipelines
- Research applications requiring gradients through audio processing
