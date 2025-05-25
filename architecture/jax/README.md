# Faust JAX Backend Documentation

The JAX backend allows Faust to generate Python code that uses JAX and Flax for efficient numerical computation with automatic differentiation support.

## Installation

### Prerequisites

1. Python 3.9 or later
2. Faust compiler built with JAX backend support (included in regular builds)

### Installing JAX Dependencies

JAX and its ecosystem can be installed in two ways:

**Option 1 - [JAX AI Stack](https://github.com/jax-ml/jax-ai-stack) (for new environments):**
```bash
pip install jax-ai-stack
```
This installs JAX, Flax, Optax, and other useful libraries with pinned compatible versions.

**Option 2 - Minimal Installation (recommended for CI/existing environments):**
```bash
pip install --upgrade jax jaxlib flax
```
This installs the latest versions of core dependencies.

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
input_audio = jnp.zeros((model.num_inputs, n_samples))
input_audio = input_audio.at[:, 0].set(1.0)  # impulse

# Initialize and run the model
variables = model.init({'params': key}, input_audio, n_samples)
output_audio, mod_vars = model.apply(variables, input_audio, n_samples, mutable='intermediates')

# output_audio shape: (num_outputs, n_samples)
```

### Generator Example (No Input)

For DSPs that generate audio without inputs (e.g., synthesizers, oscillators):

```python
import jax
import jax.numpy as jnp
from jax import random
from mydsp import MyDSP

# Initialize the model
model = MyDSP(sample_rate=48000)
key = random.PRNGKey(0)

# For generators, pass None as input and specify length with T
n_samples = 48000  # 1 second
variables = model.init({'params': key}, None, n_samples)
output_audio, mod_vars = model.apply(variables, None, n_samples, mutable='intermediates')

# output_audio shape: (num_outputs, n_samples)
```

**Note**: The module should handle `None` input gracefully:
- When `x is None`, the module should generate `T` samples
- No transpose operations should be performed on `None`
- The module should internally create appropriate zero inputs if needed

### Real-time Processing (Future Enhancement)

The current JAX backend processes entire audio buffers at once using `jax.lax.scan` internally. For true real-time processing with block-wise computation, the backend should be enhanced to work more like Flax's `RNNBase` pattern.

**Proposed API (not yet implemented):**

```python
import jax
import jax.numpy as jnp
from jax import random
from mydsp import MyDSP

# Initialize model
model = MyDSP(sample_rate=48000)
key = random.PRNGKey(0)

# Initialize parameters with dummy input
dummy_input = jnp.zeros((model.getNumInputs(), 1))
variables = model.init({'params': key}, dummy_input, 1)

# Initialize carry state for real-time processing
carry = model.initialize_carry(key, (model.getNumInputs(),))

# Process audio block by block
block_size = 512
for block_idx in range(num_blocks):
    # Get input block (e.g., from audio interface)
    input_block = get_audio_input()  # shape: (num_inputs, block_size)
    
    # Process block and get updated state
    (output_block, new_carry), _ = model.apply(
        variables, 
        carry, 
        input_block,
        method=model.process_block,
        mutable='intermediates'
    )
    
    # Update carry for next iteration
    carry = new_carry
    
    # Send output to audio interface
    send_audio_output(output_block)
```

**Implementation Ideas:**

1. **Stateful Processing**: The module should expose methods like:
   - `initialize_carry()`: Create initial state (delays, oscillator phases, filter states)
   - `process_block(carry, inputs)`: Process one block and return `(outputs, new_carry)`

2. **Efficient State Management**: 
   - State should be a PyTree that can be efficiently updated
   - Only the necessary state (delays, integrators) should be carried between blocks
   - State size should be minimal for low latency

3. **Compatibility**:
   - The existing `__call__` method could internally use `process_block` with `scan`
   - This maintains backward compatibility while enabling new use cases

4. **JIT Compilation**:
   ```python
   @jax.jit
   def process_block_jit(variables, carry, input_block):
       return model.apply(variables, carry, input_block, 
                         method=model.process_block, 
                         mutable='intermediates')
   ```

This pattern would enable:
- True real-time processing with low latency
- Integration with audio streaming frameworks
- Stateful processing in interactive applications
- Efficient block-wise gradient computation for online learning

## Architecture

The JAX backend generates:

1. A Flax `nn.Module` class with:
   - `__call__(x, T)` method that accepts:
     - `x`: Input audio tensor of shape `(num_inputs, num_samples)` or `None` for generators
     - `T`: Number of samples to process (used when `x is None`)
   - `initialize()` method for state initialization
   - Static `tick()` method for DSP computation
   - `num_inputs` and `num_outputs` properties (Pythonic style)
   - `getNumInputs()` and `getNumOutputs()` methods (for backward compatibility)
   - `getJSON()` method returning DSP metadata

2. Efficient JAX operations using `jnp` (JAX numpy)

3. Support for both single and double precision (controlled by `-single` or `-double` flags)

4. Automatic handling of generators vs processors:
   - When `getNumInputs() == 0`, the module works as a generator
   - The module creates zero inputs internally when needed
   - `T` parameter determines output length for generators

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
