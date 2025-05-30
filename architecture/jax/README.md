# Faust JAX Backend Documentation

The JAX backend allows Faust to generate Python code that uses JAX and Flax for efficient numerical computation with automatic differentiation support. For more information on developing backends for Faust, check out [`compiler/generator/template/README.md`](https://github.com/grame-cncm/faust/tree/master-dev/compiler/generator/template) and the related C++ files.

## Installing JAX Dependencies

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

### Recommended Dependencies

For full functionality:
```bash
# For loading audio files (used in both architectures)
pip install librosa

# For real-time audio streaming (used in jax-realtime.py)
pip install sounddevice
```

## Usage

Generate JAX code from a Faust DSP file:

```bash
faust -lang jax mydsp.dsp -cn MyDSP -o mydsp.py
```

**Important**: If running Faust from the project root directory, you need to specify the libraries path:

```bash
./build/bin/faust -lang jax -I libraries mydsp.dsp -cn MyDSP -o mydsp.py
```

Options:
- `-lang jax`: Use the JAX backend
- `-I libraries`: Include path for Faust libraries (required when running from build directory)
- `-cn MyDSP`: Sets the class name (default is `mydsp`)
- `-o mydsp.py`: Specifies the output file

The generated code creates a Flax `nn.Module` that can be used in JAX programs. You can also verify its basic execution with:
```bash
python3 mydsp.py
```
and check how to use it:
```bash
python3 mydsp.py --help
```

### Basic Example

```python
import jax
import jax.numpy as jnp
from jax import random
from mydsp import MyDSP

# Initialize the model
model = MyDSP(sample_rate=48000)
key = random.key(0)

# Create input (channels x samples)
n_samples = 48000  # 1 second

input_audio = jnp.zeros((model.num_inputs, n_samples))
input_audio = input_audio.at[:, 0].set(1.0)  # impulse on all channels

# Initialize and run the model
variables = model.init({'params': key}, input_audio, length=n_samples)
output_audio, mod_vars = model.apply(variables, input_audio, length=n_samples, mutable="intermediates")
# output_audio shape: (num_outputs, n_samples)

# if bargraphs are in the DSP code:
bargraphs = mod_vars["intermediates"]

# if no bargraphs are needed, just run this:
output_audio = model.apply(variables, input_audio, length=n_samples)

# For generators, pass None as input and specify `length`
variables = model.init({'params': key}, None, n_samples)
output_audio = model.apply(variables, None, length=n_samples)
```

**Note**: The module should handle `None` input gracefully:
- When `x is None`, the module should generate `n_samples` samples
- No transpose operations should be performed on `None`
- The module should internally create appropriate zero inputs if needed

### Real-time Processing

The JAX backend now supports real-time audio processing with block-wise computation and proper state management. This API enables low-latency processing similar to Flax's `RNNBase` pattern.

**Using the Real-time API:**

```python
import jax
import jax.numpy as jnp
from jax import random
from mydsp import MyDSP

# Initialize model
model = MyDSP(sample_rate=48000)
key = random.key(0)

BLOCK_SIZE = 512

# JIT compile the process method
@jax.jit
def process_block_jit(carry, inputs: jnp.ndarray, rng: jax.Array):
   return model.apply(variables, carry, inputs, length=BLOCK_SIZE, method="process_block", rngs={"rng_stream": rng})

# Initialize carry state
carry, variables = model.init_with_output({"params": key, "rng_stream": key}, method="initialize_carry")

rng = jax.random.key(0)

# Process audio block by block
for block_idx in range(num_blocks):
    # Get input block (e.g., from audio interface)
    input_block = get_audio_input()  # shape: (num_inputs, BLOCK_SIZE)
    
    # Process block and get updated state
    rng, subkey = jax.random.split(rng)
    (output_block, carry), _ = process_block_jit(carry, input_block, subkey)
    
    # Send output to audio interface
    send_audio_output(output_block)
```

**Available Methods:**

1. **`initialize_carry(self)`**: Creates initial state for real-time processing
   - Returns: Dictionary containing all stateful components (delays, filter states, etc.)

1. **`process_block(carry, inputs)`**: Processes one block of audio
   - `carry`: State dictionary from previous block
   - `inputs`: Input block of shape `(num_inputs, block_size)`
   - `length`: output length if `inputs` is None
   - `unroll`: Unroll size for `nn.scan`
   - Returns: Tuple `(outputs, new_carry)` where outputs has shape `(num_outputs, block_size)`

1. **`__call__(x, length, unroll)`**: Basic offline audio processing without receiving and giving a carry state
   - `x`: Input audio tensor of shape `(num_inputs, num_samples)` or `None` for generators
   - `length`: Number of samples to process (used when `x is None`)
   - `unroll`: Unroll size for `nn.scan`

**Available properties:**

- `num_inputs` and `num_outputs` properties
- `getJSON()` method returning DSP metadata


## Features

- **JIT Compilation**: Generated code is compatible with `jax.jit` and `nn.jit` for performance
- **Automatic Differentiation**: Can be used with `jax.grad` and other JAX transformations
- **Vectorization**: Compatible with `jax.vmap` for batch processing
- **State Management**: Proper handling of delays and stateful operations
- **RNG Support**: Compatible with Flax's RNG system via `self.make_rng("rng_stream")` for stochastic DSPs

## Performance Optimizations

### Single-Sample Delay Optimization

The JAX backend includes a special optimization for single-sample delays (`x'` in Faust), which are very common in DSP code. Instead of using arrays with roll operations, single-sample delays are implemented as scalar state variables.

#### How it Works

For a Faust expression like `x - x'`, the standard approach would generate:
```python
# Standard approach (inefficient)
# In `initialize`:
state["fVec0"] = np.zeros((2,), dtype=np.float32)  # Array of size 2
# In `tick`:
state["fVec0"] = state["fVec0"].at[0].set(input)  # Array update
output = input - state["fVec0"][1]                # Array access
state["fVec0"] = jnp.roll(state["fVec0"], 1)      # Expensive roll operation
```

The optimized JAX backend generates:
```python
# Optimized approach
# In `initialize`:
state["fVec0"] = np.float32(0)         # Scalar, not array
# In `tick`:
fVec0_temp = state["fVec0"]            # Store previous value
state["fVec0"] = input                 # Direct scalar update
output = input - fVec0_temp            # Use previous value
```

## Testing

Run the JAX backend tests:

```bash
cd tests/impulse-tests
make jax
```

There are several todo items marked in `tests/impulse-tests/Make.jax`.

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
