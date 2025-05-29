# Faust JAX Backend Documentation

The JAX backend allows Faust to generate Python code that uses JAX and Flax for efficient numerical computation with automatic differentiation support. For more information on developing backends for Faust, check out [`compiler/generator/template/README.md`](https://github.com/grame-cncm/faust/tree/master-dev/compiler/generator/template).

## Available Architecture Files

### `minimal.py`
Minimal architecture for basic usage. Includes:
- All UI element handlers (sliders, buttons, soundfiles, etc.)
- Basic `__call__` method for processing
- Generator support (0-input DSPs)
- Soundfile loading with librosa
- `initialize_carry()`: Creates initial state for block-wise processing
- `process_block()`: Processes audio block-by-block with state management
- Real-time audio streaming example using `sounddevice`

### UI Element Handlers

An architecture file must implement these methods:

- **`add_slider()`**: Continuous parameters with linear/exp/log scaling
- **`add_hslider()`/`add_vslider()`**: Horizontal/vertical sliders
- **`add_button()`**: Momentary buttons (0 or 1)
- **`add_checkbox()`**: Toggle switches (0 or 1)
- **`add_nentry()`**: Numerical entries with discrete steps
- **`add_soundfile()`**: Audio file loading and playback
- **`add_hbargraph()`/`add_vbargraph()`**: Output value displays

It is optional to override these methods:
- **`load_soundfile()`**: Helper for loading audio files from disk

## Installation

### Prerequisites

1. Python 3.9 or later
2. Faust compiler built with JAX backend support (included in regular builds)

### Building Faust with JAX Backend

To build Faust with only the JAX backend:

```bash
# From the Faust root directory
cd build
cmake . -Bfaustdir -DCMAKE_BUILD_TYPE=Release -C ./backends/jax-only.cmake
cmake --build faustdir --config=Release

# Optional: To build only the compiler (faster):
# cmake --build faustdir --config=Release --target faust
```

To build Faust with JAX and other backends:

```bash
# From the Faust root directory
cd build
cmake . -Bfaustdir -DCMAKE_BUILD_TYPE=Release -C ./backends/all.cmake
cmake --build faustdir --config=Release
```

The Faust executable will be in `build/bin/faust`. Verify JAX backend is included:

```bash
./build/bin/faust -v
# Should show "DSP to JAX" in the embedded backends list
```

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
input_audio = input_audio.at[:, 0].set(1.0)  # impulse

# Initialize and run the model
variables = model.init({'params': key}, input_audio, n_samples)
output_audio, mod_vars = model.apply(variables, input_audio, n_samples, mutable="intermediates")
# output_audio shape: (num_outputs, n_samples)

# if bargraphs are in the DSP code:
bargraphs = mod_vars["intermediates"]
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
key = random.key(0)

# For generators, pass None as input and specify `length`
n_samples = 48000  # 1 second
variables = model.init({'params': key}, None, n_samples)
output_audio = model.apply(variables, None, length=n_samples)

# output_audio shape: (num_outputs, n_samples)
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

block_size = 512

# Initialize parameters with dummy input
dummy_input = jnp.zeros((model.getNumInputs(), block_size))
variables = model.init({"params": key, "rng_stream": key}, dummy_input, length=1)

# Initialize carry state
carry = model.apply(variables, method="initialize_carry")

# JIT compile the process method
@jax.jit
def process_block_jit(carry, inputs: jnp.ndarray, rng: jax.Array):
   return model.apply(variables, carry, inputs, length=BLOCK_SIZE, unroll=unroll, method="process_block", rngs={"rng_stream": rng})

rng = jax.random.key(0)

# Process audio block by block
for block_idx in range(num_blocks):
    # Get input block (e.g., from audio interface)
    input_block = get_audio_input()  # shape: (num_inputs, block_size)
    
    # Process block and get updated state
    rng, subkey = jax.random.split(rng)
    (output_block, carry), _ = process_block_jit(carry, input_block, subkey)
    
    # Send output to audio interface
    send_audio_output(output_block)
```

**Available Methods:**

1. **`initialize_carry(x, length: int)`**: Creates initial state for real-time processing
   - `x`:
   - `length`: 
   - Returns: Dictionary containing all stateful components (delays, filter states, etc.)

1. **`process_block(carry, inputs)`**: Processes one block of audio
   - `carry`: State dictionary from previous block
   - `inputs`: Input block of shape `(num_inputs, block_size)`
   - `length`: 
   - `unroll`:
   - Returns: Tuple `(outputs, new_carry)` where outputs has shape `(num_outputs, block_size)`

## Architecture

The JAX backend generates:

1. A Flax `nn.Module` class with:
   - `__call__(x, length, unroll)` method that accepts:
     - `x`: Input audio tensor of shape `(num_inputs, num_samples)` or `None` for generators
     - `length`: Number of samples to process (used when `x is None`)
     - `unroll`: Unroll size for `nn.scan`
   - `initialize()` method for state initialization
   - Static `tick()` method for DSP computation
   - `num_inputs` and `num_outputs` properties
   - `getJSON()` method returning DSP metadata

2. Efficient JAX operations using `jnp` (JAX numpy)

3. Support for both single and double precision (controlled by `-single` or `-double` flags)

4. Automatic handling of generators vs processors:
   - When `.num_inputs == 0`, the module works as a generator
   - The module creates zero inputs internally when needed
   - `T` parameter determines output length for generators

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
