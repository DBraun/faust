# Faust JAX Backend Documentation

The JAX backend allows Faust to generate Python code that uses JAX and Flax (NNX) for efficient numerical computation with automatic differentiation support.
For more information on developing backends for Faust, check out [`compiler/generator/template/README.md`](https://github.com/grame-cncm/faust/tree/master-dev/compiler/generator/template) and the related C++ files.

## Installing JAX Dependencies

JAX and its ecosystem should be installed via the [JAX AI Stack](https://github.com/jax-ml/jax-ai-stack):

```bash
pip install jax-ai-stack
```

For GPU support:
```bash
pip install jax-ai-stack "jax[cuda]"  # JAX + AI stack with GPU/CUDA support
pip install jax-ai-stack "jax[tpu]"  # JAX + AI stack with TPU support
```

### Recommended Dependencies

For full functionality:
```bash
# For loading audio files via the soundfile primitive
pip install librosa

# For Linux
pip install soundfile

# For real-time audio streaming
pip install sounddevice
```

## Usage

Generate JAX code from a Faust DSP file:

**Important**: If running Faust from the project root directory, you need to specify the libraries path:

```bash
./build/bin/faust -lang jax -I libraries my_example.dsp -cn MyExample -o my_example.py
```

Options:
- `-lang jax`: Use the JAX backend
- `-I libraries`: Include path for Faust libraries (you can omit this entirely if `faust` has been fully installed)
- `-a architecture/jax/minimal.py`: The architecture file (use `-a jax/minimal.py` if `faust` has been fully installed)
- `-cn MyExample`: Sets the class name (default is `mydsp`)
- `-o my_example.py`: Specifies the output file

The generated code creates a Flax `nnx.Module` that can be used in JAX programs. You can also verify its basic execution with:
```bash
python3 my_example.py
```
and check how to use it:
```bash
python3 my_example.py --help
```

### Basic Example

```python
import jax
from jax import numpy as jnp, random
from my_example import MyExample

# Pick some seed values
rngs = nnx.Rngs(0, params=42, rng_stream=1337)

sample_rate = 48_000

# Initialize the model
model = MyExample(sample_rate=sample_rate, rngs=rngs)

# Create input (channels x samples)
n_samples = int(sample_rate*1.0)  # 1 second

input_audio = jnp.zeros((model.num_inputs, n_samples))
input_audio = input_audio.at[:, 0].set(1.0)  # impulse on all channels

# Run the model
output_audio = model(input_audio)
assert output_audio.shape == (model.num_outputs, n_samples)

# For generators, pass None as input and specify `length`
output_audio = model(None, length=n_samples)

# If bargraphs are in the DSP code:
# todo: this feature is currently disabled in jax_code_container.cpp
output_audio, mod_vars = model.apply(variables, input_audio, mutable="intermediates")
bargraphs = mod_vars["intermediates"]
```

**How it works:**

**`__call__(x: jnp.ndarray, length: int, unroll: int)`**: Basic offline audio processing without receiving and returning a carry state
- `x`: Input audio tensor of shape `(num_inputs, num_samples)` or `None` for generators
- `length`: Number of samples to process (necessary when `x is None`)
- `unroll`: Unroll size for `jax.lax.scan`

**Available properties:**

- `num_inputs` and `num_outputs` are constant properties about the number of input and output channels for the DSP.
- `json_metadata` method returning DSP metadata

### Real-time Processing

The JAX backend supports real-time audio processing with block-wise computation and proper state management.
This API enables low-latency processing similar to Flax's `RNNBase` pattern.

**Using the Real-time API:**

```python
import jax
from jax import numpy as jnp, random
from my_example import MyExample

# Pick some seed values
rngs = nnx.Rngs(0, params=42, rng_stream=1337)

# Initialize model
model = MyExample(sample_rate=48000, rngs=rngs)

BLOCK_SIZE = 512

# Initialize carry state
carry = model.initialize_carry()

# JIT compile the process method
@jax.jit
def process_block_jit(carry, inputs: jnp.ndarray):
   outputs, new_carry = model.process_block(
      carry,
      inputs,
      length=block_size,
      unroll=unroll,
   )
   return outputs, new_carry

# Process audio block by block
for block_idx in range(num_blocks):
    # Get input block (e.g., from audio interface)
    input_block = get_audio_input()  # shape: (num_inputs, BLOCK_SIZE)
    
    # Process block and get updated state
    output_block, carry = process_block_jit(carry, input_block)
    # output_block is (num_outputs, BLOCK_SIZE)
    
    # Send output to audio interface. In reality, audio interfaces use a callback strategy.
    send_audio_output(output_block)
```

**How it works:**

1. **`initialize_carry(self)`**: Creates initial state for real-time processing
   - Returns: Dictionary containing all stateful components (delays, filter states, etc.)

1. **`process_block(carry, inputs: jnp.ndarray, length: int = None, unroll: int = 1)`**: Processes one block of audio
   - `carry`: State dictionary from previous block
   - `inputs`: Input block of shape `(num_inputs, block_size)`
   - `length`: output length if `inputs` is None. This is like the block size.
   - `unroll`: Unroll size for `jax.lax.scan`
   - Returns: Tuple `(outputs, new_carry)` where outputs has shape `(num_outputs, block_size)`

## Features

- **JIT Compilation**: Generated code is compatible with `jax.jit` and `nnx.jit` for performance
- **Automatic Differentiation**: Can be used with `jax.grad`, `nnx.value_and_grad` and other JAX/Flax [transformations](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html)
- **Vectorization**: Compatible with `jax.vmap` and `nnx.vmap` for batch processing
- **State Management**: Proper handling of delays and stateful operations
- **RNG Support**: Compatible with Flax's RNG system for stochastic DSPs

### Random Number Generation

The JAX backend supports native JAX random number generation through foreign functions. You can use JAX's PRNG system by declaring `random_uniform` as a foreign function:

```faust
import("stdfaust.lib");

// Declare JAX's random uniform function as a foreign function
random_uniform = ffunction(float random_uniform(), <math.h>, "");

// Use it to generate random values in [-1, 1]
process = random_uniform, random_uniform;  // stereo noise
```

This generates code that calls `self.random_uniform(rngs())` where `rngs()` is a helper function that provides fresh RNG keys from JAX's PRNG system. Each call to `random_uniform` receives a fresh RNG subkey, ensuring different random values even when used multiple times in the same expression.

The `self.random_uniform()` method is implemented in the architecture file and returns uniform random values in the range [-1, 1], compatible with Faust's standard noise generators. This approach provides:
- Proper JAX PRNG state management
- Reproducibility with seed control
- Different values for each channel (no caching of foreign function calls in JAX backend)
- Compatibility with JAX transformations like `vmap` and `jit`

**Note**: Standard Faust noise functions like `no.noise` still use the linear congruential generator (LCG) for compatibility. Use the foreign function approach shown above for JAX PRNG.

## Performance Optimizations

### Delay Line Optimization

The JAX backend implements intelligent delay line optimization to minimize expensive `jnp.roll` operations. This is controlled by the `-mcd` (max copy delay) compiler flag.

#### The `-mcd` Flag

The `-mcd` flag determines when to use circular buffers vs roll operations:

```bash
# Default: delays ≤16 use roll, delays >16 use circular buffers
./build/bin/faust -lang jax mydsp.dsp -o mydsp.py

# Force more delays to use roll operations (may reduce performance)
./build/bin/faust -lang jax -mcd 64 mydsp.dsp -o mydsp.py

# Force more delays to use circular buffers (may improve performance)
./build/bin/faust -lang jax -mcd 8 mydsp.dsp -o mydsp.py
```

#### Implementation Strategies

**Roll Operations** (for delays ≤ `-mcd`):
```python
# Initialization
state["fVec0"] = np.zeros((delay+1,), dtype=np.float64)

# Per-sample tick
state["fVec0"] = state["fVec0"].at[0].set(inputs[0])  # Write at position 0
_result0 = state["fVec0"][delay]                       # Read from fixed position
state["fVec0"] = jnp.roll(state["fVec0"], 1)          # O(n) shift operation
```

**Circular Buffers** (for delays > `-mcd`):
```python
# Initialization
state["fVec0"] = np.zeros((next_pow2,), dtype=np.float64)  # Size = next power of 2
state["IOTA0"] = 0

# Per-sample tick
write_idx = (state["IOTA0"] & mask)                    # Modulo using bit mask
read_idx = ((state["IOTA0"] - delay) & mask)           # Delayed position
state["fVec0"] = state["fVec0"].at[write_idx].set(inputs[0])
_result0 = state["fVec0"][read_idx]                    # O(1) operations
state["IOTA0"] = (state["IOTA0"] + 1)
```

#### Performance Characteristics

| Strategy | Write | Read | Shift | Memory | Best For |
|----------|-------|------|-------|--------|----------|
| Roll | O(1) | O(1) | O(n) per sample | Exact (delay+1) | Very small delays (≤8) |
| Circular | O(1) | O(1) | None | Power of 2 | Larger delays (>8) |

#### Optimization Guidelines

Based on benchmarking results:

1. **Simple delays (≤8 samples)**: Use `-mcd 4` or `-mcd 8`
   - Roll operations are fast for small arrays
   - Better cache locality

2. **Medium delays (8-32 samples)**: Default `-mcd 16` is usually optimal
   - Balances performance vs complexity

3. **Large delays (>32 samples)**: Use circular buffers (low `-mcd`)
   - Roll becomes prohibitively expensive
   - Essential for reverbs and long delay lines

4. **Multiple parallel delays**: Use `-mcd 4` to `-mcd 8`
   - Forces most delays to use circular buffers
   - Can improve performance by 2-3x

5. **Complex reverbs**: May need higher `-mcd` for correctness
   - Some algorithms depend on specific delay semantics
   - Test with impulse responses when changing `-mcd`

For detailed information about delay line optimization, see `DELAY_LINES.md`.

### Benchmarking Tools

The `minimal.py` architecture includes benchmarking capabilities:

```bash
# Benchmark with timing statistics
python mydsp.py --benchmark 100 --jit -d 1.0

# Display model structure
python mydsp.py --tabulate

# Test real-time performance
python mydsp.py --realtime --block-size 512
```

The benchmark mode provides:
- Average, min, max execution times
- Throughput (samples/sec)
- Real-time factor (can it run in real-time?)
- Proper use of `block_until_ready()` for accurate JIT timing

A comprehensive delay benchmark script is available in `tests/jax-tests/benchmark_delays.py` which automatically tests different `-mcd` values and provides optimization recommendations

## Testing

Run the JAX backend tests:

```bash
cd tests/impulse-tests
make jax
```

There are several todo items marked in `tests/impulse-tests/Make.jax`.

### Impulse Test Architecture

The `tests/impulse-tests/archs/impulsejax.py` file is a specialized architecture for impulse response testing. It has specific requirements:

1. **Output Formatting**: Must match the reference format exactly with proper spacing:
   ```
   number_of_inputs  :   1    # 3 spaces after colon
   number_of_outputs :   1    # 3 spaces after colon  
   number_of_frames  :   60000 # 3 spaces after colon
   ```

2. **Soundfile Workaround**: Like `minimal.py`, it includes the soundfile state workaround described below.

## Polyphony Support

The JAX backend supports polyphonic DSPs. Polyphony in JAX is naturally handled using `vmap` for efficient vectorized processing across multiple voices.

## Troubleshooting

### Performance

For optimal performance:
- Use `jax.jit` or `nnx.jit`
- Consider using GPU acceleration with large batch sizes.

## Integration with Machine Learning

The JAX backend is particularly useful for:
- Differentiable audio effects
- Neural audio synthesis
- Audio processing in ML pipelines
- Research applications requiring gradients through audio processing

## Available Architecture File(s)

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

### Soundfile Handling Implementation Note

The current JAX backend has a known issue with soundfile handling that requires a workaround in the architecture files:

#### The Issue

The Faust compiler generates code that expects soundfiles to be in the `state` dictionary:
```python
# In generated tick function:
fSoundfile0ca = state["fSoundfile0"]
```

However, the generated `_initialize_carry()` method does NOT include soundfiles in the state initialization. This is a bug because soundfiles are immutable data (read-only audio buffers) that should not be part of the carry state.

#### Current Workaround

Architecture files work around this by manually adding soundfiles to the state in `initialize_carry()`:
```python
def initialize_carry(self):
    state = self._initialize_carry()
    
    # Add soundfiles to state if they exist
    for attr_name in dir(self):
        if attr_name.startswith("fSoundfile"):
            state[attr_name] = getattr(self, attr_name)
    
    return state
```

#### Why This is Suboptimal

Soundfiles should NOT be in the carry state because:
- They are immutable (never modified during processing)
- They are not part of the signal processing state
- Including them unnecessarily passes large static data through JAX's scan operations
- This can impact performance, especially with large soundfiles

#### Ideal Solution

The Faust compiler should be fixed to either:
1. Access soundfiles directly from `self` in the tick function
2. Pass soundfiles through the `params` dictionary (like UI parameters)

Until the compiler is fixed, the workaround must remain in place to ensure soundfile-based DSPs work correctly.

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

# Limitations and todos:

* The JAX backend generates some unused variables in the output python code. These should be eliminated.
* By default, the `soundfile` primitive is *not* a learnable parameter. You can make it learnable by adding prefixing the label with "param:" such as `soundfile("param:Tango[url:{'tango.wav'}]", 2)`
* * This should be revised into `soundfile("Tango[param:1][url:{'tango.wav'}]", 2)`
* The `waveform` primitive in Faust doesn't become a Flax parameter in the generated Module. It would be useful to have a way to turn it into a learnable parameter instead of just a constant array.
* It would be useful to use metadata to enable or disable a parameter. For example, currently by default, all `hslider` are learnable, but we could have `hslider("foo[param:0]", 0.5, 0, 1, .01)` be frozen.
* We could try to generate more efficient code in the `tick` function, but it's possible that XLA is already equivalently optimizing the code for us when we use JIT. We could look at the HLO or other to investigate.
* * Try to minimize the amount of `jnp.roll` operations (see `DELAY_LINES.md`)
* * Avoid unnecessary casts from `bool` to `jnp.int32`
* Bargraph support is incomplete. Ideally we would use `nnx.Module.sow` within a scan (see https://github.com/google/flax/discussions/4799)