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
# For loading audio files via the soundfile primitive
pip install librosa

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

The generated code creates a Flax `nn.Module` that can be used in JAX programs. You can also verify its basic execution with:
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
import jax.numpy as jnp
from jax import random
from my_example import MyExample

# Initialize the model
model = MyExample(sample_rate=48000)
key = random.key(0)

# Create input (channels x samples)
n_samples = 48000  # 1 second

input_audio = jnp.zeros((model.num_inputs, n_samples))
input_audio = input_audio.at[:, 0].set(1.0)  # impulse on all channels

# Initialize and run the model
variables = model.init({'params': key}, input_audio)
output_audio, mod_vars = model.apply(variables, input_audio, mutable="intermediates")
assert output_audio.shape == (model.num_outputs, n_samples)

# if bargraphs are in the DSP code:
bargraphs = mod_vars["intermediates"]

# if no bargraphs are needed, just run this:
output_audio = model.apply(variables, input_audio)

# For generators, pass None as input and specify `length`
variables = model.init({'params': key}, None, length=n_samples)
output_audio = model.apply(variables, None, length=n_samples)
```

**How it works:**

**`__call__(x, length, unroll)`**: Basic offline audio processing without receiving and giving a carry state
- `x`: Input audio tensor of shape `(num_inputs, num_samples)` or `None` for generators
- `length`: Number of samples to process (used when `x is None`)
- `unroll`: Unroll size for `nn.scan`

**Available properties:**

- `num_inputs` and `num_outputs` are constant properties about the number of input and output channels for the DSP.
- `json_metadata` method returning DSP metadata

### Real-time Processing

The JAX backend supports real-time audio processing with block-wise computation and proper state management. This API enables low-latency processing similar to Flax's `RNNBase` pattern.

**Using the Real-time API:**

```python
import jax
import jax.numpy as jnp
from jax import random
from my_example import MyExample

# Initialize model
model = MyExample(sample_rate=48000)
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

**How it works:**

1. **`initialize_carry(self)`**: Creates initial state for real-time processing
   - Returns: Dictionary containing all stateful components (delays, filter states, etc.)

1. **`process_block(carry, inputs)`**: Processes one block of audio
   - `carry`: State dictionary from previous block
   - `inputs`: Input block of shape `(num_inputs, block_size)`
   - `length`: output length if `inputs` is None
   - `unroll`: Unroll size for `nn.scan`
   - Returns: Tuple `(outputs, new_carry)` where outputs has shape `(num_outputs, block_size)`

## Features

- **JIT Compilation**: Generated code is compatible with `jax.jit` and `nn.jit` for performance
- **Automatic Differentiation**: Can be used with `jax.grad` and other JAX transformations
- **Vectorization**: Compatible with `jax.vmap` for batch processing
- **State Management**: Proper handling of delays and stateful operations
- **RNG Support**: Compatible with Flax's RNG system via `self.make_rng("rng_stream")` for stochastic DSPs

## Performance Optimizations

### Circular Buffer Optimization for Delay Lines

The JAX backend implements intelligent delay line optimization to minimize expensive `jnp.roll` operations. This is controlled by the `-mcd` (max copy delay) compiler flag.

#### Compiler Flag: `-mcd <size>`

The `-mcd` flag controls when circular buffers are used instead of roll operations:

```bash
# Default: small delays (≤16) use roll, larger delays use circular buffers
./build/bin/faust -lang jax mydsp.dsp -o mydsp.py

# Force more delays to use roll operations (may reduce performance)
./build/bin/faust -lang jax -mcd 64 mydsp.dsp -o mydsp.py

# Force more delays to use circular buffers (may break some filter designs)
./build/bin/faust -lang jax -mcd 8 mydsp.dsp -o mydsp.py
```

#### How Circular Buffer Optimization Works

**Roll Operations** (for small delays ≤ `-mcd` size, default 16):
```python
# Traditional approach for small recursive delays
state["fRec0"] = state["fRec0"].at[0].set(new_value)
output = state["fRec0"][1] + state["fRec0"][2]  # Read delayed values
state["fRec0"] = jnp.roll(state["fRec0"], 1)    # O(n) shift operation
```

**Circular Buffers** (for larger delays > `-mcd` size):
```python
# Optimized approach using modular arithmetic
state["fVec0"] = state["fVec0"].at[state["fVec0_idx"]].set(new_value)  # Write at current index
delay1 = state["fVec0"][(((state["fVec0_idx"] - 1) + size) % size)]    # O(1) delayed access
delay2 = state["fVec0"][(((state["fVec0_idx"] - 2) + size) % size)]    # O(1) delayed access
state["fVec0_idx"] = ((state["fVec0_idx"] + 1) % size)                 # Increment index
```

#### When to Adjust `-mcd`

- **Default (16)**: Works well for most DSPs. Small recursive filters use roll, large delays use circular buffers
- **Increase `-mcd`**: If you have complex filter matrices that break with circular buffers
- **Decrease `-mcd`**: If you want maximum performance and your DSP doesn't use complex recursive structures

**Examples by DSP type:**
- **Simple delays, echo effects**: Benefit from lower `-mcd` values (more circular buffers)
- **Complex reverbs, filter banks**: May need higher `-mcd` values (preserve roll semantics)
- **Variable/modulated delays**: Always use circular buffers regardless of `-mcd`

#### How to Detect if `-mcd` is Breaking Your DSP

Several methods can help you determine if your `-mcd` setting is causing problems:

**1. Impulse Response Testing** (Most Reliable)
Compare impulse responses with different `-mcd` values:

```bash
# Test with current setting
./build/bin/faust -lang jax -mcd 16 mydsp.dsp -o mydsp_mcd16.py
python3 mydsp_mcd16.py > output_mcd16.txt

# Test with higher setting (more roll operations)
./build/bin/faust -lang jax -mcd 64 mydsp.dsp -o mydsp_mcd64.py
python3 mydsp_mcd64.py > output_mcd64.txt

# Compare outputs
diff output_mcd16.txt output_mcd64.txt
```

If the outputs differ, your DSP is likely breaking with the lower `-mcd` value.

**2. Listen for Audio Artifacts**
Process real audio and listen for:
- **Metallic/digital distortion** - Often indicates incorrect delay line behavior
- **Pitch shifting** - Can happen when circular buffer indexing is wrong
- **Loss of reverb tail** - Common in complex reverb algorithms
- **Unstable oscillations** - May indicate feedback loop issues

**3. Check for Mathematical Instability**
Monitor for numerical issues:

```python
# Add to your DSP processing loop
outputs = model.apply(variables, inputs)

# Check for NaN/Inf
if jnp.any(jnp.isnan(outputs)) or jnp.any(jnp.isinf(outputs)):
    print("WARNING: NaN or Inf detected - likely unstable!")

# Check for explosive growth
if jnp.max(jnp.abs(outputs)) > 100.0:
    print("WARNING: Output magnitude very large - possible instability!")
```

**4. DSP-Specific Guidelines**

**Increase `-mcd` (use more roll operations) if your DSP has:**
- Complex filter matrices or nested feedback
- Recursive structures with interdependent delays
- Reverb algorithms (especially algorithmic reverbs)
- Physical modeling with complex delay networks
- Variable delay lines with feedback

**Keep `-mcd` low (more circular buffers) if your DSP has:**
- Simple delay effects (echo, chorus, flanger)
- Long delay lines without complex feedback
- Pure feedforward processing
- Granular synthesis
- Simple comb filters

**5. Systematic Testing Approach**

```bash
# Start conservative (high mcd)
./build/bin/faust -lang jax -mcd 64 mydsp.dsp -o test.py
python3 test.py > baseline.txt

# Test progressively lower values
for mcd in 32 16 8 4; do
    ./build/bin/faust -lang jax -mcd $mcd mydsp.dsp -o test.py
    python3 test.py > test_$mcd.txt
    
    if ! diff baseline.txt test_$mcd.txt > /dev/null; then
        echo "DSP breaks with -mcd $mcd"
        break
    else
        echo "-mcd $mcd works fine"
    fi
done
```

**6. Look at Generated Code**
Examine the generated Python to understand what's happening:

```bash
# Generate with verbose output to see delay line decisions
./build/bin/faust -lang jax -mcd 16 mydsp.dsp -o mydsp.py

# Look for patterns in the generated code
grep -n "jnp.roll\|_idx" mydsp.py
```

**Red flags in generated code:**
- Small arrays (≤16) using circular buffer patterns instead of `jnp.roll`
- Complex interdependent delay access patterns
- Feedback loops involving circular buffer variables

**7. Reference Implementation Comparison**
Compare against the C++ version:

```bash
# Generate C++ version
./build/bin/faust -lang cpp mydsp.dsp -o mydsp.cpp

# Compare impulse responses between C++ and JAX versions
# (This requires building and running the C++ version)
```

**Quick Rule of Thumb:**
- **Start with `-mcd 16`** (the default)
- **If you hear artifacts or see different outputs**, increase to `-mcd 32` or `-mcd 64`
- **If everything sounds identical**, you can try decreasing to `-mcd 8` for better performance
- **For complex reverbs/physical models**, consider `-mcd 64` or higher

The key insight is that if your DSP works correctly with a higher `-mcd` value but breaks with a lower one, the circular buffer optimization is interfering with delay line semantics that your algorithm depends on.

### Single-Sample Delay Optimization

The JAX backend includes a special optimization for single-sample delays (`x'` in Faust), which are very common in DSP code. Instead of using arrays with roll operations, single-sample delays are implemented as scalar state variables.

#### How it Works

For a Faust expression like `x - x'`, the standard approach would generate:
```python
# Standard approach (inefficient)
# In `initialize_carry`:
state["fVec0"] = np.zeros((2,), dtype=np.float32)  # Array of size 2
# In `tick`:
state["fVec0"] = state["fVec0"].at[0].set(input0)  # Array update
output = input0 - state["fVec0"][1]                # Array access
state["fVec0"] = jnp.roll(state["fVec0"], 1)      # Expensive roll operation
```

The optimized JAX backend generates:
```python
# Optimized approach
# In `initialize_carry`:
state["fVec0"] = np.float32(0)  # Scalar, not array
# In `tick`:
fVec0_temp = state["fVec0"]     # Store previous value
state["fVec0"] = input0          # Direct scalar update
output = input0 - fVec0_temp     # Use previous value
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

* By default, the `soundfile` primitive is *not* a learnable parameter. You can make it learnable by adding prefixing the label with "param:" such as `soundfile("param:Tango[url:{'tango.wav'}]", 2)`
* * This should be revised into `soundfile("Tango[param:1][url:{'tango.wav'}]", 2)`
* The `waveform` primitive in Faust doesn't become a Flax parameter in the generated Module. It would be useful to have a way to turn it into a learnable parameter instead of just a constant array.
* It would be useful to use metadata to enable or disable a parameter. For example, currently by default, all `hslider` are learnable, but we could have `hslider("foo[param:0]", 0.5, 0, 1, .01)` be frozen.
* We could try to generate more efficient code in the `tick` function, but it's possible that XLA is already equivalently optimizing the code for us when we use JIT. We could look at the HLO or other to investigate.
* * Try to minimize the amount of `jnp.roll` operations.
* * Avoid unnecessary casts from `bool` to `jnp.int32`
