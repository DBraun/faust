# Faust JAX Backend Documentation

## Table of Contents

- [Installing JAX Dependencies](#installing-jax-dependencies)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Real-time Processing](#real-time-processing)
- [NNX vs Linen](#nnx-vs-linen)
  - [Generated-Module API Stability](#generated-module-api-stability)
- [Features](#features)
  - [Random Number Generation](#random-number-generation)
- [Performance Optimizations](#performance-optimizations)
  - [Delay Line Optimization](#delay-line-optimization)
  - [Benchmarking Tools](#benchmarking-tools)
- [Testing](#testing)
- [Polyphony Support](#polyphony-support)
- [Saving and Loading Parameters](#saving-and-loading-parameters)
  - [Scope: parameters only, not training checkpoints](#scope-parameters-only-not-training-checkpoints)
  - [Functional training with outer `jax.jit`](#functional-training-with-outer-jaxjit)
- [Troubleshooting](#troubleshooting)
- [Available Architecture Files](#available-architecture-files)
  - [UI Element Handlers](#ui-element-handlers)
  - [Soundfile Handling Note](#soundfile-handling-note)
- [Installation](#installation)
- [Differentiable DSP (DDSP) and Gradient-Based Optimization](#differentiable-dsp-ddsp-and-gradient-based-optimization)
  - [Parameter Types and Differentiability](#parameter-types-and-differentiability)
  - [Basic Gradient Example](#basic-gradient-example)
  - [Gradient Flow Through Parameter Constraints](#gradient-flow-through-parameter-constraints)
  - [Discrete Parameters and Gumbel-Softmax](#discrete-parameters-and-gumbel-softmax)
  - [Best Practices for DDSP](#best-practices-for-ddsp)
- [Parameter Naming: Zones vs Labels](#parameter-naming-zones-vs-labels)
  - [Parameter Representations: Normalized vs Physical](#parameter-representations-normalized-vs-physical)
- [Batch Processing and RL Integration](#batch-processing-and-rl-integration)
- [Limitations and Todos](#limitations-and-todos)

The JAX backend allows Faust to generate Python code that uses JAX and Flax for efficient numerical computation with automatic differentiation support. Two framework variants are available:

- **`-lang nnx`** (NNX) — Uses Flax NNX (`nnx.Module`). Modern, recommended for new projects.
- **`-lang linen`** (Linen) — Uses Flax Linen (`nn.Module`). For compatibility with existing Linen codebases.

Both backends share the same code generator (~95% shared) and produce identical numerical results. The key difference is the module framework and how parameters/state are routed.

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
# NNX (default, modern)
./build/bin/faust -lang nnx -I libraries my_example.dsp -cn MyExample -o my_example.py

# Linen (legacy framework compatibility)
./build/bin/faust -lang linen -I libraries my_example.dsp -cn MyExample -o my_example.py
```

Options:
- `-lang nnx`: Use the JAX/NNX backend
- `-lang linen`: Use the JAX/Linen backend
- `-I libraries`: Include path for Faust libraries (you can omit this entirely if `faust` has been fully installed)
- `-a architecture/jax/minimal.py`: NNX architecture file (use `-a jax/minimal.py` if `faust` has been fully installed)
- `-a architecture/jax/minimal_linen.py`: Linen architecture file
- `-cn MyExample`: Sets the class name (default is `mydsp`)
- `-o my_example.py`: Specifies the output file

The generated code creates a Flax module (`nnx.Module` or `nn.Module`) that can be used in JAX programs. You can verify its basic execution with:
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

# For generators, pass zero channels but with a valid length
assert input_audio.shape[0] == 0
output_audio = model(input_audio)

# If the DSP contains bargraphs (hbargraph/vbargraph), opt in with return_bargraphs=True:
model = MyExample(sample_rate=sample_rate, return_bargraphs=True, rngs=rngs)

# NNX: per-sample bargraph values are returned as extra outputs
output_audio, bargraph_data = model(input_audio)
# bargraph_data is a dict mapping zone names to arrays of shape (num_samples,)
# With process_block:
outputs, new_carry, bargraph_data = model.process_block(carry, input_audio)
# The tuple size is always consistent: with return_bargraphs=True, bargraph_data
# is always returned (empty dict {} if the DSP has no bargraphs).
# With return_bargraphs=False (default), bargraph_data is never returned.

# Linen: bargraph values live in the carry dict (last sample's value only)
outputs, new_carry = model.process_block(carry, input_audio)
level = new_carry["fHbargraph0"]  # scalar: value from the last sample
```

**How it works:**

**`__call__(x: jnp.ndarray, unroll: int)`**: Basic offline audio processing without receiving and returning a carry state
- `x`: Input audio tensor of shape `(num_inputs, num_samples)` or `None` for generators
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
UNROLL = 1

# Initialize carry state
carry = model.initialize_carry()

# JIT compile the process method
@partial(jax.jit, donate_argnums=(0,))
def process_block_jit(carry, inputs: jnp.ndarray):
   outputs, new_carry = model.process_block(
      carry,
      inputs,
      unroll=UNROLL,
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

1. **`process_block(carry, inputs: jnp.ndarray, unroll: int = 1)`**: Processes one block of audio
   - `carry`: State dictionary from previous block
   - `inputs`: Input block of shape `(num_inputs, block_size)`
   - `unroll`: Unroll size for `jax.lax.scan`
   - Returns: `(outputs, new_carry)` by default, or `(outputs, new_carry, bargraph_data)` when `return_bargraphs=True`

## NNX vs Linen

Both backends produce the same numerical output and share the same architecture file API. Choose based on your project's framework:

| | NNX (`-lang nnx`) | Linen (`-lang linen`) |
|---|---|---|
| Module base class | `nnx.Module` | `nn.Module` |
| Architecture file | `minimal.py` | `minimal_linen.py` |
| Parameter storage | `params["fHslider0"]` / `state["fRec0"]` | `state["fHslider0"]` / `state["fRec0"]` |
| Tick signature | `tick(self, params, state, inputs, rng)` | `tick(self, state, inputs, rng)` |
| Scan function | `nnx.scan` | `jax.lax.scan` |
| Parameter wrappers | `nnx.Param`, `nnx.Variable` | Plain attributes |
| RNG management | `nnx.Rngs` (built-in) | `nnx.Rngs` (standalone utility) |
| Bargraph output | Per-sample dict via `return_bargraphs=True` | Last value in carry (`new_carry["fHbargraph0"]`) |

**Key difference**: NNX separates UI parameters into a `params` dict and state variables into a `state` dict. Linen puts everything in a single `state` dict (params are merged into the carry before the scan loop).

**When to use Linen**: If your existing codebase uses Flax Linen (`nn.Module`) and you want to integrate Faust DSPs without migrating to NNX.

### Generated-Module API Stability

Downstream projects write wrappers and base classes that mirror the generated
module's interface so that hand-written JAX processors and Faust-compiled ones
are interchangeable. The following surface should therefore be treated as a
contract and changed only deliberately:

- Constructor: `MyDSP(sample_rate=..., rngs=...)` plus the `faust_float`,
  `soundfile_dirs`, `use_magic_clamp`, and `return_bargraphs` keywords
- Processing: `__call__(inputs, params=..., normalized_params=..., ...)` and
  `initialize_carry()` / `process_block(carry, inputs, ...)`
- Introspection and addressing: `num_inputs`, `num_outputs`,
  `get_parameter_metadata()`, `label_to_zone()`, `params_from_labels()`,
  `with_defaults()`, `unnormalize()` / `unnormalize_params()`
- Zone naming (`fHslider0`, `fEntry0`, ...) and the normalized `[0, 1]`
  parameter convention

## Features

- **JIT Compilation**: Generated code is compatible with `jax.jit` and `nnx.jit` for performance
- **Automatic Differentiation**: Can be used with `jax.grad`, `nnx.value_and_grad` and other JAX/Flax [transformations](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html)
- **Magic-Clamp Gradients**: Conditional straight-through estimator enabled by default for better parameter optimization (can be disabled by initializing the DSP Module with `use_magic_clamp=False`)
- **Vectorization**: Compatible with `jax.vmap` and `nnx.vmap` for batch processing
- **State Management**: Proper handling of delays and stateful operations
- **RNG Support**: Compatible with Flax's RNG system for stochastic DSPs

### Random Number Generation

The JAX backend supports native JAX random number generation through foreign functions:

| Function | Distribution | Range/Parameters |
|----------|--------------|------------------|
| `random_uniform()` | Uniform | U(0, 1) |
| `random_normal()` | Normal/Gaussian | N(0, 1) |
| `random_exponential(rate)` | Exponential | rate λ (mean = 1/λ) |
| `random_bernoulli(p)` | Bernoulli | probability p of returning 1 |
| `random_beta(a, b)` | Beta | shape parameters α, β in [0, 1] |

**Example usage:**

```faust
import("stdfaust.lib");

// Declare foreign functions
random_uniform = ffunction(float random_uniform(), <math.h>, "");
random_normal = ffunction(float random_normal(), <math.h>, "");
random_exponential = ffunction(float random_exponential(float), <math.h>, "");
random_bernoulli = ffunction(float random_bernoulli(float), <math.h>, "");
random_beta = ffunction(float random_beta(float, float), <math.h>, "");

// White noise in [-1, 1]
white_noise = 2 * random_uniform - 1;

// Gaussian noise
gaussian_noise = random_normal;

// Exponential with slider-controlled rate
rate = hslider("rate", 1.0, 0.1, 10.0, 0.1);
exp_noise = random_exponential(rate);

process = white_noise, gaussian_noise;
```

**Benefits:**
- Proper JAX PRNG state management (stateless, splittable)
- Reproducibility with seed control
- Compatibility with JAX transformations (`vmap`, `jit`, `grad`)

**Note**: Standard Faust noise functions like `no.noise` still use LCG for compatibility.

## Performance Optimizations

### Delay Line Optimization

The JAX backend implements intelligent delay line optimization to minimize expensive `jnp.roll` operations. This is controlled by the `-mcd` (max copy delay) compiler flag.

#### The `-mcd` Flag

The `-mcd` flag determines when to use circular buffers vs roll operations:

```bash
# Default: delays ≤16 use roll, delays >16 use circular buffers
./build/bin/faust -lang nnx mydsp.dsp -o mydsp.py

# Force more delays to use roll operations (may reduce performance)
./build/bin/faust -lang nnx -mcd 64 mydsp.dsp -o mydsp.py

# Force more delays to use circular buffers (may improve performance)
./build/bin/faust -lang nnx -mcd 8 mydsp.dsp -o mydsp.py
```

#### Implementation Strategies

**Roll Operations** (for delays ≤ `-mcd`):
```python
# Initialization
state["fVec0"] = np.zeros((delay+1,), dtype=np.float64)

# Per-sample tick
state["fVec0"] = state["fVec0"].at[0].set(inputs[0])  # Write at position 0
_result0 = state["fVec0"][delay]                      # Read from fixed position
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

Run the backend impulse tests:

```bash
cd tests/impulse-tests

# NNX backend
make nnx

# Linen backend
make linen
```

### Impulse Test Architecture

The `tests/impulse-tests/archs/impulsennx.py` (NNX) and `tests/impulse-tests/archs/impulsejax_linen.py` (Linen) files are specialized architectures for impulse response testing. They have specific requirements:

1. **Output Formatting**: Must match the reference format exactly with proper spacing:
   ```
   number_of_inputs  :   1    # 3 spaces after colon
   number_of_outputs :   1    # 3 spaces after colon  
   number_of_frames  :   60000 # 3 spaces after colon
   ```

2. **Soundfile Workaround**: Like `minimal.py`, it includes the soundfile state workaround (see [Soundfile Handling Note](#soundfile-handling-note)).

## Polyphony Support

The JAX backend supports polyphonic DSPs. Polyphony in JAX is naturally handled using `jax.vmap` for efficient vectorized processing across multiple voices.

### Widget Modulation

Faust's **widget modulation** lets you turn UI parameters into signal inputs. The named-route syntax replaces each widget with a signal input using a `replace = !,_;` identity:

```faust
import("stdfaust.lib");

freq = hslider("freq", 440, 20, 20000, 1);
gain = hslider("gain", 0.5, 0, 1, 0.01);
gate = hslider("gate", 0, 0, 1, 1);

synth = os.osc(freq) * en.ar(0.05, 2.0, gate) * gain;

replace = !,_;
process = ["freq": replace, "gain": replace, "gate": replace -> synth];
```

Each `"name": replace` entry drops the widget's default value and substitutes a signal input. The compiled module has `num_inputs=3` (freq, gain, gate) and no UI parameters.

Compile with:

```bash
faust -lang nnx poly_synth.dsp -cn SynthVoice -o poly_synth.py
```

### Single Voice

Instantiate the module and run a single voice:

```python
import jax
import jax.numpy as jnp
from jax import random
from poly_synth import SynthVoice

sample_rate = 44100
num_frames = 1024

model = SynthVoice(sample_rate=sample_rate)
state = model.initialize_carry()

# inputs: (num_inputs, num_frames) — [freq, gain, gate]
inputs = jnp.stack([
    jnp.full(num_frames, 440.0),   # freq
    jnp.full(num_frames, 0.8),     # gain
    jnp.full(num_frames, 1.0),     # gate
])

audio, state = model.process_block(state, inputs)  # audio shape: (1, num_frames)
```

### Polyphony with `jax.vmap`

Use `jax.vmap` to vectorize `process_block` across N voices in parallel, then sum the outputs.

> **Note:** `process_block` uses Flax NNX's `Rngs` internally, which has mutable state that cannot live inside `jax.vmap`. Pass explicit JAX random keys instead — `process_block` accepts a raw `jax.Array` key and passes it through without mutation.

```python
num_voices = 4

# Create one model, replicate state for each voice
model = SynthVoice(sample_rate=sample_rate)
states = jax.tree.map(lambda x: jnp.stack([x] * num_voices), model.initialize_carry())

# Per-voice inputs: (num_voices, num_inputs, num_frames)
freqs = jnp.array([261.63, 329.63, 392.00, 523.25])  # C major chord
voice_inputs = jnp.stack([
    jnp.stack([jnp.full(num_frames, f),
               jnp.full(num_frames, 0.5),
               jnp.full(num_frames, 1.0)])
    for f in freqs
])  # shape: (4, 3, num_frames)

# Per-voice RNG keys (explicit JAX keys avoid Flax Rngs mutation inside vmap)
rng_keys = random.split(random.key(0), num_voices)

# Wrap process_block to pass the RNG key explicitly
def poly_step(carry, inputs, rng_key):
    return model.process_block(carry, inputs, rngs=rng_key)

voices, states = jax.vmap(poly_step)(states, voice_inputs, rng_keys)
# voices: (num_voices, 1, num_frames)

# Sum voices for polyphonic output
output = jnp.sum(voices, axis=0)  # shape: (1, num_frames)
```

Because `vmap` compiles into a single fused kernel, this runs at near-constant cost regardless of voice count on GPU/TPU.

### Polyphony under outer `jax.jit` (and backprop into shared params)

The closure example above bakes the model's parameters in as constants. For
training — or simply to follow the "outer `jax.jit`" convention — split the model
once into `(graphdef, state)` and run a plain `jax.jit` that merges inside. The
merged model is a *free variable* of `poly_step`, so its parameters are **shared
(broadcast) across all voices** while `carry`/`inputs`/`rng` are mapped:

```python
from flax import nnx

graphdef, params, rest = nnx.split(model, nnx.Param, ...)  # params = shared, learnable

@jax.jit
def poly_render(params, carries, voice_inputs, rng_keys):
    m = nnx.merge(graphdef, params, rest)
    def poly_step(carry, inputs, rng_key):
        return m.process_block(carry, inputs, rngs=rng_key)
    voices, _ = jax.vmap(poly_step, in_axes=(0, 0, 0))(carries, voice_inputs, rng_keys)
    return jnp.sum(voices, axis=0)  # polyphonic mix: (num_outputs, num_frames)
```

Because the same `params` feed every voice, **gradients accumulate** the per-voice
contributions onto the one shared parameter set — exactly what you want when
fitting a polyphonic synth:

```python
def loss(params, carries, voice_inputs, rng_keys, target):
    return jnp.mean((poly_render(params, carries, voice_inputs, rng_keys) - target) ** 2)

grads = jax.grad(loss)(params, carries, voice_inputs, rng_keys, target)
# reverse-mode AD flows through jax.vmap and the inner scan; one optimizer step on
# `params` (e.g. optax) lowers the loss. The trained params persist with
# save_params / load_params below, independent of the voice count.
```

## Saving and Loading Parameters

Trained (or edited) parameters can be written to a portable, framework-agnostic
[`safetensors`](https://github.com/huggingface/safetensors) file and reloaded
later. Faust UI parameters are commonly 0-dim scalars, which safetensors cannot
store directly; the helpers reshape such leaves to `(1,)` on save and restore their
original shape on load (the affected keys are recorded in the file metadata).

**NNX** (`minimal.py`) — instance methods on the model (state lives in the module):

```python
model.save_params("synth.safetensors")   # writes every nnx.Param leaf

fresh = mydsp(sample_rate=44100, rngs=nnx.Rngs(0))
fresh.load_params("synth.safetensors")    # in-place; same compiled DSP structure
```

`save_params` also records the DSP class name and the zone-to-label map (e.g.
`fHslider0` → `"Synth/Oscillator/Frequency"`) in the file's metadata header.
By default `load_params` verifies that the file's parameter names *and* UI
labels exactly match the model, and raises `UnknownParameterError` otherwise.
This matters because zone names are generic — almost every DSP has an
`fHslider0` — so without the check, a file saved from a different DSP would
load without complaint and silently produce wrong parameter values. To
deliberately transfer parameters between related DSP variants, opt out with:

```python
model.load_params("other_dsp.safetensors", strict=False)  # loads only overlapping params
```

**Linen** (`minimal_linen.py`) — module-level functions (params are external):

```python
from my_compiled_dsp import save_params, load_params

variables = model.init(jax.random.key(0), x)
save_params(variables, "synth.safetensors")

# Recommended: pass a reference pytree so a file from the wrong DSP is rejected
variables = load_params("synth.safetensors", expected=model.init(jax.random.key(0), x))
y = model.apply(variables, x)
```

### Scope: parameters only, not training checkpoints

The safetensors helpers are a *distribution* format: they persist the learnable
parameters and nothing else, so a file is small, portable, and readable from any
framework. They are deliberately **not** a training-checkpoint system — they do
not save optimizer state, learning-rate schedules, RNG streams, or data-iterator
positions, and the Faust repository does not provide a training loop. For
resumable training, manage full checkpoints in your own code with a dedicated
library such as [Orbax](https://orbax.readthedocs.io/) (save
`nnx.state((model, optimizer))`), and export to safetensors with `save_params`
when you want to publish or share the trained result.

### Functional training with outer `jax.jit`

NNX modules can be trained with a plain `jax.jit` (faster than `nnx.jit`, which
traverses the module graph in Python on every call — see the
[Flax performance guide](https://flax.readthedocs.io/en/latest/guides/performance.html)).
Split the model and optimizer **once**, then merge inside the jitted step and return
the updated `nnx.state`:

```python
import optax
from functools import partial

optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
graphdef, state = nnx.split((model, optimizer))   # once, before the loop

@partial(jax.jit, donate_argnums=(1,))
def train_step(graphdef, state, batch):
    model, opt = nnx.merge(graphdef, state)
    loss, grads = nnx.value_and_grad(lambda m: loss_fn(m, batch))(model)
    opt.update(model=model, grads=grads)
    return nnx.state((model, opt)), loss

for batch in dataset:
    state, loss = train_step(graphdef, state, batch)

nnx.update((model, optimizer), state)   # sync the live objects when done
```

## Troubleshooting

### Performance

For optimal performance:
- Prefer an outer `jax.jit` over `nnx.jit` (split once, merge inside — see [Functional training with outer `jax.jit`](#functional-training-with-outer-jaxjit)).
- Consider using GPU acceleration with large batch sizes.

## Available Architecture Files

### `minimal.py` (NNX)
Architecture for `-lang nnx`. Uses Flax NNX (`nnx.Module`). Includes:
- All UI element handlers (sliders, buttons, soundfiles, etc.)
- Basic `__call__` method for processing
- Generator support (0-input DSPs)
- Soundfile loading with librosa
- `initialize_carry()`: Creates initial state for block-wise processing
- `process_block()`: Processes audio block-by-block with state management
- Real-time audio streaming example using `sounddevice`

### `minimal_linen.py` (Linen)
Architecture for `-lang linen`. Uses Flax Linen (`nn.Module`). Same feature set as `minimal.py` with Linen-specific adaptations:
- Uses `jax.lax.scan` instead of `nnx.scan`
- Parameters stored as plain attributes (no `nnx.Param` wrappers)
- Single `state` dict for all variables (no params/state split)
- Uses `nnx.Rngs` as a standalone RNG utility

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

### Soundfile Handling Note

**Known Issue**: The compiler generates code expecting soundfiles in the `state` dictionary, but `_initialize_carry()` doesn't include them. Architecture files work around this by manually adding soundfiles to state in `initialize_carry()`. This is suboptimal since soundfiles are immutable and shouldn't be in the carry state. The compiler should be fixed to access soundfiles directly from `self`.

## Installation

### Prerequisites

1. Python 3.11 or later
2. Faust compiler built with JAX backend support (included in regular builds)

### Building Faust with JAX Backend

To build Faust with only the JAX backend — that is, both the `nnx` and `linen`
variants (`NNX_BACKEND` + `LINEN_BACKEND`) and no other backends:

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

The Faust executable will be in `build/bin/faust`. Verify backends are included:

```bash
./build/bin/faust -v
# Should show "DSP to NNX" and "DSP to Linen" in the embedded backends list
```

## Differentiable DSP (DDSP) and Gradient-Based Optimization

The JAX backend supports automatic differentiation, enabling gradient-based optimization of DSP parameters. This is useful for applications like:
- Neural audio synthesis (learning synthesis parameters from examples)
- Adaptive audio effects (optimizing effect parameters for specific inputs)
- System identification (learning filter coefficients to match target responses)
- Differentiable audio processors in neural network pipelines

### Parameter Types and Differentiability

All Faust UI parameters become learnable by default (as `nnx.Param` in NNX, or plain attributes in Linen):

- **Sliders** (`hslider`, `vslider`): Continuous parameters normalized to [0, 1]
  - Support linear, logarithmic, and exponential scaling
  - Clipped to valid range before unnormalization
- **Number Entries** (`nentry`): Discrete parameters using Gumbel-softmax (see [Discrete Parameters](#discrete-parameters-and-gumbel-softmax))
- **Buttons/Checkboxes**: Binary parameters (0 or 1)
- **Soundfiles**: Audio buffers that can optionally be made learnable
  - Non-learnable by default: `soundfile("Tango[url:{'tango.wav'}]", 2)`
  - Learnable with `[param:1]`: `soundfile("Tango[param:1][url:{'tango.wav'}]", 2)`

### Basic Gradient Example

```python
import jax.numpy as jnp
from flax import nnx
import optax
from my_filter import MyFilter  # Compiled Faust DSP with parameters

# Initialize model and optimizer
rngs = nnx.Rngs(0, params=42, rng_stream=0)
model = MyFilter(sample_rate=48000, faust_float=jnp.float32, rngs=rngs)
model.train()
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)

# Prepare data
inputs = jnp.zeros((model.num_inputs, 4096))       # Input signal
target_output = jnp.zeros((model.num_outputs, 4096))  # Target signal

# JIT-compiled train step (recommended pattern)
@nnx.jit
def train_step(model, optimizer, inputs, target):
    def loss_fn(model):
        outputs = model(inputs)
        return jnp.mean((outputs - target) ** 2)  # MSE loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Training loop
for step in range(1000):
    loss = train_step(model, optimizer, inputs, target_output)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")
```

### Gradient Flow Through Parameter Constraints

Faust parameters are bounded (e.g., slider ranges). The JAX backend enforces these constraints using clipping.

**Magic-Clamp (Default):**

By default, the JAX backend uses a ["magic-clamp"](https://github.com/ben-hayes/magic-clamp) conditional straight-through estimator that improves gradient flow:

- **Inside bounds**: Gradients flow normally
- **Outside bounds moving toward valid region**: Gradients flow (allows recovery)
- **Outside bounds moving further away**: Gradients are zeroed

This prevents optimization from getting stuck when parameters start outside bounds or hit boundaries during training.

**Standard Clipping (Optional):**

You can disable magic-clamp if you prefer standard JAX clipping behavior:

```python
# Disable magic-clamp
model = MyDSP(sample_rate=48000, use_magic_clamp=False)
```

With standard clipping:
- **Inside bounds**: Gradients flow normally
- **At/beyond bounds**: Gradients are zeroed

This can cause optimization issues when parameters hit their bounds, as learning stops.

**When Magic-Clamp Helps Most:**
- Parameters initialized outside valid range
- Aggressive learning rates that overshoot bounds
- Multi-parameter optimization with coupling effects
- Transfer learning where pretrained values may be out of range

The implementation uses `jax.custom_vjp` to define a conditional straight-through estimator. See `minimal.py` for the full implementation.

### Discrete Parameters and Gumbel-Softmax

The `nentry` primitive generates discrete parameters using Gumbel-softmax:

- **Training**: Uses soft sampling for gradient flow (requires `nentry` RNG stream)
- **Inference**: Uses hard argmax for deterministic selection

To enable Gumbel-softmax during training:

```python
# Create RNGs with nentry (named to avoid collision with jax.random.gumbel)
rngs = nnx.Rngs(0, params=42, rng_stream=0, nentry=123)

# The model will automatically use Gumbel-softmax when in training mode and if nentry RNG is available
model = MyDSP(sample_rate=48000, rngs=rngs)
model.train()
```

#### Configuring Temperature

The temperature parameter (`tau`) controls the softness of the sampling:
- High temperature (τ > 1): Softer, more uniform sampling
- Low temperature (τ → 0): Harder, closer to argmax

You can configure temperature using metadata:

```faust
// Fixed temperature at 1.0 (default)
mode = nentry("mode", 0, 0, 3, 1);

// Fixed temperature with custom initial value
mode = nentry("mode[tau_init:2.0]", 0, 0, 3, 1);

// Learnable temperature (optimizable via gradient descent)
mode = nentry("mode[tau:learnable]", 0, 0, 3, 1);

// Learnable with custom initial value
mode = nentry("mode[tau:learnable][tau_init:0.5]", 0, 0, 3, 1);
```

### Future Enhancements

The following features are planned for better DDSP support:

1. **Parameter freezing**: Selectively freeze parameters
   ```faust
   // Example syntax (not yet implemented - requires compiler support)
   fixed_param = hslider("Fixed[param:0]", 0.5, 0, 1, 0.01);
   ```

3. **Metadata-driven gradient config**: Use UI metadata to specify gradient behavior per-parameter
   ```faust
   // Example syntax (not yet implemented - requires compiler support)
   cutoff = hslider("Cutoff[gradient:conditional_ste]", 1000, 20, 20000, 1);
   ```

### Best Practices for DDSP

1. **Normalization**: Parameters are auto-normalized to [0, 1], making optimization easier
2. **Learning rates**: Start with small learning rates (1e-4 to 1e-3) for audio parameters
3. **Gradient clipping**: Consider using gradient clipping to prevent instability
4. **JIT compilation**: Always JIT your training loop for performance
5. **Batch processing**: Process multiple examples in parallel when possible
6. **Loss functions**: Use perceptual losses (spectral, multi-scale) for audio tasks
7. **Numerical precision**: Use `float32` for most cases, `float64` if needed

### Resources

- [JAX documentation on automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Flax NNX documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)
- Hayes, B., (2025, July). Magic Clamp. [GitHub Repository] URL: https://github.com/ben-hayes/magic-clamp/
- Yang, Y., Jin, Z., Barnes, C., & Finkelstein, A. (2023, November). White Box Search Over Audio Synthesizer Parameters. In ISMIR (pp. 190-196).

## Parameter Naming: Zones vs Labels

The generated module uses two naming conventions for parameters:

### Zones (Internal Names)

**Zones** are the internal variable names generated by the Faust compiler. They follow a predictable pattern based on the UI element type:

| UI Element | Zone Pattern | Example |
|------------|--------------|---------|
| `hslider` | `fHslider0`, `fHslider1`, ... | `fHslider18` |
| `vslider` | `fVslider0`, `fVslider1`, ... | `fVslider0` |
| `nentry` | `fEntry0`, `fEntry1`, ... | `fEntry2` |
| `button` | `fButton0`, `fButton1`, ... | `fButton0` |
| `checkbox` | `fCheckbox0`, `fCheckbox1`, ... | `fCheckbox0` |

Zones are used as keys in the `params` dict passed to `__call__()`:

```python
# Using zones directly
params = {'fHslider0': 1000.0, 'fHslider1': 440.0}
audio = model(inputs, params=params)
```

### Labels (Human-Readable Names)

**Labels** (specifically `full_label`) are the human-readable UI paths defined in your Faust code. They reflect the hierarchical structure of your UI:

```faust
// Faust code
freq = hslider("Synth/Oscillator/Frequency", 440, 20, 20000, 1);
gain = hslider("Synth/Amp/Gain", 0.5, 0, 1, 0.01);
```

These become full_labels:
- `"Synth/Oscillator/Frequency"`
- `"Synth/Amp/Gain"`

### Converting Between Zones and Labels

The module provides methods to work with both naming conventions:

**`label_to_zone(label)`** - Convert a single label to its zone:
```python
zone = model.label_to_zone('Synth/Oscillator/Frequency')
# Returns: 'fHslider0'
```

**`params_from_labels(label_params)`** - Convert an entire dict from labels to zones:
```python
# Using human-readable labels (e.g., from a config file)
label_params = {
    'Synth/Oscillator/Frequency': 880.0,
    'Synth/Amp/Gain': 0.8,
}

# Convert to zone-keyed params
zone_params = model.params_from_labels(label_params)
# Returns: {'fHslider0': 880.0, 'fHslider1': 0.8}

# Use with the model
audio = model(inputs, params=zone_params)
```

**`get_parameter_metadata()`** - Get full metadata for all parameters:
```python
metadata = model.get_parameter_metadata()
# {
#     'fHslider0': {
#         'full_label': 'Synth/Oscillator/Frequency',
#         'label': 'Frequency',
#         'type': 'hslider',
#         'min': 20.0,
#         'max': 20000.0,
#         'default': 440.0,
#         ...
#     },
#     ...
# }
```

### Merging Partial Parameters with Defaults

When you only want to override some parameters, use **`with_defaults(partial_params)`**:

```python
# Only specify the parameters you care about
partial = model.params_from_labels({
    'Synth/Oscillator/Frequency': 880.0,
})

# Fill in the rest with defaults
full_params = model.with_defaults(partial)

# Now full_params contains ALL parameters
audio = model(inputs, params=full_params)
```

This method validates that all provided keys are valid zones, raising `UnknownParameterError` if not:

```python
# This raises UnknownParameterError
model.with_defaults({'invalid_zone': 1.0})
```

### When to Use Which

| Use Case | Approach |
|----------|----------|
| Quick prototyping | Use zones directly with `params=` |
| Config files / CSV data | Use `params_from_labels()` to convert |
| Partial overrides | Use `with_defaults()` to fill in missing params |
| RL / ML training | Use `normalized_params=` with zones |
| Introspection | Use `get_parameter_metadata()` |

### Parameter Representations: Normalized vs Physical

Parameters have two representations depending on how you access them:

#### Continuous Parameters (hslider, vslider, button, checkbox)

| Representation | Type | Range | Example |
|----------------|------|-------|---------|
| **Normalized** | scalar | [0, 1] | `0.5` |
| **Physical** | scalar | [min, max] | `1000.0` (for freq slider 20-20000 Hz) |

```python
# Normalized (what the module stores internally)
norm_value = model.fHslider0[...]  # e.g., 0.5

# Physical (what the DSP actually uses)
phys_params = model.unnormalize()
phys_value = phys_params['fHslider0']  # e.g., 1000.0
```

#### Discrete Parameters (nentry)

Discrete parameters use Gumbel-softmax for differentiability. Their representation is more complex:

| Representation | Type | Description |
|----------------|------|-------------|
| **Normalized** | `logits` array + `tau` scalar | Logits: one value per option; Tau: temperature |
| **Physical** | scalar | The discrete step value (integer-valued float) |

```python
# Example: nentry with 4 options (values 0, 1, 2, 3)

# Normalized representation (what the module stores)
logits = model.fEntry0_logits[...]  # e.g., [10.0, 0.0, 0.0, 0.0] → option 0 selected
tau = model.fEntry0_tau[...]        # e.g., 1.0

# Physical representation (after unnormalize)
phys_params = model.unnormalize()
phys_value = phys_params['fEntry0']  # e.g., 0.0 (the actual discrete value)

# In training mode with nentry RNG: soft value (e.g., 0.3)
# In eval mode: hard argmax (e.g., 0.0, 1.0, 2.0, or 3.0)
```

#### Quick Reference Table

| Parameter Type | Normalized Keys | Physical Key | Physical Type |
|----------------|-----------------|--------------|---------------|
| `hslider` | `fHslider0` (scalar) | `fHslider0` | scalar in [min, max] |
| `vslider` | `fVslider0` (scalar) | `fVslider0` | scalar in [min, max] |
| `button` | `fButton0` (scalar) | `fButton0` | 0.0 or 1.0 |
| `checkbox` | `fCheckbox0` (scalar) | `fCheckbox0` | 0.0 or 1.0 |
| `nentry` | `fEntry0_logits` (array) + `fEntry0_tau` (scalar) | `fEntry0` | discrete step value |

#### Passing Parameters to the Model

The `__call__` and `process_block` methods support three parameter modes:

```python
# Mode 1: Use module's own parameters (default)
outputs = model(inputs)

# Mode 2: Physical values (use params=)
physical_params = {
    'fHslider0': 1000.0,          # Continuous: actual value
    'fEntry0': 2.0,               # Discrete: step value (integer-valued)
}
output = model(inputs, params=physical_params)

# Mode 3: Normalized values (use normalized_params=)
normalized_params = {
    'fHslider0': 0.5,             # Continuous: [0, 1]
    'fEntry0_logits': jnp.array([0.0, 0.0, 10.0, 0.0]),  # Discrete: logits
    'fEntry0_tau': 1.0,           # Discrete: temperature
}
output = model(inputs, normalized_params=normalized_params)
```

## Batch Processing and RL Integration

The JAX backend supports efficient batch processing with different parameters per batch item, designed for reinforcement learning and neural network-driven synthesis.

### RL Integration

RL policies typically output:
- **Continuous parameters** in [0, 1] using Beta distribution
- **Categorical parameters** as logits (for Gumbel-softmax) or discrete values

The `normalized_params` argument accepts these directly:

```python
# Continuous policy outputs normalized params [0, 1]
continuous_output = beta_policy(observation)  # [batch, num_continuous] in [0, 1]

# Categorical policy outputs logits (for discrete choices)
categorical_logits = categorical_policy(observation)  # [batch, num_options]

# Convert to dict format
continuous_names = list(model.get_continuous_params().keys())  # defaults to normalized=True
categorical_names = list(model.get_categorical_params().keys())

continuous_batch = {name: continuous_output[:, i] for i, name in enumerate(continuous_names)}
categorical_batch = {f"{name}_logits": categorical_logits for name in categorical_names}

# Combine and batch process with vmap
normalized_batch = {**continuous_batch, **categorical_batch}

@jax.vmap
def render(norm_params, inputs, rng):
    return model(inputs, normalized_params=norm_params, rngs=rng)

audio = render(normalized_batch, inputs_batch, rngs_batch)
```

### Helper Methods for Batch Processing

**Get continuous parameters (for RL policies using Beta/Gaussian distributions):**
```python
# Normalized [0, 1] values (default)
continuous = model.get_continuous_params()
# {'fHslider0': 0.5, 'fHslider1': 0.3, ...}

# Batched normalized values
continuous = model.get_continuous_params(batch_size=4)
# {'fHslider0': [0.5, 0.5, ...], ...}

# Physical values (unnormalized)
continuous = model.get_continuous_params(normalized=False)
# {'fHslider0': 1000.0, 'fHslider1': 440.0, ...}
```

**Get categorical parameters (for discrete choices using Categorical/Gumbel-softmax):**
```python
# Returns logits and tau for each nentry
categorical = model.get_categorical_params()
# {'fEntry0': {'logits': Array([1., 0., 0., 0.]), 'tau': 1.0}}

# Batched
categorical = model.get_categorical_params(batch_size=4)
# {'fEntry0': {'logits': [[1., 0., ...], ...], 'tau': [1.0, ...]}}
```

**Unnormalize parameters:**
```python
# Convert normalized params (from RL policy) to physical values
normalized = {'fHslider0': 0.5, 'fEntry0_logits': jnp.array([2.0, 0.1])}
physical = model.unnormalize_params(normalized)
# Use with params argument: model(inputs, params=physical)
```

### Auto-Fill Behavior

When using `normalized_params`, missing parameters are automatically filled with module defaults:
- Provide only continuous params → categorical uses defaults
- Provide only categorical → continuous uses defaults
- Omit any param → uses module default for that param

### Practical Examples

See `tests/jax-tests/rl_demo/` for complete demonstrations:
- `rl_policy_demo.py` - RL training with Beta distribution policies
- `categorical_demo.py` - Handling discrete parameters with `ba.selectn`
- `synth.dsp` - Example synthesizer with both parameter types

## Limitations and Todos

* `nnx.tabulate` currently has compatibility issues with scan-based models (see [Flax issue #5067](https://github.com/google/flax/issues/5067)). The `--tabulate` flag may not work until this is resolved.
* The `waveform` primitive in Faust doesn't become a Flax parameter in the generated Module. It would be useful to have a way to turn it into a learnable parameter instead of just a constant array.
* The `[param:0]`/`[param:1]` metadata is implemented for soundfiles but not yet for sliders, buttons, or nentry. It would be useful to support `hslider("foo[param:0]", 0.5, 0, 1, .01)` to freeze individual parameters.
* We could try to generate more efficient code in the `tick` function, but it's possible that XLA is already equivalently optimizing the code for us when we use JIT. We could look at the HLO or other to investigate.
* * Avoid unnecessary casts from `bool` to `jnp.int32`
* NNX bargraph support uses a closure-based scan pattern that returns per-sample values as extra outputs, controlled by the `return_bargraphs` constructor flag (default `False`). When `True`, the return tuple always includes a bargraph dict (empty `{}` if the DSP has no bargraphs), giving a consistent return shape. Compatible with `jax.vmap`. An alternative `sow`-based approach would be cleaner but is blocked by Flax NNX `Rngs` trace-level conflicts inside `vmap` (see https://github.com/google/flax/discussions/4799). Linen bargraphs only expose the last sample's value via the carry dict; per-sample history could be added with `sow` + `variable_axes={'intermediates': 0}` (see https://github.com/google/flax/discussions/3727).
