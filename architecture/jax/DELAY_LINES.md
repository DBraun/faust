### Delay Line Implementation in the JAX Backend

The JAX backend uses three strategies for delay lines, chosen by the maximum delay depth (mxd) and the `-mcd` (max copy delay) compiler flag (default: 16):

1. **Direct element copies** for very small delays (mxd 1-2)
2. **Shift register** (`jnp.roll`) for small delays (mxd 3 to mcd-1)
3. **Circular buffers** (IOTA-indexed ring buffers) for large delays (mxd >= mcd)

The `-mcd` flag is not JAX-specific -- all Faust backends use it -- but it has important performance implications for JAX because `jnp.roll` is O(n) while direct copies and circular buffer access are O(1).

#### The Three Strategies

##### Strategy 1a: Direct Element Copy (mxd == 1 or mxd == 2)

For delays of 1 or 2 samples, the compiler generates direct `.at[].set()` copies instead of `jnp.roll`. This avoids allocating a new array for trivially small shifts.

**Example: Single-sample delay** (`_ <: _ - mem`, i.e. `y[n] = x[n] - x[n-1]`):

```python
# _initialize_carry:
state["fVec0"] = np.zeros((2,), dtype=np.float64)

# tick:
state["fVec0"] = state["fVec0"].at[0].set(input0)
output0 = input0 - state["fVec0"][1]
state["fVec0"] = state["fVec0"].at[1].set(state["fVec0"][0])  # copy [0] -> [1]
```

**Example: Biquad lowpass filter** (`fi.lowpass(2, 1000)`, mxd == 2):

A second-order IIR filter implements the difference equation:

    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

The Faust compiler restructures this into `fRec0` with delay depth 2:

```python
# _initialize_carry:
state["fRec0"] = np.zeros((3,), dtype=np.float64)

# tick (simplified, constants folded for clarity):
state["fRec0"] = state["fRec0"].at[0].set(
    input0 - C2 * (C3 * state["fRec0"][2] + C4 * state["fRec0"][1])
)
output0 = C2 * (state["fRec0"][2] + state["fRec0"][0] + 2.0 * state["fRec0"][1])
state["fRec0"] = state["fRec0"].at[2].set(state["fRec0"][1])  # copy [1] -> [2]
state["fRec0"] = state["fRec0"].at[1].set(state["fRec0"][0])  # copy [0] -> [1]
```

##### Strategy 1b: Shift Register (`jnp.roll`, mxd 3 to mcd-1)

For delays of 3 or more samples (but below `-mcd`), the compiler uses `jnp.roll` to shift the entire array by one position. The array is written at index 0, past values are read from higher indices.

```python
# For a 4th-order FIR filter (mxd == 4):
state["fVec0"] = state["fVec0"].at[0].set(input0)
output0 = 0.2*state["fVec0"][0] + 0.3*state["fVec0"][1] + ...
state["fVec0"] = jnp.roll(state["fVec0"], 1)  # shift all elements by 1
```

##### Strategy 2: Circular Buffer (IOTA-indexed)

Used when the maximum delay depth is >= `-mcd` (default: >= 16 samples).

Instead of shifting the whole array, a shared integer counter `IOTA0` tracks the current write position. The buffer is sized to the next power of two, and all index arithmetic uses bitwise AND (`& (size-1)`) for O(1) wrapping.

**Example: 1-second feedback delay**

`process = + ~ @(44100) * 0.5` implements: y[n] = x[n] + 0.5 * y[n-44100]

With a maximum delay of 44100 samples (well above the default `-mcd` of 16), the buffer is sized to 65536 (next power of two):

```python
# _initialize_carry:
state["fRec0"] = np.zeros((65536,), dtype=np.float64)
state["IOTA0"] = 0

# tick:
state["fRec0"] = state["fRec0"].at[
    (state["IOTA0"] & 65535).astype(jnp.int32)
].set(input0 + 0.5 * state["fRec0"][
    ((state["IOTA0"] - 44101) & 65535).astype(jnp.int32)
])
output0 = state["fRec0"][(state["IOTA0"] & 65535).astype(jnp.int32)]
state["IOTA0"] = state["IOTA0"] + 1
```

Key details:
- `IOTA0` is a **shared** counter across all circular buffers in the DSP, not per-array.
- The mask `& 65535` is equivalent to `% 65536` but faster.
- Reads from the past use `(IOTA0 - delay) & mask`.

**Example: Variable delay**

`process = + ~ @(delay_time) * 0.5` where `delay_time` is a slider. The maximum possible delay determines the buffer size at compile time; the actual delay is computed at runtime:

```python
# tick:
iSlow0 = jnp.int32(params["fHslider0"]) + 1
state["fRec0"] = state["fRec0"].at[
    (state["IOTA0"] & 65535).astype(jnp.int32)
].set(input0 + 0.5 * state["fRec0"][
    ((state["IOTA0"] - iSlow0) & 65535).astype(jnp.int32)
])
```

#### Mixed Usage in Real DSPs

Complex DSPs often use both strategies. For example, `freeverb.dsp` generates code with 40 `jnp.roll` calls (for its short allpass filter delays) and 50 `IOTA` references (for its longer comb filter delay lines).

#### Compiler Flag: `-mcd <size>`

```bash
# Default (16): small delays use roll, large delays use circular buffers
faust -lang jax mydsp.dsp -a architecture/jax/minimal.py

# Lower threshold: more circular buffers (better performance for large DSPs)
faust -lang jax -mcd 8 mydsp.dsp -a architecture/jax/minimal.py

# Higher threshold: more roll operations
faust -lang jax -mcd 64 mydsp.dsp -a architecture/jax/minimal.py
```

Both strategies produce identical numerical results -- the choice is purely about performance. The default of 16 is a good balance for most DSPs.

#### Verifying Correctness

The impulse test suite (`tests/impulse-tests/Make.jax`) validates that JAX output matches the C++ reference implementation for 73 DSP files. To test a specific DSP:

```bash
# Compare JAX output against C++ reference
cd tests/impulse-tests
faust -lang jax dsp/echo.dsp -a archs/impulsejax.py -double > ir/jax/double/jax_echo.py
python3 ir/jax/double/jax_echo.py > ir/jax/double/echo.ir
./filesCompare ir/jax/double/echo.ir reference/echo.ir
```
