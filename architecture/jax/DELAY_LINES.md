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

# Default behavior sends an impulse to all input channels
python3 mydsp_mcd16.py -o output_mcd16.wav

# Test with higher setting (more roll operations)
./build/bin/faust -lang jax -mcd 64 mydsp.dsp -o mydsp_mcd64.py
python3 mydsp_mcd64.py -o output_mcd64.wav

# Then use numpy to compare the wav files.
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
python3 test.py -o baseline.wav

# Test progressively lower values
for mcd in 32 16 8 4; do
    ./build/bin/faust -lang jax -mcd $mcd mydsp.dsp -o test.py
    python3 test.py -o test_$mcd.wav
    
    # todo: explain how to use numpy to compare wav files.
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
