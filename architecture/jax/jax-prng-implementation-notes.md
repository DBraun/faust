# JAX PRNG Implementation Notes

## Problem
The Faust noise library uses a linear congruential generator (LCG) with the formula:
```
random = +(seed) ~ *(1103515245); // where seed = 12345
```

Without PRNG detection, this would generate the following JAX code:
```python
state["iRec0"] = state["iRec0"].at[0].set(((1103515245 * state["iRec0"][1]) + 12345))
```

This is problematic because:
1. It stores state in the `state` dictionary, which is incompatible with `nn.scan`
2. It doesn't use JAX's proper PRNG system designed for parallel and reproducible random number generation
3. It prevents the use of JAX's functional programming model

## Solution: Stateless PRNG Implementation

The JAX backend now includes built-in LCG pattern detection and replacement. When the compiler detects LCG patterns, it automatically replaces them with stateless JAX PRNG calls.

### Current Implementation

The pattern detection is implemented in `/compiler/generator/jax/jax_instructions.hh` and includes:

1. **Simple LCG patterns**: Direct `no.noise` usage
2. **Chained LCG patterns**: Complex patterns like those in `no.noises(N,i)`
3. **Temporary variable tracking**: Handles intermediate calculations

When detected, the compiler generates stateless code:
```python
# JAX PRNG noise generation (replacing LCG)
noise0 = jax.random.randint(self.make_rng("rng_stream"), (), 0, 2147483647, dtype=jnp.int32)
```

### Limitations

The pattern detection is fragile and depends on specific FIR patterns. Changes to:
- `noises.lib` implementation
- Compiler optimization levels
- DSP structure

may cause the detection to fail, resulting in state-based noise generation.

See `architecture/jax/README.md` for user-facing documentation and troubleshooting.
