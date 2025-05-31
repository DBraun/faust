# JAX Backend Implementation Notes

### Key Technical Details

#### JAX Arrays are Immutable
JAX arrays cannot be modified in-place. Instead of `array[i] = value`, we must use:
- `array = array.at[i].set(value)` for JAX arrays
- Or use NumPy arrays during initialization (mutable), then convert to JAX
### Building Faust

To rebuild the Faust compiler after making changes:
```bash
cd /Users/braun/GitHub/faust/build
make -j8  # Parallel build with 8 cores
```

### Debugging Approach

#### Iterative Testing Strategy
1. **Focus on specific failing tests** - Instead of running the full test suite, test individual DSP files:
   ```bash
   # Generate JAX code for a specific file
   ./build/bin/faust -lang jax dsp/table2.dsp -a archs/impulsejax.py -double > ir/jax/double/jax_table2.py
   
   # Run the test
   python3 ir/jax/double/jax_table2.py > ir/jax/double/table2.ir
   
   # Compare output
   ./filesCompare ir/jax/double/table2.ir reference/table2.ir
   ```

2. **Use debug output** to understand what's being generated:
   ```bash
   ./build/bin/faust -lang jax dsp/table2.dsp -d > /tmp/table2_debug.txt
   ```

3. **Compare with C++ backend** to understand expected behavior:
   ```bash
   ./build/bin/faust -lang cpp dsp/table2.dsp -o /tmp/table2.cpp
   ```

#### Avoiding Command Line Pollution
- Use `2>&1` to capture both stdout and stderr
- Pipe to `tail` or `head` to see relevant portions: `make test 2>&1 | tail -20`
- Use temporary files for debug output: `> /tmp/debug.txt`
- Grep for specific patterns: `grep -B5 -A5 "pattern" file`
- Use `&&` to chain successful commands: `command1 && echo "✅ Success"`

