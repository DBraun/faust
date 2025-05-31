# JAX Backend Tests

This directory contains tests for the Faust JAX backend using the minimal.py architecture file.

## Structure

- `dsp/` - Directory containing DSP test files
- `generated/` - Directory for generated Python files and test results (created by Makefile)
- `Makefile` - Build and test automation

## Usage

### Running Tests

Run all tests:
```bash
make
# or
make test
```

Test a specific DSP file:
```bash
make test-simple_gain    # Compile and test simple_gain.dsp
make test-stereo_pan     # Compile and test stereo_pan.dsp
```

Only compile a DSP file without running:
```bash
make compile-simple_gain  # Only compile, don't run
```

Run a previously compiled Python file:
```bash
make run-simple_gain      # Run the already compiled simple_gain.py
```

Clean all generated files:
```bash
make clean
```

Show help and all available commands:
```bash
make help
```

### Adding New Tests

1. Add a `.dsp` file to the `dsp/` directory
2. Run `make` to test all files, or `make test-<filename>` to test just your new file

### Test Output

The Makefile will:
1. Compile each `.dsp` file in `dsp/` to a Python file using the JAX backend
2. Run each generated Python file with `--duration 0.1 --jit`
3. Report success/failure for each test
4. Provide a summary of all test results

Generated files in the `generated/` directory:
- `*.py` - Generated Python code (kept for debugging)
- `*.test` - Test result (contains "PASSED" or "FAILED: reason")
- `*.compile.log` - Compilation output
- `*.output.log` - Runtime output

Note: Generated Python files are preserved (not automatically deleted) to help with debugging.

## Options

You can customize the build with environment variables:

```bash
# Use a different Faust compiler
FAUST=../../build/bin/faust make

# Use a different Python interpreter
PYTHON=python3.11 make

# Add additional Faust options
FAUSTOPTIONS="-lang jax -a ../../architecture/jax/minimal.py -double" make
```

## Example DSP Files

Place test DSP files in the `dsp/` directory. For example:

```faust
// dsp/simple_gain.dsp
process = *(0.5);
```

```faust
// dsp/sine_oscillator.dsp
import("stdfaust.lib");
process = os.osc(440);
```

## Requirements

- Faust compiler built with JAX backend support
- Python 3 with the following packages:
  - `jax` and `jaxlib` - JAX framework
  - `numpy` - Numerical operations
  - `tqdm` - Progress bars for JIT benchmarking
  - `flax` - Neural network library used by the JAX backend
  - `librosa` (optional) - For loading audio files
  - `scipy` (optional) - For saving WAV output files

Install requirements:
```bash
pip install jax jaxlib numpy tqdm flax librosa scipy
```

## CI/CD Integration

This test suite is integrated with GitHub Actions via `.github/workflows/jax-tests.yml`. The workflow:
1. Builds Faust once and shares the build artifacts
2. Runs two parallel test jobs:
   - JAX impulse tests
   - JAX minimal architecture tests (this test suite)
3. Tests are run on Ubuntu with Python 3.11

## Troubleshooting

If a test fails:
1. Check the compilation log: `generated/<testname>.compile.log`
2. Check the runtime output: `generated/<testname>.output.log`
3. Examine the generated Python code: `generated/<testname>.py`
4. The test result file `generated/<testname>.test` will indicate if it was a compilation or runtime error

Common issues:
- **Missing function error (e.g., `fillmydspSIG0SIG0`)**: The JAX backend may have incomplete support for certain DSP primitives like waveforms
- **JIT compilation errors**: Ensure the minimal.py architecture file has correct JAX syntax
- **Import errors**: Install all required Python packages listed above
