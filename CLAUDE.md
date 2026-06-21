# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Faust (Functional Audio Stream) is a functional programming language for real-time signal processing and synthesis. This repository contains the Faust compiler that translates DSP specifications into efficient code for various languages (C++, C, JAVA, LLVM IR, WebAssembly, NNX, Rust, etc.).

## Key Development Commands

### Building the Compiler

**IMPORTANT: Build location matters!** The Faust repository has two different Makefiles:
- **Root Makefile** (`/Makefile`) - Legacy, has targets like `make compiler`, `make developer`
- **Build directory** (`/build/Makefile`) - **RECOMMENDED**, use this for all builds

Always build from the `/build` directory:

```bash
cd build

# Standard builds (from /build directory)
make all           # Default build (no LLVM backend)
make all -j8       # Parallel build (adjust based on CPU cores)
make full          # All backends including LLVM (requires llvm-config)
make full -j8      # Parallel build with all backends

# Quiet mode (recommended for LLMs to reduce output processing)
make all -j8 -s               # Silent mode, shows only errors
make all -j8 VERBOSE=0        # Alternative quiet mode for CMake builds

# Specific targets
make faust         # Build only the Faust compiler
make staticlib     # Build libfaust static library
make dynamiclib    # Build libfaust dynamic library

# Clean builds
make clean         # Clean build artifacts
make distclean     # Deep clean (removes faustdir)
```

**From root directory (legacy):**
```bash
# These targets only work from the root directory, not /build
make compiler      # Basic compiler without LLVM backend
make most          # Compiler with LLVM backend and static libraries
make developer     # Full build with all backends
make debug         # Creates debug build in faustdebug folder

# Platform-specific (macOS only)
make universal     # macOS universal binary mode
make native        # Revert to native mode
```

**Best practice:** Use `cd build && make all -j8 -s` for fast, quiet builds.

**Troubleshooting:**
- **Error: "Cannot find program llvm-config"** - Use `make all` instead of `make full`. The `all` target builds without LLVM backend, which is sufficient for most development (including NNX backend).
- **Build time** - First build takes 3-5 minutes on modern hardware with `-j8`. Incremental rebuilds are much faster.
- **WSL2 performance** - Builds may be slower on WSL2 due to filesystem overhead. Consider using `/home/` paths instead of `/mnt/c/` for better performance.

### Running Tests

```bash
# Integration tests (impulse response comparison)
cd tests
make -C impulse-tests  # Test all backends

# NNX backend tests
cd tests/jax-tests
make test              # Run all NNX tests (use 5 minute timeout)
make test-table        # Run specific test (e.g., table.dsp)
make compile-table     # Only compile without running

# Important: Generated Python files in tests/jax-tests/generated/ are created 
# only when compiling or testing. They don't exist by default. Always:
# - Use make compile-<test> to generate the .py file before reading it
# - Or check if file exists first
# Clean generated files: rm generated/*.py generated/*.test generated/*.output.log

# IMPORTANT: When running `make test` for all NNX tests, use a 5 minute timeout
# as there are many tests and they take time to complete:
# Example: timeout=300000 (5 minutes in milliseconds)

# Other test types
cd tests
make -C compile-tests  # Compilation tests
make -C error-tests    # Error handling tests
make -C architecture-tests && ./testserver  # Architecture tests
```

### Code Quality

```bash
make format        # Format code with clang-format
make doc           # Generate Doxygen documentation
```

## Architecture Overview

### Core Compiler Structure (`/compiler`)

The compiler follows a traditional pipeline architecture:

1. **Lexer/Parser** (`/parser`) - Flex/Bison based, generates AST
2. **Box Algebra** (`/boxes`) - High-level signal flow representation
3. **Signal Processing** (`/signals`) - Converts boxes to typed signal trees
4. **Type System** (`/signals/sigtyperules.cpp`) - Infers and checks types
5. **Normalization** (`/normalize`) - Simplifies expressions
6. **Code Generation** (`/generator`) - Backend-specific code generation

### Key Components

- **Global State** (`global.hh/cpp`) - Single mutable object `gGlobal` manages compiler-wide state
- **Tree Structure** (`/tlib`) - Core data structure for AST and signals
- **Interval Analysis** (`/interval`) - Computes signal bounds for optimization
- **Architecture Files** (`/architecture`) - Platform-specific runtime templates

### Backend Code Generators

Located in `/generator`:
- `cpp_code_container.*` - C++ backend
- `c_code_container.*` - C backend  
- `llvm_code_container.*` - LLVM IR backend
- `wasm_*_code_container.*` - WebAssembly backends
- `rust_code_container.*` - Rust backend
- `nnx_code_container.*` - NNX backend (with circular buffer optimization)
- `julia_code_container.*` - Julia backend
- `template.*` - A template starting point for making a new backend

#### NNX Backend Circular Buffer Optimization

The NNX backend implements a hybrid approach for delay line optimization:

**Circular Buffers** (O(1) performance):
- Used for larger delay lines (> gMaxCopyDelay, typically > 16 samples)
- Applied to variable delay lines (e.g., modulated delays, chorus effects)
- Replaces expensive `jnp.roll` operations with modular arithmetic
- Examples: comb_delay1.dsp, echo.dsp, freeverb.dsp, smoothdelay.dsp

**Roll Operations** (preserved for compatibility):
- Used for small recursive delay arrays (≤ 16 samples)
- Maintained for IIR filters where shift semantics are integral
- Ensures compatibility with complex filter feedback structures
- Examples: tf_exp.dsp (biquad sections), vcf_wah_pedals.dsp, zita_rev1.dsp

**Implementation files:**
- `compiler/generator/instructions_compiler_nnx.cpp/hh` - Compiler logic
- `compiler/generator/nnx/nnx_instructions.hh` - Visitor for array access conversion

### Adding New Features

When adding language features (see `adding-feature.md`):

1. **Lexer/Parser**: Add tokens in `faustparser.y` and patterns in `faustlexer.l`
2. **Signal Constructors**: Define in `signals.cpp/hh` with global symbols
3. **Type Rules**: Update `sigtyperules.cpp` for type inference
4. **Code Generation**: Modify backend-specific files in `/generator`
5. **Additional Updates**:
   - `subsignals.cpp` - Signal extraction
   - `sigToGraph.cpp` - Signal visualization
   - `ppbox.cpp` - Pretty printing

## Important Conventions

### Memory Management
- Uses custom `Tree` smart pointers with reference counting
- Global state in `gGlobal` object to avoid true globals

### Error Handling
- Errors reported via `errormsg.hh` functions
- Compilation stops on first error in most cases

### Testing Philosophy
- Impulse response tests compare output against reference files
- Tests should cover all backends when adding features

### Building Faust
- **ALWAYS** run `make` from the `/build` directory, not the root directory
- Use `cd build && make all -j8 -s` for fast, quiet parallel builds
- The `-s` flag enables silent mode, reducing output by ~95% (helpful for LLMs)
- Build artifacts go to `build/bin/` (compiler) and `build/lib/` (libraries)

## Development Workflow

### Branch Management
- `master` - Stable releases only
- `master-dev` - Main development branch (use for PRs)
- Feature branches - Name as `feature.description.NNN`

### Submodule Updates
```bash
# Update all submodules
git submodule update --remote --merge

# Update specific submodules
git submodule update --remote --merge libraries
git submodule update --remote --merge tools/faust2ck
```

### Common Development Tasks

**Debugging Parser Issues:**
```bash
# Regenerate parser files
cd compiler/parser
make clean
make
```

**Testing Single DSP File:**
```bash
# Help info
./build/bin/faust --help

# Quick compile test
./build/bin/faust test.dsp

# Use cpp backend, architecture directory, architecture file, library include directory, and write to file
./build/bin/faust -lang cpp -A architecture -a alsa-gtk.cpp test.dsp -I libraries -o test.cpp
```

## Version Information

Current version: 2.80.7 (defined in `/version` and Makefiles)

Version updates require changes to:
- `/version`
- `/Makefile`
- `/build/Makefile`