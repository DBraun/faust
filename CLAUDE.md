# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Faust (Functional Audio Stream) is a functional programming language for real-time signal processing and synthesis. This repository contains the Faust compiler that translates DSP specifications into efficient code for various languages (C++, C, JAVA, LLVM IR, WebAssembly, JAX, Rust, etc.).

## Key Development Commands

### Building the Compiler

```bash
# Standard builds
make compiler      # Basic compiler without LLVM backend
make most          # Compiler with LLVM backend and static libraries
make developer     # Full build with all backends (recommended for development)
make -j8           # Parallel build (adjust number based on CPU cores)

# Debug build
make debug         # Creates debug build in faustdebug folder

# Platform-specific builds
make universal     # macOS universal binary mode
make native        # Revert to native mode
```

### Running Tests

```bash
# Integration tests (impulse response comparison)
cd tests
make -C impulse-tests  # Test all backends

# Other test types
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
- `jax_code_container.*` - JAX backend
- `julia_code_container.*` - Julia backend
- `template.*` - A template starting point for making a new backend

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
