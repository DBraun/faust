# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Faust (Functional Audio Stream) is a functional programming language for real-time signal processing and synthesis. The compiler translates DSP specifications into efficient code for various target languages and platforms.

## Build System

### Main Build Targets
- `make compiler` - Basic Faust compiler with OSC/HTTP libraries
- `make most` - Compiler with LLVM backend + static libraries  
- `make developer` - All backends + static libraries (recommended for development)
- `make all` - All backends + static/dynamic libraries
- `make world` - Full installation with additional tools

### Development Commands
- `make updatesubmodules` - Update git submodules (libraries, faust2ck, etc.)
- `make parser` - Regenerate parser from lex/yacc files
- `make clean` - Remove object files (keep build configurations)
- `make distclean` - Remove entire build/faustdir folder
- `make format` - Run clang-format on source files
- `make doc` - Generate documentation with doxygen

### Testing
- `make -C tests/impulse-tests` - Test code generation correctness
- `make -C tests/compile-tests` - Verify examples compile across backends
- `make -C tests/llvm-tests` - LLVM backend tests
- `make -C tests/interp-tests` - Interpreter backend tests

### GitHub Actions Workflows
- `libfaust.yml` - Production workflow (complex, legacy Docker/QEMU for aarch64)
- `libfaust-simplified.yml` - Simplified workflow using native aarch64 runners

## Architecture

### Core Components

**Compiler (`/compiler/`)**
- Entry point: `main.cpp`
- Parser: Flex/Bison-based in `parser/` (`.l` and `.y` files)
- Signal processing: Core representation in `signals/`
- Code generation: Multiple backends in `generator/`
- Optimization: Transformation passes in `transform/`, `normalize/`

**Pipeline: `.dsp` → Lexing → Parsing → Box Algebra → Signal Processing → Type Inference → Optimization → Code Generation**

**Architecture Files (`/architecture/`)**
- Platform-specific wrappers (JACK, ALSA, CoreAudio, etc.)
- Plugin formats (VST, AU, LV2, Max/MSP, etc.)
- Target platforms (iOS, Android, WebAssembly, etc.)

**Libraries (`/libraries/`)** - Git submodule
- DSP functions organized by category (filters, oscillators, effects)
- Well-documented modular library system

**Tools (`/tools/`)**
- `faust2xxx` scripts - Deploy to 100+ targets
- Physical modeling tools (`physicalModeling/`)
- Benchmarking (`benchmark/`)

### Key Code Generation Backends (`/compiler/generator/`)
- **C/C++**: Traditional compiled output
- **LLVM**: JIT compilation for libfaust
- **WebAssembly**: Browser deployment  
- **Interpreter**: Bytecode execution
- **Language targets**: Java, Rust, Julia, C#, etc.

## Development Workflow

### Adding Features
Comprehensive guide in `/adding-feature.md` covers:
1. Lexer/Parser modifications (`compiler/parser/`)
2. Signal constructors (`signals/`)
3. Type system integration (`signals/sigtyperules.cpp`)
4. Code generation for each backend

### Submodule Management
Key submodules requiring periodic updates:
- `libraries/` - DSP library functions
- `tools/faust2ck/` - ChucK integration
- `architecture/android/app/oboe` - Android audio
- `architecture/max-msp/py2max` - Max/MSP Python integration

Use `git submodule update --remote --merge [path]` to sync.

### Branch Strategy
- `master-dev` - Main development branch (use for PRs)
- `master` - Stable release branch

## Code Organization

**Signal Processing Pipeline:**
1. Box algebra (`boxes/`) - High-level primitives
2. Signal representation (`signals/`) - Lower-level signals  
3. Type inference (`signals/sigtype*`) - Type system
4. Optimization passes (`transform/`, `normalize/`)
5. Backend code emission (`generator/`)

**Key Files for Contributors:**
- Parser grammar: `compiler/parser/faustparser.y`
- Lexer tokens: `compiler/parser/faustlexer.l` 
- Signal constructors: `compiler/signals/signals.cpp`
- Type rules: `compiler/signals/sigtyperules.cpp`
- Code generation: `compiler/generator/compile_scal.cpp`

## Dependencies and Installation

CMake-based build system (since v2.5.18). Most dependencies are optional and enable specific backends:
- LLVM - For libfaust and JIT compilation
- Various audio libraries (JACK, ALSA, etc.) for platform support
- Qt/GTK - For GUI applications via faust2xxx scripts

Installation: `make install` (after building)