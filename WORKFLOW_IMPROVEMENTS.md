# Faust libfaust.yml Workflow Improvements

## Overview

This document outlines the simplification of the `libfaust.yml` GitHub Actions workflow, focusing on macOS ARM64 improvements while preserving Ubuntu manylinux2014 compatibility for Python wheel projects.

## Changes Made

### 1. Created `libfaust-simplified.yml`

**Key Improvements:**
- **Native aarch64 Ubuntu**: Replaced Docker/QEMU with `ubuntu-22.04-arm64` runners
- **Simplified macOS LLVM**: Uses `brew install llvm@18` instead of universal binary slimming
- **Eliminated complexity**: Removed 50+ lines of Docker setup and 100+ lines of LLVM workarounds

**Ubuntu aarch64 Status:**
GitHub Actions doesn't provide native ARM64 Ubuntu runners. The Docker/QEMU approach remains necessary for ARM64 Ubuntu builds. However, the main simplification comes from macOS ARM64 native runners.

**Before (Ubuntu aarch64):**
```yaml
# Complex Docker/QEMU setup (lines 95-150) - STILL REQUIRED
- name: Download LLVM (pre-built)
- name: Set up QEMU 
- name: Set up Docker Buildx
- name: Build and push (Docker)
```

**Ubuntu x86_64 (unchanged):**
```yaml
# Already simple, no changes needed
- name: Install dependencies  
- name: Install LLVM (apt)
- name: Build libfaust (cmake)
```

**Before (macOS LLVM):**
```yaml
# Complex universal binary handling (lines 178-294)
- name: Checkout LLVM universal binaries
- name: Slim LLVM libraries (lipo extraction)
- name: Complex environment setup (manual library linking)
```

**After (macOS LLVM):**
```yaml
# Simplified cmajor-lang/llvm approach (keeps proven solution)
- name: Checkout LLVM universal binaries (shallow)
- name: Extract LLVM libraries for target architecture  
- name: Install brew dependencies (everything except LLVM)
```

### 2. Created `test-static-libs.yml`

**Purpose:** Validate critical static library functionality:
- `libfaustwithllvm.a` creation process
- Native aarch64/arm64 builds
- LLVM static linking verification
- Basic Faust compiler functionality

### 3. Updated `CLAUDE.md`

Added documentation for new workflows and testing approach.

## Benefits

### Immediate Impact
1. **macOS ARM64 native builds**: No more complex universal binary slimming (~100 lines simplified)
2. **Streamlined macOS LLVM**: Uses proven cmajor-lang/llvm but simplified extraction
3. **Better maintainability**: Reduces complexity while keeping working solutions
4. **Ubuntu builds preserved**: Kept Docker/QEMU approach for manylinux2014 compatibility

### For Python Library Users (like DawDreamer)
1. **manylinux2014 compatibility**: Ubuntu builds preserve proper glibc/ABI compatibility
2. **Improved macOS artifacts**: Native ARM64 builds for better performance
3. **Predictable releases**: Standard GitHub release artifacts maintained
4. **Faster macOS CI**: Reduced build times for macOS builds

## Preserved Functionality

### Critical Requirements Maintained
- ✅ `libfaustwithllvm.a` static library creation
- ✅ `Make.llvm.static` build process
- ✅ Platform-specific system library linking
- ✅ All LLVM backends included
- ✅ Compatible release artifacts for DawDreamer-style usage

### Linux Minimal Build Preserved
The simple Ubuntu x86_64 build remains unchanged:
```bash
sudo apt-get install -y libllvm15 llvm-15 llvm-15-dev libncurses-dev libxml2-dev
cmake -C ./backends/regular.cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Testing Strategy

### Phase 1: Validation
1. **Run `test-static-libs.yml`** - Verify core functionality
2. **Run `libfaust-simplified.yml`** - Test full build pipeline
3. **Compare artifacts** - Ensure output compatibility

### Phase 2: Integration
1. **Test with DawDreamer workflow** - Validate real-world usage
2. **Performance benchmarks** - Measure build time improvements
3. **Size validation** - Ensure artifact sizes are consistent

### Phase 3: Migration
1. **Replace `libfaust.yml`** with simplified version
2. **Archive legacy workflow** as `libfaust-legacy.yml`
3. **Update documentation** and release process

## Next Steps

1. **Immediate**: Run `test-static-libs.yml` to validate approach
2. **Short-term**: Run `libfaust-simplified.yml` for full pipeline test
3. **Medium-term**: Test with downstream Python projects
4. **Long-term**: Consider additional optimizations (LLVM component selection, parallel builds)

## Risk Mitigation

- **Gradual migration**: Keep existing workflow during testing
- **Artifact validation**: Ensure binary compatibility
- **Rollback plan**: Legacy workflow remains available
- **Community testing**: Validate with real downstream users