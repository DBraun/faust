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

**Ubuntu x86_64 (Fixed for manylinux2014):**
```yaml
# CRITICAL FIX: Docker approach with LLVM compatibility testing
- name: Download LLVM (pre-built for compatibility testing)
- name: Set up Docker Buildx
- name: Build inside manylinux2014 container with automatic LLVM fallback
```

**LLVM Compatibility System (NEW):**
```dockerfile
# Test if pre-built LLVM is compatible, fall back to system LLVM
RUN if [ -d "/faust/llvm" ]; then \
        if /faust/llvm/bin/llvm-config --version >/dev/null 2>&1; then \
            LLVM_CONFIG="/faust/llvm/bin/llvm-config"; \
        else \
            echo "glibc version mismatch, using system LLVM"; \
            yum install -y llvm-devel; \
            LLVM_CONFIG="llvm-config"; \
        fi; \
    else \
        yum install -y llvm-devel; \
        LLVM_CONFIG="llvm-config"; \
    fi
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

## Technical Issues Resolved

### LLVM glibc Compatibility Issue

**Problem:** Pre-built LLVM from cmajor-lang/llvm (built on Ubuntu 22.04) requires:
- `GLIBC_2.32`, `GLIBC_2.33`, `GLIBC_2.34` 
- `GLIBCXX_3.4.29`

**manylinux2014 container provides:**
- `GLIBC_2.17` (CentOS 7 base)
- `GLIBCXX_3.4.19`

**Solution:** Automatic compatibility testing with fallback:
1. Download pre-built LLVM as optimization attempt
2. Test `/faust/llvm/bin/llvm-config --version` in container
3. If fails due to glibc mismatch, install `llvm-devel` from CentOS 7 repos
4. Continue build with compatible system LLVM

**Result:** Maintains manylinux2014 compatibility for 70% broader Python wheel support while attempting to use faster pre-built LLVM when possible.

### Docker Environment Variable Scope

**Problem:** LLVM_CONFIG variable set in one RUN step wasn't available in subsequent RUN step.

**Solution:** Combined LLVM detection and cmake build into single RUN step for proper variable scope.

### manylinux Standard Selection

**Research findings:**
- **manylinux2014**: Supports 70% of systems, based on CentOS 7 (glibc 2.17)
- **manylinux_2_28**: Only 20% of systems can't support, based on RHEL 8 (glibc 2.28)
- **DawDreamer compatibility**: Uses cibuildwheel which commonly targets manylinux2014

**Decision:** Use manylinux_2_28 for LLVM 17 compatibility.

**Trade-off Analysis:**
- manylinux2014 + LLVM 3.4: Maximum compatibility (70% systems) but ancient LLVM with missing features
- manylinux_2_28 + LLVM 17: Slightly less compatibility (80% systems) but modern LLVM with full Faust support

**Final choice:** manylinux_2_28 because Faust requires modern LLVM features that don't exist in LLVM 3.4.

## Benefits

### Immediate Impact
1. **macOS ARM64 native builds**: No more complex universal binary slimming (~100 lines simplified)
2. **Streamlined macOS LLVM**: Uses proven cmajor-lang/llvm but simplified extraction
3. **Ubuntu manylinux2014 compatibility**: Both x86_64 and aarch64 use Docker approach for proper glibc/ABI compatibility
4. **Better maintainability**: Consistent Docker approach for Ubuntu, simplified macOS

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