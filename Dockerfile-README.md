# Faust Docker Build System

This document explains the Docker-based build system used for creating manylinux-compatible libfaust distributions.

## Overview

The Docker builds ensure broad Linux compatibility by using manylinux standards, which provide predictable glibc versions and ABI compatibility across different Linux distributions.

## Docker Files

### `Dockerfile-x86_64`
Builds libfaust for x86_64 architecture using manylinux2014 (CentOS 7 base, glibc 2.17).

### `Dockerfile-aarch64` 
Builds libfaust for ARM64 architecture using manylinux_2_28 (RHEL 8 base, glibc 2.28).

## LLVM Compatibility System

### The Challenge
Different LLVM builds have varying glibc requirements:

**Previous Ubuntu build:**
- Built on: Ubuntu 22.04 (glibc 2.35)
- Requires: `GLIBC_2.32`, `GLIBC_2.33`, `GLIBC_2.34`, `GLIBCXX_3.4.29`
- **Incompatible** with manylinux containers

**cmajor-lang/llvm Linux x64 build (SOLUTION):**
- Built for: GNU/Linux 3.2.0 
- Requires: `GLIBC_2.15`, `GLIBCXX_3.4.21`
- **Compatible** with both manylinux2014 and manylinux_2_28

**manylinux containers:**
- manylinux2014: CentOS 7 (glibc 2.17, GLIBCXX_3.4.19)
- manylinux_2_28: RHEL 8 (glibc 2.28, GLIBCXX_3.4.25)

### The Solution
Use cmajor-lang/llvm static libraries with manual cmake configuration:

```dockerfile
RUN if [ -d "/faust/llvm" ] && [ -d "/faust/llvm/lib" ]; then \
        echo "Using cmajor-lang/llvm static libraries (GLIBC_2.15 compatible)"; \
        LLVM_DIR="/faust/llvm"; \
        LLVM_INCLUDE_DIR="$LLVM_DIR/include"; \
        LLVM_LIB_DIR="$LLVM_DIR/lib"; \
        cmake ... \
            -DUSE_LLVM_CONFIG=OFF \
            -DLLVM_DIR="$LLVM_DIR" \
            -DLLVM_INCLUDE_DIRS="$LLVM_INCLUDE_DIR" \
            -DLLVM_LIBRARY_DIRS="$LLVM_LIB_DIR"; \
    else \
        echo "Fallback: installing system LLVM"; \
        yum install -y llvm-devel clang-devel && \
        cmake ... -DUSE_LLVM_CONFIG=ON -DLLVM_CONFIG="llvm-config"; \
    fi
```

### Key Features
1. **Better compatibility**: cmajor-lang/llvm requires only GLIBC_2.15 
2. **Static libraries**: Reliable linking with .a files instead of dynamic libraries
3. **Manual configuration**: Direct cmake setup instead of llvm-config dependency
4. **Proven solution**: Used successfully by Cmajor project across platforms
5. **Graceful fallback**: Uses system LLVM if cmajor build unavailable

## manylinux Standards

### manylinux_2_28 (Used for x86_64 and aarch64)
- **Base**: RHEL 8  
- **glibc**: 2.28
- **Compatibility**: ~80% of Linux systems (20% can't support)
- **LLVM support**: Compatible with LLVM 17+ (requires glibc 2.28+)
- **Trade-off**: Modern LLVM features vs. maximum compatibility

### manylinux2014 (Previously considered)
- **Base**: CentOS 7
- **glibc**: 2.17
- **Compatibility**: ~70% of Linux systems
- **LLVM limitation**: Only supports LLVM 3.4, missing modern Faust requirements
- **Why not used**: Faust's LLVM backend requires features not available in LLVM 3.4

## Build Process

### Stage 1: Base Dependencies
```dockerfile
RUN yum install -y libxml2-devel ncurses-devel libmicrohttpd-devel git cmake pkgconfig
```

### Stage 2: Custom ncurses
Built from source to ensure `libncurses.a` is available for static linking:
```dockerfile
RUN git clone https://github.com/mirror/ncurses.git
RUN ./configure --prefix=/usr/local/ncurses/6_4 --with-shared --enable-pc-files
```

### Stage 3: LLVM Detection & Build
Combined step that tests LLVM compatibility and builds libfaust:
- Tests pre-built LLVM compatibility
- Falls back to system LLVM if needed
- Runs cmake with appropriate LLVM configuration
- Builds both dynamic and static libraries

### Stage 4: Library Processing
```dockerfile
# Keep newest libfaust.so version
RUN mv $(ls -1 libfaust.so.* | tail -n1) libfaust.so
RUN strip --strip-unneeded libfaust.so
```

### Stage 5: Cleanup & Packaging
Removes build artifacts to minimize final image size and creates distribution zip.

## Static Library Creation

The `Make.llvm.static` target creates `libfaustwithllvm.a` by combining:
- `libfaust.a` (core Faust functionality)
- LLVM static libraries (for JIT compilation)
- System libraries (ncurses, xml2, etc.)

Platform-specific combining:
- **macOS**: Uses `libtool -static`
- **Linux**: Uses `ar` archiver

## Integration with GitHub Actions

### Workflow Triggers
```yaml
push:
  paths:
    - 'Dockerfile-*'  # Docker build changes
    - 'build/**'      # Build system changes
    - '.github/workflows/libfaust-simplified.yml'
```

### Build Matrix
```yaml
matrix:
  include:
    - name: ubuntu-x86_64
      dockerfile: Dockerfile-x86_64
      platform: linux/amd64
    - name: ubuntu-aarch64  
      dockerfile: Dockerfile-aarch64
      platform: linux/arm64
```

### QEMU for ARM64
ARM64 builds use QEMU emulation since GitHub doesn't provide native ARM64 Ubuntu runners:
```yaml
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3
  with:
    platforms: arm64
```

## Troubleshooting

### Common Issues

**LLVM glibc errors:**
```
/faust/llvm/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.33' not found
```
*Solution*: The automatic fallback should handle this. If not, check that system LLVM is being installed.

**Missing libncurses.a:**
*Solution*: Custom ncurses build provides this. Check that configure runs with `--with-shared`.

**Docker environment variable scope:**
*Solution*: Ensure LLVM detection and cmake build are in the same RUN step.

### Debugging
To debug LLVM compatibility locally:
```bash
# Test pre-built LLVM in container
docker run --rm -v "$PWD:/faust" quay.io/pypa/manylinux2014_x86_64 \
  /faust/llvm/bin/llvm-config --version

# Check glibc requirements
objdump -T /faust/llvm/bin/llvm-config | grep GLIBC
```

## Future Improvements

1. **LLVM component selection**: Only include needed LLVM components to reduce size
2. **Parallel builds**: Use `-j$(nproc)` for faster compilation  
3. **Layer caching**: Optimize Dockerfile for better layer reuse
4. **Multi-stage builds**: Separate build and runtime environments