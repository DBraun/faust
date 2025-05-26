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
Pre-built LLVM binaries from cmajor-lang/llvm are built on modern Ubuntu systems with newer glibc versions:
- Built on: Ubuntu 22.04 (glibc 2.35)
- Requires: `GLIBC_2.32`, `GLIBC_2.33`, `GLIBC_2.34`, `GLIBCXX_3.4.29`

But manylinux containers use older base systems:
- manylinux2014: CentOS 7 (glibc 2.17, GLIBCXX_3.4.19)
- manylinux_2_28: RHEL 8 (glibc 2.28, GLIBCXX_3.4.25)

### The Solution
Automatic compatibility testing with graceful fallback:

```dockerfile
RUN if [ -d "/faust/llvm" ]; then \
        echo "Testing pre-built LLVM compatibility..."; \
        chmod u+x /faust/llvm/bin/llvm-config; \
        if /faust/llvm/bin/llvm-config --version >/dev/null 2>&1; then \
            echo "Pre-built LLVM is compatible"; \
            LLVM_CONFIG="/faust/llvm/bin/llvm-config"; \
        else \
            echo "Pre-built LLVM is incompatible (glibc version mismatch)"; \
            echo "Installing system LLVM instead"; \
            yum install -y llvm-devel; \
            LLVM_CONFIG="llvm-config"; \
        fi; \
    else \
        echo "No pre-built LLVM found, installing system LLVM"; \
        yum install -y llvm-devel; \
        LLVM_CONFIG="llvm-config"; \
    fi && \
    # Use LLVM_CONFIG immediately in same RUN step
    cmake ... -DLLVM_CONFIG="$LLVM_CONFIG"
```

### Key Features
1. **Automatic detection**: Tests if pre-built LLVM works in the container
2. **Graceful fallback**: Uses system LLVM if pre-built version is incompatible  
3. **Single RUN step**: Ensures environment variables persist correctly
4. **Clear logging**: Shows which LLVM version is being used and why

## manylinux Standards

### manylinux2014 (Used for x86_64)
- **Base**: CentOS 7
- **glibc**: 2.17
- **Compatibility**: ~70% of Linux systems
- **Python wheels**: Widely supported by cibuildwheel
- **Use case**: Maximum compatibility for projects like DawDreamer

### manylinux_2_28 (Used for aarch64)
- **Base**: RHEL 8  
- **glibc**: 2.28
- **Compatibility**: ~80% of Linux systems (20% can't support)
- **Trade-off**: Newer features vs. compatibility

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