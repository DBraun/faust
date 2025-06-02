# Waveform Extraction Approach in JAX Backend

## Overview

This document describes both the current approach and a proposed future approach for handling waveform data extraction in the JAX backend.

## Current Implementation (Quick Fix)

The JAX backend currently uses a pragmatic approach to handle tables used in inline subcontainers that aren't declared globally. This was implemented as a quick fix to resolve test failures.

### Table Detection

A `TableDetector` visitor scans inline subcontainer code for table usage:

```cpp
struct TableDetector : public DispatchVisitor {
    std::set<std::string> fTablesUsed;
    
    virtual void visit(StoreVarInst* inst) {
        string varname = inst->fAddress->getName();
        if ((varname.find("ftbl0") == 0 || varname.find("itbl0") == 0) &&
            varname.find("_idx") == std::string::npos) {
            fTablesUsed.insert(varname);
        }
        DispatchVisitor::visit(inst);
    }
    
    virtual void visit(LoadVarInst* inst) {
        string varname = inst->fAddress->getName();
        if ((varname.find("ftbl0") == 0 || varname.find("itbl0") == 0) &&
            varname.find("_idx") == std::string::npos) {
            fTablesUsed.insert(varname);
        }
        DispatchVisitor::visit(inst);
    }
};
```

### Table Declaration

When undeclared tables are found, they're declared with default sizes:

```cpp
if (!found) {
    tab(n + 2, *fOut);
    *fOut << "# Table used in inline subcontainer but not declared globally";
    tab(n + 2, *fOut);
    // Determine size and type from usage pattern (default to 65537 for sine tables)
    if (tableName.find("ftbl0") == 0) {
        *fOut << tableName << " = np.zeros((65537,), dtype=np.float" 
              << (gGlobal->gFloatSize == 1 ? "32" : "64") << ")";
    } else {
        *fOut << tableName << " = np.zeros((65537,), dtype=np.int32)";
    }
}
```

### Pattern Matching

The implementation uses pattern matching to identify different types of data:

1. **Static Tables**: `ftbl0*` and `itbl0*` patterns
2. **Waveform Data**: Variables containing "Wave" or "SIG" in their names
3. **Index Variables**: Variables ending with "_idx"

### Limitations

1. **Hard-coded Size**: Uses 65537 (standard oscillator table size) for all tables
2. **Limited Pattern Recognition**: Only recognizes basic table name patterns
3. **Incomplete Subcontainer Support**: Can't handle complex initialization patterns

## Future Implementation (Proposed)

A more sophisticated approach would use a `WaveformDataExtractor` to analyze the FIR and extract initialization patterns.

### Architecture

```cpp
struct WaveformDataExtractor : public DispatchVisitor {
    struct WaveformInfo {
        std::string tableName;           // e.g., "itbl0mydspSIG0"
        std::string varType;             // "int32" or "float32/64"
        std::vector<int> intData;        // For integer waveforms
        std::vector<float> floatData;    // For float waveforms
        std::string initPattern;         // "direct", "computed", "interpolated"
        int tableSize;                   // Size of the target table
        
        // For computed patterns (like ba.tabulate)
        std::string sourceTable;         // Source table for interpolation
        ValueInst* computeExpression;    // Expression used in computation
    };
    
    std::map<std::string, WaveformInfo> fWaveforms;
    
    // Extract data from literal arrays
    virtual void visit(Int32ArrayNumInst* inst);
    virtual void visit(FloatArrayNumInst* inst);
    
    // Track table assignments
    virtual void visit(StoreVarInst* inst);
    
    // Analyze loops for initialization patterns
    virtual void visit(ForLoopInst* inst);
};
```

### Pattern Recognition

The extractor would recognize:

1. **Direct Initialization**: Waveform literals like `waveform{0, 1, 0, -1}`
2. **Computed Initialization**: Patterns from `ba.tabulate` and similar functions
3. **Standard Patterns**: Common patterns like SIG0 (incrementing integers) and SIG1 (incrementing floats)

### Benefits

1. **Accurate Sizes**: Extract actual table sizes from the code
2. **Proper Initialization**: Generate correct initialization patterns
3. **Better Error Handling**: Detect and report unsupported patterns
4. **Extensible**: Easy to add new patterns

## Table Naming Conventions

### Static Tables (Read-only)
- **Float**: `ftbl0<classname>SIG<n>` (e.g., `ftbl0mydspSIG0`)
- **Integer**: `itbl0<classname>SIG<n>` (e.g., `itbl0mydspSIG1`)
- **Size**: Typically 65537 for oscillators (2^16 + 1)

### Read-Write Tables
- **Float**: `ftbl<n>` where n > 0 (e.g., `ftbl1`, `ftbl2`)
- **Integer**: `itbl<n>` where n > 0 (e.g., `itbl1`, `itbl2`)
- **Size**: Variable based on usage

### Waveform Data
- **Pattern**: `<f|i><classname>Wave<n>` or `<f|i><classname>SIG<n>Wave<n>`
- **Example**: `fmydspWave0`, `imydspSIG1Wave0`

## Common Issues and Solutions

### Issue 1: Undefined Table References
**Symptom**: `NameError: name 'ftbl0mydspSIG0' is not defined`
**Solution**: Use TableDetector to find and declare missing tables

### Issue 2: Incorrect Table Size
**Symptom**: `IndexError: index 65536 is out of bounds`
**Solution**: Use size 65537 instead of 65536 for oscillator tables

### Issue 3: Missing Waveform Data
**Symptom**: Tables initialized with zeros instead of proper waveform
**Solution**: Extract waveform data from global declarations and inline it

## Testing

The implementation is tested with:
- `comb_delay1.dsp`, `comb_delay2.dsp` - Delay lines with tables
- `table.dsp`, `table1.dsp` - Basic table operations
- `tester.dsp`, `tester2.dsp` - Complex table usage
- `waveform4.dsp` - Multiple waveform types
- `waveform_tabulate.dsp` - ba.tabulate usage (currently failing)

## Handling ba.tabulate and Complex Patterns

### The ba.tabulate Challenge

The `ba.tabulate` function creates a new table by interpolating values from an existing table. In the C++ backend, this generates:

1. A subcontainer class (e.g., `mydspSIG0SIG0`) with fill and init methods
2. A source table (e.g., `itbl0mydspSIG0SIG0`) with the original waveform
3. A destination table (e.g., `itbl1mydspSIG0`) filled by interpolation

Example from `waveform_tabulate.dsp`:
```faust
mytable = rdtable(waveform{0, 1, 0, -1});
mytable_tabulated(x) = ba.tabulate(1, mytable, 4, 0, 4, x).lin;
```

This currently generates undefined function calls in JAX:
```python
fillmydspSIG0SIG0(dsp, np.int32(4), self._itbl0mydspSIG0SIG0)
instanceInitmydspSIG0SIG0(dsp, sample_rate)
```

### Implementation Plan

#### Phase 1: Pattern Detection and Replacement

Create a visitor to detect and replace subcontainer function calls:

```cpp
struct SubcontainerFunctionReplacer : public BasicCloneVisitor {
    JAXCodeContainer* fContainer;
    std::map<std::string, SubcontainerInfo> fSubcontainers;
    
    struct SubcontainerInfo {
        std::string className;      // e.g., "mydspSIG0SIG0"
        std::string sourceTable;    // e.g., "itbl0mydspSIG0SIG0"
        std::string destTable;      // e.g., "itbl1mydspSIG0"
        int tableSize;              // e.g., 4
        ValueInst* waveformData;    // The waveform literal
        std::string interpolation;  // "lin", "cos", "cubic"
        float factor;               // Interpolation factor (e.g., 1.333334)
    };
    
    virtual StatementInst* visit(VoidFunCallInst* inst) {
        // Pattern: fill<className>(dsp, size, table)
        if (inst->fName.find("fill") == 0) {
            return generateTableFill(inst);
        }
        // Pattern: instanceInit<className>(dsp, sample_rate)
        else if (inst->fName.find("instanceInit") == 0) {
            // These can often be no-ops for simple tables
            return IB::genCommentInst("// Subcontainer init handled inline");
        }
        return BasicCloneVisitor::visit(inst);
    }
    
    StatementInst* generateTableFill(VoidFunCallInst* inst) {
        // Extract info from the function call
        auto info = extractSubcontainerInfo(inst);
        
        if (info.waveformData) {
            // Direct waveform initialization
            return generateDirectWaveformInit(info);
        } else {
            // Look for interpolation pattern in static init
            return generateInterpolationInit(info);
        }
    }
};
```

#### Phase 2: Waveform Data Extraction

Enhance the waveform extraction to capture:

1. **Waveform literals**: `waveform{0, 1, 0, -1}`
2. **Table relationships**: Which tables are derived from others
3. **Interpolation parameters**: Factor, type (linear/cosine/cubic)

```cpp
struct EnhancedWaveformExtractor : public DispatchVisitor {
    // Track waveform literals
    virtual void visit(Int32ArrayNumInst* inst) {
        fCurrentWaveform.intData = inst->fValues;
        fCurrentWaveform.type = "int32";
    }
    
    // Track table initialization loops
    virtual void visit(ForLoopInst* inst) {
        // Detect patterns like:
        // for (int i1 = 0; i1 < 4; i1++) {
        //     itbl1[i1] = itbl0[int(1.333334 * i1)];
        // }
        if (isTableInitLoop(inst)) {
            extractInterpolationPattern(inst);
        }
    }
    
    void extractInterpolationPattern(ForLoopInst* loop) {
        // Analyze the loop body to extract:
        // 1. Source and destination tables
        // 2. Interpolation factor
        // 3. Interpolation type (linear, etc.)
    }
};
```

#### Phase 3: Code Generation Strategy

Generate the initialization inline in the appropriate context:

```python
# In setup() for static tables:
# Initialize source waveform
self._itbl0mydspSIG0SIG0 = np.array([0, 1, 0, -1], dtype=np.int32)

# In _initialize_carry() for read-write tables:
# Linear interpolation with factor 1.333334
for i in range(4):
    src_pos = 1.333334 * i
    src_idx = int(src_pos)
    frac = src_pos - src_idx
    
    # Bounds checking
    idx_low = np.clip(src_idx, 0, 3)
    idx_high = np.clip(src_idx + 1, 0, 3)
    
    # Linear interpolation
    low_val = self._itbl0mydspSIG0SIG0[idx_low]
    high_val = self._itbl0mydspSIG0SIG0[idx_high]
    state["itbl1mydspSIG0"][i] = int(low_val + frac * (high_val - low_val))
```

#### Phase 4: Integration Points

1. **Modify `inlineSubcontainersFunCalls()`**:
```cpp
BlockInst* JAXCodeContainer::inlineSubcontainersFunCalls(BlockInst* block) {
    // First pass: extract subcontainer information
    SubcontainerAnalyzer analyzer;
    block->accept(&analyzer);
    
    // Second pass: replace function calls with inline code
    SubcontainerFunctionReplacer replacer(this, analyzer.getInfo());
    return replacer.getCode(block);
}
```

2. **Update table detection**:
```cpp
// Add to TableDetector
virtual void visit(VoidFunCallInst* inst) {
    // Extract table names from function arguments
    if (inst->fName.find("fill") == 0 && inst->fArgs.size() >= 3) {
        if (LoadVarInst* table = dynamic_cast<LoadVarInst*>(inst->fArgs[2])) {
            fTablesUsed.insert(table->fAddress->getName());
        }
    }
}
```

### Expected Output

For `waveform_tabulate.dsp`, the generated code should be:

```python
def setup(self):
    # Initialize static tables
    itbl0mydspSIG0SIG0 = np.zeros((4,), dtype=np.int32)
    
    # Initialize waveform data directly
    itbl0mydspSIG0SIG0[:] = [0, 1, 0, -1]
    
    # Convert to JAX array
    self._itbl0mydspSIG0SIG0 = jnp.array(itbl0mydspSIG0SIG0)

def _initialize_carry(self, x: jnp.ndarray, length: int):
    state = {}
    
    # Initialize read-write tables
    state["itbl1mydspSIG0"] = np.zeros((4,), dtype=np.int32)
    
    # ba.tabulate interpolation (factor: 1.333334)
    for i in range(4):
        src_pos = np.float32(1.333334) * i
        src_idx = int(src_pos)
        frac = src_pos - src_idx
        
        idx_low = np.maximum(0, np.minimum(src_idx, 3))
        idx_high = np.maximum(0, np.minimum(src_idx + 1, 3))
        
        low_val = self._itbl0mydspSIG0SIG0[idx_low]
        high_val = self._itbl0mydspSIG0SIG0[idx_high]
        state["itbl1mydspSIG0"][i] = int(low_val + frac * (high_val - low_val))
    
    return state
```

### Testing Strategy

1. **Unit tests** for pattern detection
2. **Integration tests** with various ba.tabulate parameters
3. **Comparison tests** against C++ backend output
4. **Performance benchmarks** for initialization overhead

## Future Work

1. **Implement SubcontainerFunctionReplacer** for proper function call handling
2. **Support all interpolation types** (linear, cosine, cubic, etc.)
3. **Handle nested ba.tabulate** calls
4. **Support other complex patterns** (ba.take, ba.subseq, etc.)
5. **Generate proper error messages** for unsupported patterns
6. **Optimize initialization** for large tables
7. **Add caching** for repeated patterns