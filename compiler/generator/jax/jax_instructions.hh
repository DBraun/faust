/************************************************************************
 ************************************************************************
    FAUST compiler
    Copyright (C) 2021 GRAME, Centre National de Creation Musicale
    ---------------------------------------------------------------------
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 ************************************************************************
 ************************************************************************/

#ifndef _JAX_INSTRUCTIONS_H
#define _JAX_INSTRUCTIONS_H

#include <string>

#include "struct_manager.hh"
#include "text_instructions.hh"
#include "jax_state_manager.hh"
#include "jax_variable_classifier.hh"

// Visitor used to initialize array fields into the DSP structure
struct JAXInitFieldsVisitor : public DispatchVisitor {
    std::ostream* fOut;
    int           fTab;
    std::set<std::string>* fScalarDelayVars;
    std::set<std::string>* fCircularBufferVars;
    std::map<std::string, int>* fDelayLineSizes;
    JAXVariableClassifier fClassifier;

    JAXInitFieldsVisitor(std::ostream* out, int tab = 0, 
                         std::set<std::string>* scalarDelayVars = nullptr,
                         std::set<std::string>* circularBufferVars = nullptr,
                         std::map<std::string, int>* delayLineSizes = nullptr,
                         const std::string& className = "mydsp") 
        : fOut(out), fTab(tab), fScalarDelayVars(scalarDelayVars), 
          fCircularBufferVars(circularBufferVars), fDelayLineSizes(delayLineSizes),
          fClassifier(className) {}

    virtual void visit(DeclareVarInst* inst)
    {        
        // Check if this is a scalar delay variable - skip if so (handled in StoreVarInst)
        if (fScalarDelayVars && inst->fAddress) {
            if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
                if (fScalarDelayVars->find(named->fName) != fScalarDelayVars->end()) {
                    return;  // Skip - will be initialized as scalar in StoreVarInst
                }
            }
        }
        
        ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(inst->fType);
        if (array_type) {
            tab(fTab, *fOut);
            inst->fAddress->accept(this);
            *fOut << " = ";
            if (inst->fValue) {
                inst->fValue->accept(this);
            } else {
                ZeroInitializer(fOut, inst->fType);
            }
            
            // Initialize circular buffer index if needed
            if (fCircularBufferVars && inst->fAddress) {
                if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
                    if (fCircularBufferVars->find(named->fName) != fCircularBufferVars->end()) {
                        *fOut << " ";
                        tab(fTab, *fOut);
                        *fOut << "# Initialize circular buffer index\n";
                        tab(fTab, *fOut);
                        *fOut << "state[\"" << named->fName << "_idx\"] = np.int64(0) ";
                    }
                }
            }
        } else {
            // Handle non-array struct variables (like IOTA)
            if (inst->fAddress) {
                if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
                    // Skip UI parameters - they're handled separately
                    auto category = fClassifier.classifyVariable(named->fName);
                    if (category == JAXVariableClassifier::VarCategory::UI_PARAMETER) {
                        return;  // Skip UI parameters
                    }
                    
                    if (named->isStruct() || named->isStaticStruct()) {
                        tab(fTab, *fOut);
                        inst->fAddress->accept(this);
                        *fOut << " = ";
                        // Initialize based on type
                        if (isIntType(inst->fType->getType())) {
                            *fOut << "np.int32(0)";
                        } else if (isRealType(inst->fType->getType())) {
                            if (gGlobal->gFloatSize == 1) {
                                *fOut << "np.float32(0)";
                            } else {
                                *fOut << "np.float64(0)";
                            }
                        } else if (inst->fValue) {
                            inst->fValue->accept(this);
                        } else {
                            *fOut << "None";  // Default for unknown types
                        }
                        *fOut << " ";
                    }
                }
            }
        }
    }

    virtual void visit(NamedAddress* named)
    {
        // kStaticStruct are actually merged in the main DSP
        if (named->isStruct() || named->isStaticStruct()) {
            *fOut << "state[\"";
        }
        *fOut << named->fName;
        if (named->isStruct() || named->isStaticStruct()) {
            *fOut << "\"]";
        }
    }

    static void ZeroInitializer(std::ostream* fOut, Typed* typed)
    {
        ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(typed);
        faustassert(array_type);

        // Always use numpy for initialization (static context)
        if (isIntPtrType(typed->getType())) {
            // Create a temporary JAXStringTypeManager to use its helper methods
            JAXStringTypeManager typeManager("", "", "");
            *fOut << typeManager.generateZeroArray(array_type->fSize, Typed::kInt32, true);
        } else if (isRealPtrType(typed->getType())) {
            // Create a temporary JAXStringTypeManager to use its helper methods
            JAXStringTypeManager typeManager("", "", "");
            Typed::VarType floatType = (gGlobal->gFloatSize == 1) ? Typed::kFloat : Typed::kDouble;
            *fOut << typeManager.generateZeroArray(array_type->fSize, floatType, true);
        }
    }

    virtual void visit(StoreVarInst* inst)
    {
        // Check if this is a scalar delay initialization
        if (fScalarDelayVars && inst->fAddress) {
            if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
                if (fScalarDelayVars->find(named->fName) != fScalarDelayVars->end()) {
                    // Initialize scalar delay variable
                    tab(fTab, *fOut);
                    inst->fAddress->accept(this);
                    *fOut << " = ";
                    // Determine type from the value
                    JAXStringTypeManager typeManager("", "", "");
                    if (Int32NumInst* intVal = dynamic_cast<Int32NumInst*>(inst->fValue)) {
                        *fOut << typeManager.wrapLiteral(std::to_string(intVal->fNum), Typed::kInt32, true);
                    } else if (FloatNumInst* floatVal = dynamic_cast<FloatNumInst*>(inst->fValue)) {
                        *fOut << typeManager.wrapLiteral(checkFloat(floatVal->fNum), Typed::kFloat, true);
                    } else if (DoubleNumInst* doubleVal = dynamic_cast<DoubleNumInst*>(inst->fValue)) {
                        *fOut << typeManager.wrapLiteral(checkDouble(doubleVal->fNum), Typed::kDouble, true);
                    } else {
                        // Default case - determine from global float size
                        Typed::VarType floatType = (gGlobal->gFloatSize == 1) ? Typed::kFloat : Typed::kDouble;
                        *fOut << typeManager.wrapLiteral("0", floatType, true);
                    }
                    return;
                }
            }
        }
        // Not a scalar delay - use default behavior
        DispatchVisitor::visit(inst);
    }
    
    // Needed for waveforms
    virtual void visit(Int32ArrayNumInst* inst)
    {
        JAXStringTypeManager typeManager("", "", "");
        *fOut << typeManager.getTypePrefix(true) << ".array(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << inst->fNumTable[i];
            sep = ',';
        }
        *fOut << "], dtype=" << typeManager.getDTypeString(Typed::kInt32, true) << ")";
    }

    virtual void visit(FloatArrayNumInst* inst)
    {
        JAXStringTypeManager typeManager("", "", "");
        *fOut << typeManager.getTypePrefix(true) << ".array(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkFloat(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=" << typeManager.getFloatTypeString(true) << ")";
    }

    virtual void visit(DoubleArrayNumInst* inst)
    {
        JAXStringTypeManager typeManager("", "", "");
        *fOut << typeManager.getTypePrefix(true) << ".array(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkDouble(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=" << typeManager.getDTypeString(Typed::kDouble, true) << ")";
    }
};

/**
 * JAX Instruction Visitor with Circular Buffer Support
 * 
 * This visitor generates JAX/Python code from Faust IR instructions, with special
 * handling for optimized delay line access patterns. It implements:
 * 
 * - Circular buffer array access conversion for efficient delay line operations
 * - Variable classification for proper state management in JAX modules
 * - Dynamic index handling for modulated delays (e.g., chorus, flangers)
 * - Integration with JAX's immutable array semantics (.at[].set() operations)
 */
class JAXInstVisitor : public TextInstVisitor {
   public:
    std::set<std::string> fScalarDelayVars;      // Single-sample delay variables (optimized as scalars)
    std::set<std::string> fUIParamVars;          // UI parameter variables
    std::set<std::string> fConstantVars;         // Constant variables
    std::set<std::string> fCircularBufferVars;   // Delay lines using circular buffer optimization
    std::map<std::string, int> fDelayLineSizes;  // Buffer sizes for circular buffer index calculations
    
    // State management (mutable so it can be modified in const contexts)
    mutable JAXStateManager fStateManager;
    
    // Variable classification
    JAXVariableClassifier fClassifier;
    
   private:
    // Helper to get the JAX type manager
    JAXStringTypeManager* getJAXTypeManager() const {
        return static_cast<JAXStringTypeManager*>(fTypeManager);
    }
    
    /*
     Global functions names table as a static variable in the visitor
     so that each function prototype is generated as most once in the module.
     */
    static std::map<std::string, bool> gFunctionSymbolTable;

    // Polymorphic math functions
    std::map<std::string, std::string> gPolyMathLibTable;

    // bool for "is storing left-hand-side".
    // Suppose the output code will be `state['foo'] = bar`.
    // This boolean indicates that we are starting this line but haven't yet reached the equals
    // sign.
    bool fIsStoringLhs = false;

    // bool for "will set array".
    // jax has a special syntax for setting items of arrays:
    // https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    // This bool helps us know that we're going to use the .at[X] operator followed by the set(Y)
    // operator. This bool is used in tandem with fIsStoringLhs.
    bool fWillSetArray = false;

    // This bool is not related to fIsStoringLhs or fWillSetArray.
    // It is used so that we don't cast to integers in the condition of a while (cond) loop.
    bool fIsDoingWhile = false;
    
    // Track when we're in array index context to avoid wrapping integers
    bool fIsArrayIndex = false;

    std::set<std::string> fLogSet;  // set of widget zone having a log UI scale
    std::set<std::string> fExpSet;  // set of widget zone having an exp UI scale

   public:
    using TextInstVisitor::visit;
    
    // Map to store pfPerm initialization values
    std::map<std::string, ValueInst*> fPfPermInitValues;

    // DEPRECATED: These boolean flags are replaced by fStateManager
    // bool fUseNumpy = true;
    // bool fInSetup = false;
    // bool fInTick = false;
    // bool fInStaticInit = false;
    // bool fInInlineSubcontainer = false;
    // std::set<std::string> fInlineSubcontainerLocals;
    
    // Helper method to check if we should use numpy (for backward compatibility)
    bool fUseNumpy() const { return fStateManager.useNumpy(); }

    JAXInstVisitor(std::ostream* out, const std::string& struct_name, int tab = 0)
        : TextInstVisitor(out, ".", new JAXStringTypeManager(xfloat(), "*", struct_name), tab),
          fClassifier(struct_name)
    {
        // Mark all math.h functions as generated...
        gFunctionSymbolTable["abs"] = true;

        gFunctionSymbolTable["max_i"] = true;
        gFunctionSymbolTable["min_i"] = true;

        gFunctionSymbolTable["max_f"] = true;
        gFunctionSymbolTable["min_f"] = true;

        gFunctionSymbolTable["max_"] = true;
        gFunctionSymbolTable["min_"] = true;

        gFunctionSymbolTable["max_l"] = true;
        gFunctionSymbolTable["min_l"] = true;

        // Float version
        gFunctionSymbolTable["fabsf"]      = true;
        gFunctionSymbolTable["acosf"]      = true;
        gFunctionSymbolTable["asinf"]      = true;
        gFunctionSymbolTable["atanf"]      = true;
        gFunctionSymbolTable["atan2f"]     = true;
        gFunctionSymbolTable["ceilf"]      = true;
        gFunctionSymbolTable["cosf"]       = true;
        gFunctionSymbolTable["expf"]       = true;
        gFunctionSymbolTable["exp10f"]     = false;
        gFunctionSymbolTable["floorf"]     = true;
        gFunctionSymbolTable["fmodf"]      = true;
        gFunctionSymbolTable["logf"]       = true;
        gFunctionSymbolTable["log10f"]     = true;
        gFunctionSymbolTable["powf"]       = true;
        gFunctionSymbolTable["remainderf"] = true;
        gFunctionSymbolTable["rintf"]      = true;
        gFunctionSymbolTable["roundf"]     = true;
        gFunctionSymbolTable["sinf"]       = true;
        gFunctionSymbolTable["sqrtf"]      = true;
        gFunctionSymbolTable["tanf"]       = true;

        // Hyperbolic
        gFunctionSymbolTable["acoshf"] = true;
        gFunctionSymbolTable["asinhf"] = true;
        gFunctionSymbolTable["atanhf"] = true;
        gFunctionSymbolTable["coshf"]  = true;
        gFunctionSymbolTable["sinhf"]  = true;
        gFunctionSymbolTable["tanhf"]  = true;

        // Double version
        gFunctionSymbolTable["fabs"]      = true;
        gFunctionSymbolTable["acos"]      = true;
        gFunctionSymbolTable["asin"]      = true;
        gFunctionSymbolTable["atan"]      = true;
        gFunctionSymbolTable["atan2"]     = true;
        gFunctionSymbolTable["ceil"]      = true;
        gFunctionSymbolTable["cos"]       = true;
        gFunctionSymbolTable["exp"]       = true;
        gFunctionSymbolTable["exp10"]     = false;
        gFunctionSymbolTable["floor"]     = true;
        gFunctionSymbolTable["fmod"]      = true;
        gFunctionSymbolTable["log"]       = true;
        gFunctionSymbolTable["log10"]     = true;
        gFunctionSymbolTable["pow"]       = true;
        gFunctionSymbolTable["remainder"] = true;
        gFunctionSymbolTable["rint"]      = true;
        gFunctionSymbolTable["round"]     = true;
        gFunctionSymbolTable["sin"]       = true;
        gFunctionSymbolTable["sqrt"]      = true;
        gFunctionSymbolTable["tan"]       = true;

        // Hyperbolic
        gFunctionSymbolTable["acosh"] = true;
        gFunctionSymbolTable["asinh"] = true;
        gFunctionSymbolTable["atanh"] = true;
        gFunctionSymbolTable["coshf"] = true;
        gFunctionSymbolTable["sinh"]  = true;
        gFunctionSymbolTable["tanh"]  = true;

        // Quad version
        gFunctionSymbolTable["fabsl"]      = true;
        gFunctionSymbolTable["acosl"]      = true;
        gFunctionSymbolTable["asinl"]      = true;
        gFunctionSymbolTable["atanl"]      = true;
        gFunctionSymbolTable["atan2l"]     = true;
        gFunctionSymbolTable["ceill"]      = true;
        gFunctionSymbolTable["cosl"]       = true;
        gFunctionSymbolTable["expl"]       = true;
        gFunctionSymbolTable["exp10l"]     = false;
        gFunctionSymbolTable["floorl"]     = true;
        gFunctionSymbolTable["fmodl"]      = true;
        gFunctionSymbolTable["logl"]       = true;
        gFunctionSymbolTable["log10l"]     = true;
        gFunctionSymbolTable["powl"]       = true;
        gFunctionSymbolTable["remainderl"] = true;
        gFunctionSymbolTable["rintl"]      = true;
        gFunctionSymbolTable["roundl"]     = true;
        gFunctionSymbolTable["sinl"]       = true;
        gFunctionSymbolTable["sqrtl"]      = true;
        gFunctionSymbolTable["tanl"]       = true;

        // Hyperbolic
        gFunctionSymbolTable["acoshl"] = true;
        gFunctionSymbolTable["asinhl"] = true;
        gFunctionSymbolTable["atanhl"] = true;
        gFunctionSymbolTable["coshl"]  = true;
        gFunctionSymbolTable["sinhl"]  = true;
        gFunctionSymbolTable["tanhl"]  = true;

        // Polymath mapping int version
        gPolyMathLibTable["abs"]   = "jnp.abs";
        gPolyMathLibTable["max_i"] = "jnp.maximum";
        gPolyMathLibTable["min_i"] = "jnp.minimum";

        // Polymath mapping float version
        gPolyMathLibTable["max_f"] = "jnp.maximum";
        gPolyMathLibTable["min_f"] = "jnp.minimum";

        gPolyMathLibTable["fabsf"]  = "jnp.abs";
        gPolyMathLibTable["acosf"]  = "jnp.arccos";
        gPolyMathLibTable["asinf"]  = "jnp.arcsin";
        gPolyMathLibTable["atanf"]  = "jnp.arctan";
        gPolyMathLibTable["atan2f"] = "jnp.arctan2";
        gPolyMathLibTable["ceilf"]  = "jnp.ceil";
        gPolyMathLibTable["cosf"]   = "jnp.cos";
        gPolyMathLibTable["expf"]   = "jnp.exp";
        gPolyMathLibTable["exp2f"]  = "jnp.exp2";
        gPolyMathLibTable["exp10f"] = "jnp.exp10f";
        gPolyMathLibTable["floorf"] = "jnp.floor";
        gPolyMathLibTable["fmodf"]  = "jnp.mod";
        gPolyMathLibTable["logf"]   = "jnp.log";
        gPolyMathLibTable["log2f"]  = "jnp.log2";
        gPolyMathLibTable["log10f"] = "jnp.log10";
        gPolyMathLibTable["powf"]   = "jnp.power";
        gPolyMathLibTable["remainderf"] =
            "remainder";  // todo: we currently rely on a custom remainder implementation in the
                          // architecture file.
        gPolyMathLibTable["rintf"]  = "jnp.rint";
        gPolyMathLibTable["roundf"] = "jnp.round";
        gPolyMathLibTable["sinf"]   = "jnp.sin";
        gPolyMathLibTable["sqrtf"]  = "jnp.sqrt";
        gPolyMathLibTable["tanf"]   = "jnp.tan";

        // Hyperbolic
        gPolyMathLibTable["acoshf"] = "jnp.arccosh";
        gPolyMathLibTable["asinhf"] = "jnp.arcsinh";
        gPolyMathLibTable["atanhf"] = "jnp.arctanh";
        gPolyMathLibTable["coshf"]  = "jnp.cosh";
        gPolyMathLibTable["sinhf"]  = "jnp.sinh";
        gPolyMathLibTable["tanhf"]  = "jnp.tanh";

        gPolyMathLibTable["isnanf"]    = "jnp.isnan";
        gPolyMathLibTable["isinff"]    = "jnp.isinf";
        gPolyMathLibTable["copysignf"] = "jnp.copysign";

        // Polymath mapping double version
        gPolyMathLibTable["max_"] = "jnp.maximum";
        gPolyMathLibTable["min_"] = "jnp.minimum";

        gPolyMathLibTable["fabs"]  = "jnp.abs";
        gPolyMathLibTable["acos"]  = "jnp.arccos";
        gPolyMathLibTable["asin"]  = "jnp.arcsin";
        gPolyMathLibTable["atan"]  = "jnp.arctan";
        gPolyMathLibTable["atan2"] = "jnp.arctan2";
        gPolyMathLibTable["ceil"]  = "jnp.ceil";
        gPolyMathLibTable["cos"]   = "jnp.cos";
        gPolyMathLibTable["exp"]   = "jnp.exp";
        gPolyMathLibTable["exp2"]  = "jnp.exp2";
        gPolyMathLibTable["exp10"] = "jnp.exp10";
        gPolyMathLibTable["floor"] = "jnp.floor";
        gPolyMathLibTable["fmod"]  = "jnp.mod";
        gPolyMathLibTable["log"]   = "jnp.log";
        gPolyMathLibTable["log2"]  = "jnp.log2";
        gPolyMathLibTable["log10"] = "jnp.log10";
        gPolyMathLibTable["pow"]   = "jnp.power";
        gPolyMathLibTable["remainder"] =
            "remainder";  // todo: we currently rely on a custom remainder implementation in the
                          // architecture file.
        gPolyMathLibTable["rint"]  = "jnp.rint";
        gPolyMathLibTable["round"] = "jnp.round";
        gPolyMathLibTable["sin"]   = "jnp.sin";
        gPolyMathLibTable["sqrt"]  = "jnp.sqrt";
        gPolyMathLibTable["tan"]   = "jnp.tan";

        // Hyperbolic
        gPolyMathLibTable["acosh"] = "jnp.arccosh";
        gPolyMathLibTable["asinh"] = "jnp.arcsinh";
        gPolyMathLibTable["atanh"] = "jnp.arctanh";
        gPolyMathLibTable["cosh"]  = "jnp.cosh";
        gPolyMathLibTable["sinh"]  = "jnp.sinh";
        gPolyMathLibTable["tanh"]  = "jnp.tanh";

        gPolyMathLibTable["isnan"]    = "jnp.isnan";
        gPolyMathLibTable["isinf"]    = "jnp.isinf";
        gPolyMathLibTable["copysign"] = "jnp.copysign";
    }

    virtual ~JAXInstVisitor() {}

    virtual void visit(LoadVarInst* inst)
    {
        // JAX Circular Buffer Array Access Handler
        //
        // This visitor intercepts array load operations and converts them to circular buffer
        // indexing when appropriate. This provides significant performance benefits by
        // replacing O(n) jnp.roll operations with O(1) modular arithmetic.
        //
        // The circular buffer approach works by:
        // 1. Maintaining a current write index for each delay line
        // 2. Converting array[delay] to array[(current_idx - delay + size) % size]
        // 3. Incrementing the write index after each sample
        //
        // This is applied selectively to avoid breaking recursive filter structures
        // that depend on the specific semantics of roll operations.
        
        if (IndexedAddress* indexed = dynamic_cast<IndexedAddress*>(inst->fAddress)) {
            if (NamedAddress* named = dynamic_cast<NamedAddress*>(indexed->fAddress)) {
                if (fCircularBufferVars.find(named->fName) != fCircularBufferVars.end()) {
                    // This array is marked for circular buffer optimization
                    int bufferSize = fDelayLineSizes[named->fName];
                    
                    if (Int32NumInst* constIndex = dynamic_cast<Int32NumInst*>(indexed->getIndex())) {
                        // Constant index access to circular buffer
                        int delay = constIndex->fNum;
                        
                        if (delay == 0) {
                            // Direct access at current write index (most recent sample)
                            *fOut << "state[\"" << named->fName << "\"][state[\"" << named->fName << "_idx\"]]";
                        } else {
                            // Access with delay: read from (current_idx - delay + buffer_size) % buffer_size
                            // Adding buffer_size ensures the result is positive before modulo
                            *fOut << "state[\"" << named->fName << "\"][(((state[\"" << named->fName << "_idx\"] - " 
                                  << delay << ") + " << bufferSize << ") % " << bufferSize << ")]";
                        }
                        return;
                    } else {
                        // Variable index access to circular buffer (e.g., modulated delay time)
                        // This handles cases like comb_delay1.dsp where delay time is computed dynamically
                        *fOut << "state[\"" << named->fName << "\"][((state[\"" << named->fName << "_idx\"] - ";
                        indexed->getIndex()->accept(this);
                        *fOut << " + " << bufferSize << ") % " << bufferSize << ").astype(jnp.int32)]";
                        return;
                    }
                }
            }
        }
        
        // Default handling for non-circular buffer access (including roll-based arrays)
        TextInstVisitor::visit(inst);
    }

    virtual void visit(AddMetaDeclareInst* inst)
    {
        if (inst->fKey == "scale") {
            if (inst->fValue == "exp") {
                fExpSet.emplace(inst->fZone);
            } else if (inst->fValue == "log") {
                fLogSet.emplace(inst->fZone);
            } else {
                // it's linear by default
            }
        }
    }

    virtual void visit(OpenboxInst* inst)
    {
        *fOut << "ui_path.append(" << quote(inst->fName) << ")";
        EndLine(' ');
    }

    virtual void visit(CloseboxInst* inst)
    {
        *fOut << "ui_path.pop()";
        tab(fTab, *fOut);
    }

    virtual void visit(AddButtonInst* inst)
    {
        *fOut << "self.add_button(" << quote(inst->fZone) << ", ui_path, "
              << quote(inst->fLabel) << ", unnorm_funcs)";
        EndLine(' ');
        // Note: UI parameter tracking is now handled by the classifier
    }

    virtual void visit(AddSliderInst* inst)
    {
        std::string scaleMode = "";
        if (fExpSet.count(inst->fZone)) {
            scaleMode = "\"exp\"";
        } else if (fLogSet.count(inst->fZone)) {
            scaleMode = "\"log\"";
        } else {
            scaleMode = "\"linear\"";
        }

        switch (inst->fType) {
            case AddSliderInst::kHorizontal:
                // clang-format off
                *fOut << "self.add_hslider(" 
                    << quote(inst->fZone) << ", ui_path, "
                    << quote(inst->fLabel) << ", "
                    << checkReal(inst->fInit) << ", "
                    << checkReal(inst->fMin) << ", "
                    << checkReal(inst->fMax) << ", "
                    << "unnorm_funcs, "
                    << scaleMode << ")";
                break;
                // clang-format on
            case AddSliderInst::kVertical:
                // clang-format off
                *fOut << "self.add_vslider(" 
                    << quote(inst->fZone) << ", ui_path, "
                    << quote(inst->fLabel) << ", "
                    << checkReal(inst->fInit) << ", "
                    << checkReal(inst->fMin) << ", "
                    << checkReal(inst->fMax) << ", "
                    << "unnorm_funcs, "
                    << scaleMode << ")";
                break;
                // clang-format on
            case AddSliderInst::kNumEntry:
                // clang-format off
                *fOut << "self.add_nentry(" 
                    << quote(inst->fZone) << ", ui_path, "
                    << quote(inst->fLabel) << ", "
                    << checkReal(inst->fInit) << ", "
                    << checkReal(inst->fMin) << ", "
                    << checkReal(inst->fMax) << ", "
                    << checkReal(inst->fStep) << ", unnorm_funcs, \"linear\")";
                break;
                // clang-format on
        }
        EndLine(' ');
        // Track this UI parameter
        fUIParamVars.insert(inst->fZone);
    }

    virtual void visit(AddBargraphInst* inst)
    {
        // Always use setup-style - bargraphs are output-only
        *fOut << "self.add_" << ((inst->fType == AddBargraphInst::kHorizontal) ? "h" : "v") 
              << "bargraph(" << quote(inst->fZone) << ", ui_path, "
              << quote(inst->fLabel) << ", " 
              << checkReal(inst->fMin) << ", " 
              << checkReal(inst->fMax) << ", unnorm_funcs)";
        EndLine(' ');
    }

    virtual void visit(AddSoundfileInst* inst)
    {
        // Always use setup-style (no state parameter)
        *fOut << "self.add_soundfile(" << quote(inst->fSFZone) << ", ui_path, "
              << quote(inst->fLabel) << ", " << quote(inst->fURL) << ", unnorm_funcs)";
        EndLine(' ');
    }

    virtual void visit(Int32NumInst* inst) 
    { 
        if (fIsArrayIndex) {
            *fOut << inst->fNum;
        } else {
            *fOut << getJAXTypeManager()->wrapLiteral(std::to_string(inst->fNum), Typed::kInt32, fUseNumpy());
        }
    }

    virtual void visit(Int64NumInst* inst) 
    { 
        if (fIsArrayIndex) {
            *fOut << inst->fNum;
        } else {
            *fOut << getJAXTypeManager()->wrapLiteral(std::to_string(inst->fNum), Typed::kInt64, fUseNumpy());
        }
    }

    virtual void visit(FloatNumInst* inst) 
    { 
        *fOut << getJAXTypeManager()->wrapLiteral(checkFloat(inst->fNum), Typed::kFloat, fUseNumpy());
    }

    virtual void visit(DoubleNumInst* inst) 
    { 
        *fOut << getJAXTypeManager()->wrapLiteral(checkDouble(inst->fNum), Typed::kDouble, fUseNumpy());
    }

    virtual void visit(Int32ArrayNumInst* inst)
    {
        *fOut << getJAXTypeManager()->getTypePrefix(fUseNumpy()) << ".array(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << inst->fNumTable[i];
            sep = ',';
        }
        *fOut << "], dtype=" << getJAXTypeManager()->getDTypeString(Typed::kInt32, fUseNumpy()) << ")";
    }

    virtual void visit(FloatArrayNumInst* inst)
    {
        *fOut << getJAXTypeManager()->getTypePrefix(fUseNumpy()) << ".array(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkFloat(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=" << getJAXTypeManager()->getFloatTypeString(fUseNumpy()) << ")";
    }

    virtual void visit(DoubleArrayNumInst* inst)
    {
        *fOut << getJAXTypeManager()->getTypePrefix(fUseNumpy()) << ".array(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkDouble(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=" << getJAXTypeManager()->getDTypeString(Typed::kDouble, fUseNumpy()) << ")";
    }

    virtual void visit(BinopInst* inst)
    {
        if (inst->fOpcode == kXOR) {
            *fOut << "(";
            inst->fInst1->accept(this);
            *fOut << " ^ ";
            inst->fInst2->accept(this);
            *fOut << ")";
        } else {
            // Operator prededence is not like C/C++, so for simplicity, we keep the fully
            // parenthezid version
            *fOut << "(";
            inst->fInst1->accept(this);
            *fOut << " ";
            *fOut << gBinOpTable[inst->fOpcode]->fName;
            *fOut << " ";
            inst->fInst2->accept(this);
            *fOut << ")";

            bool opCodeIsBoolean = inst->fOpcode >= kGT && inst->fOpcode <= kXOR;
            if (opCodeIsBoolean && !fIsDoingWhile) {
                // these opcodes (>,>=,<,<= etc.) result in bools which should be re-cast to
                // integers
                *fOut << ".astype(jnp.int32)";
            }
        }
    }

    virtual void visit(DeclareVarInst* inst)
    {        
        if (inst->fAddress->isStaticStruct()) {
            *fOut << fTypeManager->generateType(inst->fType, inst->getName());
            // Allocation is actually done in JAXInitFieldsVisitor
        } else {
            *fOut << fTypeManager->generateType(inst->fType, inst->getName());
            if (inst->fValue) {
                *fOut << " = ";
                inst->fValue->accept(this);
            }
        }
        EndLine(' ');
    }

    virtual void visitAux(RetInst* inst, bool gen_empty)
    {
        if (inst->fResult) {
            *fOut << "return ";
            inst->fResult->accept(this);
            EndLine(' ');
        } else if (gen_empty) {
            *fOut << "return";
            EndLine(' ');
        }
    }

    virtual void visit(DropInst* inst)
    {
        if (inst->fResult) {
            inst->fResult->accept(this);
            EndLine(' ');
        }
    }

    virtual void visit(DeclareFunInst* inst)
    {
        // Already generated
        if (gFunctionSymbolTable.find(inst->fName) != gFunctionSymbolTable.end()) {
            return;
        } else {
            gFunctionSymbolTable[inst->fName] = true;
        }

        *fOut << "def " << inst->fName;
        generateFunDefArgs(inst);
        generateFunDefBody(inst);
    }

    virtual void visit(DeclareBufferIterators* inst)
    {
        // Don't generate if no channels
        if (inst->fChannels == 0) {
            return;
        }

        for (int i = 0; i < inst->fChannels; ++i) {
            *fOut << inst->fBufferName1 << i << " = " << inst->fBufferName2 << "[ " << i << ":"
                  << i + 1 << ",:]";
            tab(fTab, *fOut);
        }
    }

    virtual void generateFunDefBody(DeclareFunInst* inst)
    {
        if (inst->fCode->fCode.size() == 0) {
            *fOut << "):";
            fTab++;
            tab(fTab, *fOut);
            *fOut << "pass";
            fTab--;
            tab(fTab, *fOut);
            tab(fTab, *fOut);
        } else {
            // Function body
            *fOut << "):";
            fTab++;
            tab(fTab, *fOut);
            inst->fCode->accept(this);
            fTab--;
            back(1, *fOut);
            tab(fTab, *fOut);
        }
    }

    virtual void visit(NamedAddress* named)
    {
        const std::string& varName = named->fName;
        auto category = fClassifier.classifyVariable(varName);
        
        // Special case: fSampleRate always uses self.sample_rate
        if (varName == "fSampleRate") {
            *fOut << "self.sample_rate";
            return;
        }
        
        // Special case: inline subcontainer local variables
        if (fStateManager.inInlineSubcontainer() && 
            fStateManager.isInlineSubcontainerLocal(varName)) {
            *fOut << varName;
            return;
        }
        
        // Special case: pfPerm variables are state variables, not constants
        if (varName.find("pfPerm") == 0) {
            if ((named->isStruct() || named->isStaticStruct()) && 
                !fStateManager.inSetup() && !fStateManager.inStaticInit()) {
                *fOut << "state[\"" << varName << "\"]";
            } else {
                *fOut << varName;
            }
            return;
        }
        
        switch (category) {
            case JAXVariableClassifier::VarCategory::UI_PARAMETER:
                // UI parameters are accessed from params dict in tick
                *fOut << "params[\"" << varName << "\"]";
                break;
                
            case JAXVariableClassifier::VarCategory::SOUNDFILE:
                // Soundfile variables are module attributes
                *fOut << "self." << varName;
                break;
                
            case JAXVariableClassifier::VarCategory::CONSTANT:
                // Constants are instance attributes with _ prefix
                // Also check if it's been tracked during compilation
                if (fConstantVars.find(varName) != fConstantVars.end()) {
                    *fOut << "self._" << varName;
                } else {
                    *fOut << "self._" << varName;
                }
                break;
                
            case JAXVariableClassifier::VarCategory::STATIC_TABLE:
                // Static tables: local in static init, instance attr elsewhere
                if (fStateManager.inStaticInit()) {
                    *fOut << varName;
                } else {
                    *fOut << "self._" << varName;
                }
                break;
                
            case JAXVariableClassifier::VarCategory::WAVEFORM_DATA:
                // Waveform data: local in tick, instance attr elsewhere
                if (fStateManager.inTick()) {
                    // In tick, waveform data is converted to local JAX arrays
                    *fOut << varName;
                } else {
                    // In all other contexts (setup, static_init, inline_subcontainer),
                    // waveform data is accessed as instance attributes
                    *fOut << "self._" << varName;
                }
                break;
                
            case JAXVariableClassifier::VarCategory::READ_WRITE_TABLE:
                // Read-write tables in static init shouldn't happen
                if (fStateManager.inStaticInit()) {
                    *fOut << varName;  // Will cause runtime error
                } else if ((named->isStruct() || named->isStaticStruct()) && 
                          !fStateManager.inSetup() && !fStateManager.inStaticInit()) {
                    *fOut << "state[\"" << varName << "\"]";
                } else {
                    *fOut << varName;
                }
                break;
                
            case JAXVariableClassifier::VarCategory::BARGRAPH:
                // Bargraphs are always local variables
                *fOut << varName;
                break;
                
            case JAXVariableClassifier::VarCategory::DELAY_LINE:
                // Delay lines: handle special cases for static init
                if (fStateManager.inStaticInit()) {
                    *fOut << varName;  // Local in static init
                } else if ((named->isStruct() || named->isStaticStruct()) && 
                          !fStateManager.inSetup() && !fStateManager.inStaticInit()) {
                    *fOut << "state[\"" << varName << "\"]";
                } else {
                    *fOut << varName;
                }
                break;
                
            case JAXVariableClassifier::VarCategory::STATE_VARIABLE:
                // State variables like IOTA
                if ((named->isStruct() || named->isStaticStruct()) && 
                    !fStateManager.inSetup() && !fStateManager.inStaticInit()) {
                    *fOut << "state[\"" << varName << "\"]";
                } else {
                    *fOut << varName;
                }
                break;
                
            case JAXVariableClassifier::VarCategory::INDEX_VARIABLE:
            case JAXVariableClassifier::VarCategory::TEMPORARY:
            case JAXVariableClassifier::VarCategory::OTHER:
            default:
                // Handle special cases for setup context
                if (fStateManager.inSetup() && varName.find("SIG") != std::string::npos) {
                    *fOut << varName;  // Local temporary in setup
                } else if ((named->isStruct() || named->isStaticStruct()) && 
                          !fStateManager.inSetup() && !fStateManager.inStaticInit()) {
                    *fOut << "state[\"" << varName << "\"]";
                } else {
                    *fOut << varName;
                }
                break;
        }
    }

    /*
    Indexed address can actually be values in an array or fields in a struct type
    */
    virtual void visit(IndexedAddress* indexed)
    {
        
        if (NamedAddress* named = dynamic_cast<NamedAddress*>(indexed->fAddress)) {            
            // Check if this is a scalar delay variable
            if (isScalarDelayVar(named->fName)) {
                // For scalar delays in static init, use local variable
                if (fStateManager.inStaticInit() && (named->fName.find("Vec") != std::string::npos || named->fName.find("Rec") != std::string::npos)) {
                    *fOut << named->fName;
                } else {
                    *fOut << "state[\"" << named->fName << "\"]";
                }
                return;
            }
        }
        
        if (fUseNumpy()) {
            indexed->fAddress->accept(this);
            DeclareStructTypeInst* struct_type = isStructType(indexed->getName());
            if (struct_type) {
                Int32NumInst* field_index = static_cast<Int32NumInst*>(indexed->getIndex());
                *fOut << "[\"" << struct_type->fType->getName(field_index->fNum) << "\"]";
            } else {
                Int32NumInst* field_index = dynamic_cast<Int32NumInst*>(indexed->getIndex());
                if (field_index) {
                    *fOut << "[" << field_index->fNum << "]";
                } else {
                    *fOut << "[";
                    fIsArrayIndex = true;
                    indexed->getIndex()->accept(this);
                    fIsArrayIndex = false;
                    *fOut << "]";
                }
            }

        } else {
            // JAX mode - use .at[].set() syntax for array modifications
            indexed->fAddress->accept(this);
            DeclareStructTypeInst* struct_type = isStructType(indexed->getName());
            if (struct_type) {
                Int32NumInst* field_index = static_cast<Int32NumInst*>(indexed->getIndex());
                *fOut << "[\"" << struct_type->fType->getName(field_index->fNum) << "\"]";
            } else {
                if (fIsStoringLhs) {
                    fWillSetArray = true;
                    return;
                }

                if (fWillSetArray) {
                    *fOut << ".at";
                    fWillSetArray = false;
                }

                Int32NumInst* field_index = dynamic_cast<Int32NumInst*>(indexed->getIndex());
                if (field_index) {
                    *fOut << "[" << field_index->fNum << "]";
                } else {
                    *fOut << "[";
                    fIsArrayIndex = true;
                    indexed->getIndex()->accept(this);
                    fIsArrayIndex = false;
                    *fOut << "]";
                }
            }
        }
    }

    virtual void visit(LoadVarAddressInst* inst) { faustassert(false); }
    
    // Helper to determine if a variable is integer type based on naming convention
    bool isIntegerVariable(const std::string& name) {
        return !name.empty() && name[0] == 'i';
    }
    
    // Helper to determine if a variable is float type based on naming convention
    bool isFloatVariable(const std::string& name) {
        return !name.empty() && name[0] == 'f';
    }
    
    // Helper to check if value expression is likely integer type
    bool isIntegerExpression(ValueInst* inst) {
        // Check for integer literals
        if (dynamic_cast<Int32NumInst*>(inst) || dynamic_cast<Int64NumInst*>(inst)) {
            return true;
        }
        
        // Check for binary operations that typically produce integers
        if (BinopInst* binop = dynamic_cast<BinopInst*>(inst)) {
            // Subtraction of integers produces integer
            if (binop->fOpcode == kSub) {
                // Check if both operands are integers
                bool op1_int = dynamic_cast<Int32NumInst*>(binop->fInst1) != nullptr;
                bool op2_int = false;
                if (LoadVarInst* load = dynamic_cast<LoadVarInst*>(binop->fInst2)) {
                    if (NamedAddress* named = dynamic_cast<NamedAddress*>(load->fAddress)) {
                        op2_int = isIntegerVariable(named->fName);
                    }
                }
                return op1_int && op2_int;
            }
        }
        
        // Check for loads of variables
        if (LoadVarInst* load = dynamic_cast<LoadVarInst*>(inst)) {
            if (NamedAddress* named = dynamic_cast<NamedAddress*>(load->fAddress)) {
                // Check if it's an integer variable
                if (isIntegerVariable(named->fName)) {
                    return true;
                }
                // Check if it's a temp variable that contains an integer (like fTemp0)
                // Temp variables that start with 'f' but are assigned integer expressions
                // This is a heuristic - we assume fTemp variables assigned from integer
                // expressions are integer typed even though they have 'f' prefix
                if (named->fName.find("Temp") != std::string::npos && 
                    named->fName[0] == 'f') {
                    // For now, we'll mark this as potentially integer
                    // A more robust solution would track the type of temp variables
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // Helper to check if a variable is a scalar delay
    bool isScalarDelayVar(const std::string& name) {
        return fScalarDelayVars.find(name) != fScalarDelayVars.end();
    }
    
    // Helper to check if a variable is a bargraph
    bool isBargraphVar(const std::string& name) {
        return fClassifier.classifyVariable(name) == JAXVariableClassifier::VarCategory::BARGRAPH;
    }

    virtual void visit(StoreVarInst* inst)
    {
        // Check if we're storing to a local variable in static init
        if (fStateManager.inStaticInit() && inst->fAddress) {
            if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
                if (named->fName.find("Vec") != std::string::npos || named->fName.find("Rec") != std::string::npos) {
                    // Direct assignment to local variable
                    *fOut << named->fName << " = ";
                    inst->fValue->accept(this);
                    EndLine(' ');
                    return;
                }
            }
            // Also check IndexedAddress for scalar delay stores
            else if (IndexedAddress* indexed = dynamic_cast<IndexedAddress*>(inst->fAddress)) {
                if (NamedAddress* named = dynamic_cast<NamedAddress*>(indexed->fAddress)) {
                    if (isScalarDelayVar(named->fName) && 
                        (named->fName.find("Vec") != std::string::npos || named->fName.find("Rec") != std::string::npos)) {
                        // Direct assignment to local scalar delay variable
                        *fOut << named->fName << " = ";
                        inst->fValue->accept(this);
                        EndLine(' ');
                        return;
                    }
                }
            }
        }
        
        // Check if we're storing to a scalar delay variable
        if (IndexedAddress* indexed = dynamic_cast<IndexedAddress*>(inst->fAddress)) {
            if (NamedAddress* named = dynamic_cast<NamedAddress*>(indexed->fAddress)) {
                if (isScalarDelayVar(named->fName)) {
                    // Generate scalar assignment for scalar delays
                    *fOut << "state[\"" << named->fName << "\"]";
                    *fOut << " = ";
                    
                    // Check if we need type casting
                    bool needsCast = false;
                    std::string castType;
                    
                    // Check if we're storing to a float variable but have an integer expression
                    // EXCEPTION: Never cast _idx variables to float - they must always remain integers
                    if (isFloatVariable(named->fName) && isIntegerExpression(inst->fValue)) {
                        // Check if this is an index variable (ends with _idx)
                        if (named->fName.length() > 4 && named->fName.substr(named->fName.length() - 4) == "_idx") {
                            needsCast = false;  // Index variables must remain integers
                        } else {
                            needsCast = true;
                            // Determine float precision based on global settings
                            Typed::VarType floatType = (gGlobal->gFloatSize == 1) ? Typed::kFloat : Typed::kDouble;
                            castType = getJAXTypeManager()->getDTypeString(floatType, fUseNumpy());
                        }
                    }
                    
                    if (needsCast) {
                        *fOut << castType << "(";
                        inst->fValue->accept(this);
                        *fOut << ")";
                    } else {
                        inst->fValue->accept(this);
                    }
                    
                    EndLine(' ');
                    return;
                }
            }
        }
        
        // Check if we're storing to a bargraph variable
        std::string targetVar;
        if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
            targetVar = named->fName;
            auto varCategory = fClassifier.classifyVariable(targetVar);
            
            if (varCategory == JAXVariableClassifier::VarCategory::BARGRAPH) {
                // Generate temporary variable and sow() call for bargraph
                *fOut << targetVar << " = ";
                
                // Check if we need type casting
                bool needsCast = false;
                std::string castType;
                
                // Check if we're storing to a float variable but have an integer expression
                if (isFloatVariable(targetVar) && isIntegerExpression(inst->fValue)) {
                    needsCast = true;
                    // Determine float precision based on global settings
                    Typed::VarType floatType = (gGlobal->gFloatSize == 1) ? Typed::kFloat : Typed::kDouble;
                    castType = getJAXTypeManager()->getDTypeString(floatType, fUseNumpy());
                }
                
                if (needsCast) {
                    *fOut << castType << "(";
                    inst->fValue->accept(this);
                    *fOut << ")";
                } else {
                    inst->fValue->accept(this);
                }
                
                // Generate sow() call
                tab(fTab, *fOut);
                *fOut << "self.sow(\"intermediates\", \"" << targetVar << "\", " << targetVar << ")";
                EndLine(' ');
                return;
            }
        }
        
        // Normal store operation
        fIsStoringLhs = true;
        inst->fAddress->accept(this);
        fIsStoringLhs = false;
        *fOut << " = ";

        if (fWillSetArray && !fUseNumpy()) {
            inst->fAddress->accept(this);
            *fOut << ".set(";
            inst->fValue->accept(this);
            *fOut << ")";
        } else {
            // Reset fWillSetArray if we're in numpy mode
            fWillSetArray = false;
            // Check if we need type casting
            bool needsCast = false;
            std::string castType;
            
            // Check if we're storing to a float variable but have an integer expression
            // EXCEPTION: Never cast _idx variables to float - they must always remain integers
            if (!targetVar.empty() && isFloatVariable(targetVar) && isIntegerExpression(inst->fValue)) {
                // Check if this is an index variable (ends with _idx)
                if (targetVar.length() > 4 && targetVar.substr(targetVar.length() - 4) == "_idx") {
                    needsCast = false;  // Index variables must remain integers
                } else {
                    needsCast = true;
                    // Determine float precision based on global settings
                    Typed::VarType floatType = (gGlobal->gFloatSize == 1) ? Typed::kFloat : Typed::kDouble;
                    castType = getJAXTypeManager()->getDTypeString(floatType, fUseNumpy());
                }
            }
            
            if (needsCast) {
                *fOut << castType << "(";
                inst->fValue->accept(this);
                *fOut << ")";
            } else {
                inst->fValue->accept(this);
            }
        }

        EndLine(' ');
    }

    virtual void visit(::CastInst* inst)
    {
        if (isIntType(inst->fType->getType())) {
            *fOut << getJAXTypeManager()->getIntTypeString(fUseNumpy()) << "(";
            inst->fInst->accept(this);
            *fOut << ")";
        } else {
            *fOut << fTypeManager->generateType(inst->fType) << "(";
            inst->fInst->accept(this);
            *fOut << ")";
        }
    }

    virtual void visit(BitcastInst* inst) { faustassert(false); }

    virtual void visitCond(ValueInst* cond)
    {
        *fOut << "(";
        cond->accept(this);
        *fOut << " != 0)";
    }

    virtual void visit(Select2Inst* inst)
    {
        *fOut << "jnp.where(";
        visitCond(inst->fCond);
        *fOut << ", ";
        inst->fThen->accept(this);
        *fOut << ", ";
        inst->fElse->accept(this);
        *fOut << ")";
    }

    // Generate standard funcall (not 'method' like funcall...)
    virtual void visit(FunCallInst* inst)
    {
        std::string name = (gPolyMathLibTable.find(inst->fName) != gPolyMathLibTable.end())
                               ? gPolyMathLibTable[inst->fName]
                               : inst->fName;
        if (fUseNumpy() && name.rfind("jnp.") == 0) {
            // turn "jnp." into "np."
            name = name.substr(1, name.size() - 1);
        }
        *fOut << name << "(";
        // Compile parameters
        generateFunCallArgs(inst->fArgs.begin(), inst->fArgs.end(), inst->fArgs.size());
        *fOut << ")";
    }

    virtual void visit(IfInst* inst)
    {
        *fOut << "if ";
        visitCond(inst->fCond);
        *fOut << ":";
        fTab++;
        tab(fTab, *fOut);
        inst->fThen->accept(this);
        fTab--;
        back(1, *fOut);
        if (inst->fElse->fCode.size() > 0) {
            *fOut << "else:";
            fTab++;
            tab(fTab, *fOut);
            inst->fElse->accept(this);
            fTab--;
            back(1, *fOut);
        }
        tab(fTab, *fOut);
    }

    virtual void visit(ForLoopInst* inst)
    {
        // Don't generate empty loops...
        if (inst->fCode->size() == 0) {
            return;
        }

        fIsDoingWhile = true;

        fFinishLine = false;
        inst->fInit->accept(this);
        tab(fTab, *fOut);
        *fOut << "while ";
        inst->fEnd->accept(this);
        fIsDoingWhile = false;
        *fOut << ":";
        tab(fTab, *fOut);
        fFinishLine = true;
        fTab++;
        tab(fTab, *fOut);
        inst->fCode->accept(this);
        tab(fTab, *fOut);
        inst->fIncrement->accept(this);
        fTab--;
        back(1, *fOut);
        tab(fTab, *fOut);
    }

    virtual void visit(SimpleForLoopInst* inst)
    {
        // Don't generate empty loops...
        if (inst->fCode->size() == 0) {
            return;
        }
        
        // Skip loops based on context
        if (fStateManager.inStaticInit() || fStateManager.inInlineSubcontainer()) {
            // Check if this loop accesses tables we should skip
            struct TableChecker : public DispatchVisitor {
                bool fAccessesRWTable = false;
                bool fAccessesStaticTable = false;
                
                virtual void visit(IndexedAddress* indexed) {
                    if (NamedAddress* named = dynamic_cast<NamedAddress*>(indexed->fAddress)) {
                        if ((named->fName.find("ftbl0") == 0 && named->fName.find("SIG") != std::string::npos) || 
                            (named->fName.find("itbl0") == 0 && named->fName.find("SIG") != std::string::npos)) {
                            fAccessesStaticTable = true;
                        } else if (named->fName.find("ftbl") == 0 || named->fName.find("itbl") == 0) {
                            fAccessesRWTable = true;
                        }
                    }
                }
                
                // Also check LoadVarInst for table accesses
                virtual void visit(LoadVarInst* inst) {
                    if (NamedAddress* named = dynamic_cast<NamedAddress*>(inst->fAddress)) {
                        if ((named->fName.find("ftbl0") == 0 && named->fName.find("SIG") != std::string::npos) || 
                            (named->fName.find("itbl0") == 0 && named->fName.find("SIG") != std::string::npos)) {
                            fAccessesStaticTable = true;
                        } else if (named->fName.find("ftbl") == 0 || named->fName.find("itbl") == 0) {
                            fAccessesRWTable = true;
                        }
                    }
                }
                
                // Also check StoreVarInst for table accesses
                virtual void visit(StoreVarInst* inst) {
                    if (inst->fAddress) {
                        if (IndexedAddress* indexed = dynamic_cast<IndexedAddress*>(inst->fAddress)) {
                            visit(indexed);
                        }
                    }
                }
            };
            
            TableChecker checker;
            inst->fCode->accept(&checker);
            
            // In static init, skip read-write table loops
            if (fStateManager.inStaticInit() && checker.fAccessesRWTable) {
                tab(fTab, *fOut);
                *fOut << "# Skipping loop that fills read-write table - handled in _initialize_carry";
                tab(fTab, *fOut);
                return;
            }
            
            // In inline subcontainer (_initialize_carry), skip static table loops
            if (fStateManager.inInlineSubcontainer() && checker.fAccessesStaticTable) {
                tab(fTab, *fOut);
                *fOut << "# Skipping loop that fills static table - already handled in setup";
                tab(fTab, *fOut);
                *fOut << "pass";
                EndLine(' ');
                return;
            }
        }
        
        *fOut << "for " << inst->getName() << " in ";

        if (inst->fReverse) {
            // todo:
            *fOut << "reverse(";
            Int32NumInst* lower_bound = dynamic_cast<Int32NumInst*>(inst->fLowerBound);
            faustassert(lower_bound);
            *fOut << lower_bound->fNum << ":";
            Int32NumInst* upper_bound = dynamic_cast<Int32NumInst*>(inst->fUpperBound);
            if (upper_bound) {
                // If an Int32NumInst, we just generate it without any type information
                // (see visit(Int32NumInst* inst) which adds type information that we don't want
                // here)
                *fOut << upper_bound->fNum;
            } else {
                inst->fUpperBound->accept(this);
            }
            *fOut << ")";
        } else {
            Int32NumInst* lower_bound = dynamic_cast<Int32NumInst*>(inst->fLowerBound);
            faustassert(lower_bound);
            Int32NumInst* upper_bound = dynamic_cast<Int32NumInst*>(inst->fUpperBound);
            if (upper_bound) {
                *fOut << "range(" << lower_bound->fNum << ", " << upper_bound->fNum;
                if (upper_bound->fNum <= lower_bound->fNum) {
                    *fOut << ", -1";
                }
            } else {
                *fOut << "range(" << lower_bound->fNum << ", ";
                inst->fUpperBound->accept(this);
            }
            *fOut << "):";
        }

        fTab++;
        tab(fTab, *fOut);
        inst->fCode->accept(this);
        fTab--;
        back(1, *fOut);
        tab(fTab, *fOut);
    }

    static void cleanup() { gFunctionSymbolTable.clear(); }
};

#endif
