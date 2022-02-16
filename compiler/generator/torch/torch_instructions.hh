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

#ifndef _TORCH_INSTRUCTIONS_H
#define _TORCH_INSTRUCTIONS_H

#include <string>

#include "text_instructions.hh"
#include "struct_manager.hh"

using namespace std;

// Visitor used to initialize array fields into the DSP structure
struct TorchInitFieldsVisitor : public DispatchVisitor {
    std::ostream* fOut;
    int           fTab;
    
    TorchInitFieldsVisitor(std::ostream* out, int tab = 0) : fOut(out), fTab(tab) {}
    
    virtual void visit(DeclareVarInst* inst)
    {
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
        }
    }
    
    virtual void visit(NamedAddress* named)
    {
        // kStaticStruct are actually merged in the main DSP
        if (named->getAccess() & Address::kStruct || named->getAccess() & Address::kStaticStruct) {
            *fOut << "self.";
        }
        *fOut << named->fName;
    }
    
    static void ZeroInitializer(std::ostream* fOut, Typed* typed)
    {
        ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(typed);
        faustassert(array_type);
        if (isIntPtrType(typed->getType())) {
            *fOut << "torch.zeros(" << array_type->fSize << ", dtype=torch.int32, device=device)";
        } else if (isFloatType(typed->getType())) {
            *fOut << "torch.zeros(" << array_type->fSize << ", dtype=torch.float32, device=device)";
        } else {
            *fOut << "torch.zeros(" << array_type->fSize << ", dtype=torch.float64, device=device)";
        }
    }
    
    // Needed for waveforms
    
    virtual void visit(Int32ArrayNumInst* inst)
    {
        *fOut << "torch.tensor(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << inst->fNumTable[i];
            sep = ',';
        }
        *fOut << "], dtype=torch.int32, device=device)";
    }
    
    virtual void visit(FloatArrayNumInst* inst)
    {
        *fOut << "torch.tensor(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkFloat(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=torch.float32, device=device)";
    }
    
    virtual void visit(DoubleArrayNumInst* inst)
    {
        *fOut << "torch.tensor(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkDouble(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=torch.float64, device=device)";
    }
    
};

class TorchInstVisitor : public TextInstVisitor {
   private:
     
    /*
     Global functions names table as a static variable in the visitor
     so that each function prototype is generated as most once in the module.
     */
    static map<string, bool> gFunctionSymbolTable;

    // Polymorphic math functions
    map<string, string> gPolyMathLibTable;
    
    bool fMutateFun;
    
    string cast2FAUSTFLOAT(const string& str) { return "FAUSTFLOAT(" + str + ")"; }
    
   public:
    using TextInstVisitor::visit;

    bool is_resetting_ui = false;

    TorchInstVisitor(std::ostream* out, const string& struct_name, int tab = 0, bool mutate_fun = false)
        : TextInstVisitor(out, ".", new TorchStringTypeManager(xfloat(), "*", struct_name), tab), fMutateFun(mutate_fun)
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
        gFunctionSymbolTable["exp10f"]     = true;
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
        gFunctionSymbolTable["exp10"]     = true;
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
        gFunctionSymbolTable["coshf"]  = true;
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
        gFunctionSymbolTable["exp10l"]     = true;
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
        gPolyMathLibTable["abs"]   = "torch.abs";
        gPolyMathLibTable["max_i"] = "torch.max";
        gPolyMathLibTable["min_i"] = "torch.min";

        // Polymath mapping float version
        gPolyMathLibTable["max_f"] = "torch.max";
        gPolyMathLibTable["min_f"] = "torch.min";

        gPolyMathLibTable["fabsf"]      = "torch.abs";
        gPolyMathLibTable["acosf"]      = "torch.acos";
        gPolyMathLibTable["asinf"]      = "torch.asin";
        gPolyMathLibTable["atanf"]      = "torch.atan";
        gPolyMathLibTable["atan2f"]     = "torch.atan";
        gPolyMathLibTable["ceilf"]      = "torch.ceil";
        gPolyMathLibTable["cosf"]       = "torch.cos";
        gPolyMathLibTable["expf"]       = "torch.exp";
        gPolyMathLibTable["exp2f"]      = "torch.exp2";
        gPolyMathLibTable["exp10f"]     = "torch.exp10f";
        gPolyMathLibTable["floorf"]     = "torch.floor";
        gPolyMathLibTable["fmodf"]      = "mod";
        gPolyMathLibTable["logf"]       = "torch.log";
        gPolyMathLibTable["log2f"]      = "torch.log2";
        gPolyMathLibTable["log10f"]     = "torch.log10";
        gPolyMathLibTable["powf"]       = "torch.pow";
        gPolyMathLibTable["remainderf"] = "torch.remainder";
        gPolyMathLibTable["rintf"]      = "rint";
        gPolyMathLibTable["roundf"]     = "round";
        gPolyMathLibTable["sinf"]       = "torch.sin";
        gPolyMathLibTable["sqrtf"]      = "torch.sqrt";
        gPolyMathLibTable["tanf"]       = "torch.tan";
                             
        // Hyperbolic
        gPolyMathLibTable["acoshf"]     = "torch.acosh";
        gPolyMathLibTable["asinhf"]     = "torch.asinh";
        gPolyMathLibTable["atanhf"]     = "torch.atanh";
        gPolyMathLibTable["coshf"]      = "torch.cosh";
        gPolyMathLibTable["sinhf"]      = "torch.sinh";
        gPolyMathLibTable["tanhf"]      = "torch.tanh";
    
        gPolyMathLibTable["isnanf"]     = "torch.isnan";
        gPolyMathLibTable["isinff"]     = "torch.isinf";
        gPolyMathLibTable["copysignf"]  = "torch.copysign";
        
        // Polymath mapping double version
        gPolyMathLibTable["max_"] = "max";
        gPolyMathLibTable["min_"] = "min";

        gPolyMathLibTable["fabs"]      = "torch.abs";
        gPolyMathLibTable["acos"]      = "torch.acos";
        gPolyMathLibTable["asin"]      = "torch.asin";
        gPolyMathLibTable["atan"]      = "torch.atan";
        gPolyMathLibTable["atan2"]     = "torch.atan";
        gPolyMathLibTable["ceil"]      = "torch.ceil";
        gPolyMathLibTable["cos"]       = "torch.cos";
        gPolyMathLibTable["exp"]       = "torch.exp";
        gPolyMathLibTable["exp2"]      = "torch.exp2";
        gPolyMathLibTable["exp10"]     = "torch.exp10";
        gPolyMathLibTable["floor"]     = "torch.floor";
        gPolyMathLibTable["fmod"]      = "mod";
        gPolyMathLibTable["log"]       = "torch.log";
        gPolyMathLibTable["log2"]      = "torch.log2";
        gPolyMathLibTable["log10"]     = "torch.log10";
        gPolyMathLibTable["pow"]       = "torch.pow";
        gPolyMathLibTable["remainder"] = "remainder";
        gPolyMathLibTable["rint"]      = "rint";
        gPolyMathLibTable["round"]     = "torch.round";
        gPolyMathLibTable["sin"]       = "torch.sin";
        gPolyMathLibTable["sqrt"]      = "torch.sqrt";
        gPolyMathLibTable["tan"]       = "torch.tan";
    
        // Hyperbolic
        gPolyMathLibTable["acosh"]     = "torch.acosh";
        gPolyMathLibTable["asinh"]     = "torch.asinh";
        gPolyMathLibTable["atanh"]     = "torch.atanh";
        gPolyMathLibTable["cosh"]      = "torch.cosh";
        gPolyMathLibTable["sinh"]      = "torch.sinh";
        gPolyMathLibTable["tanh"]      = "torch.tanh";

        gPolyMathLibTable["isnan"]     = "torch.isnan";
        gPolyMathLibTable["isinf"]     = "torch.isinf";
        gPolyMathLibTable["copysign"]  = "torch.copysign";
    }

    virtual ~TorchInstVisitor() {}

    virtual void visit(AddMetaDeclareInst* inst)
    {
        // Special case
        if (inst->fZone == "0") {
            *fOut << "self.declare(ui_interface, dummy, " << quote(inst->fKey)
            << ", " << quote(inst->fValue) << ")";
        } else {
            *fOut << "self.declare(ui_interface, self." << inst->fZone << ", "
            << quote(inst->fKey) << ", " << quote(inst->fValue) << ")";
        }
        EndLine(' ');
    }

    virtual void visit(OpenboxInst* inst)
    {
        switch (inst->fOrient) {
            case OpenboxInst::kVerticalBox:
                *fOut << "ui_interface.openVerticalBox(";
                break;
            case OpenboxInst::kHorizontalBox:
                *fOut << "ui_interface.openHorizontalBox(";
                break;
            case OpenboxInst::kTabBox:
                *fOut << "ui_interface.openTabBox(";
                break;
        }
        *fOut << quote(inst->fName) << ")";
        EndLine(' ');
    }

    virtual void visit(CloseboxInst* inst)
    {
        *fOut << "ui_interface.closeBox()";
        tab(fTab, *fOut);
    }
    
    virtual void visit(AddButtonInst* inst)
    {
        if (inst->fType == AddButtonInst::kDefaultButton) {
            *fOut << "ui_interface.addButton(";
        } else {
            *fOut << "ui_interface.addCheckButton(";
        }
        // todo:
        *fOut << quote(inst->fLabel) << ", self." << inst->fZone << ")";
        EndLine(' ');
    }

    virtual void visit(AddSliderInst* inst)
    {
        switch (inst->fType) {
            case AddSliderInst::kHorizontal:
                *fOut << "ui_interface.addHorizontalSlider(";
                break;
            case AddSliderInst::kVertical:
                *fOut << "ui_interface.addVerticalSlider(";
                break;
            case AddSliderInst::kNumEntry:
                *fOut << "ui_interface.addNumEntry(";
                break;
        }
        // todo: fix this
        *fOut << quote(inst->fLabel) << ", self." << inst->fZone << ", "
              << cast2FAUSTFLOAT(checkReal(inst->fInit)) << ", "
              << cast2FAUSTFLOAT(checkReal(inst->fMin)) << ", "
              << cast2FAUSTFLOAT(checkReal(inst->fMax)) << ", "
              << cast2FAUSTFLOAT(checkReal(inst->fStep)) << ")";
        EndLine(' ');
    }

    virtual void visit(AddBargraphInst* inst)
    {
        switch (inst->fType) {
            case AddBargraphInst::kHorizontal:
                *fOut << "ui_interface.addHorizontalBargraph(";
                break;
            case AddBargraphInst::kVertical:
                *fOut << "ui_interface.addVerticalBargraph(";
                break;
        }
        // todo: fix this
        *fOut << quote(inst->fLabel) << ", self." << inst->fZone << ", "
              << cast2FAUSTFLOAT(checkReal(inst->fMin)) << ", "
              << cast2FAUSTFLOAT(checkReal(inst->fMax)) << ")";
        EndLine(' ');
    }

    virtual void visit(AddSoundfileInst* inst)
    {
        // Not supported for now
        throw faustexception("ERROR : 'soundfile' primitive not yet supported for Torch\n");
    }
    
    // virtual void visit(Int32NumInst* inst) { *fOut << "torch.tensor([" << inst->fNum << "], dtype=torch.int32, device=device)"; }
    
    // virtual void visit(Int64NumInst* inst) { *fOut << "torch.tensor([" << inst->fNum << "], dtype=torch.int64, device=device)"; }

    virtual void visit(Int32NumInst* inst) { *fOut << inst->fNum; }
    
    virtual void visit(Int64NumInst* inst) { *fOut << inst->fNum; }
    
    virtual void visit(Int32ArrayNumInst* inst)
    {
        *fOut << "torch.tensor(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << inst->fNumTable[i];
            sep = ',';
        }
        *fOut << "], dtype=torch.int32, device=device)";
    }
    
    virtual void visit(FloatArrayNumInst* inst)
    {
        *fOut << "torch.tensor(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkFloat(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=torch.float32, device=device)";
    }
    
    virtual void visit(DoubleArrayNumInst* inst)
    {
        *fOut << "torch.tensor(";
        char sep = '[';
        for (size_t i = 0; i < inst->fNumTable.size(); i++) {
            *fOut << sep << checkDouble(inst->fNumTable[i]);
            sep = ',';
        }
        *fOut << "], dtype=torch.float64, device=device)";
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
            // Operator prededence is not like C/C++, so for simplicity, we keep the fully parenthezid version
            *fOut << "(";
            inst->fInst1->accept(this);
            *fOut << " ";
            *fOut << gBinOpTable[inst->fOpcode]->fName;
            *fOut << " ";
            inst->fInst2->accept(this);
            *fOut << ")";
        }
    }
   
    virtual void visit(DeclareVarInst* inst)
    {
        if (inst->fAddress->getAccess() & Address::kStaticStruct) {
            *fOut << fTypeManager->generateType(inst->fType, inst->fAddress->getName());
            // Allocation is actually done in TorchInitFieldsVisitor
        } else {
            *fOut << fTypeManager->generateType(inst->fType, inst->fAddress->getName());
            if (inst->fValue) {
                *fOut << " = ";
                inst->fValue->accept(this);
            }
        }
        EndLine(' ');
    }

    // virtual void visit(RetInst* inst)
    // {
    //     *fOut << "return ";
    //     if (inst->fResult) {
    //         inst->fResult->accept(this);
    //     } else {
    //         *fOut << "output0";
    //     }
    //     EndLine();
    // }

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
        if (inst->fNumChannels == 0) return;
    
        for (int i = 0; i < inst->fNumChannels; ++i) {
            *fOut << inst->fBufferName1 << i << " = " << inst->fBufferName2 << "[:, " << i << "]";
            tab(fTab, *fOut);
        }
    }
    
    virtual void generateFunDefBody(DeclareFunInst* inst)
    {
        if (inst->fCode->fCode.size() == 0) {
            *fOut << "):" << endl;  // Pure prototype
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
        // kStaticStruct are actually merged in the main DSP
        if (named->getAccess() & Address::kStruct || named->getAccess() & Address::kStaticStruct) {
            *fOut << "self.";
        }
        *fOut << named->fName;
    }
    
    /*
    Indexed address can actually be values in an array or fields in a struct type
    */
    virtual void visit(IndexedAddress* indexed)
    {
        indexed->fAddress->accept(this);
        DeclareStructTypeInst* struct_type = isStructType(indexed->getName());
        if (struct_type) {
            Int32NumInst* field_index = static_cast<Int32NumInst*>(indexed->fIndex);
            *fOut << "." << struct_type->fType->getName(field_index->fNum);
        } else {
            *fOut << "[";
            Int32NumInst* field_index = dynamic_cast<Int32NumInst*>(indexed->fIndex);
            if (field_index) {
                *fOut << field_index->fNum << "]";
            } else {
                indexed->fIndex->accept(this);
                *fOut << "]";
            }
        }
    }

    virtual void visit(LoadVarAddressInst* inst)
    {
        faustassert(false);
    }
    
    virtual void visit(StoreVarInst* inst)
    {
        inst->fAddress->accept(this);
        *fOut << " = ";

        if (is_resetting_ui) {
            *fOut << "torch.nn.Parameter(torch.tensor(";
        }

        inst->fValue->accept(this);

        if (is_resetting_ui) {
            *fOut << "))";
        }

        EndLine(' ');
    }
      
    virtual void visit(::CastInst* inst)
    {
        if (isIntType(inst->fType->getType())) {
            // todo: better casting?
            // fTypeManager->generateType(inst->fType) might be torch.int32
            // *fOut << fTypeManager->generateType(inst->fType) << "(math.floor(";
            *fOut << "int(math.floor(";
            inst->fInst->accept(this);
            *fOut << "))";
        } else {
            *fOut << fTypeManager->generateType(inst->fType) << "(";
            inst->fInst->accept(this);
            *fOut << ")";
        }
    }

    // TODO : does not work, put this code in a function
    virtual void visit(BitcastInst* inst)
    {}
    
    virtual void visitCond(ValueInst* cond)
    {
        *fOut << "(";
        cond->accept(this);
        *fOut << " != 0)";
    }
    
    virtual void visit(Select2Inst* inst)
    {
        *fOut << "(";
        inst->fThen->accept(this);
        *fOut << " if ";
        visitCond(inst->fCond);
        *fOut << " else ";
        inst->fElse->accept(this);
        *fOut << ")";
    }

    // Generate standard funcall (not 'method' like funcall...)
    virtual void visit(FunCallInst* inst)
    {
        string name = (gPolyMathLibTable.find(inst->fName) != gPolyMathLibTable.end()) ? gPolyMathLibTable[inst->fName] : inst->fName;
        *fOut << name << "(";
        // Compile parameters
        generateFunCallArgs(inst->fArgs.begin(), inst->fArgs.end(), inst->fArgs.size());
        *fOut << ")";
    }
    
    virtual void visit(IfInst* inst)
    {
        *fOut << "if ";
        visitCond(inst->fCond);
        fTab++;
        tab(fTab, *fOut);
        inst->fThen->accept(this);
        fTab--;
        back(1, *fOut);
        if (inst->fElse->fCode.size() > 0) {
            *fOut << "elif";
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
        if (inst->fCode->size() == 0) return;

        fFinishLine = false;
        inst->fInit->accept(this);
        tab(fTab, *fOut);
        *fOut << "while ";
        inst->fEnd->accept(this);
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
        if (inst->fCode->size() == 0) return;
        
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
                // (see visit(Int32NumInst* inst) which adds type information that we don't want here)
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
                *fOut << "range(" << lower_bound->fNum << ", " << upper_bound->fNum-1;

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
