/************************************************************************
 ************************************************************************
    FAUST compiler
    Copyright (C) 2022 GRAME, Centre National de Creation Musicale
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

#include "signal2Torch.hh"

#include <stdlib.h>
#include <cstdlib>
#include <map>
#include "Text.hh"
#include "global.hh"
#include "ppsig.hh"
#include "property.hh"
#include "signals.hh"
#include "sigtyperules.hh"
#include "tlib.hh"
#include "tree.hh"
#include "treeTransform.hh"

#include "ensure.hh"
//#include "exception.hh"
//#include "fir_to_fir.hh"
#include "floats.hh"
//#include "global.hh"
#include "instructions.hh"
#include "old_occurences.hh"

//-------------------------SignalVisitor-------------------------------
// An identity transformation on signals. Can be used to test
// that everything works, and as a pattern for real transformations.
//----------------------------------------------------------------------

// TO COMPLETE
static const char* binopname[] = {"add", "sub", "mul", "div", "%", "<<", ">>", "ge", "le", "geq", "leq", "==", "!=", "&", "|", "^"};

///*****************************************************************************
// vector name property
// *****************************************************************************/
//
///**
// * Set the vector name property of a signal, the name of the vector used to
// * store the previous values of the signal to implement a delay.
// * @param sig the signal expression.
// * @param vname the string representing the vector name.
// * @return true is already compiled
// */
//void Signal2Torch::setVectorNameProperty(Tree sig, const string& vname)
//{
//    faustassert(vname.size() > 0);
//    fVectorProperty.set(sig, vname);
//}
//
///**
// * Get the vector name property of a signal, the name of the vector used to
// * store the previous values of the signal to implement a delay.
// * @param sig the signal expression.
// * @param vname the string where to store the vector name.
// * @return true if the signal has this property, false otherwise
// */
//
//bool Signal2Torch::getVectorNameProperty(Tree sig, string& vname)
//{
//    return fVectorProperty.get(sig, vname);
//}
//
//void Signal2Torch::setTableNameProperty(Tree sig, const string& name)
//{
//    faustassert(name.size() > 0);
//    fTableProperty.set(sig, name);
//}
//
//bool Signal2Torch::getTableNameProperty(Tree sig, string& name)
//{
//    return fTableProperty.get(sig, name);
//}
//
void Signal2Torch::sig2Torch(Tree L, ofstream& fout)
{
    fOut << "\n##\n";
    fOut << "def forward(self, x):\n";
    tab(++indent, fOut);
    while (!isNil(L)) {
        self(hd(L)); // comment
        L = tl(L);
        if (!isNil(L)) fOut << ", ";
    }
    tab(--indent, fOut);
    fOut << "\n##\n";
    fout << fOut.str();
}

void Signal2Torch::generateFConst(Tree sig, Tree type, const string& file, const string& name_aux)
{
    // Special case for 02/25/19 renaming
    string name = (name_aux == "fSamplingFreq") ? "fSampleRate" : name_aux;
    fOut << "(";
    if (name_aux == "fSamplingFreq") {
        fOut << "SAMPLERATE";
    } else {
        fOut << name_aux;
    }
    fOut << ")";
}
//
//#define _DNF_ 1
//
//
//#if _DNF_
//#define CND2CODE dnf2code
//#else
//#define CND2CODE cnf2code
//#endif
//
//ValueInst* Signal2Torch::dnf2code(Tree cc)
//{
//    if (cc == gGlobal->nil) return InstBuilder::genNullValueInst();
//    Tree c1 = hd(cc);
//    cc      = tl(cc);
//    if (cc == gGlobal->nil) {
//        return and2code(c1);
//    } else {
//        return InstBuilder::genOr(and2code(c1), dnf2code(cc));
//    }
//}
//
//ValueInst* Signal2Torch::and2code(Tree cs)
//{
//    if (cs == gGlobal->nil) return InstBuilder::genNullValueInst();
//    Tree c1 = hd(cs);
//    cs      = tl(cs);
//    if (cs == gGlobal->nil) {
//        return self(c1);
//    } else {
//        return InstBuilder::genAnd(self(c1), and2code(cs));
//    }
//}
//
//ValueInst* Signal2Torch::cnf2code(Tree cs)
//{
//    if (cs == gGlobal->nil) return InstBuilder::genNullValueInst();
//    Tree c1 = hd(cs);
//    cs      = tl(cs);
//    if (cs == gGlobal->nil) {
//        return or2code(c1);
//    } else {
//        return InstBuilder::genAnd(or2code(c1), cnf2code(cs));
//    }
//}
//
//ValueInst* Signal2Torch::or2code(Tree cs)
//{
//    if (cs == gGlobal->nil) return InstBuilder::genNullValueInst();
//    Tree c1 = hd(cs);
//    cs      = tl(cs);
//    if (cs == gGlobal->nil) {
//        return self(c1);
//    } else {
//        return InstBuilder::genOr(self(c1), or2code(cs));
//    }
//}
//
//// Temporary implementation for test purposes
//ValueInst* Signal2Torch::getConditionCode(Tree sig)
//{
//    Tree cc = fConditionProperty[sig];
//    if ((cc != 0) && (cc != gGlobal->nil)) {
//        CND2CODE(cc);
//    } else {
//        return InstBuilder::genNullValueInst();
//    }
//}

/*****************************************************************************
 RECURSIONS
 *****************************************************************************/

///**
// * Generate code for a projection of a group of mutually recursive definitions
// */
//void Signal2Torch::generateRecProj(Tree sig, Tree r, int i)
//{
//    string     vname;
//    Tree       var, le;
//    ValueInst* res;
//
//    if (!getVectorNameProperty(sig, vname)) {
//        ensure(isRec(r, var, le));
//        generateRec(r, var, le, i);
//        ensure(getVectorNameProperty(sig, vname));
//    }
//}

///**
// * Generate code for a group of mutually recursive definitions
// */
//void Signal2Torch::generateRec(Tree sig, Tree var, Tree le, int index)
//{
//    int N = len(le);
//
//    ValueInst*             res = nullptr;
//    vector<bool>           used(N);
//    vector<int>            delay(N);
//    vector<string>         vname(N);
//    vector<Typed::VarType> ctype(N);
//
//    // Prepare each element of a recursive definition
//    for (int i = 0; i < N; i++) {
//        Tree e = sigProj(i, sig);  // recreate each recursive definition
//        if (fOccMarkup->retrieve(e)) {
//            // This projection is used
//            used[i] = true;
//            getTypedNames(getCertifiedSigType(e), "Rec", ctype[i], vname[i]);
//            setVectorNameProperty(e, vname[i]);
//            delay[i] = fOccMarkup->retrieve(e)->getMaxDelay();
//        } else {
//            // This projection is not used therefore
//            // we should not generate code for it
//            used[i] = false;
//        }
//    }
//
//    // Generate delayline for each element of a recursive definition
//    for (int i = 0; i < N; i++) {
//        if (used[i]) {
//            Address::AccessType var_access;
//            ValueInst*          ccs = getConditionCode(nth(le, i));
//            generateDelayLine(self(nth(le, i)), ctype[i], vname[i], delay[i], var_access, ccs);
//        }
//    }
//}

/*****************************************************************************
 CACHE CODE
 *****************************************************************************/

//void Signal2Torch::getTypedNames(::Type t, const string& prefix, Typed::VarType& ctype, string& vname)
//{
//    if (t->nature() == kInt) {
//        ctype = Typed::kInt32;
//        vname = subst("i$0", gGlobal->getFreshID(prefix));
//    } else {
//        ctype = itfloat();
//        vname = subst("f$0", gGlobal->getFreshID(prefix));
//    }
//}

void Signal2Torch::visit(Tree sig)
{
    int    i;
    double r;
    Tree   c, sel, x, y, z, u, v, var, le, label, id, ff, largs, type, name, file, sf;
    
    if (getUserData(sig)) {
        for (Tree b : sig->branches()) {
            self(b);
        }
        return;
    } else if (isSigInt(sig, &i)) {
        fOut << "torch.Tensor([" << i << "], dtype=torch.int32)";
        return;
    } else if (isSigReal(sig, &r)) {
        fOut << "torch.Tensor([" << r << "], dtype=torch.float32)";
        return;
    } else if (isSigWaveform(sig)) {
        fOut << "isSigWaveform";
        return;
    } else if (isSigInput(sig, &i)) {
        tab(++indent, fOut);
        //fOut << "isSigInput( " << i << " )";
        fOut << "x[:,  " << i << " ]";
        tab(--indent, fOut);
        return;
    } else if (isSigOutput(sig, &i, x)) {
        fOut << "isSigOutput( ";
        self(x);
        fOut << " )";
        return;
    } else if (isSigDelay1(sig, x)) {
        fOut << "isSigDelay1( ";
        self(x);
        fOut << " )";
        return;
    } else if (isSigDelay(sig, x, y)) {
        fOut << "isSigDelay( ";
        self(x);
        fOut << " , ";
        self(y);
        fOut << " )";        
        return;
    } else if (isSigPrefix(sig, x, y)) {
        fOut << "isSigPrefix( ";
        self(x);
        fOut << " , ";
        self(y);
        fOut << " )";   
        return;
    } else if (isSigIota(sig, x)) {
        fOut << "isSigIota( ";
        self(x);
        fOut << " )";
        return;
    } else if (isSigBinOp(sig, &i, x, y)) {

        if (std::strcmp(binopname[i], "mul") == 0) {
            fOut << "(";
            self(x);
            fOut << "*";
            self(y);
            fOut << ")";
            return;
        }

        if (std::strcmp(binopname[i], "div") == 0) {
            fOut << "(";
            self(x);
            fOut << "/";
            self(y);
            fOut << ")";
            return;
        }

        if (std::strcmp(binopname[i], "add") == 0) {
            fOut << "(";
            self(x);
            fOut << "+";
            self(y);
            fOut << ")";
            return;
        }

        if (std::strcmp(binopname[i], "sub") == 0) {
            fOut << "(";
            self(x);
            fOut << "-";
            self(y);
            fOut << ")";
            return;
        }

        fOut << "torch." << binopname[i] << "(";
        self(x);
        fOut << ", ";
        self(y);
        fOut << ")";
        return;
    }
    
    // Foreign functions
    else if (isSigFFun(sig, ff, largs)) {
        fOut << "isSigFFun.";
        self(ff);
        fOut << "(";
        mapself(largs);
        fOut << ")";
        return;
    } else if (isSigFConst(sig, type, name, file)) {
        generateFConst(sig, type, tree2str(file), tree2str(name));
        return;
    } else if (isSigFVar(sig, type, name, file)) {
        fOut << "isSigFVar(";
        //self(name);
        fOut << ")";
        return;
    }
    
    // Tables
    else if (isSigTable(sig, id, x, y)) {
        fOut << "isSigTable(";
        self(x);
        fOut << ", ";
        self(y);
        fOut << ")";
        return;
    } else if (isSigWRTbl(sig, id, x, y, z)) {
        fOut << "isSigWRTbl(";
        self(x);
        fOut << ", ";
        self(y);
        fOut << ", ";
        self(z);
        fOut << ")";
        return;
    } else if (isSigRDTbl(sig, x, y)) {
        fOut << "isSigRDTbl(";
        self(x);
        fOut << ", ";
        self(y);
        fOut << ")";
        return;
    }
    
    // Doc
    else if (isSigDocConstantTbl(sig, x, y)) {
        fOut << "isSigDocConstantTbl(";
        self(x);
        fOut << ", ";
        self(y);
        fOut << ")";
        return;
    } else if (isSigDocWriteTbl(sig, x, y, u, v)) {
        self(x);
        self(y);
        self(u);
        self(v);
        return;
    } else if (isSigDocAccessTbl(sig, x, y)) {
        self(x);
        self(y);
        return;
    }
    
    // Select2 (and Select3 expressed with Select2)
    else if (isSigSelect2(sig, sel, x, y)) {
        fOut << "torch.where(";
        self(sel);
        fOut << ", ";
        self(x);
        fOut << ", ";
        self(y);
        fOut << ")";
        return;
    }
    
    // Table sigGen
    else if (isSigGen(sig, x)) {
        if (fVisitGen) {
            fOut << "isSigGen(";
            self(x);
            fOut << ")";
            return;
        } else {
            fOut << "notIsGen";
            return;
        }
    }
    
    // Recursive signals
    else if (isProj(sig, &i, x)) {

        fOut << "isProj(" << i << ", ";
        tab(++indent, fOut);
        self(x);
        tab(--indent, fOut);
        fOut << ")\n";
        return;
    } else if (isRec(sig, var, le)) {
        fOut << "isRec(le = ";
        tab(++indent, fOut);
        mapself(le);
        tab(indent, fOut);
        //fOut << ", var = ";
        //self(var);
        tab(--indent, fOut);
        fOut << ")\n";

        return;
    }
    
    // Int and Float Cast
    else if (isSigIntCast(sig, x)) {
        fOut << "(";
        self(x);
        fOut << ").type(torch.int32)";
        return;
    } else if (isSigFloatCast(sig, x)) {
        fOut << "(";
        self(x);
        fOut << ").type(torch.float32)";
        return;
    }
    
    // UI
    else if (isSigButton(sig, label)) {
        return;
    } else if (isSigCheckbox(sig, label)) {
        return;
    } else if (isSigVSlider(sig, label, c, x, y, z)) {
        self(c), self(x), self(y), self(z);
        return;
    } else if (isSigHSlider(sig, label, c, x, y, z)) {

        fOut << "torch.nn.Parameter(";
        self(c);
        fOut << ")";
        
        // todo: use the min, max, step
        //self(x), self(y), self(z);
        return;
    } else if (isSigNumEntry(sig, label, c, x, y, z)) {
        self(c), self(x), self(y), self(z);
        return;
    } else if (isSigVBargraph(sig, label, x, y, z)) {
        self(x), self(y), self(z);
        return;
    } else if (isSigHBargraph(sig, label, x, y, z)) {
        self(x), self(y), self(z);
        return;
    }
    
    // Soundfile length, rate, channels, buffer
    else if (isSigSoundfile(sig, label)) {
        return;
    } else if (isSigSoundfileLength(sig, sf, x)) {
        self(sf), self(x);
        return;
    } else if (isSigSoundfileRate(sig, sf, x)) {
        self(sf), self(x);
        return;
    } else if (isSigSoundfileBuffer(sig, sf, x, y, z)) {
        self(sf), self(x), self(y), self(z);
        return;
    }
    
    // Attach, Enable, Control
    else if (isSigAttach(sig, x, y)) {
        self(x), self(y);
        return;
    } else if (isSigEnable(sig, x, y)) {
        self(x), self(y);
        return;
    } else if (isSigControl(sig, x, y)) {
        self(x), self(y);
        return;
    }
    
    else if (isNil(sig)) {
        // now nil can appear in table write instructions
        fOut << "isNil";
        return;
    } else {
        stringstream error;
        error << __FILE__ << ":" << __LINE__ << " ERROR : unrecognized signal : " << *sig << endl;
        throw faustexception(error.str());
    }
}
