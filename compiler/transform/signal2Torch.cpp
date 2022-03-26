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

#include <torch/csrc/jit/api/module.h>
//#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/frontend/parser.h>

#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/jit/frontend/tree_views.h>

//#include <ATen/core/symbol.h>
//#include <ATen/record_function.h>
//#include <c10/util/Exception.h>
//#include <c10/util/StringUtil.h>
//#include <c10/util/irange.h>

using namespace torch;
using torch::jit::Module;
using torch::jit::Maybe;
using torch::jit::Expr;
using torch::jit::ExprStmt;
using torch::jit::Stmt;
using torch::jit::List;
using torch::jit::ClassDef;
using torch::jit::ClassType;
using torch::jit::ParserImpl;
using torch::jit::Ident;
using torch::jit::SourceRange;
using torch::jit::TreeRef;
using torch::jit::TreeList;
using torch::jit::Compound;
using torch::jit::Const;
using torch::jit::ListLiteral;
using torch::jit::SliceExpr;
using torch::jit::Subscript;
using torch::jit::TK_LIST;

//using torch::jit::get_python_cu;

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

TreeRef
Signal2Torch::parseStatements(Tree L, bool expect_indent = false, bool in_class = false)
{
    TreeList stmts;
    while (!isNil(L)) {
        auto stmt = parseStmt(hd(L));
        stmts.push_back(stmt);
        L = tl(L);
    }
    return create_compound(TK_LIST, _sourceRange, std::move(stmts));
}

void Signal2Torch::mapself(Tree lt)
{
    // todo: this is wrong
    if (!isNil(lt)) {
        auto blah = parseStmt(hd(lt));
        mapself(tl(lt));
    }
    return;
}

void Signal2Torch::sig2Torch(Tree L, ofstream& fout)
{

    const auto  name       = Ident::create(_sourceRange, "something");
    Maybe<Expr> superclass = Maybe<Expr>::create(_sourceRange);
    const auto  statements = parseStatements(L, /*expect_indent=*/true, /*in_class=*/true);

    auto classDef = ClassDef::create(_sourceRange, name, superclass, List<Stmt>(statements));

    //auto cu        = get_python_cu();
    //auto className = c10::QualifiedName("MyName");
    //auto classType = ClassType::create(className, cu,
    //                            /* is_module = */ false,
    //                            /* doc_string = */ "", getUnresolvedClassAttributes(classDef));

    //auto module = Module(cu, classType);

    //fout << fOut.str();
}

TreeRef
Signal2Torch::generateFConst(Tree sig, Tree type, const string& file, const string& name_aux)
{
    // Special case for 02/25/19 renaming
    string name = (name_aux == "fSamplingFreq") ? "fSampleRate" : name_aux;
    if (name_aux == "fSamplingFreq") {
        fOut << "SAMPLERATE";
    } else {
        fOut << name_aux;
    }

    // todo
    return Const::create(_sourceRange, "1");
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

TreeRef
Signal2Torch::parseStmt(Tree sig, bool in_class=false)
{
    int    i;
    double r;
    Tree   c, sel, x, y, z, u, v, var, le, label, id, ff, largs, type, name, file, sf;
    
    if (getUserData(sig)) {
        // todo: do what here?
        TreeList stmts;
        for (Tree b : sig->branches()) {
            stmts.push_back(parseStmt(b));
        }
        return create_compound(TK_LIST, _sourceRange, std::move(stmts));
    } else if (isSigInt(sig, &i)) {
        return Const::create(_sourceRange, std::to_string(i));
    } else if (isSigReal(sig, &r)) {
        // for double vals
        return Const::create(_sourceRange, std::to_string(r));
    } else if (isSigWaveform(sig)) {
        // TODO: push back a constant array
        //return;
    } else if (isSigInput(sig, &i)) {
        //fOut << "isSigInput( " << i << " )";
        //fOut << "x[:,  " << i << " ]";
        // todo: slice operator
        auto maybe_first = Maybe<Expr>::create(_sourceRange);
        auto maybe_second = Maybe<Expr>::create(_sourceRange);
        auto maybe_third  = Maybe<Expr>::create(_sourceRange);
        auto subscript_exprs = torch::jit::List<Expr>(SliceExpr::create(_sourceRange, maybe_first, maybe_second, maybe_third));

        auto input_expr = Const::create(_sourceRange, "0"); // todo: what's an Expr to refer to the arg input?

        auto sliced_input = Subscript::create(_sourceRange, input_expr, subscript_exprs);
        return sliced_input;
    } else if (isSigOutput(sig, &i, x)) {
        // todo: do what here?
        return parseStmt(x);
    } else if (isSigDelay1(sig, x)) {
        // todo: do what here?
        //return Delay::create(x);
        //return;
    } else if (isSigDelay(sig, x, y)) {
        // todo: do what here?
        //return Delay::create(x, y);
        //return;
    } else if (isSigPrefix(sig, x, y)) {
        // todo: do what here?
        // return SigPrefix::create(x, y);
        //return;
    } else if (isSigIota(sig, x)) {
        // todo: do what here?
        //return SigIota::create(x);
        //return;
    } else if (isSigBinOp(sig, &i, x, y)) {
        // todo: use parseStmt(y)
        if (std::strcmp(binopname[i], "mul") == 0) {
            return create_compound('*', _sourceRange, {parseStmt(x)});
        }

        if (std::strcmp(binopname[i], "div") == 0) {
            return create_compound('/', _sourceRange, {parseStmt(x)});
        }

        if (std::strcmp(binopname[i], "add") == 0) {
            return create_compound('+', _sourceRange, {parseStmt(x)});
        }

        if (std::strcmp(binopname[i], "sub") == 0) {
            return create_compound('-', _sourceRange, {parseStmt(x)});
        }

        return create_compound(i, _sourceRange, {parseStmt(x)});  // todo: get the right int kind. refer to binopname
    }
    
    // Foreign functions
    else if (isSigFFun(sig, ff, largs)) {
        // todo: do what here?
        parseStmt(ff);
        mapself(largs);
        //return;
    } else if (isSigFConst(sig, type, name, file)) {
        // todo: do what here?
        return generateFConst(sig, type, tree2str(file), tree2str(name));
    } else if (isSigFVar(sig, type, name, file)) {
        // todo: do what here?
        //self(name);
        //return;
    }
    
    // Tables
    else if (isSigTable(sig, id, x, y)) {
        // todo: do what here?
        //self(x);
        //self(y);
        //return;
    } else if (isSigWRTbl(sig, id, x, y, z)) {
        // todo: do what here?
        //self(x);
        //self(y);
        //self(z);
        //return;
    } else if (isSigRDTbl(sig, x, y)) {
        // todo: do what here?
        //self(x);
        //self(y);
        //return;
    }
    
    // Doc
    else if (isSigDocConstantTbl(sig, x, y)) {
        // todo: do what here?
        //self(x);
        //self(y);
        //return;
    } else if (isSigDocWriteTbl(sig, x, y, u, v)) {
        // todo: do what here?
        //self(x);
        //self(y);
        //self(u);
        //self(v);
        //return;
    } else if (isSigDocAccessTbl(sig, x, y)) {
        // todo: do what here?
        //self(x);
        //self(y);
        //return;
    }
    
    // Select2 (and Select3 expressed with Select2)
    else if (isSigSelect2(sig, sel, x, y)) {
        // todo: do a ternary?
        // torch.where( sel, x, y);
        //self(sel);
        //self(x);
        //self(y);
        //return;
    }
    
    // Table sigGen
    else if (isSigGen(sig, x)) {
        if (fVisitGen) {
            //self(x);
            //return;
        } else {
            //return;
        }
    }
    
    // Recursive signals
    else if (isProj(sig, &i, x)) {
        // todo: do what here?
        //self(x);
        //return;
    } else if (isRec(sig, var, le)) {
        // todo: do what here?
        mapself(le);
        //self(var);
        //return;
    }
    
    // Int and Float Cast
    else if (isSigIntCast(sig, x)) {
        // todo: do what here?
        //self(x);
        //return;
    } else if (isSigFloatCast(sig, x)) {
        // todo: do what here?
        //self(x);
        return;
    }
    
    // UI
    else if (isSigButton(sig, label)) {
        //return;
    } else if (isSigCheckbox(sig, label)) {
        //return;
    } else if (isSigVSlider(sig, label, c, x, y, z)) {
        //self(c), self(x), self(y), self(z);
        //return;
    } else if (isSigHSlider(sig, label, c, x, y, z)) {
        // todo: do what here?
        // torch.nn.Parameter
        //self(c);
        
        // todo: use the min, max, step
        //self(x), self(y), self(z);
        //return;
    } else if (isSigNumEntry(sig, label, c, x, y, z)) {
        //self(c), self(x), self(y), self(z);
        //return;
    } else if (isSigVBargraph(sig, label, x, y, z)) {
        //self(x), self(y), self(z);
        //return;
    } else if (isSigHBargraph(sig, label, x, y, z)) {
        //self(x), self(y), self(z);
        //return;
    }
    
    // Soundfile length, rate, channels, buffer
    else if (isSigSoundfile(sig, label)) {
        //return;
    } else if (isSigSoundfileLength(sig, sf, x)) {
        //self(sf), self(x);
        //return;
    } else if (isSigSoundfileRate(sig, sf, x)) {
        //self(sf), self(x);
        //return;
    } else if (isSigSoundfileBuffer(sig, sf, x, y, z)) {
        //self(sf), self(x), self(y), self(z);
        //return;
    }
    
    // Attach, Enable, Control
    else if (isSigAttach(sig, x, y)) {
        //self(x), self(y);
        //return;
    } else if (isSigEnable(sig, x, y)) {
        //self(x), self(y);
        //return;
    } else if (isSigControl(sig, x, y)) {
        //self(x), self(y);
        //return;
    }
    
    else if (isNil(sig)) {
        // now nil can appear in table write instructions
        return;
    } else {
        stringstream error;
        error << __FILE__ << ":" << __LINE__ << " ERROR : unrecognized signal : " << *sig << endl;
        throw faustexception(error.str());
    }

    return Const::create(_sourceRange, "0");
}
