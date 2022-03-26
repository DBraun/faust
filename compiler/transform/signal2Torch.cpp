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

#include <torch/csrc/jit/frontend/tracer.h>

//-------------------------SignalVisitor-------------------------------
// An identity transformation on signals. Can be used to test
// that everything works, and as a pattern for real transformations.
//----------------------------------------------------------------------

// TO COMPLETE
static const char* binopname[] = {"add", "sub", "mul", "div", "%", "<<", ">>", "ge", "le", "geq", "leq", "==", "!=", "&", "|", "^"};

TreeRef
Signal2Torch::parseStatements(Tree L, bool expect_indent, bool in_class)
{
    TreeList stmts;
    while (!isNil(L)) {
        std::cerr << "parseStatements1(" << std::endl;
        auto stmt = parseStmt(hd(L));
        stmts.push_back(stmt);
        std::cerr << "parseStatements2(" << std::endl;
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

#include <torch/torch.h>

void Signal2Torch::sig2Torch(Tree L, ofstream& fout)
{

    std::cerr << "sig2Torch(" << std::endl;

    auto source  = std::make_shared<torch::jit::Source>("");
    _sourceRange = SourceRange(source, 0, 0);

    const auto  name       = Ident::create(_sourceRange, "something");
    Maybe<Expr> superclass = Maybe<Expr>::create(_sourceRange);
    const auto  statements = parseStatements(L, /*expect_indent=*/true, /*in_class=*/true);

    std::cerr << "ClassDef::create" << std::endl;
    auto classDef = ClassDef::create(_sourceRange, name, superclass, torch::jit::List<Stmt>(statements));

    torch::jit::Node* node = nullptr;
    std::shared_ptr<torch::jit::tracer::TracingState> tracer_state;
    std::cerr << "isTracing" << std::endl;
    if (torch::jit::tracer::isTracing()) {
        std::cerr << "getTracingState" << std::endl;
        tracer_state = torch::jit::tracer::getTracingState();
        
        at::Symbol op_name;
        op_name = torch::jit::Symbol::fromQualString("aten::__ilshift__");
        node    = tracer_state->graph->create(op_name, /*num_outputs=*/0);
        torch::jit::tracer::recordSourceLocation(node);

        auto myTensor = torch::zeros({1, 100});

        torch::jit::tracer::addInputs(node, "self", myTensor);
        //torch::jit::tracer::addInputs(node, "input_signal", other);  // todo
        tracer_state->graph->insertNode(node);

        torch::jit::tracer::setTracingState(nullptr);
    }
    //TypeDefault::__ilshift__(self, other);  // todo
    if (tracer_state) {
        torch::jit::tracer::setTracingState(std::move(tracer_state));
        fout << tracer_state->graph->toString();
        //torch::jit::tracer::addOutput(node, self);  // todo
    }

    //tracer_state->graph->print(fout);
    

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

  template <typename T>
List<T> parseList(int begin, int sep, int end, T (ParserImpl::*parse)())
{
    auto           r = L.cur().range;
    std::vector<T> elements;
    parseSequence(begin, sep, end, [&] { elements.emplace_back((this->*parse)()); });
    return List<T>::create(r, elements);
}

TreeRef
Signal2Torch::parseStmt(Tree sig, bool in_class)
{
    std::cerr << "parseStmt(" << std::endl;
    int    i;
    double r;
    Tree   c, sel, x, y, z, u, v, var, le, label, id, ff, largs, type, name, file, sf;
    
    if (getUserData(sig)) {
        // todo: do what here?
        
        TreeList stmts;
        for (Tree b : sig->branches()) {
            std::cerr << "getUserData(" << std::endl;
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
        std::cerr << "isSigInput(" << std::endl;
        //fOut << "isSigInput( " << i << " )";
        //fOut << "x[:,  " << i << " ]";
        // todo: slice operator
        auto maybe_first  = Const::create(_sourceRange, std::to_string(i));
        auto maybe_second = Maybe<Expr>::create(_sourceRange);
        auto maybe_third  = Maybe<Expr>::create(_sourceRange);
        std::cerr << "SliceExpr::create(" << std::endl;
        //auto sliceexpr = SliceExpr::create(_sourceRange, maybe_first, maybe_second, maybe_third);
        std::cerr << "torch::jit::List<Expr>" << std::endl;
        auto sliceexpr = std::vector<torch::jit::Expr>();
        std::cerr << "maybe_first" << std::endl;
        sliceexpr.push_back(maybe_first);
        //std::cerr << "maybe_second" << std::endl;
        //sliceexpr.push_back((torch::jit::Expr)maybe_second);
        //std::cerr << "maybe_third" << std::endl;
        //sliceexpr.push_back((torch::jit::Expr)maybe_third);
        std::cerr << "torch::jit::List<Expr>::create" << std::endl;
        auto subscript_exprs = torch::jit::List<Expr>::create(_sourceRange, sliceexpr);
        auto input_expr = Const::create(_sourceRange, "0"); // todo: what's an Expr to refer to the arg input?
        std::cerr << "isSigInput-Subscript::create" << std::endl;
        auto sliced_input = Subscript::create(_sourceRange, input_expr, subscript_exprs);
        std::cerr << "isSigInput-return" << std::endl;
        return sliced_input;
    } else if (isSigOutput(sig, &i, x)) {
        std::cerr << "isSigOutput(" << std::endl;
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
        std::cerr << "isSigBinOp(" << std::endl;
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
        std::cerr << "isSigFFun(" << std::endl;
        // todo: do what here?
        parseStmt(ff);
        mapself(largs);
        //return;
    } else if (isSigFConst(sig, type, name, file)) {
        // todo: do what here?
        std::cerr << "isSigFConst(" << std::endl;
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
        std::cerr << "isRec(" << std::endl;
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
        //return;
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
        //return;
    } else {
        stringstream error;
        error << __FILE__ << ":" << __LINE__ << " ERROR : unrecognized signal : " << *sig << endl;
        throw faustexception(error.str());
    }

    std::cerr << "Const::create(" << std::endl;
    return Const::create(_sourceRange, "0");
}
