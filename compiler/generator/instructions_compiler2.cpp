/************************************************************************
 ************************************************************************
 FAUST compiler
 Copyright (C) 2017-2022 GRAME, Centre National de Creation Musicale
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
//#include "instructions_compiler.cpp"
#include "ensure.hh"
#include "instructions_compiler2.hh"
#include "instructions.hh"
#include "sigtyperules.hh"
#include "ppsig.hh"
#include "prim2.hh"
#include "torch_code_container.hh"

static inline BasicTyped* genBasicFIRTyped(int sig_type)
{
    return InstBuilder::genBasicTyped((sig_type == kInt) ? Typed::kInt32 : itfloat());
}

InstructionsCompiler2::InstructionsCompiler2(CodeContainer* container) : InstructionsCompiler(container)
{
}

//StatementInst* InstructionsCompiler2::generateInitArray(const string& vname, Typed::VarType ctype, int delay)
//{
//    ValueInst*  init  = InstBuilder::genTypedZero(ctype);
//    BasicTyped* typed = InstBuilder::genBasicTyped(ctype);
//    string      index = gGlobal->getFreshID("l");
//
//    // Generates table declaration
//    pushDeclare(InstBuilder::genDecStructVar(vname, InstBuilder::genArrayTyped(typed, delay)));
//
//    ValueInst* upperBound = InstBuilder::genInt32NumInst(delay);
//    // Generates init table loop
//    SimpleForLoopInst* loop = InstBuilder::genSimpleForLoopInst(index, upperBound);
//
//    LoadVarInst* loadVarInst = InstBuilder::genLoadVarInst(InstBuilder::genNamedAddress(index, Address::kLoop));
//    loop->pushFrontInst(InstBuilder::genStoreArrayStructVar(vname, loadVarInst, init));
//    return loop;
//}
//
//StatementInst* InstructionsCompiler2::generateShiftArray(const string& vname, int delay)
//{
//    string index = gGlobal->getFreshID("j");
//
//    ValueInst*         upperBound  = InstBuilder::genInt32NumInst(delay);
//    ValueInst*         lowerBound  = InstBuilder::genInt32NumInst(1);
//    
//    SimpleForLoopInst* loop        = InstBuilder::genSimpleForLoopInst(index, upperBound, lowerBound, true);
//    LoadVarInst*       loadVarInst = InstBuilder::genLoadVarInst(InstBuilder::genNamedAddress(index, Address::kLoop));
//    ValueInst*         load_value2 = InstBuilder::genSub(loadVarInst, InstBuilder::genInt32NumInst(1));
//    ValueInst*         load_value3 = InstBuilder::genLoadArrayStructVar(vname, load_value2);
//
//    loop->pushFrontInst(InstBuilder::genStoreArrayStructVar(vname, loadVarInst, load_value3));
//    return loop;
//}
//
//StatementInst* InstructionsCompiler2::generateCopyArray(const string& vname_to, const string& vname_from, int size)
//{
//    string index = gGlobal->getFreshID("j");
//
//    ValueInst*         upperBound  = InstBuilder::genInt32NumInst(size);
//    SimpleForLoopInst* loop        = InstBuilder::genSimpleForLoopInst(index, upperBound);
//    LoadVarInst*       loadVarInst = InstBuilder::genLoadVarInst(InstBuilder::genNamedAddress(index, Address::kLoop));
//    ValueInst*         load_value  = InstBuilder::genLoadArrayStructVar(vname_from, loadVarInst);
//
//    loop->pushFrontInst(InstBuilder::genStoreArrayStackVar(vname_to, loadVarInst, load_value));
//    return loop;
//}

ValueInst* InstructionsCompiler2::generateDelayLine(ValueInst* exp, Typed::VarType ctype, const string& vname, int mxd,
                                                    Address::AccessType& var_access, ValueInst* ccs)
{
    if (mxd == 0) {
        // Generate scalar use
        if (dynamic_cast<NullValueInst*>(ccs)) {
            pushComputeDSPMethod(InstBuilder::genDecStackVar(vname, InstBuilder::genBasicTyped(ctype), exp));
        } else {
            pushPreComputeDSPMethod(InstBuilder::genDecStackVar(vname, InstBuilder::genBasicTyped(ctype),
                                                                InstBuilder::genTypedZero(ctype)));
            pushComputeDSPMethod(InstBuilder::genControlInst(ccs, InstBuilder::genStoreStackVar(vname, exp)));
        }

    } else if (mxd < gGlobal->gMaxCopyDelay) {
        // Generates table init
        pushClearMethod(generateInitArray(vname, ctype, mxd + 1));

        string            funname = "delay";
        list<ValueInst*>  args_value;
        list<NamedTyped*> args_types;

        args_types.push_back(InstBuilder::genNamedTyped("dummy0", genBasicFIRTyped(kReal)));
        args_value.push_back(exp);

        args_types.push_back(InstBuilder::genNamedTyped("dummy1", genBasicFIRTyped(kReal)));
        args_value.push_back(InstBuilder::genInt32NumInst(mxd));

        pushComputeDSPMethod(InstBuilder::genStoreStackVar(vname, InstBuilder::genFunCallInst(funname, args_value)));

    } else {
        int N = pow2limit(mxd + 1);
        if (N <= gGlobal->gMaskDelayLineThreshold) {
            ensureIotaCode();

            // Generates table init
            pushClearMethod(generateInitArray(vname, ctype, N));

            // Generate table use
            if (gGlobal->gComputeIOTA) {  // Ensure IOTA base fixed delays are computed once
                if (fIOTATable.find(N) == fIOTATable.end()) {
                    string   iota_name = subst("i$0", gGlobal->getFreshID(fCurrentIOTA + "_temp"));
                    FIRIndex value2    = FIRIndex(InstBuilder::genLoadStructVar(fCurrentIOTA)) & FIRIndex(N - 1);

                    pushPreComputeDSPMethod(InstBuilder::genDecStackVar(iota_name, InstBuilder::genInt32Typed(),
                                                                        InstBuilder::genInt32NumInst(0)));
                    pushComputeDSPMethod(
                        InstBuilder::genControlInst(ccs, InstBuilder::genStoreStackVar(iota_name, value2)));

                    fIOTATable[N] = iota_name;
                }

                pushComputeDSPMethod(InstBuilder::genControlInst(
                    ccs, InstBuilder::genStoreArrayStructVar(vname, InstBuilder::genLoadStackVar(fIOTATable[N]), exp)));

            } else {
                FIRIndex value2 = FIRIndex(InstBuilder::genLoadStructVar(fCurrentIOTA)) & FIRIndex(N - 1);
                pushComputeDSPMethod(
                    InstBuilder::genControlInst(ccs, InstBuilder::genStoreArrayStructVar(vname, value2, exp)));
            }
        } else {
            // 'select' based delay
            string widx_tmp_name = vname + "_widx_tmp";
            string widx_name     = vname + "_widx";

            // Generates table write index
            pushDeclare(InstBuilder::genDecStructVar(widx_name, InstBuilder::genInt32Typed()));
            pushInitMethod(InstBuilder::genStoreStructVar(widx_name, InstBuilder::genInt32NumInst(0)));

            // Generates table init
            pushClearMethod(generateInitArray(vname, ctype, mxd + 1));

            // int w = widx;
            pushComputeDSPMethod(InstBuilder::genControlInst(
                ccs, InstBuilder::genDecStackVar(widx_tmp_name, InstBuilder::genBasicTyped(Typed::kInt32),
                                                 InstBuilder::genLoadStructVar(widx_name))));

            // dline[w] = v;
            pushComputeDSPMethod(InstBuilder::genControlInst(
                ccs, InstBuilder::genStoreArrayStructVar(vname, InstBuilder::genLoadStackVar(widx_tmp_name), exp)));

            // w = w + 1;
            FIRIndex widx_tmp1 = FIRIndex(InstBuilder::genLoadStackVar(widx_tmp_name));
            pushPostComputeDSPMethod(
                InstBuilder::genControlInst(ccs, InstBuilder::genStoreStackVar(widx_tmp_name, widx_tmp1 + 1)));

            // w = ((w == delay) ? 0 : w);
            FIRIndex widx_tmp2 = FIRIndex(InstBuilder::genLoadStackVar(widx_tmp_name));
            pushPostComputeDSPMethod(InstBuilder::genControlInst(
                ccs,
                InstBuilder::genStoreStackVar(widx_tmp_name, InstBuilder::genSelect2Inst(widx_tmp2 == FIRIndex(mxd + 1),
                                                                                         FIRIndex(0), widx_tmp2))));
            // *widx = w
            pushPostComputeDSPMethod(InstBuilder::genControlInst(
                ccs, InstBuilder::genStoreStructVar(widx_name, InstBuilder::genLoadStackVar(widx_tmp_name))));
        }
    }

    return exp;
}

/**
 * Generate code for accessing a delayed signal. The generated code depend of
 * the maximum delay attached to exp.
 */
ValueInst* InstructionsCompiler2::generateDelay(Tree sig, Tree exp, Tree delay)
{
    ValueInst* code = CS(exp);  // Ensure exp is compiled to have a vector name
    int        mxd  = fOccMarkup->retrieve(exp)->getMaxDelay();
    string     vname;

    if (!getVectorNameProperty(exp, vname)) {
        if (mxd == 0) {
            // cerr << "it is a pure zero delay : " << code << endl;
            return code;
        } else {
            stringstream error;
            error << "ERROR : no vector name for : " << ppsig(exp) << endl;
            throw faustexception(error.str());
        }
    }

    if (mxd == 0) {
        // not a real vector name but a scalar name
        return InstBuilder::genLoadStackVar(vname);

    } else if (mxd < gGlobal->gMaxCopyDelay) {
        int d;
        if (isSigInt(delay, &d)) {
            //return InstBuilder::genLoadArrayStructVar(vname, CS(delay));
            return InstBuilder::genLoadStackVar(vname);
        } else {
            return generateCacheCode(sig, InstBuilder::genLoadArrayStructVar(vname, CS(delay)));
        }
    } else {
        int N = pow2limit(mxd + 1);
        if (N <= gGlobal->gMaskDelayLineThreshold) {
            ensureIotaCode();

            FIRIndex value2 = (FIRIndex(InstBuilder::genLoadStructVar(fCurrentIOTA)) - CS(delay)) & FIRIndex(N - 1);
            return generateCacheCode(sig, InstBuilder::genLoadArrayStructVar(vname, value2));
        } else {
            string ridx_name = gGlobal->getFreshID(vname + "_ridx_tmp");

            // int ridx = widx - delay;
            FIRIndex widx1 = FIRIndex(InstBuilder::genLoadStructVar(vname + "_widx"));
            pushComputeDSPMethod(
                InstBuilder::genDecStackVar(ridx_name, InstBuilder::genBasicTyped(Typed::kInt32), widx1 - CS(delay)));

            // dline[((ridx < 0) ? ridx + delay : ridx)];
            FIRIndex ridx1 = FIRIndex(InstBuilder::genLoadStackVar(ridx_name));
            FIRIndex ridx2 = FIRIndex(InstBuilder::genSelect2Inst(ridx1 < 0, ridx1 + FIRIndex(mxd + 1), ridx1));
            return generateCacheCode(sig, InstBuilder::genLoadArrayStructVar(vname, ridx2));
        }
    }
}

/**
 * Generate code for a projection of a group of mutually recursive definitions
 */
ValueInst* InstructionsCompiler2::generateRecProj(Tree sig, Tree r, int i)
{
    string     vname;
    Tree       var, le;
    ValueInst* res;

    if (!getVectorNameProperty(sig, vname)) {
        ensure(isRec(r, var, le));
        res = generateRec(sig, r, var, le, i);
        ensure(getVectorNameProperty(sig, vname));
    } else {
        res = InstBuilder::genNullValueInst();  // Result not used
    }
    return res;
}

/**
 * Generate code for a group of mutually recursive definitions
 */
ValueInst* InstructionsCompiler2::generateRec(Tree other, Tree sig, Tree var, Tree le, int index)
{
    int N = len(le);

    ValueInst*             res = nullptr;
    vector<bool>           used(N);
    vector<int>            delay(N);
    vector<string>         vname(N);
    vector<Typed::VarType> ctype(N);

    // Prepare each element of a recursive definition
    for (int i = 0; i < N; i++) {
        Tree e = sigProj(i, sig);  // recreate each recursive definition
        if (fOccMarkup->retrieve(e)) {
            // This projection is used
            used[i] = true;
            getTypedNames(getCertifiedSigType(e), "Rec", ctype[i], vname[i]);
            setVectorNameProperty(e, vname[i]);
            delay[i] = fOccMarkup->retrieve(e)->getMaxDelay();

            // extra stuff

            //{
            //    string            fun_name = "fun_rec0";
            //    list<NamedTyped*> args;
            //    // todo: figure out the args from the Tree and append them?
            //    args.push_back(InstBuilder::genNamedTyped("channel", Typed::kInt32));

            //    BlockInst* block = InstBuilder::genBlockInst();

            //    // todo: add the Tree to the block as code?
            //    auto the_stream = new ostringstream();
            //    auto container2 = dynamic_cast<TorchCodeContainer*> (TorchCodeContainer::createContainer(
            //        gGlobal->gClassName, 1, 1, the_stream));
            //    int numInputs = 0;
            //    int numOutputs = 0;
            //    //evaluateBlockDiagram(e, numInputs, numOutputs);

            //    InstructionsCompiler2* compiler2 = new InstructionsCompiler2(container2);
            //    compiler2->compileMultiSignal(e);

            //    block->pushBackInst(container2->get_fComputeBlockInstructions());

            //    // Return "rec_result" result
            //    block->pushBackInst(InstBuilder::genRetInst(InstBuilder::genLoadStackVar("rec_result")));

            //    // Creates function
            //    FunTyped* fun_type = InstBuilder::genFunTyped(args, InstBuilder::genInt32Typed(), FunTyped::kDefault);
            //    DeclareFunInst* funInst = InstBuilder::genDeclareFunInst(fun_name, fun_type, block);

            //    pushComputeBlockMethod(funInst);
            //}

            //Address::AccessType var_access;
            //ValueInst*          ccs = getConditionCode(nth(le, i));
           // ValueInst*          ccs = generateCode(nth(le, i));
            //NullStatementInst*  exp_inst = new NullStatementInst();
            //RetInst*            exp_inst       = new RetInst(ccs);
            //pushComputeBlockMethod(InstBuilder::genControlInst(ccs, exp_inst));
            //generateCode(other);

            //////auto* block = ;
            //BlockInst* block     = InstBuilder::genBlockInst();
            //block->pushBackInst(ccs);
            ////ValueInst* valueinst = InstBuilder::genBlockInst(block);
            //pushComputeBlockMethod(block);


            //list<ValueInst*>  args_value;
            //list<NamedTyped*> args_types;

            //int blah = e->arity() - 1;
            //for (int i = 0; i < blah; i++) {

            //    args_value.push_back(CS(e->branch(i)));

            //    int         sig_argtype = kReal;  // todo
            //    BasicTyped* argtype = genBasicFIRTyped(sig_argtype);

            //    args_types.push_back(InstBuilder::genNamedTyped("dummy" + to_string(i), argtype));

            //    //Tree parameter = nth(e, i);
            //    //// Reversed...
            //    //int         sig_argtype = ffargtype(e, (ffarity(e) - 1) - i);
            //    //BasicTyped* argtype     = genBasicFIRTyped(sig_argtype);
            //    //args_types.push_back(InstBuilder::genNamedTyped("dummy" + to_string(i), argtype));
            //    //args_value.push_back(InstBuilder::genCastInst(CS(parameter), argtype));
            //}

            //// Add function declaration

            //int       sig_type = kReal; // todo
            //FunTyped* fun_type = InstBuilder::genFunTyped(args_types, genBasicFIRTyped(sig_type));

            //pushComputeBlockMethod(InstBuilder::genDeclareFunInst("rec_func0", fun_type));

        } else {
            // This projection is not used therefore
            // we should not generate code for it
            used[i] = false;
        }
    }

    // Generate delayline for each element of a recursive definition
    for (int i = 0; i < N; i++) {
        if (used[i]) {
            Address::AccessType var_access;
            ValueInst*          ccs = getConditionCode(nth(le, i));
            if (index == i) {
                res = generateDelayLine(CS(nth(le, i)), ctype[i], vname[i], delay[i], var_access, ccs);
            } else {
                generateDelayLine(CS(nth(le, i)), ctype[i], vname[i], delay[i], var_access, ccs);
            }
        }
    }

    return res;
}