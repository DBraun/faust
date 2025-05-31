/************************************************************************
 ************************************************************************
 FAUST compiler
 Copyright (C) 2017-2021 GRAME, Centre National de Creation Musicale
 ---------------------------------------------------------------------
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 ************************************************************************
 ************************************************************************/

#include "instructions_compiler_jax.hh"
#include "ppsig.hh"
#include "sigtyperules.hh"
#include "jax_instructions.hh"

using namespace std;

StatementInst* InstructionsCompilerJAX::generateShiftArray(const string& vname, int delay)
{
    Values truncated_args;
    truncated_args.push_back(IB::genLoadArrayStructVar(vname));
    truncated_args.push_back(IB::genLoadStackVar("1"));
    return IB::genStoreArrayStructVar(vname,
                                      IB::genFunCallInst(string("jnp.roll"), truncated_args));
}

ValueInst* InstructionsCompilerJAX::generateDelayLine(ValueInst* exp, BasicTyped* ctype,
                                                      const string& vname, int mxd,
                                                      Address::AccessType& access, ValueInst* ccs)
{
    if (mxd == 0) {
        // Generate scalar use
        if (dynamic_cast<NullValueInst*>(ccs)) {
            pushComputeDSPMethod(IB::genDecStackVar(vname, ctype, exp));
        } else {
            pushPreComputeDSPMethod(IB::genDecStackVar(vname, ctype, IB::genTypedZero(ctype)));
            pushComputeDSPMethod(IB::genControlInst(ccs, IB::genStoreStackVar(vname, exp)));
        }

    } else if (mxd == 1) {
        // Special optimization for single-sample delays in JAX
        // Use array type for compatibility, but mark as scalar for JAX generation
        generateInitArray(vname, ctype, 1);
        
        // Mark this as a scalar delay variable
        fScalarDelayVars.insert(vname);
        
        // Also add to global JAX visitor if available
        if (gGlobal->gJAXVisitor) {
            static_cast<JAXInstVisitor*>(gGlobal->gJAXVisitor)->fScalarDelayVars.insert(vname);
        }
        
        // In the compute method:
        // Store current state value in a temporary variable, then update state
        string temp_name = vname + "_temp";
        
        // 1. Store current state value in the temporary variable BEFORE any computation
        // Use pushPreComputeDSPMethod to ensure it happens before the signal is used
        pushPreComputeDSPMethod(IB::genControlInst(
            ccs, IB::genDecStackVar(temp_name, ctype,
                IB::genLoadArrayStructVar(vname, IB::genInt32NumInst(0)))));
        
        // 2. Update state with new value
        pushComputeDSPMethod(IB::genControlInst(
            ccs, IB::genStoreArrayStructVar(vname, IB::genInt32NumInst(0), exp)));
        
        // 3. The delayed value is in the temporary variable
        // (will be used by generateDelayAccess)
        
    } else if (mxd < gGlobal->gMaxCopyDelay) {
        // Generates table init
        generateInitArray(vname, ctype, mxd + 1);

        // Generate table use
        pushComputeDSPMethod(IB::genControlInst(
            ccs, IB::genStoreArrayStructVar(vname, IB::genInt32NumInst(0), exp)));

        // Generates post processing copy code to update delay values
        pushPostComputeDSPMethod(IB::genControlInst(ccs, generateShiftArray(vname, mxd)));

    } else {
        int N = pow2limit(mxd + 1);
        if (N <= gGlobal->gMaskDelayLineThreshold) {
            ensureIotaCode();

            // Generates table init
            generateInitArray(vname, ctype, N);

            // Generate table use
            if (gGlobal->gComputeIOTA) {  // Ensure IOTA base fixed delays are computed once
                if (fIOTATable.find(N) == fIOTATable.end()) {
                    string   iota_name = subst("i$0", gGlobal->getFreshID(fCurrentIOTA + "_temp"));
                    FIRIndex value2 =
                        FIRIndex(IB::genLoadStructVar(fCurrentIOTA)) & FIRIndex(N - 1);

                    pushPreComputeDSPMethod(
                        IB::genDecStackVar(iota_name, IB::genInt32Typed(), IB::genInt32NumInst(0)));
                    pushComputeDSPMethod(
                        IB::genControlInst(ccs, IB::genStoreStackVar(iota_name, value2)));

                    fIOTATable[N] = iota_name;
                }

                pushComputeDSPMethod(IB::genControlInst(
                    ccs,
                    IB::genStoreArrayStructVar(vname, IB::genLoadStackVar(fIOTATable[N]), exp)));

            } else {
                FIRIndex value2 = FIRIndex(IB::genLoadStructVar(fCurrentIOTA)) & FIRIndex(N - 1);
                pushComputeDSPMethod(
                    IB::genControlInst(ccs, IB::genStoreArrayStructVar(vname, value2, exp)));
            }
        } else {
            // 'select' based delay
            string widx_tmp_name = vname + "_widx_tmp";
            string widx_name     = vname + "_widx";

            // Generates table write index
            pushDeclare(IB::genDecStructVar(widx_name, IB::genInt32Typed()));
            pushInitMethod(IB::genStoreStructVar(widx_name, IB::genInt32NumInst(0)));

            // Generates table init
            generateInitArray(vname, ctype, mxd + 1);

            // int w = widx;
            pushComputeDSPMethod(IB::genControlInst(
                ccs, IB::genDecStackVar(widx_tmp_name, IB::genBasicTyped(Typed::kInt32),
                                        IB::genLoadStructVar(widx_name))));

            // dline[w] = v;
            pushComputeDSPMethod(IB::genControlInst(
                ccs, IB::genStoreArrayStructVar(vname, IB::genLoadStackVar(widx_tmp_name), exp)));

            // w = w + 1;
            FIRIndex widx_tmp1 = FIRIndex(IB::genLoadStackVar(widx_tmp_name));
            pushPostComputeDSPMethod(
                IB::genControlInst(ccs, IB::genStoreStackVar(widx_tmp_name, widx_tmp1 + 1)));

            // w = ((w == delay) ? 0 : w);
            FIRIndex widx_tmp2 = FIRIndex(IB::genLoadStackVar(widx_tmp_name));
            pushPostComputeDSPMethod(IB::genControlInst(
                ccs, IB::genStoreStackVar(widx_tmp_name,
                                          IB::genSelect2Inst(widx_tmp2 == FIRIndex(mxd + 1),
                                                             FIRIndex(0), widx_tmp2))));
            // *widx = w
            pushPostComputeDSPMethod(IB::genControlInst(
                ccs, IB::genStoreStructVar(widx_name, IB::genLoadStackVar(widx_tmp_name))));
        }
    }

    return exp;
}

ValueInst* InstructionsCompilerJAX::generateDelayAccess(Tree sig, Tree exp, Tree delay)
{
    ValueInst* code = CS(exp);  // Ensure exp is compiled to have a vector name
    int        mxd  = fOccMarkup->retrieve(exp)->getMaxDelay();
    string     vname;

    if (!getVectorNameProperty(exp, vname)) {
        if (mxd == 0) {
            return code;
        } else {
            cerr << "ASSERT : no vector name for : " << ppsig(exp, MAX_ERROR_SIZE) << endl;
            faustassert(false);
        }
    }

    if (mxd == 0) {
        // not a real vector name but a scalar name
        return IB::genLoadStackVar(vname);
    } else if (mxd == 1) {
        int d;
        if (isSigInt(delay, &d) && d == 1) {
            // For single-sample delays, we stored the delayed value in a temporary variable
            string temp_name = vname + "_temp";
            return IB::genLoadStackVar(temp_name);
        } else {
            // Variable delay or delay != 1, use array access
            return IB::genLoadArrayStructVar(vname, IB::genInt32NumInst(0));
        }
    } else {
        // For all other cases, use the default implementation
        return InstructionsCompiler::generateDelayAccess(sig, exp, delay);
    }
}

ValueInst* InstructionsCompilerJAX::generateSoundfile(Tree sig, Tree path)
{
    string varname = gGlobal->getFreshID("fSoundfile");
    string SFcache = varname + "ca";

    fUITree.addUIWidget(reverse(tl(path)), uiWidget(hd(path), tree(varname), sig));

    pushDeclare(IB::genDecStructVar(varname, IB::genBasicTyped(Typed::kSound_ptr)));

    if (gGlobal->gUseDefaultSound) {
        BlockInst* block = IB::genBlockInst();
        block->pushBackInst(IB::genStoreStructVar(varname, IB::genLoadGlobalVar("defaultsound")));

        pushResetUIInstructions(IB::genIfInst(
            IB::genEqual(
                IB::genCastInst(IB::genLoadStructVar(varname), IB::genBasicTyped(Typed::kUint_ptr)),
                IB::genTypedZero(Typed::kSound_ptr)),
            block, IB::genBlockInst()));
    }

    // In JAX, always use struct variables (state dictionary) instead of stack variables
    pushDeclare(IB::genDecStructVar(SFcache, IB::genBasicTyped(Typed::kSound_ptr)));
    pushComputeBlockMethod(IB::genStoreStructVar(SFcache, IB::genLoadStructVar(varname)));

    return IB::genLoadStructVar(varname);
}
