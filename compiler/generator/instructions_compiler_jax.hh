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

#ifndef _INSTRUCTION_COMPILER_JAX_H
#define _INSTRUCTION_COMPILER_JAX_H

#include "instructions_compiler.hh"

/**
 * JAX-specific instructions compiler with circular buffer optimization
 * 
 * This compiler extends the base InstructionsCompiler to provide optimized delay line
 * handling for the JAX backend. It implements a hybrid approach:
 * 
 * 1. Circular Buffers: Used for larger delay lines and variable delays to replace
 *    expensive O(n) jnp.roll operations with O(1) index arithmetic
 * 
 * 2. Roll Operations: Preserved for small recursive delay arrays (typically in IIR
 *    filters) where the shift semantics are integral to the algorithm
 * 
 * This approach maintains compatibility with complex filter designs while optimizing
 * performance for simple delay operations.
 */
class InstructionsCompilerJAX : public InstructionsCompiler {
   private:
    std::set<std::string> fScalarDelayVars;     // Track single-sample delay variables (optimized as scalars)
    std::set<std::string> fCircularBufferVars;  // Track delay lines using circular buffer optimization
    std::map<std::string, int> fDelayLineSizes; // Track buffer sizes for circular buffer index calculations
    
   public:
    InstructionsCompilerJAX(CodeContainer* container) : InstructionsCompiler(container) {}
    
    const std::set<std::string>& getScalarDelayVars() const { return fScalarDelayVars; }
    const std::set<std::string>& getCircularBufferVars() const { return fCircularBufferVars; }
    const std::map<std::string, int>& getDelayLineSizes() const { return fDelayLineSizes; }

    StatementInst* generateShiftArray(const std::string& vname, int delay) override;

    ValueInst* generateDelayLine(ValueInst* exp, BasicTyped* ctype, const std::string& vname,
                                 int mxd, Address::AccessType& access, ValueInst* ccs) override;
    
    ValueInst* generateDelayAccess(Tree sig, Tree exp, Tree delay) override;

    ValueInst* generateSoundfile(Tree sig, Tree path) override;
    
    // Override to handle self.random_uniform with proper RNG splitting
    ValueInst* generateFFun(Tree sig, Tree ff, Tree largs) override;
};

#endif
