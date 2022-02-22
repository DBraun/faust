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

#ifndef _INSTRUCTION_COMPILER2_H
#define _INSTRUCTION_COMPILER2_H

#include "instructions_compiler.hh"

// To be used with Torch backend

class InstructionsCompiler2 : public InstructionsCompiler {
   public:
    InstructionsCompiler2(CodeContainer* container);

    ValueInst* generateDelayLine(ValueInst* exp, Typed::VarType ctype, const string& vname, int mxd,
                                 Address::AccessType& var_access, ValueInst* ccs);
    ValueInst* generateDelay(Tree sig, Tree exp, Tree delay);

/**
     * Generate code for a group of mutually recursive definitions
     */
    ValueInst* generateRec(Tree sig, Tree var, Tree le, int index);

   //private:
   // StatementInst* generateInitArray(const string& vname, Typed::VarType ctype, int delay);

   // StatementInst* generateShiftArray(const string& vname, int delay);

   // StatementInst* generateCopyArray(const string& vname_to, const string& vname_from, int size);

};

#endif
