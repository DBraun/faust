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

#ifndef __SIG2TORCH__
#define __SIG2TORCH__

#include <stdlib.h>
#include <cstdlib>
#include <sstream>
#include "property.hh"
#include "sigtyperules.hh"
#include "tree.hh"
#include "treeTraversal.hh"

//-------------------------Signal2Torch-------------------------------
// Transforms signals to Torch code (see: https://pytorch.org)
//----------------------------------------------------------------------

class Signal2Torch : public TreeTraversal {
    bool fVisitGen;
    std::stringstream fOut;

    int indent = 0;

   public:
    Signal2Torch() : fVisitGen(false) {}
    
    void sig2Torch(Tree L, ofstream& fout);

    void generateFConst(Tree sig, Tree type, const string& file, const string& name_aux);

   protected:
    void visit(Tree);
};

#endif
