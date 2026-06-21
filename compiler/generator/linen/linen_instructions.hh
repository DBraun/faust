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

#ifndef _LINEN_INSTRUCTIONS_H
#define _LINEN_INSTRUCTIONS_H

#include "nnx_base_instructions.hh"

/**
 * Linen init fields visitor.
 * All struct variables (UI params and state) route to state[""].
 * Cache variables (ending in "ca") are bare locals.
 */
struct LinenInitFieldsVisitor : public NNXBaseInitFieldsVisitor {
    using NNXBaseInitFieldsVisitor::NNXBaseInitFieldsVisitor;

    void visit(NamedAddress* named) override
    {
        bool isCacheVar = false;
        if (!(named->isStruct() || named->isStaticStruct())) {
            std::string name = named->fName;
            isCacheVar = (name.length() > 2 && name.substr(name.length() - 2) == "ca");
        }

        // In Linen, everything struct goes to state (no params/state split)
        if (named->isStruct() || named->isStaticStruct()) {
            *fOut << "state[\"" << named->fName << "\"]";
        } else if (isCacheVar) {
            *fOut << named->fName;
        } else {
            *fOut << named->fName;
        }
    }
};

/**
 * Linen main instruction visitor.
 * All struct variables (UI params and state) route to state[""].
 * Cache variables (ending in "ca") are bare locals.
 */
class LinenInstVisitor : public NNXBaseInstVisitor {
   public:
    using NNXBaseInstVisitor::NNXBaseInstVisitor;

    void visit(NamedAddress* named) override
    {
        bool isCacheVar = false;
        if (!(named->isStruct() || named->isStaticStruct())) {
            std::string name = named->fName;
            isCacheVar = (name.length() > 2 && name.substr(name.length() - 2) == "ca");
        }

        // In Linen, everything struct goes to state (no params/state split)
        if (named->isStruct() || named->isStaticStruct()) {
            *fOut << "state[\"" << named->fName << "\"]";
        } else if (isCacheVar) {
            *fOut << named->fName;
        } else {
            *fOut << named->fName;
        }
    }
};

#endif
