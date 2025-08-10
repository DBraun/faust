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

#ifndef _JAX_INSTRUCTIONS_H
#define _JAX_INSTRUCTIONS_H

#include "jax_base_instructions.hh"

/**
 * JAX/NNX init fields visitor.
 * Routes UI parameters to params[""], state variables to state[""],
 * cache variables (ending in "ca") as bare locals.
 */
struct JAXInitFieldsVisitor : public JAXBaseInitFieldsVisitor {
    using JAXBaseInitFieldsVisitor::JAXBaseInitFieldsVisitor;

    void visit(NamedAddress* named) override
    {
        // Check if this is a UI parameter
        bool isUIParam = false;
        bool isCacheVar = false;
        if (named->isStruct() || named->isStaticStruct()) {
            std::string name = named->fName;
            isUIParam = (name.find("fButton") == 0) ||
                       (name.find("fCheckbox") == 0) ||
                       (name.find("fVslider") == 0) ||
                       (name.find("fHslider") == 0) ||
                       (name.find("fEntry") == 0) ||
                       (name.find("fVbargraph") == 0) ||
                       (name.find("fHbargraph") == 0);
        } else {
            std::string name = named->fName;
            isCacheVar = (name.length() > 2 && name.substr(name.length() - 2) == "ca");
        }

        // kStaticStruct are actually merged in the main DSP
        if (named->isStruct() || named->isStaticStruct()) {
            if (isUIParam) {
                *fOut << "params[\"";
            } else {
                *fOut << "state[\"";
            }
            *fOut << named->fName;
            *fOut << "\"]";
        } else if (isCacheVar) {
            *fOut << named->fName;
        } else {
            *fOut << named->fName;
        }
    }
};

/**
 * JAX/NNX main instruction visitor.
 * Routes UI parameters to params[""], state variables to state[""],
 * cache variables (ending in "ca") as bare locals.
 */
class JAXInstVisitor : public JAXBaseInstVisitor {
   public:
    using JAXBaseInstVisitor::JAXBaseInstVisitor;

    void visit(NamedAddress* named) override
    {
        // Check if this is a UI parameter
        bool isUIParam = false;
        bool isCacheVar = false;
        if (named->isStruct() || named->isStaticStruct()) {
            std::string name = named->fName;
            isUIParam = (name.find("fButton") == 0) ||
                       (name.find("fCheckbox") == 0) ||
                       (name.find("fVslider") == 0) ||
                       (name.find("fHslider") == 0) ||
                       (name.find("fEntry") == 0) ||
                       (name.find("fVbargraph") == 0) ||
                       (name.find("fHbargraph") == 0);
        } else {
            std::string name = named->fName;
            isCacheVar = (name.length() > 2 && name.substr(name.length() - 2) == "ca");
        }

        // kStaticStruct are actually merged in the main DSP
        if (named->isStruct() || named->isStaticStruct()) {
            if (isUIParam) {
                *fOut << "params[\"";
            } else {
                *fOut << "state[\"";
            }
            *fOut << named->fName;
            *fOut << "\"]";
        } else if (isCacheVar) {
            *fOut << named->fName;
        } else {
            *fOut << named->fName;
        }
    }
};

#endif
