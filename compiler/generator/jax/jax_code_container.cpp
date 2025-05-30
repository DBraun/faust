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

#include "jax_code_container.hh"
#include "Text.hh"
#include "exception.hh"
#include "fir_function_builder.hh"
#include "floats.hh"
#include "global.hh"
#include "instructions.hh"

using namespace std;

/*
 JAX backend and module description:

 - Whereas a normal code container would generate a "compute" method, we generate
   a one-sample loop "tick" method. Our hard-coded "compute" method __call__ is implemented
   in an architecture file. It uses JAX's scan function in conjunction with the generated tick
 function.
 - Inside "__call__" and before using "scan", we setup the arrays, soundfiles, user interface
 parameters, and other state variables.
 - One tricky part of JAX is modifying arrays in-place:
   https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
   Whereas C++ would look like
   `fRec1[0] = fTemp0`
   in JAX we have to do
   `state["fRec1"] = state["fRec1"].at[0].set(fTemp0)`
   Also, this at-and-set operation is slow, so we only use it inside the tick method.
   This is why in all other places (like initializing sound files which are arrays),
   we use numpy arrays instead of jnp arrays. It's best to just look at the generated code and
 notice how the jnp prefix is used differently than the np prefix.
 - In order to simplify global array typing, subcontainers are actually merged in the main DSP
 structure:
    - so 'mergeSubContainers' is used
    - global variables are added in the DSP structure
    - the JAXInitFieldsVisitor class does initialisation for waveforms. This makes it easy to use
 numpy instead of jax when initializing arrays (good for speed). We also use fUseNumpy in this
 decision making. We convert the numpy arrays to jax numpy before they're used in the tick method.
    - the fGlobalDeclarationInstructions contains global functions and variables. It is "manually"
 used to generate global functions and move global variables declaration at DSP structure level.
*/

map<string, bool> JAXInstVisitor::gFunctionSymbolTable;

dsp_factory_base* JAXCodeContainer::produceFactory()
{
    return new text_dsp_factory_aux(
        fKlassName, "", "",
        ((dynamic_cast<ostringstream*>(fOut)) ? dynamic_cast<ostringstream*>(fOut)->str() : ""),
        "");
}

JAXCodeContainer::JAXCodeContainer(const std::string& name, int numInputs, int numOutputs,
                                   std::ostream* out)
{
    // Mandatory
    initialize(numInputs, numOutputs);
    fKlassName = name;
    fOut       = out;

    // Allocate one static visitor to be shared by main module and sub containers
    if (!gGlobal->gJAXVisitor) {
        gGlobal->gJAXVisitor = new JAXInstVisitor(out, name);
    }
}

CodeContainer* JAXCodeContainer::createScalarContainer(const string& name, int sub_container_type)
{
    return new JAXScalarCodeContainer(name, 0, 1, fOut, sub_container_type);
}

CodeContainer* JAXCodeContainer::createContainer(const string& name, int numInputs, int numOutputs,
                                                 ostream* dst)
{
    CodeContainer* container;

    if (gGlobal->gOpenCLSwitch) {
        throw faustexception("ERROR : OpenCL not supported for JAX\n");
    }
    if (gGlobal->gCUDASwitch) {
        throw faustexception("ERROR : CUDA not supported for JAX\n");
    }

    if (gGlobal->gOpenMPSwitch) {
        throw faustexception("ERROR : OpenMP not supported for JAX\n");
    } else if (gGlobal->gSchedulerSwitch) {
        throw faustexception("ERROR : Scheduler not supported for JAX\n");
    } else if (gGlobal->gVectorSwitch) {
        throw faustexception("ERROR : Vector not supported for JAX\n");
    } else {
        container = new JAXScalarCodeContainer(name, numInputs, numOutputs, dst, kInt);
    }

    return container;
}

inline string flattenJSONforPython(const string& src)
{
    string dst;
    for (size_t i = 0; i < src.size(); i++) {
        switch (src[i]) {
            case '"':
                dst += "\\\"";
                break;
            case '\\':
                dst += "/";
                break;
            case '\'':
                dst += "'";
                break;
            default:
                dst += src[i];
                break;
        }
    }
    return dst;
}

void JAXCodeContainer::produceClass()
{
    int n = 0;

    // Print header
    *fOut << "\"\"\"" << endl << "Code generated with Faust version " << FAUSTVERSION << endl;
    *fOut << "Compilation options: ";
    stringstream stream;
    gGlobal->printCompilationOptions(stream);
    *fOut << stream.str();
    tab(n, *fOut);
    *fOut << "\"\"\"";
    tab(n, *fOut);

    if (gGlobal->gFloatSize == 2) {
        tab(n, *fOut);
        *fOut << "# enable double precision: "
                 "https://jax.readthedocs.io/en/latest/notebooks/"
                 "Common_Gotchas_in_JAX.html#double-64bit-precision";
        tab(n, *fOut);
        *fOut << "import jax";
        tab(n, *fOut);
        *fOut << "jax.config.update(\"jax_enable_x64\", True)";
        tab(n, *fOut);
        *fOut << "FAUSTFLOAT = jnp.float64";
        tab(n, *fOut);
        *fOut << "FAUSTINT = jnp.int64";
        tab(n, *fOut);
    } else {
        tab(n, *fOut);
        *fOut << "# enable single precision";
        tab(n, *fOut);
        *fOut << "FAUSTFLOAT = jnp.float32";
        tab(n, *fOut);
        *fOut << "FAUSTINT = jnp.int32";
        tab(n, *fOut);
    }

    // Merge sub containers
    mergeSubContainers();
    
    // Extract pfPerm initialization values BEFORE generating methods
    // This ensures they're available when _initialize_carry is generated
    JAXInstVisitor* jaxVisitor = static_cast<JAXInstVisitor*>(gGlobal->gJAXVisitor);
    struct PfPermExtractor : public DispatchVisitor {
        JAXInstVisitor* fJaxVisitor;
        
        PfPermExtractor(JAXInstVisitor* visitor) : fJaxVisitor(visitor) {}
        
        virtual void visit(StoreVarInst* inst) {
            string varname = inst->fAddress->getName();
            if (varname.find("pfPerm") == 0) {
                fJaxVisitor->fPfPermInitValues[varname] = inst->fValue;
            }
        }
    };
    
    PfPermExtractor extractor(jaxVisitor);
    fInitInstructions->accept(&extractor);

    // Missing math function
    tab(n, *fOut);
    *fOut << "def remainder(x, y):";
    tab(n + 1, *fOut);
    *fOut << "\"\"\"C++ std::remainder implemented with jax numpy\"\"\"";
    tab(n + 1, *fOut);
    *fOut << "quo = jnp.round(x/y)";
    tab(n + 1, *fOut);
    *fOut << "return x - quo * y";
    tab(n + 1, *fOut);

    // Functions
    tab(n, *fOut);
    gGlobal->gJAXVisitor->Tab(n);

    *fOut << "class " << fKlassName << "(nn.Module):";
    tab(n + 1, *fOut);

    // Fields
    gGlobal->gJAXVisitor->Tab(n + 1);

    tab(n + 1, *fOut);
    *fOut << "sample_rate: int";
    tab(n + 1, *fOut);
    *fOut << "soundfile_dirs: list[str] = dataclasses.field(default_factory=list)";

    tab(n + 1, *fOut);
    gGlobal->gJAXVisitor->Tab(n);

    tab(n + 1, *fOut);
    produceInfoFunctions(n + 1, "", "self", false, FunTyped::kDefault, gGlobal->gJAXVisitor);

    tab(n + 1, *fOut);
    *fOut << "# fmt: off";

    tab(n + 1, *fOut);
    *fOut << "def _initialize_carry(self, x: jnp.ndarray, length: int):";
    {
        tab(n + 2, *fOut);
        *fOut << "state = {}";
        tab(n + 2, *fOut);
        
        // Initialize scalar delays BEFORE they're used in inline subcontainers
        JAXInstVisitor* jaxVisitor = static_cast<JAXInstVisitor*>(gGlobal->gJAXVisitor);
        
        // Collect all delay variables first to avoid duplicates
        std::set<std::string> processedDelays;
        
        // Process scalar delays from JAXInstVisitor
        if (!jaxVisitor->fScalarDelayVars.empty()) {
            tab(n + 2, *fOut);
            *fOut << "# scalar delay initializations:";
            for (const auto& varName : jaxVisitor->fScalarDelayVars) {
                if (processedDelays.find(varName) == processedDelays.end()) {
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varName << "\"] = ";
                    // Determine type based on variable name prefix
                    if (varName[0] == 'i' || varName.find("_idx") != std::string::npos) {
                        *fOut << "np.int32(0)";
                    } else if (gGlobal->gFloatSize == 1) {
                        *fOut << "np.float32(0)";
                    } else {
                        *fOut << "np.float64(0)";
                    }
                    processedDelays.insert(varName);
                }
            }
            tab(n + 2, *fOut);
        }
        
        // Also check struct member variables for delay arrays, IOTA variables, and pfPerm variables
        for (const auto& it : fDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                // Look for state variables in struct members, skip if already processed
                // Exclude static tables (ftbl0*) as they should be in setup
                if ((varname.find("Rec") != std::string::npos || 
                     varname.find("Vec") != std::string::npos ||
                     varname.find("IOTA") != std::string::npos ||
                     varname.find("pfPerm") != std::string::npos ||
                     // Include table variables
                     (varname.find("tbl") != std::string::npos && varname.find("ftbl0") != 0) ||
                     // Include waveform arrays 
                     (varname.find("mydspWave") != std::string::npos && varname.find("SIG") == std::string::npos)) && 
                    varname.find("ftbl0") != 0 &&
                    processedDelays.find(varname) == processedDelays.end()) {
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varname << "\"] = ";
                    ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(decl->fType);
                    if (array_type) {
                        // Check if this is an integer table
                        if (varname[0] == 'i' || varname.find("itbl") == 0) {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.int32)";
                        } else if (gGlobal->gFloatSize == 1) {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float32)";
                        } else {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float64)";
                        }
                    } else {
                        // Scalar variables - IOTA variables are always integers
                        if (varname[0] == 'i' || varname.find("IOTA") != std::string::npos || varname.find("_idx") != std::string::npos) {
                            *fOut << "np.int32(0)";
                        } else if (varname.find("pfPerm") == 0 && jaxVisitor->fPfPermInitValues.find(varname) != jaxVisitor->fPfPermInitValues.end()) {
                            // Special handling for pfPerm variables - check their initialization value type
                            ValueInst* init_val = jaxVisitor->fPfPermInitValues[varname];
                            if (dynamic_cast<Int32NumInst*>(init_val) || dynamic_cast<Int64NumInst*>(init_val) || dynamic_cast<BoolNumInst*>(init_val)) {
                                *fOut << "np.int32(0)";
                            } else if (gGlobal->gFloatSize == 1) {
                                *fOut << "np.float32(0)";
                            } else {
                                *fOut << "np.float64(0)";
                            }
                        } else if (gGlobal->gFloatSize == 1) {
                            *fOut << "np.float32(0)";
                        } else {
                            *fOut << "np.float64(0)";
                        }
                    }
                    processedDelays.insert(varname);
                }
            }
        }
        
        // Include all global variables that should be in state
        for (const auto& it : fGlobalDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                // Include variables that change during execution and aren't static
                if (// Read-write tables (not static tables like ftbl0*)
                    (varname.find("ftbl") == 0 && varname.find("ftbl0") != 0) ||
                    // Integer tables (itbl)
                    (varname.find("itbl") == 0) ||
                    // Waveform arrays that are accessed via index in tick (not static)
                    (varname.find("mydspWave") != std::string::npos && varname.find("SIG") == std::string::npos) ||
                    // Index variables for waveforms (not static/subcontainer ones)
                    (varname.find("_idx") != std::string::npos && varname.find("fmydspSIG0") != 0) ||
                    // Delay arrays (fRec, iRec, fVec, iVec, etc.)
                    (varname.find("Rec") != std::string::npos || varname.find("Vec") != std::string::npos)) {
                    
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varname << "\"] = ";
                    ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(decl->fType);
                    if (array_type) {
                        // Check if this is an integer table
                        if (varname[0] == 'i' || varname.find("itbl") == 0) {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.int32)";
                        } else if (gGlobal->gFloatSize == 1) {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float32)";
                        } else {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float64)";
                        }
                    } else {
                        // Scalar variables (like index)
                        if (varname[0] == 'i' || varname.find("_idx") != std::string::npos) {
                            *fOut << "np.int32(0)";
                        } else if (gGlobal->gFloatSize == 1) {
                            *fOut << "np.float32(0)";
                        } else {
                            *fOut << "np.float64(0)";
                        }
                    }
                }
            }
        }
        
        // Index variables and read-write tables are handled above
        
        // Static tables and waveforms are moved to setup method
        
        // Initialize state variables for read-write tables and indices
        for (const auto& it : fGlobalDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                // Initialize read-write tables that change during execution
                if (varname.find("ftbl") == 0 && varname.find("ftbl0") != 0) {
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varname << "\"] = ";
                    ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(decl->fType);
                    if (array_type) {
                        if (gGlobal->gFloatSize == 1) {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float32)";
                        } else {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float64)";
                        }
                    }
                } else if (varname.find("_idx") != std::string::npos && varname.find("fmydspSIG0") != 0) {
                    // Index variables for read-write operations
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varname << "\"] = np.int32(0)";
                }
            }
        }
        
        // Also check struct declarations for additional state variables
        for (const auto& it : fDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                // Initialize read-write tables that change during execution
                if (varname.find("ftbl") == 0 && varname.find("ftbl0") != 0 &&
                    processedDelays.find(varname) == processedDelays.end()) {
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varname << "\"] = ";
                    ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(decl->fType);
                    if (array_type) {
                        if (gGlobal->gFloatSize == 1) {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float32)";
                        } else {
                            *fOut << "np.zeros((" << array_type->fSize << ",), dtype=np.float64)";
                        }
                    }
                    processedDelays.insert(varname);
                } else if (varname.find("_idx") != std::string::npos && varname.find("fmydspSIG0") != 0 &&
                           processedDelays.find(varname) == processedDelays.end()) {
                    // Index variables for read-write operations
                    tab(n + 2, *fOut);
                    *fOut << "state[\"" << varname << "\"] = np.int32(0)";
                    processedDelays.insert(varname);
                }
            }
        }
        
        tab(n + 2, *fOut);
        *fOut << "return state";
        tab(n + 1, *fOut);
    }
    back(1, *fOut);

    // todo: enable later
    // // JSON generation
    // tab(n + 1, *fOut);
    // *fOut << "def getJSON(self):";
    // {
    //     string json;
    //     if (gGlobal->gFloatSize == 1) {
    //         json = generateJSON<float>();
    //     } else {
    //         json = generateJSON<double>();
    //     }
    //     tab(n + 2, *fOut);
    //     *fOut << "json_str = \"\"\"" << flattenJSONforPython(json) << "\"\"\"";
    //     tab(n + 2, *fOut);
    //     *fOut << "return json.loads(json_str)";
    //     tab(n + 1, *fOut);
    // }

    // Setup method
    tab(n + 1, *fOut);
    *fOut << "def setup(self):";
    {
        JAXInstVisitor* jaxVisitor = static_cast<JAXInstVisitor*>(gGlobal->gJAXVisitor);

        // Initialize constants as instance attributes
        tab(n + 2, *fOut);
        *fOut << "# Initialize constants as instance attributes";
        tab(n + 2, *fOut);
        *fOut << "self._fSampleRate = self.sample_rate";
        // Track fSampleRate as a constant
        jaxVisitor->fConstantVars.insert("fSampleRate");
        
        // Initialize static waveforms and tables as instance attributes
        for (const auto& it : fGlobalDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                // Static tables and waveforms become instance attributes
                if (varname.find("ftbl0") == 0 || 
                    (varname.find("fmydspWave") == 0 && varname.find("_idx") == std::string::npos)) {
                    tab(n + 2, *fOut);
                    *fOut << "self._" << varname << " = ";
                    ArrayTyped* array_type = dynamic_cast<ArrayTyped*>(decl->fType);
                    if (array_type) {
                        if (gGlobal->gFloatSize == 1) {
                            *fOut << "jnp.zeros((" << array_type->fSize << ",), dtype=jnp.float32)";
                        } else {
                            *fOut << "jnp.zeros((" << array_type->fSize << ",), dtype=jnp.float64)";
                        }
                    }
                    // Track as constant variable
                    jaxVisitor->fConstantVars.insert(varname);
                }
            }
        }
        
        // Initialize waveform data as instance attributes
        std::map<std::string, ValueInst*> waveformData;
        for (const auto& it : fGlobalDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                if ((varname.find("fmydspWave") == 0 || varname.find("fmydspSIG") == 0) && 
                    varname.find("_idx") == std::string::npos && decl->fValue) {
                    waveformData[varname] = decl->fValue;
                }
            }
        }
        
        // Initialize waveform data as instance attributes
        for (const auto& kv : waveformData) {
            tab(n + 2, *fOut);
            *fOut << "self._" << kv.first << " = ";
            // Use JAX arrays for waveform data since they'll be used in tick
            jaxVisitor->fUseNumpy = false;
            kv.second->accept(jaxVisitor);
            // Track as constant variable
            jaxVisitor->fConstantVars.insert(kv.first);
        }
        
        // Initialize read-write tables and index variables in setup
        // Process global declarations again to look for read-write state vars that need initialization
        for (const auto& it : fGlobalDeclarationInstructions->fCode) {
            if (DeclareVarInst* decl = dynamic_cast<DeclareVarInst*>(it)) {
                string varname = decl->fAddress->getName();
                // Initialize read-write tables that aren't static
                if (varname.find("ftbl") == 0 && varname.find("ftbl0") != 0) {
                    tab(n + 2, *fOut);
                    *fOut << "# Initialize read-write table: " << varname;
                    // This is moved to _initialize_carry for state management
                } else if (varname.find("_idx") != std::string::npos && varname.find("fmydspSIG0") != 0) {
                    tab(n + 2, *fOut);
                    *fOut << "# Initialize index variable: " << varname;
                    // This is moved to _initialize_carry for state management
                }
            }
        }
        
        // Initialize static tables from waveforms
        for (const auto& kv : waveformData) {
            if (kv.first.find("fmydspSIG") == 0 && kv.first.find("Wave0") != std::string::npos) {
                // Extract the base name for the corresponding table
                size_t pos = kv.first.find("Wave0");
                if (pos != std::string::npos) {
                    std::string baseName = kv.first.substr(1, pos - 1); // Remove 'f' prefix
                    std::string tableName = "ftbl0" + baseName;
                    
                    // Check if this static table exists
                    if (jaxVisitor->fConstantVars.find(tableName) != jaxVisitor->fConstantVars.end()) {
                        tab(n + 2, *fOut);
                        *fOut << "# Fill static table " << tableName;
                        tab(n + 2, *fOut);
                        *fOut << "_idx = 0";
                        tab(n + 2, *fOut);
                        *fOut << "for i in range(len(self._" << tableName << ")):";
                        tab(n + 3, *fOut);
                        *fOut << "self._" << tableName << " = self._" << tableName << ".at[i].set(self._" << kv.first << "[_idx])";
                        tab(n + 3, *fOut);
                        *fOut << "_idx = (_idx + 1) % len(self._" << kv.first << ")";
                        tab(n + 2, *fOut);
                        break;
                    }
                }
            }
        }
        // Note: instance clear is handled in _initialize_carry
        
        // Initialize unnormalization functions dictionary
        tab(n + 2, *fOut);
        *fOut << "# Initialize unnormalization functions dictionary";
        tab(n + 2, *fOut);
        *fOut << "unnorm_funcs = {}";
        
        tab(n + 2, *fOut);
        *fOut << "# Initialize UI parameters as instance attributes";
        tab(n + 2, *fOut);
        *fOut << "ui_path = []";
        tab(n + 2, *fOut);
        gGlobal->gJAXVisitor->Tab(n + 2);
        generateUserInterface(gGlobal->gJAXVisitor);
        
        // Store the unnorm_funcs dictionary
        tab(n + 2, *fOut);
        *fOut << "# Store unnormalization functions";
        tab(n + 2, *fOut);
        *fOut << "self._unnorm_funcs = unnorm_funcs";
        
        // Initialize constants from init instructions
        tab(n + 2, *fOut);
        *fOut << "# Initialize constants";
        // Process init instructions to find constant initializations
        struct ConstantInitExtractor : public DispatchVisitor {
            std::ostream* fOut;
            int fTab;
            JAXInstVisitor* fJaxVisitor;
            
            ConstantInitExtractor(std::ostream* out, int tab, JAXInstVisitor* visitor) 
                : fOut(out), fTab(tab), fJaxVisitor(visitor) {}
                
            virtual void visit(StoreVarInst* inst) {
                string varname = inst->fAddress->getName();
                if (varname.find("Const") != std::string::npos) {
                    // Add to constant vars BEFORE visiting the instruction
                    fJaxVisitor->fConstantVars.insert(varname);
                    tab(fTab, *fOut);
                    // Don't add "self._" here, the visitor will handle it
                    inst->accept(fJaxVisitor);
                } else if (varname.find("pfPerm") == 0) {
                    // For pfPerm variables, store the initialization value for later use
                    // Don't generate code here - it will be handled in _initialize_carry
                    // Store the initialization instruction for later processing
                    fJaxVisitor->fPfPermInitValues[varname] = inst->fValue;
                }
            }
        };
        
        // Set setup context flag
        jaxVisitor->fInSetup = true;
        ConstantInitExtractor extractor(fOut, n + 2, jaxVisitor);
        fInitInstructions->accept(&extractor);
        jaxVisitor->fInSetup = false;
        
        // For JAX, we need to handle table initialization differently
        // The init instructions may contain C++ style function calls that don't exist in Python
        // Instead, we'll generate the table filling code directly
        
        // Note: Table initialization is now handled in _initialize_carry
    }

    // Compute
    generateCompute(n + 1);
}

void JAXCodeContainer::generateCompute(int n)
{
    tab(n, *fOut);
    *fOut << "def tick(self, params: dict, state: dict, inputs: jnp.array):";
    tab(n + 1, *fOut);

    tab(n + 1, *fOut);
    gGlobal->gJAXVisitor->Tab(n + 1);

    // Generates local variables declaration and setup
    gGlobal->gJAXVisitor->fUseNumpy = false;
    generateComputeBlock(gGlobal->gJAXVisitor);

    auto loop = fCurLoop->generateOneSample();
    loop->accept(gGlobal->gJAXVisitor);

    generatePostComputeBlock(gGlobal->gJAXVisitor);
    gGlobal->gJAXVisitor->fUseNumpy = true;

    tab(n, *fOut);
    *fOut << "# fmt: on";
}

void JAXCodeContainer::generateSR()
{
    if (!fGeneratedSR) {
        pushDeclare(IB::genDecStructVar("fSampleRate", IB::genInt32Typed()));
    }
    pushPreInitMethod(
        IB::genStoreStructVar("fSampleRate", IB::genLoadFunArgsVar("self.sample_rate")));
}

void JAXCodeContainer::produceInfoFunctions(int tabs, const string& classname, const string& obj,
                                           bool ismethod, FunTyped::FunAttribute funtype,
                                           TextInstVisitor* producer, const string& in_fun,
                                           const string& out_fun)
{
    // Generate as properties instead of methods for JAX/Flax
    *fOut << "@property";
    tab(tabs, *fOut);
    *fOut << "def num_inputs(self):";
    tab(tabs + 1, *fOut);
    *fOut << "return " << fNumInputs;
    tab(tabs, *fOut);
    
    tab(tabs, *fOut);
    *fOut << "@property";
    tab(tabs, *fOut);
    *fOut << "def num_outputs(self):";
    tab(tabs + 1, *fOut);
    *fOut << "return " << fNumOutputs;
    tab(tabs, *fOut);
}

// Scalar
JAXScalarCodeContainer::JAXScalarCodeContainer(const string& name, int numInputs, int numOutputs,
                                               std::ostream* out, int sub_container_type)
    : JAXCodeContainer(name, numInputs, numOutputs, out)
{
    fSubContainerType = sub_container_type;
}
