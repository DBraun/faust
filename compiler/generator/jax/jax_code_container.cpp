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

using namespace std;

/*
 JAX backend implementation with Flax NNX support:

 ### Core Architecture:
 - Generates a one-sample "tick" method instead of a traditional "compute" method
 - Uses JAX's nnx.scan for efficient loop processing over audio blocks
 - Architecture files (minimal.py, impulsejax.py) provide the __call__ wrapper
 - Flax NNX modules handle parameter management and state initialization

 ### Key Design Decisions:
 - **Flax NNX Integration**: Uses nnx.Module, nnx.Param, nnx.Variable for parameter/state management
 - **Immutable Arrays**: JAX arrays require `.at[index].set(value)` instead of in-place updates
 - **Parameter Separation**: UI parameters stored in `params` dict, state variables in `state` dict
 - **Cache Variables**: Soundfile cache variables (ending in "ca") are local vars, not state entries
 - **NumPy Initialization**: Use mutable NumPy arrays during setup, convert to JAX in tick method

 ### Code Generation:
 - `tick(params, state, inputs, rng)` signature for NNX compatibility
 - `_initialize_carry()` method sets up initial state dictionary
 - `build_interface()` method creates UI parameter structure
 - UI parameters accessed via `params["fVslider0"]`, state via `state["fRec0"]`
 - Soundfiles initialized in `initialize_carry()` and copied to state

 ### Subcontainer Handling:
 - All subcontainers merged into main DSP structure for simplified typing
 - Global variables moved to DSP structure level
 - JAXInitFieldsVisitor handles waveform initialization with NumPy for speed

 ### Delay Line Optimization:
 - Small delays (≤16 samples) use jnp.roll operations
 - Large delays use circular buffer indexing for O(1) performance
 - Controlled by -mcd compiler flag (default 16)
*/

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
    } else {
        tab(n, *fOut);
        *fOut << "# enable single precision";
        tab(n, *fOut);
        *fOut << "FAUSTFLOAT = jnp.float32";
        tab(n, *fOut);
    }

    // Merge sub containers
    mergeSubContainers();

    // Missing math function
    tab(n, *fOut);
    *fOut << "def remainder(x, y):";
    tab(n + 1, *fOut);
    *fOut << "\"\"\"C++ std::remainder implemented with jax numpy\"\"\"";
    tab(n + 1, *fOut);
    *fOut << "quo = jnp.round(x/y)";
    tab(n + 1, *fOut);
    *fOut << "return x - quo * y";
    tab(n, *fOut);
    tab(n, *fOut);

    // Functions
    tab(n, *fOut);
    gGlobal->gJAXVisitor->Tab(n);

    *fOut << "class " << fKlassName << "(nnx.Module):";
    tab(n + 1, *fOut);

    tab(n + 1, *fOut);
    gGlobal->gJAXVisitor->Tab(n);

    // Generate __init__ method for NNX
    tab(n + 1, *fOut);
    *fOut << "def __init__(self, sample_rate: int, faust_float = jnp.float32, soundfile_dirs: List[str] = None, use_magic_clamp: bool = True, return_bargraphs: bool = False, rngs: rnglib.Rngs | rnglib.RngStream | None = None):";
    tab(n + 2, *fOut);
    *fOut << "self.sample_rate = sample_rate";
    tab(n + 2, *fOut);
    *fOut << "self.soundfile_dirs = soundfile_dirs or []";
    tab(n + 2, *fOut);
    *fOut << "self.faust_float = faust_float";
    tab(n + 2, *fOut);
    *fOut << "self.rng_collection = \"default\"";
    tab(n + 2, *fOut);
    *fOut << "# Handle RNG types following NNX pattern (see nnx.Dropout)";
    tab(n + 2, *fOut);
    *fOut << "# Note: We store Rngs directly (not forked) because Faust modules may need";
    tab(n + 2, *fOut);
    *fOut << "# multiple RNG collections (e.g., 'default' for random_*, 'nentry' for nentry)";
    tab(n + 2, *fOut);
    *fOut << "if isinstance(rngs, rnglib.Rngs):";
    tab(n + 3, *fOut);
    *fOut << "self.rngs = rngs";
    tab(n + 2, *fOut);
    *fOut << "elif isinstance(rngs, rnglib.RngStream):";
    tab(n + 3, *fOut);
    *fOut << "self.rngs = rngs.fork()";
    tab(n + 2, *fOut);
    *fOut << "elif rngs is None:";
    tab(n + 3, *fOut);
    *fOut << "self.rngs = nnx.data(None)";
    tab(n + 2, *fOut);
    *fOut << "else:";
    tab(n + 3, *fOut);
    *fOut << "raise TypeError(f'rngs must be Rngs, RngStream or None, got {type(rngs).__name__}')";
    tab(n + 2, *fOut);
    *fOut << "self.deterministic = False";
    tab(n + 2, *fOut);
    *fOut << "self.use_magic_clamp = use_magic_clamp";
    tab(n + 2, *fOut);
    *fOut << "self.return_bargraphs = return_bargraphs";
    tab(n + 2, *fOut);
    *fOut << "self._parameter_metadata = {}";
    tab(n + 2, *fOut);
    *fOut << "self._unnorm_funcs = {}";
    tab(n + 2, *fOut);
    *fOut << "self.num_inputs = " << fNumInputs;
    tab(n + 2, *fOut);
    *fOut << "self.num_outputs = " << fNumOutputs;
    tab(n + 2, *fOut);
    tab(n + 2, *fOut);
    *fOut << "# Build UI interface";
    tab(n + 2, *fOut);
    *fOut << "ui_path = []";
    tab(n + 2, *fOut);
    *fOut << "unnorm_funcs = {}";
    tab(n + 2, *fOut);
    *fOut << "self.build_interface(ui_path, unnorm_funcs)";
    tab(n + 2, *fOut);
    *fOut << "self._unnorm_funcs = unnorm_funcs";
    tab(n + 1, *fOut);
    tab(n + 1, *fOut);

    *fOut << "def _initialize_carry(self):";
    {
        tab(n + 2, *fOut);
        *fOut << "state = {}";
        tab(n + 2, *fOut);
        tab(n + 2, *fOut);
        *fOut << "# global declarations:";
        JAXInitFieldsVisitor initializer(fOut, n + 2);
        generateDeclarations(&initializer);
        // Generate global variables initialisation
        for (const auto& it : fGlobalDeclarationInstructions->fCode) {
            if (dynamic_cast<DeclareVarInst*>(it)) {
                it->accept(&initializer);
            }
        }
        tab(n + 2, *fOut);
        tab(n + 2, *fOut);
        *fOut << "# init constants:";
        tab(n + 2, *fOut);
        gGlobal->gJAXVisitor->Tab(n + 2);
        inlineSubcontainersFunCalls(fInitInstructions)->accept(gGlobal->gJAXVisitor);
        tab(n + 2, *fOut);
        *fOut << "# inline subcontainers:";
        tab(n + 2, *fOut);
        gGlobal->gJAXVisitor->Tab(n + 2);
        inlineSubcontainersFunCalls(fStaticInitInstructions)->accept(gGlobal->gJAXVisitor);
        tab(n + 2, *fOut);
        *fOut << "# instance clear:";
        tab(n + 2, *fOut);
        generateClear(gGlobal->gJAXVisitor);
        tab(n + 2, *fOut);
        
        // TODO: Initialize bargraphs if needed
        
        tab(n + 2, *fOut);
        *fOut << "return state";
        tab(n + 1, *fOut);
    }
    back(1, *fOut);

    // JSON generation
    tab(n + 1, *fOut);
    *fOut << "@property";
    tab(n + 1, *fOut);
    *fOut << "def json_metadata(self):";
    {
        string json;
        if (gGlobal->gFloatSize == 1) {
            json = generateJSON<float>();
        } else {
            json = generateJSON<double>();
        }
        tab(n + 2, *fOut);
        *fOut << "json_str = \"\"\"" << flattenJSONforPython(json) << "\"\"\"";
        tab(n + 2, *fOut);
        *fOut << "return json.loads(json_str)";
        tab(n + 1, *fOut);
    }

    // User interface
    tab(n + 1, *fOut);
    *fOut << "def build_interface(self, ui_path: List[str], unnorm_funcs: Dict[str, Tuple[str, Callable]]) -> None:";
    tab(n + 2, *fOut);
    gGlobal->gJAXVisitor->Tab(n + 2);
    generateUserInterface(gGlobal->gJAXVisitor);
    tab(n + 2, *fOut);
    *fOut << "return";

    // Compute
    tab(n + 1, *fOut);
    generateCompute(n + 1);
    tab(n, *fOut);
}

void JAXCodeContainer::generateCompute(int n)
{
    // Generates declaration
    tab(n, *fOut);
    *fOut << "def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None):";
    tab(n + 1, *fOut);
    
    // Generate RNG helper function for random calls
    tab(n + 1, *fOut);
    *fOut << "# Helper function to get RNG keys";
    tab(n + 1, *fOut);
    *fOut << "rngs = nnx.Rngs(rng) if rng is not None else nnx.data(None)";
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
}

void JAXCodeContainer::generateSR()
{
    if (!fGeneratedSR) {
        pushDeclare(IB::genDecStructVar("fSampleRate", IB::genInt32Typed()));
    }
    pushPreInitMethod(
        IB::genStoreStructVar("fSampleRate", IB::genLoadFunArgsVar("self.sample_rate")));
}

// Scalar
JAXScalarCodeContainer::JAXScalarCodeContainer(const string& name, int numInputs, int numOutputs,
                                               std::ostream* out, int sub_container_type)
    : JAXCodeContainer(name, numInputs, numOutputs, out)
{
    fSubContainerType = sub_container_type;
}
