# ************************************************************************
# FAUST Architecture File
# Copyright (C) 2025 GRAME, Centre National de Creation Musicale
# ---------------------------------------------------------------------

# This is sample code. This file is provided as an example of minimal
# FAUST architecture file. Redistribution and use in source and binary
# forms, with or without modification, in part or in full are permitted.
# In particular you can create a derived work of this FAUST architecture
# and distribute that work under terms of your choice.

# This sample code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ************************************************************************

import json
import dataclasses
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import jax
from jax import numpy as jnp, random
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from
from flax.typing import Dtype

try:
	import librosa
except ImportError:
	print("Warning: librosa not installed. Soundfile loading will return dummy data.")
	print("Install with: pip install librosa")
	librosa = None

# Generated code
"""
Code generated with Faust version 2.81.3
Compilation options: -a ../../architecture/jax/minimal.py -lang jax -it -ct 1 -es 1 -mcd 16 -mdd 1024 -mdy 33 -single -ftz 0 
"""

def remainder(x, y):
	"""C++ std::remainder implemented with jax numpy"""
	quo = jnp.round(x/y)
	return x - quo * y
	
class mydsp(nnx.Module):
	
	def __init__(
		self,
		sample_rate: int,
		soundfile_dirs: list[str] = None,
		faust_float: Dtype = jnp.float32,
		faust_int: Dtype = jnp.int32,
		rng_collection: str = "rng_stream",
		rngs: nnx.Rngs = None,
	):
		self._parameter_metadata = {}
		self.sample_rate = sample_rate
		self.soundfile_dirs = soundfile_dirs or []
		self.faust_float = faust_float
		self.faust_int = faust_int
		self.rng_collection = rng_collection
		self.rngs = rngs

		# fmt: off
		# Initialize static tables
		# Initialize waveform data
		# Convert static tables and waveform data to JAX arrays
		# Initialize UI parameters
		unnorm_funcs = {}
		ui_path = []
		ui_path.append("Multi Band Filter") 
		ui_path.append("peak 0") 
		self.add_nentry("fEntry19", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry18", ui_path, "freq", 1e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider9", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 1") 
		self.add_nentry("fEntry17", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry16", ui_path, "freq", 2e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider8", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 2") 
		self.add_nentry("fEntry15", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry14", ui_path, "freq", 3e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider7", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 3") 
		self.add_nentry("fEntry13", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry12", ui_path, "freq", 4e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider6", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 4") 
		self.add_nentry("fEntry11", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry10", ui_path, "freq", 5e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider5", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 5") 
		self.add_nentry("fEntry9", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry8", ui_path, "freq", 6e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider4", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 6") 
		self.add_nentry("fEntry7", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry6", ui_path, "freq", 7e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider3", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 7") 
		self.add_nentry("fEntry5", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry4", ui_path, "freq", 8e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider2", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 8") 
		self.add_nentry("fEntry3", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry2", ui_path, "freq", 9e+03, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider1", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("peak 9") 
		self.add_nentry("fEntry1", ui_path, "Q factor", 5e+01, 0.1, 1e+02, 0.1, unnorm_funcs, "linear") 
		self.add_nentry("fEntry0", ui_path, "freq", 1e+04, 2e+01, 2e+04, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider0", ui_path, "gain", 0.0, -5e+01, 5e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = (np.float32(3.1415927) / np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate)))) 
		
	@property
	def num_inputs(self):
		return 1
	
	@property
	def num_outputs(self):
		return 1
	
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize array delays
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec0"] = np.zeros((3,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = jnp.tan((self._fConst0 * params["fEntry0"])) 
		fSlow1 = params["fEntry1"] 
		fSlow2 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider0"]))) / fSlow1) 
		fSlow3 = (jnp.float32(1.0) / ((fSlow0 * (fSlow0 + fSlow2)) + jnp.float32(1.0))) 
		fSlow4 = (jnp.float32(2.0) * (jnp.power(fSlow0, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow5 = jnp.tan((self._fConst0 * params["fEntry2"])) 
		fSlow6 = params["fEntry3"] 
		fSlow7 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider1"]))) / fSlow6) 
		fSlow8 = (jnp.float32(1.0) / ((fSlow5 * (fSlow5 + fSlow7)) + jnp.float32(1.0))) 
		fSlow9 = (jnp.float32(2.0) * (jnp.power(fSlow5, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow10 = jnp.tan((self._fConst0 * params["fEntry4"])) 
		fSlow11 = params["fEntry5"] 
		fSlow12 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider2"]))) / fSlow11) 
		fSlow13 = (jnp.float32(1.0) / ((fSlow10 * (fSlow10 + fSlow12)) + jnp.float32(1.0))) 
		fSlow14 = (jnp.float32(2.0) * (jnp.power(fSlow10, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow15 = jnp.tan((self._fConst0 * params["fEntry6"])) 
		fSlow16 = params["fEntry7"] 
		fSlow17 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider3"]))) / fSlow16) 
		fSlow18 = (jnp.float32(1.0) / ((fSlow15 * (fSlow15 + fSlow17)) + jnp.float32(1.0))) 
		fSlow19 = (jnp.float32(2.0) * (jnp.power(fSlow15, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow20 = jnp.tan((self._fConst0 * params["fEntry8"])) 
		fSlow21 = params["fEntry9"] 
		fSlow22 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider4"]))) / fSlow21) 
		fSlow23 = (jnp.float32(1.0) / ((fSlow20 * (fSlow20 + fSlow22)) + jnp.float32(1.0))) 
		fSlow24 = (jnp.float32(2.0) * (jnp.power(fSlow20, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow25 = jnp.tan((self._fConst0 * params["fEntry10"])) 
		fSlow26 = params["fEntry11"] 
		fSlow27 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider5"]))) / fSlow26) 
		fSlow28 = (jnp.float32(1.0) / ((fSlow25 * (fSlow25 + fSlow27)) + jnp.float32(1.0))) 
		fSlow29 = (jnp.float32(2.0) * (jnp.power(fSlow25, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow30 = jnp.tan((self._fConst0 * params["fEntry12"])) 
		fSlow31 = params["fEntry13"] 
		fSlow32 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider6"]))) / fSlow31) 
		fSlow33 = (jnp.float32(1.0) / ((fSlow30 * (fSlow30 + fSlow32)) + jnp.float32(1.0))) 
		fSlow34 = (jnp.float32(2.0) * (jnp.power(fSlow30, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow35 = jnp.tan((self._fConst0 * params["fEntry14"])) 
		fSlow36 = params["fEntry15"] 
		fSlow37 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider7"]))) / fSlow36) 
		fSlow38 = (jnp.float32(1.0) / ((fSlow35 * (fSlow35 + fSlow37)) + jnp.float32(1.0))) 
		fSlow39 = (jnp.float32(2.0) * (jnp.power(fSlow35, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow40 = jnp.tan((self._fConst0 * params["fEntry16"])) 
		fSlow41 = params["fEntry17"] 
		fSlow42 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider8"]))) / fSlow41) 
		fSlow43 = (jnp.float32(1.0) / ((fSlow40 * (fSlow40 + fSlow42)) + jnp.float32(1.0))) 
		fSlow44 = (jnp.float32(2.0) * (jnp.power(fSlow40, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow45 = jnp.tan((self._fConst0 * params["fEntry18"])) 
		fSlow46 = params["fEntry19"] 
		fSlow47 = (jnp.power(jnp.float32(1e+01), -((jnp.float32(0.05) * params["fVslider9"]))) / fSlow46) 
		fSlow48 = (jnp.float32(1.0) / ((fSlow45 * (fSlow45 + fSlow47)) + jnp.float32(1.0))) 
		fSlow49 = (jnp.float32(2.0) * (jnp.power(fSlow45, jnp.float32(2.0)) + jnp.float32(-1.0))) 
		fSlow50 = ((fSlow45 * (fSlow45 - fSlow47)) + jnp.float32(1.0)) 
		fSlow51 = (jnp.float32(1.0) / fSlow46) 
		fSlow52 = ((fSlow45 * (fSlow45 + fSlow51)) + jnp.float32(1.0)) 
		fSlow53 = (jnp.float32(1.0) - (fSlow45 * (fSlow51 - fSlow45))) 
		fSlow54 = ((fSlow40 * (fSlow40 - fSlow42)) + jnp.float32(1.0)) 
		fSlow55 = (jnp.float32(1.0) / fSlow41) 
		fSlow56 = ((fSlow40 * (fSlow40 + fSlow55)) + jnp.float32(1.0)) 
		fSlow57 = (jnp.float32(1.0) - (fSlow40 * (fSlow55 - fSlow40))) 
		fSlow58 = ((fSlow35 * (fSlow35 - fSlow37)) + jnp.float32(1.0)) 
		fSlow59 = (jnp.float32(1.0) / fSlow36) 
		fSlow60 = ((fSlow35 * (fSlow35 + fSlow59)) + jnp.float32(1.0)) 
		fSlow61 = (jnp.float32(1.0) - (fSlow35 * (fSlow59 - fSlow35))) 
		fSlow62 = ((fSlow30 * (fSlow30 - fSlow32)) + jnp.float32(1.0)) 
		fSlow63 = (jnp.float32(1.0) / fSlow31) 
		fSlow64 = ((fSlow30 * (fSlow30 + fSlow63)) + jnp.float32(1.0)) 
		fSlow65 = (jnp.float32(1.0) - (fSlow30 * (fSlow63 - fSlow30))) 
		fSlow66 = ((fSlow25 * (fSlow25 - fSlow27)) + jnp.float32(1.0)) 
		fSlow67 = (jnp.float32(1.0) / fSlow26) 
		fSlow68 = ((fSlow25 * (fSlow25 + fSlow67)) + jnp.float32(1.0)) 
		fSlow69 = (jnp.float32(1.0) - (fSlow25 * (fSlow67 - fSlow25))) 
		fSlow70 = ((fSlow20 * (fSlow20 - fSlow22)) + jnp.float32(1.0)) 
		fSlow71 = (jnp.float32(1.0) / fSlow21) 
		fSlow72 = ((fSlow20 * (fSlow20 + fSlow71)) + jnp.float32(1.0)) 
		fSlow73 = (jnp.float32(1.0) - (fSlow20 * (fSlow71 - fSlow20))) 
		fSlow74 = ((fSlow15 * (fSlow15 - fSlow17)) + jnp.float32(1.0)) 
		fSlow75 = (jnp.float32(1.0) / fSlow16) 
		fSlow76 = ((fSlow15 * (fSlow15 + fSlow75)) + jnp.float32(1.0)) 
		fSlow77 = (jnp.float32(1.0) - (fSlow15 * (fSlow75 - fSlow15))) 
		fSlow78 = ((fSlow10 * (fSlow10 - fSlow12)) + jnp.float32(1.0)) 
		fSlow79 = (jnp.float32(1.0) / fSlow11) 
		fSlow80 = ((fSlow10 * (fSlow10 + fSlow79)) + jnp.float32(1.0)) 
		fSlow81 = (jnp.float32(1.0) - (fSlow10 * (fSlow79 - fSlow10))) 
		fSlow82 = ((fSlow5 * (fSlow5 - fSlow7)) + jnp.float32(1.0)) 
		fSlow83 = (jnp.float32(1.0) / fSlow6) 
		fSlow84 = ((fSlow5 * (fSlow5 + fSlow83)) + jnp.float32(1.0)) 
		fSlow85 = (jnp.float32(1.0) - (fSlow5 * (fSlow83 - fSlow5))) 
		fSlow86 = ((fSlow0 * (fSlow0 - fSlow2)) + jnp.float32(1.0)) 
		fSlow87 = (jnp.float32(1.0) / fSlow1) 
		fSlow88 = ((fSlow0 * (fSlow0 + fSlow87)) + jnp.float32(1.0)) 
		fSlow89 = (jnp.float32(1.0) - (fSlow0 * (fSlow87 - fSlow0))) 
		fTemp0 = (fSlow49 * state["fRec9"][1]) 
		state["fRec9"] = state["fRec9"].at[0].set((inputs[0] - (fSlow48 * ((fSlow50 * state["fRec9"][2]) + fTemp0)))) 
		fTemp1 = (fSlow44 * state["fRec8"][1]) 
		state["fRec8"] = state["fRec8"].at[0].set(((fSlow48 * ((fTemp0 + (fSlow52 * state["fRec9"][0])) + (fSlow53 * state["fRec9"][2]))) - (fSlow43 * ((fSlow54 * state["fRec8"][2]) + fTemp1)))) 
		fTemp2 = (fSlow39 * state["fRec7"][1]) 
		state["fRec7"] = state["fRec7"].at[0].set(((fSlow43 * ((fTemp1 + (fSlow56 * state["fRec8"][0])) + (fSlow57 * state["fRec8"][2]))) - (fSlow38 * ((fSlow58 * state["fRec7"][2]) + fTemp2)))) 
		fTemp3 = (fSlow34 * state["fRec6"][1]) 
		state["fRec6"] = state["fRec6"].at[0].set(((fSlow38 * ((fTemp2 + (fSlow60 * state["fRec7"][0])) + (fSlow61 * state["fRec7"][2]))) - (fSlow33 * ((fSlow62 * state["fRec6"][2]) + fTemp3)))) 
		fTemp4 = (fSlow29 * state["fRec5"][1]) 
		state["fRec5"] = state["fRec5"].at[0].set(((fSlow33 * ((fTemp3 + (fSlow64 * state["fRec6"][0])) + (fSlow65 * state["fRec6"][2]))) - (fSlow28 * ((fSlow66 * state["fRec5"][2]) + fTemp4)))) 
		fTemp5 = (fSlow24 * state["fRec4"][1]) 
		state["fRec4"] = state["fRec4"].at[0].set(((fSlow28 * ((fTemp4 + (fSlow68 * state["fRec5"][0])) + (fSlow69 * state["fRec5"][2]))) - (fSlow23 * ((fSlow70 * state["fRec4"][2]) + fTemp5)))) 
		fTemp6 = (fSlow19 * state["fRec3"][1]) 
		state["fRec3"] = state["fRec3"].at[0].set(((fSlow23 * ((fTemp5 + (fSlow72 * state["fRec4"][0])) + (fSlow73 * state["fRec4"][2]))) - (fSlow18 * ((fSlow74 * state["fRec3"][2]) + fTemp6)))) 
		fTemp7 = (fSlow14 * state["fRec2"][1]) 
		state["fRec2"] = state["fRec2"].at[0].set(((fSlow18 * ((fTemp6 + (fSlow76 * state["fRec3"][0])) + (fSlow77 * state["fRec3"][2]))) - (fSlow13 * ((fSlow78 * state["fRec2"][2]) + fTemp7)))) 
		fTemp8 = (fSlow9 * state["fRec1"][1]) 
		state["fRec1"] = state["fRec1"].at[0].set(((fSlow13 * ((fTemp7 + (fSlow80 * state["fRec2"][0])) + (fSlow81 * state["fRec2"][2]))) - (fSlow8 * ((fSlow82 * state["fRec1"][2]) + fTemp8)))) 
		fTemp9 = (fSlow4 * state["fRec0"][1]) 
		state["fRec0"] = state["fRec0"].at[0].set(((fSlow8 * ((fTemp8 + (fSlow84 * state["fRec1"][0])) + (fSlow85 * state["fRec1"][2]))) - (fSlow3 * ((fSlow86 * state["fRec0"][2]) + fTemp9)))) 
		_result0 = (fSlow3 * ((fTemp9 + (fSlow88 * state["fRec0"][0])) + (fSlow89 * state["fRec0"][2]))) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec0"] = jnp.roll(state["fRec0"], 1) 
		return state, jnp.stack([_result0]) 
		
	# fmt: on
	def __repr__(self):
		name = self.__class__.__name__
		return f"{name}(sample_rate={self.sample_rate}, soundfile_dirs={self.soundfile_dirs})"

	def load_soundfile(self, filepath: str) -> Tuple[np.ndarray, int]:
		if librosa is None:
			return np.zeros((1, 1024)), self.sample_rate
		
		# soundfile_dirs should always include at least "".
		soundfile_dirs = [""] + list(self.soundfile_dirs)
		# Create a list of potential filepaths to check
		potential_paths = [Path(filepath)] if Path(filepath).is_absolute() else [Path(d) / filepath for d in soundfile_dirs]
		
		# Loop through potential paths and try to load the audio file
		for full_path in potential_paths:
			try:
				audio, sr = librosa.load(str(full_path), mono=False, sr=None)
				if audio.ndim == 1:
					audio = np.expand_dims(audio, 0)
				return audio, sr
			except FileNotFoundError:
				# If not found at this path, continue to the next
				continue

		raise FileNotFoundError(f"Could not load soundfile for path: {filepath}")
	
	def add_soundfile(self, zone: str, ui_path: list[str], label: str, url: str, unnorm_funcs: dict):
		# example url: {"tango.wav';'foo.wav';'bar/baz.wav'}
		filepaths = url[2:-2].split("';'")
		fLength, fOffset, fSR, offset = [], [], [], 0
		audio_data = [self.load_soundfile(filepath) for filepath in filepaths]
		num_chans = max([y.shape[0] for y, _ in audio_data])
		total_length = sum([y.shape[1] for y, _ in audio_data])
		fBuffers = jnp.zeros((num_chans, total_length))
		for y, sr in audio_data:
			fSR.append(sr)
			assert y.ndim == 2
			y = jnp.array(y, dtype=self.faust_float)
			fLength.append(y.shape[1])
			fOffset.append(offset)
			fBuffers = fBuffers.at[:y.shape[0],offset:offset+y.shape[1]].set(y)
			offset += y.shape[1]
		if label.startswith("param:"):
			label = label[6:]  # remove param:
			full_label = "/".join(ui_path+[label])
			fBuffers = nnx.Param(fBuffers)  # todo:
			setattr(self, "_" + full_label, fBuffers)  # todo: 
			unnorm_funcs[zone] = (zone, lambda x: x)
		else:
			full_label = "/".join(ui_path+[label])

		setattr(self, zone, {
			"fLength": jnp.array(fLength, dtype=jnp.int32),
			"fOffset": jnp.array(fOffset, dtype=jnp.int32),
			"fBuffers": fBuffers,
			"fSR": jnp.array(fSR, dtype=self.faust_float)
		})
		
		# Store parameter metadata
		self._parameter_metadata[zone] = {
			"full_label": full_label,
			"label": label,
			"type": "soundfile",
			"internal_name": zone,
		}
	
	def add_button(self, zone: str, ui_path: list[str], label: str, unnorm_funcs: dict):
		full_label = "/".join(ui_path+[label])
		setattr(self, zone, nnx.Param(jnp.zeros((), dtype=self.faust_float)))
		unnorm_funcs[full_label] = (zone, lambda x: x)
		
		# Store parameter metadata
		self._parameter_metadata[zone] = {
			"full_label": full_label,
			"label": label,
			"type": "button",
			"internal_name": zone,
			"min": 0.0,
			"max": 1.0,
			"default": 0.0,
		}
	
	def add_checkbox(self, zone: str, ui_path: list[str], label: str, unnorm_funcs: dict):
		self.add_button(zone, ui_path, label, unnorm_funcs)
		# Update type in metadata
		self._parameter_metadata[zone]["type"] = "checkbox"
	
	def add_nentry(
		self, zone: str, ui_path: List[str], label: str,
		init: float, a_min: float, a_max: float, step_size: float,
		unnorm_funcs: dict, scale_mode: str = "linear",
	):
		"""
		Gumbel-Softmax version of a FAUST nentry:
			* logits param  (num_steps,)
			* learnable temperature τ
			* optional Gumbel noise from self.rngs.gumbel()
		Returns a *soft* value in the physical range.

		todo: this implementation may be problematic for custom value nentry like:
		`foo = nentry("foo[style:menu{'low':0;'mid':5;'high':7}]",0,0,7,1)`
		"""
		faust_float = self.faust_float
		# ---------- set up grid ----------
		full_label = "/".join(ui_path + [label])
		num_steps  = int(round((a_max - a_min) / step_size)) + 1
		init_step  = int(round((init - a_min) / step_size))
		step_values = jnp.arange(num_steps, dtype=faust_float) * faust_float(step_size) + faust_float(a_min)

		# ---------- parameters ----------
		# (1) logits, initialised to favour the initial step
		logits = jnp.zeros((num_steps,), dtype=faust_float)
		logits = logits.at[init_step].set(faust_float(5.0))  # bias ≈ exp(5) ≈ 148

		logits_zone = zone + "_logits"
		setattr(self, logits_zone, nnx.Param(logits))

		# temperature (optional learnable scalar)
		# tau = nnx.Param(jnp.ones((), dtype=faust_float))
		tau = 1.0  # TODO: user should be able to configure via UI Label metadata:
		# https://faustdoc.grame.fr/manual/syntax/#ui-label-metadata

		# Store nentry metadata as attributes. TODO: necessary?
		setattr(self, f"_{zone}_step_values", step_values)
		setattr(self, f"_{zone}_tau", tau)
		
		# Add unnormalization lambda for nentry
		def make_nentry_unnorm(zone, tau, step_values):
			def unnorm_nentry(logits):
				# todo: finish Gumbel-softmax for NNX
				# # Gumbel-softmax computation
				# if hasattr(self.rngs, 'gumbel'):  # training
				# 	gumbel_noise = random.gumbel(
				# 		self.rngs.gumbel(), logits.shape, dtype=faust_float
				# 	)
				# 	logits_with_noise = logits + gumbel_noise
				# 	probs = nnx.softmax(logits_with_noise / tau, axis=-1)
				# 	return jnp.dot(probs, step_values)
				# else:  # inference
				index = jnp.argmax(logits, axis=-1)
				return step_values[index]

			return unnorm_nentry

		unnorm_funcs[full_label] = (zone, make_nentry_unnorm(zone, tau, step_values))
		
		# Store parameter metadata
		self._parameter_metadata[zone] = {
			"full_label": full_label,
			"label": label,
			"type": "nentry",
			"internal_name": zone,
			"min": a_min,
			"max": a_max,
			"default": init,
			"step": step_size,
			"num_options": num_steps,
			"scale_mode": scale_mode,
		}
	
	def normalize_value(self, value: float, a_min: float, a_max: float, scale_mode: str) -> float:
		"""Normalize a value from [a_min, a_max] to [0, 1] based on scale mode."""
		faust_float = self.faust_float
		if scale_mode == "linear":
			return jnp.interp(value, jnp.array([a_min, a_max], dtype=faust_float), 
							 jnp.array([faust_float(0), faust_float(1)], dtype=faust_float))
		elif scale_mode == "exp":
			# Map to [1, e], take log, then map to [0, 1]
			value_exp = jnp.interp(value, jnp.array([a_min, a_max], dtype=faust_float), 
								  jnp.array([faust_float(1), jnp.e], dtype=faust_float))
			value_log = jnp.log(value_exp)
			return jnp.interp(value_log, jnp.array([faust_float(0), faust_float(1)], dtype=faust_float), 
							 jnp.array([faust_float(0), faust_float(1)], dtype=faust_float))
		elif scale_mode == "log":
			# Map to [-4, 0], apply 10^x, then map to [0, 1]
			value_log10 = jnp.interp(value, jnp.array([a_min, a_max], dtype=faust_float), 
									jnp.array([faust_float(-4), faust_float(0)], dtype=faust_float))
			value_pow = jnp.power(faust_float(10), value_log10)
			return jnp.interp(value_pow, jnp.array([faust_float(10**-4), faust_float(1)], dtype=faust_float), 
							 jnp.array([faust_float(0), faust_float(1)], dtype=faust_float))
		else:
			raise ValueError(f"Unknown scale mode: {scale_mode}")
	
	def create_unnormalize_func(self, a_min: float, a_max: float, scale_mode: str):
		"""Create an unnormalization function for the given scale mode."""
		faust_float = self.faust_float
		if scale_mode == "linear":
			return lambda normalized: jnp.interp(
				jnp.clip(normalized, faust_float(0), faust_float(1)),
				jnp.array([faust_float(0), faust_float(1)], dtype=faust_float),
				jnp.array([a_min, a_max], dtype=faust_float)
			)
		elif scale_mode == "exp":
			return lambda normalized: jnp.interp(
				jnp.exp(jnp.clip(normalized, faust_float(0), faust_float(1))), 
				jnp.array([faust_float(1), jnp.e], dtype=faust_float), 
				jnp.array([a_min, a_max], dtype=faust_float)
			)
		elif scale_mode == "log":
			return lambda normalized: jnp.interp(
				jnp.log10(jnp.interp(
					jnp.clip(normalized, faust_float(0), faust_float(1)),
					jnp.array([faust_float(0), faust_float(1)], dtype=faust_float),
					jnp.array([faust_float(10**-4), faust_float(1)], dtype=faust_float)
				)), 
				jnp.array([faust_float(-4), faust_float(0)], dtype=faust_float), 
				jnp.array([a_min, a_max], dtype=faust_float)
			)
		else:
			raise ValueError(f"Unknown scale mode: {scale_mode}")
	
	def add_slider(self, zone: str, ui_path: list[str], label: str, init: float, a_min: float, a_max: float, unnorm_funcs: dict, scale_mode="linear"):
		"""Add a slider UI element with the specified parameters."""
		faust_float = self.faust_float
		full_label = "/".join(ui_path + [label])
		init, a_min, a_max = faust_float(init), faust_float(a_min), faust_float(a_max)
		
		# Normalize init value to [0, 1] based on scale mode
		normalized_init = self.normalize_value(init, a_min, a_max, scale_mode)
		
		# Create the normalized parameter with label as name
		setattr(self, zone, nnx.Param(normalized_init))

		# Create and store the unnormalization function
		unnorm_func = self.create_unnormalize_func(a_min, a_max, scale_mode)
		unnorm_funcs[full_label] = (zone, unnorm_func)
		
		# Store parameter metadata
		self._parameter_metadata[zone] = {
			"full_label": full_label,
			"label": label,
			"type": "slider",
			"internal_name": zone,
			"min": a_min,
			"max": a_max,
			"default": init,
			"scale_mode": scale_mode,
		}
	
	def add_hslider(self, zone: str, ui_path: list[str], label: str, init: float, a_min: float, a_max: float, unnorm_funcs: dict, scale_mode: str):
		self.add_slider(zone, ui_path, label, init, a_min, a_max, unnorm_funcs, scale_mode)
		# Update type in metadata
		self._parameter_metadata[zone]["type"] = "hslider"
	
	def add_vslider(self, zone: str, ui_path: list[str], label: str, init: float, a_min: float, a_max: float, unnorm_funcs: dict, scale_mode: str):
		self.add_slider(zone, ui_path, label, init, a_min, a_max, unnorm_funcs, scale_mode)
		# Update type in metadata
		self._parameter_metadata[zone]["type"] = "vslider"
	
	def add_hbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float, unnorm_funcs: dict):
		# Bargraphs are output-only, no parameters needed
		# But we can still store metadata
		self._parameter_metadata[zone] = {
			"full_label": "/".join(ui_path + [label]),
			"label": label,
			"type": "hbargraph",
			"internal_name": zone,
			"min": a_min,
			"max": a_max,
			"output_only": True,
		}
	
	def add_vbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float, unnorm_funcs: dict):
		# Bargraphs are output-only, no parameters needed
		# But we can still store metadata
		self._parameter_metadata[zone] = {
			"full_label": "/".join(ui_path + [label]),
			"label": label,
			"type": "vbargraph",
			"internal_name": zone,
			"min": a_min,
			"max": a_max,
			"output_only": True,
		}

	def random_uniform(self, rng: jax.Array):
		"""
		Generate a random uniform value in the range [-1, 1] using JAX's PRNG.
		"""
		return random.uniform(rng, shape=(), minval=-1, maxval=1, dtype=self.faust_float)

	def unnormalize(self) -> Dict[str, jnp.ndarray]:
		"""
		Unnormalize all UI parameters from [-1, 1] to their original ranges.
		
		Returns:
			Dictionary mapping zones to unnormalized parameter values
		"""
		params = {}
		
		# Simply use the stored unnormalization functions
		for label, (zone, unnorm_func) in self._unnorm_funcs.items():
			# Check if it's a nentry (needs module as arg)
			if hasattr(self, f"{zone}_logits"):
				logits = getattr(self, f"{zone}_logits").value
				params[zone] = unnorm_func(logits)
			elif hasattr(self, zone):
				# Regular parameter
				normalized_value = getattr(self, zone).value
				params[zone] = unnorm_func(normalized_value)
			else:
				raise ValueError(f"Zone not found: {zone}")
			# self.sow("intermediates", label, params[zone])  # todo: may need to remove this?
		
		return params
	
	def get_parameter_metadata(self) -> Dict[str, Dict[str, any]]:
		"""
		Get metadata for all parameters.
		
		Returns:
			Dictionary mapping internal names to parameter metadata
		"""
		return self._parameter_metadata.copy()

	def initialize_carry(self) -> Dict[str, jnp.ndarray]:
		"""
		Initialize the carry state for real-time processing.
			
		Returns:
			Dictionary containing all stateful components (delays, filter states, etc.)
		"""
		# Create dummy input for initialization
		dummy_x = jnp.zeros((self.num_inputs, 1), dtype=self.faust_float)
		
		# Initialize the full state using fast numpy
		state = self._initialize_carry(dummy_x, 1)
		
		# Convert numpy to JAX numpy arrays
		state = jax.tree.map(jnp.array, state)
		
		return state
	
	def process_block(
		self,
		carry: Dict[str, jnp.ndarray],
		inputs: jnp.ndarray = None,
		length: int = None,
		unroll: int = 1,
		rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
	) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
		"""
		Process one block of audio and return updated state.

		Args:
			carry: State dictionary from previous block
			inputs: Input audio block of shape (num_inputs, block_size)
			length (int): block size of generated output
			unroll (int): unroll argument for nnx.scan
			rngs: rng key.

		Returns:
			Tuple of (output_block, new_carry) where:
			- output_block has shape (num_outputs, block_size)
			- new_carry is the updated state dictionary
		"""
		if length is None and inputs is not None and hasattr(inputs, "shape"):
			length = inputs.shape[-1]

		# Unnormalize parameters once before the scan
		params = self.unnormalize()

		rngs = first_from(rngs, self.rngs, error_msg="No `rngs` argument was provided as either a __call__ argument or class attribute")		
		# Get a key from the rng_stream
		if isinstance(rngs, jax.Array):
			rng_key = rngs
		elif isinstance(rngs, rnglib.Rngs):
			rng_key = rngs[self.rng_collection]()
		elif isinstance(rngs, rnglib.RngStream):
			rng_key = rngs()
		else:
			raise TypeError(f"rngs must be JAX array, Rngs or RngStream, got {type(rngs)}")
		scan_rngs = random.split(rng_key, length)

		def scan_body(carry, x, _rng):
			new_carry, y = self.tick(params, carry, x, _rng)
			return new_carry, y

		# Handle input shape for scan
		if inputs is None or self.num_inputs == 0:
			# Generator case
			inputs = jnp.zeros((0, length), dtype=self.faust_float)

		new_carry, outputs = nnx.scan(
			scan_body,
			length=length,
			unroll=unroll,
			in_axes=(nnx.Carry, 1, 0),
			out_axes=(nnx.Carry, 1),
		)(carry, inputs, scan_rngs)

		return outputs, new_carry

	def __call__(
		self,
		inputs: jnp.ndarray,
		length: int = None,
		unroll: int = 1,
		rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
	) -> jnp.ndarray:

		if length is None and inputs is not None:
			length = inputs.shape[-1]

		rngs = first_from(rngs, self.rngs, error_msg="No `rngs` argument was provided as either a __call__ argument or class attribute")
		if isinstance(rngs, jax.Array):
			rng_key = rngs
		elif isinstance(rngs, rnglib.Rngs):
			rng_key = rngs[self.rng_collection]()
		elif isinstance(rngs, rnglib.RngStream):
			rng_key = rngs()
		else:
			raise TypeError(f"rngs must be JAX array, Rngs or RngStream, got {type(rngs)}")
		scan_rngs = random.split(rng_key, length)

		# Handle generators (no input case)
		if inputs is None:
			inputs = jnp.zeros((self.num_inputs, length), dtype=self.faust_float)

		carry = self.initialize_carry()

		# Unnormalize parameters once before the scan
		params = self.unnormalize()

		def scan_body(carry, inputs, _rng):
			new_carry, y = self.tick(params, carry, inputs, _rng)
			return new_carry, y

		# Handle input shape for scan
		if self.num_inputs == 0:
			# Generator case
			inputs = jnp.zeros((0, length), dtype=self.faust_float)

		new_carry, outputs = nnx.scan(
			scan_body,
			length=length,
			unroll=unroll,
			in_axes=(nnx.Carry, 1, 0),
			out_axes=(nnx.Carry, 1),
		)(carry, inputs, scan_rngs)

		return outputs


def test(args):

	import logging

	log_level = getattr(logging, args.log_level.upper())
	logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

	logger = logging.getLogger(__name__)

	faust_float = jnp.float64 if args.double else jnp.float32

	rngs = nnx.Rngs(args.seed, params=args.seed, rng_stream=args.seed)
	model = mydsp(sample_rate=args.sample_rate, faust_float=faust_float, rngs=rngs)

	logger.info(f"Number of input channels: {model.num_inputs}")
	logger.info(f"Number of output channels: {model.num_outputs}")

	# json_obj = model.json_metadata
	# logger.debug(f"JSON info: {json_obj}")

	key = random.key(args.seed)

	if args.input is not None:
		input_audio, _ = librosa.load(
			args.input, mono=False, sr=args.sample_rate, duration=args.duration
		)
		if input_audio.ndim == 1:
			input_audio = input_audio.unsqueeze(0)

		N_SAMPLES = input_audio.shape[1]
		N_CHANNELS = input_audio.shape[0]
		assert N_CHANNELS == model.num_inputs

		input_audio = faust_float(input_audio)
	else:
		duration_sec = args.duration or 1.0  # default to 1 second when making noise.

		N_SAMPLES = int(duration_sec * args.sample_rate)
		if isinstance(args.unroll, int):
			N_SAMPLES = (N_SAMPLES // int(args.unroll)) * int(args.unroll)
		N_CHANNELS = model.num_inputs

		if args.random:
			input_audio = random.uniform(
				key,
				shape=(N_CHANNELS, N_SAMPLES),
				minval=-1,
				maxval=1,
				dtype=faust_float,
			)
		else:
			input_audio = jnp.zeros((N_CHANNELS, N_SAMPLES), dtype=faust_float)
			input_audio = input_audio.at[:, 0].set(1.0)

	if args.verbose:
		print("model:", model)

	if args.jit:
		# For JIT, we need to handle the model call differently
		# Extract a key before JIT compilation
		if model.rngs is not None:
			# Get a single RNG key to use for the entire forward pass
			if isinstance(model.rngs, rnglib.Rngs):
				rng_key = model.rngs[model.rng_collection]()
			elif isinstance(model.rngs, rnglib.RngStream):
				rng_key = model.rngs()
			else:
				rng_key = None
		else:
			rng_key = None
		
		@jax.jit
		def forward(x: jnp.ndarray):
			# Pass the pre-extracted RNG key
			y = model(x, length=N_SAMPLES, unroll=args.unroll, rngs=rng_key)
			return y
	else:
		def forward(x: jnp.ndarray):
			y = model(x, length=N_SAMPLES, unroll=args.unroll)
			return y

	if args.benchmark:
		import tqdm

		for _ in range(3):
			y = forward(input_audio).block_until_ready()
		for _ in tqdm.trange(args.benchmark):
			y = forward(input_audio).block_until_ready()

	y = forward(input_audio)

	params = model.unnormalize()
	if args.verbose:
		print("params", params)

	assert y.ndim == 2
	assert y.shape[0] == model.num_outputs
	assert y.shape[1] == input_audio.shape[1]
	assert y.shape[1] == N_SAMPLES

	if args.output is not None:
		from scipy.io import wavfile

		output_audio = np.array(y).T
		wavfile.write(args.output, args.sample_rate, output_audio)

	logger.info("All done!")


def realtime_audio_example(
	unroll: int = 1, sample_rate: int = 48_000, block_size: int = 512, use_double=False
):
	"""
	Real-time audio streaming example using sounddevice.
	Demonstrates the real-time API with actual audio output.
	"""
	try:
		import sounddevice as sd
	except ImportError:
		print("sounddevice not installed. Install with: pip install sounddevice")
		return

	import time

	# Initialize model
	faust_float = jnp.float64 if use_double else jnp.float32

	rngs = nnx.Rngs(0, params=0, rng_stream=0)
	model = mydsp(sample_rate=sample_rate, faust_float=faust_float, rngs=rngs)

	# Initialize carry state
	carry = model.initialize_carry()

	# JIT compile the process method
	@jax.jit
	def process_block_jit(carry, inputs: jnp.ndarray, rng_key: jax.Array):
		outputs, new_carry = model.process_block(
			carry,
			inputs,
			length=block_size,
			unroll=unroll,
			rngs=rng_key,
		)
		return outputs, new_carry

	# Create a generator for audio blocks
	def audio_generator():
		nonlocal carry
		rng_key = random.key(0)
		while True:
			# For generators, create empty input
			if model.num_inputs == 0:
				inputs = jnp.zeros((0, block_size))
			else:
				# For processors, you would get input from sounddevice
				# For this example, we'll use zeros
				inputs = jnp.zeros((model.num_inputs, block_size))

			# Process block
			subkey, rng_key = random.split(rng_key)
			outputs, carry = process_block_jit(carry, inputs, subkey)

			# Convert to numpy and reshape for sounddevice
			# sounddevice expects shape (frames, channels)
			output_np = np.asarray(outputs.T, dtype=np.float32)

			# If mono, reshape to (frames, 1)
			if output_np.ndim == 1:
				output_np = output_np.reshape(-1, 1)

			yield output_np

	# Create the audio generator
	audio_gen = audio_generator()

	# Sounddevice callback
	def callback(outdata, frames, time_info, status):
		if status:
			print(f"Sounddevice status: {status}")
		outdata[:] = next(audio_gen)

	# Start streaming
	print(f"▶ Streaming audio at {sample_rate} Hz, {block_size} samples/block")
	print(f"  Model: {model.num_inputs} inputs → {model.num_outputs} outputs")
	print("  Press Ctrl+C to stop...")

	try:
		with sd.OutputStream(
			channels=model.num_outputs,
			samplerate=sample_rate,
			blocksize=block_size,
			dtype="float32",
			callback=callback,
		):
			while True:
				time.sleep(1)
	except KeyboardInterrupt:
		print("\n⏹ Stopped.")


if __name__ == "__main__":
	import argparse

	# fmt: off
	parser = argparse.ArgumentParser(description="Run a JAX/Flax model converted from Faust code")	
	parser.add_argument("-sr", "--sample-rate", type=int, default=44100, help="Sample rate (such as 44100)")
	parser.add_argument("-d", "--duration", type=float, default=None, help="Output duration in seconds")
	parser.add_argument("--unroll", type=int, default=1, help="Unroll size (default is 1)")
	parser.add_argument("--random", default=False, action=argparse.BooleanOptionalAction, help='Whether the default audio is random. By default it"s an impulse.')
	parser.add_argument("--seed", default=0, type=int, help="Seed for random number generator (default: 0)")
	parser.add_argument("-i", "--input", type=str, default=None, help="Filepath for input audio WAV")
	parser.add_argument("-o", "--output", type=str, default=None, help="Filepath for output audio WAV")
	parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logger level (default: INFO)")
	parser.add_argument("--jit", default=False, action=argparse.BooleanOptionalAction, help="Whether to use JIT.")
	parser.add_argument("--benchmark", type=int, default=0, help="Number of loops for a speed benchmark with tqdm (default=0).")
	parser.add_argument("--platform", default="cpu", choices=["cpu", "gpu", "metal", "tpu"])
	parser.add_argument("--double", default=False, action=argparse.BooleanOptionalAction, help="Whether to enable double type (jnp.float64)")
	parser.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction, help="Whether to print the variables of the DSP")
	parser.add_argument("--realtime", default=False, action=argparse.BooleanOptionalAction, help="Run the DSP with silent input and send the output to an audio device in real-time.")
	parser.add_argument("-bs", "--block-size", type=int, default=512, help="Block size for real-time mode such as 512",)
	# fmt: on
	args = parser.parse_args()

	# Global flag to set a specific platform, must be used at startup.
	if args.double:
		jax.config.update("jax_enable_x64", True)
	jax.config.update("jax_platform_name", args.platform)

	if args.realtime:
		realtime_audio_example(
			args.unroll,
			sample_rate=args.sample_rate,
			block_size=args.block_size,
			use_double=args.double,
		)
	else:
		test(args)
