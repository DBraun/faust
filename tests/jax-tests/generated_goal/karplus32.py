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
		ui_path.append("karplus32") 
		ui_path.append("excitator") 
		self.add_hslider("fHslider4", ui_path, "excitation (samples)", 128.0, 2.0, 512.0, unnorm_funcs, "linear") 
		self.add_button("fButton0", ui_path, "play", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("noise generator") 
		self.add_hslider("fHslider3", ui_path, "level", 0.5, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		self.add_hslider("fHslider0", ui_path, "output volume", 0.4866, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.append("resonator x32") 
		self.add_hslider("fHslider2", ui_path, "attenuation", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider6", ui_path, "detune", 37.9904, 0.0, 512.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider5", ui_path, "duration (samples)", 128.0, 2.0, 512.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider1", ui_path, "polyphony", 14.0, 0.0, 32.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		
	@property
	def num_inputs(self):
		return 0
	
	@property
	def num_outputs(self):
		return 2
	
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec2"] = np.float32(0)
		state["fVec0"] = np.float32(0)
		state["iRec1"] = np.int32(0)
		# Initialize array delays
		state["fVec1"] = np.zeros((512,), dtype=np.float32)
		state["fRec0"] = np.zeros((3,), dtype=np.float32)
		state["fVec2"] = np.zeros((2048,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fVec3"] = np.zeros((4096,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fVec4"] = np.zeros((4096,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fVec5"] = np.zeros((4096,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fVec6"] = np.zeros((4096,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fVec7"] = np.zeros((4096,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fVec8"] = np.zeros((4096,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fVec9"] = np.zeros((4096,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fVec10"] = np.zeros((4096,), dtype=np.float32)
		state["fRec11"] = np.zeros((3,), dtype=np.float32)
		state["fVec11"] = np.zeros((4096,), dtype=np.float32)
		state["fRec12"] = np.zeros((3,), dtype=np.float32)
		state["fVec12"] = np.zeros((4096,), dtype=np.float32)
		state["fRec13"] = np.zeros((3,), dtype=np.float32)
		state["fVec13"] = np.zeros((4096,), dtype=np.float32)
		state["fRec14"] = np.zeros((3,), dtype=np.float32)
		state["fVec14"] = np.zeros((4096,), dtype=np.float32)
		state["fRec15"] = np.zeros((3,), dtype=np.float32)
		state["fVec15"] = np.zeros((4096,), dtype=np.float32)
		state["fRec16"] = np.zeros((3,), dtype=np.float32)
		state["fVec16"] = np.zeros((4096,), dtype=np.float32)
		state["fRec17"] = np.zeros((3,), dtype=np.float32)
		state["fVec17"] = np.zeros((1024,), dtype=np.float32)
		state["fRec18"] = np.zeros((3,), dtype=np.float32)
		state["fVec18"] = np.zeros((2048,), dtype=np.float32)
		state["fRec19"] = np.zeros((3,), dtype=np.float32)
		state["fVec19"] = np.zeros((4096,), dtype=np.float32)
		state["fRec20"] = np.zeros((3,), dtype=np.float32)
		state["fVec20"] = np.zeros((4096,), dtype=np.float32)
		state["fRec21"] = np.zeros((3,), dtype=np.float32)
		state["fVec21"] = np.zeros((4096,), dtype=np.float32)
		state["fRec22"] = np.zeros((3,), dtype=np.float32)
		state["fVec22"] = np.zeros((4096,), dtype=np.float32)
		state["fRec23"] = np.zeros((3,), dtype=np.float32)
		state["fVec23"] = np.zeros((4096,), dtype=np.float32)
		state["fRec24"] = np.zeros((3,), dtype=np.float32)
		state["fVec24"] = np.zeros((4096,), dtype=np.float32)
		state["fRec25"] = np.zeros((3,), dtype=np.float32)
		state["fVec25"] = np.zeros((4096,), dtype=np.float32)
		state["fRec26"] = np.zeros((3,), dtype=np.float32)
		state["fVec26"] = np.zeros((4096,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fVec27"] = np.zeros((4096,), dtype=np.float32)
		state["fRec28"] = np.zeros((3,), dtype=np.float32)
		state["fVec28"] = np.zeros((4096,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fVec29"] = np.zeros((4096,), dtype=np.float32)
		state["fRec30"] = np.zeros((3,), dtype=np.float32)
		state["fVec30"] = np.zeros((4096,), dtype=np.float32)
		state["fRec31"] = np.zeros((3,), dtype=np.float32)
		state["fVec31"] = np.zeros((4096,), dtype=np.float32)
		state["fRec32"] = np.zeros((3,), dtype=np.float32)
		state["fVec32"] = np.zeros((4096,), dtype=np.float32)
		state["fRec33"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = params["fHslider0"] 
		fSlow1 = params["fHslider1"] 
		fSlow2 = ((fSlow1 > jnp.float32(0.0)).astype(jnp.int32)) 
		fSlow3 = (jnp.float32(0.5) * (jnp.float32(1.0) - params["fHslider2"])) 
		fSlow4 = (jnp.float32(4.656613e-10) * params["fHslider3"]) 
		fSlow5 = params["fButton0"] 
		fSlow6 = (jnp.float32(1.0) / params["fHslider4"]) 
		fSlow7 = params["fHslider5"] 
		iSlow8 = (jnp.int32((fSlow7 + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow9 = ((fSlow1 > jnp.float32(2.0)).astype(jnp.int32)) 
		fSlow10 = params["fHslider6"] 
		iSlow11 = (jnp.int32(((fSlow7 + (jnp.float32(2.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow12 = ((fSlow1 > jnp.float32(4.0)).astype(jnp.int32)) 
		iSlow13 = (jnp.int32(((fSlow7 + (jnp.float32(4.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow14 = ((fSlow1 > jnp.float32(6.0)).astype(jnp.int32)) 
		iSlow15 = (jnp.int32(((fSlow7 + (jnp.float32(6.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow16 = ((fSlow1 > jnp.float32(8.0)).astype(jnp.int32)) 
		iSlow17 = (jnp.int32(((fSlow7 + (jnp.float32(8.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow18 = ((fSlow1 > jnp.float32(1e+01)).astype(jnp.int32)) 
		iSlow19 = (jnp.int32(((fSlow7 + (jnp.float32(1e+01) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow20 = ((fSlow1 > jnp.float32(12.0)).astype(jnp.int32)) 
		iSlow21 = (jnp.int32(((fSlow7 + (jnp.float32(12.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow22 = ((fSlow1 > jnp.float32(14.0)).astype(jnp.int32)) 
		iSlow23 = (jnp.int32(((fSlow7 + (jnp.float32(14.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow24 = ((fSlow1 > jnp.float32(16.0)).astype(jnp.int32)) 
		iSlow25 = (jnp.int32(((fSlow7 + (jnp.float32(16.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow26 = ((fSlow1 > jnp.float32(18.0)).astype(jnp.int32)) 
		iSlow27 = (jnp.int32(((fSlow7 + (jnp.float32(18.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow28 = ((fSlow1 > jnp.float32(2e+01)).astype(jnp.int32)) 
		iSlow29 = (jnp.int32(((fSlow7 + (jnp.float32(2e+01) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow30 = ((fSlow1 > jnp.float32(22.0)).astype(jnp.int32)) 
		iSlow31 = (jnp.int32(((fSlow7 + (jnp.float32(22.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow32 = ((fSlow1 > jnp.float32(24.0)).astype(jnp.int32)) 
		iSlow33 = (jnp.int32(((fSlow7 + (jnp.float32(24.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow34 = ((fSlow1 > jnp.float32(26.0)).astype(jnp.int32)) 
		iSlow35 = (jnp.int32(((fSlow7 + (jnp.float32(26.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow36 = ((fSlow1 > jnp.float32(28.0)).astype(jnp.int32)) 
		iSlow37 = (jnp.int32(((fSlow7 + (jnp.float32(28.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow38 = ((fSlow1 > jnp.float32(3e+01)).astype(jnp.int32)) 
		iSlow39 = (jnp.int32(((fSlow7 + (jnp.float32(3e+01) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow40 = ((fSlow1 > jnp.float32(1.0)).astype(jnp.int32)) 
		iSlow41 = (jnp.int32(((fSlow7 + fSlow10) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow42 = ((fSlow1 > jnp.float32(3.0)).astype(jnp.int32)) 
		iSlow43 = (jnp.int32(((fSlow7 + (jnp.float32(3.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow44 = ((fSlow1 > jnp.float32(5.0)).astype(jnp.int32)) 
		iSlow45 = (jnp.int32(((fSlow7 + (jnp.float32(5.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow46 = ((fSlow1 > jnp.float32(7.0)).astype(jnp.int32)) 
		iSlow47 = (jnp.int32(((fSlow7 + (jnp.float32(7.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow48 = ((fSlow1 > jnp.float32(9.0)).astype(jnp.int32)) 
		iSlow49 = (jnp.int32(((fSlow7 + (jnp.float32(9.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow50 = ((fSlow1 > jnp.float32(11.0)).astype(jnp.int32)) 
		iSlow51 = (jnp.int32(((fSlow7 + (jnp.float32(11.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow52 = ((fSlow1 > jnp.float32(13.0)).astype(jnp.int32)) 
		iSlow53 = (jnp.int32(((fSlow7 + (jnp.float32(13.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow54 = ((fSlow1 > jnp.float32(15.0)).astype(jnp.int32)) 
		iSlow55 = (jnp.int32(((fSlow7 + (jnp.float32(15.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow56 = ((fSlow1 > jnp.float32(17.0)).astype(jnp.int32)) 
		iSlow57 = (jnp.int32(((fSlow7 + (jnp.float32(17.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow58 = ((fSlow1 > jnp.float32(19.0)).astype(jnp.int32)) 
		iSlow59 = (jnp.int32(((fSlow7 + (jnp.float32(19.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow60 = ((fSlow1 > jnp.float32(21.0)).astype(jnp.int32)) 
		iSlow61 = (jnp.int32(((fSlow7 + (jnp.float32(21.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow62 = ((fSlow1 > jnp.float32(23.0)).astype(jnp.int32)) 
		iSlow63 = (jnp.int32(((fSlow7 + (jnp.float32(23.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow64 = ((fSlow1 > jnp.float32(25.0)).astype(jnp.int32)) 
		iSlow65 = (jnp.int32(((fSlow7 + (jnp.float32(25.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow66 = ((fSlow1 > jnp.float32(27.0)).astype(jnp.int32)) 
		iSlow67 = (jnp.int32(((fSlow7 + (jnp.float32(27.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow68 = ((fSlow1 > jnp.float32(29.0)).astype(jnp.int32)) 
		iSlow69 = (jnp.int32(((fSlow7 + (jnp.float32(29.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		fSlow70 = ((fSlow1 > jnp.float32(31.0)).astype(jnp.int32)) 
		iSlow71 = (jnp.int32(((fSlow7 + (jnp.float32(31.0) * fSlow10)) + jnp.float32(-1.5))) & jnp.int32(4095)).astype(jnp.int32) 
		iRec1_temp = state["iRec1"] 
		fVec0_temp = state["fVec0"] 
		fRec2_temp = state["fRec2"] 
		state["iRec1"] = ((jnp.int32(1103515245) * iRec1_temp) + jnp.int32(12345)) 
		state["fVec0"] = fSlow5 
		state["fRec2"] = ((fRec2_temp + (((fSlow5 - fVec0_temp) > jnp.float32(0.0)).astype(jnp.int32))) - (fSlow6 * ((fRec2_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		fTemp0 = (fSlow4 * ((state["iRec1"]) * (((state["fRec2"] > jnp.float32(0.0)).astype(jnp.int32)) + jnp.float32(1.5258789e-05)))) 
		state["fVec1"] = state["fVec1"].at[(state["IOTA0"] & 511).astype(jnp.int32)].set(((fSlow3 * (state["fRec0"][1] + state["fRec0"][2])) + fTemp0)) 
		state["fRec0"] = state["fRec0"].at[0].set(state["fVec1"][((state["IOTA0"] - iSlow8) & 511).astype(jnp.int32)]) 
		state["fVec2"] = state["fVec2"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec3"][1] + state["fRec3"][2])))) 
		state["fRec3"] = state["fRec3"].at[0].set(state["fVec2"][((state["IOTA0"] - iSlow11) & 2047).astype(jnp.int32)]) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec4"][1] + state["fRec4"][2])))) 
		state["fRec4"] = state["fRec4"].at[0].set(state["fVec3"][((state["IOTA0"] - iSlow13) & 4095).astype(jnp.int32)]) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec5"][1] + state["fRec5"][2])))) 
		state["fRec5"] = state["fRec5"].at[0].set(state["fVec4"][((state["IOTA0"] - iSlow15) & 4095).astype(jnp.int32)]) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec6"][1] + state["fRec6"][2])))) 
		state["fRec6"] = state["fRec6"].at[0].set(state["fVec5"][((state["IOTA0"] - iSlow17) & 4095).astype(jnp.int32)]) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec7"][1] + state["fRec7"][2])))) 
		state["fRec7"] = state["fRec7"].at[0].set(state["fVec6"][((state["IOTA0"] - iSlow19) & 4095).astype(jnp.int32)]) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec8"][1] + state["fRec8"][2])))) 
		state["fRec8"] = state["fRec8"].at[0].set(state["fVec7"][((state["IOTA0"] - iSlow21) & 4095).astype(jnp.int32)]) 
		state["fVec8"] = state["fVec8"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec9"][1] + state["fRec9"][2])))) 
		state["fRec9"] = state["fRec9"].at[0].set(state["fVec8"][((state["IOTA0"] - iSlow23) & 4095).astype(jnp.int32)]) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec10"][1] + state["fRec10"][2])))) 
		state["fRec10"] = state["fRec10"].at[0].set(state["fVec9"][((state["IOTA0"] - iSlow25) & 4095).astype(jnp.int32)]) 
		state["fVec10"] = state["fVec10"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec11"][1] + state["fRec11"][2])))) 
		state["fRec11"] = state["fRec11"].at[0].set(state["fVec10"][((state["IOTA0"] - iSlow27) & 4095).astype(jnp.int32)]) 
		state["fVec11"] = state["fVec11"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec12"][1] + state["fRec12"][2])))) 
		state["fRec12"] = state["fRec12"].at[0].set(state["fVec11"][((state["IOTA0"] - iSlow29) & 4095).astype(jnp.int32)]) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec13"][1] + state["fRec13"][2])))) 
		state["fRec13"] = state["fRec13"].at[0].set(state["fVec12"][((state["IOTA0"] - iSlow31) & 4095).astype(jnp.int32)]) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec14"][1] + state["fRec14"][2])))) 
		state["fRec14"] = state["fRec14"].at[0].set(state["fVec13"][((state["IOTA0"] - iSlow33) & 4095).astype(jnp.int32)]) 
		state["fVec14"] = state["fVec14"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec15"][1] + state["fRec15"][2])))) 
		state["fRec15"] = state["fRec15"].at[0].set(state["fVec14"][((state["IOTA0"] - iSlow35) & 4095).astype(jnp.int32)]) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec16"][1] + state["fRec16"][2])))) 
		state["fRec16"] = state["fRec16"].at[0].set(state["fVec15"][((state["IOTA0"] - iSlow37) & 4095).astype(jnp.int32)]) 
		state["fVec16"] = state["fVec16"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec17"][1] + state["fRec17"][2])))) 
		state["fRec17"] = state["fRec17"].at[0].set(state["fVec16"][((state["IOTA0"] - iSlow39) & 4095).astype(jnp.int32)]) 
		_result0 = (fSlow0 * ((((((((((((((((fSlow2 * state["fRec0"][0]) + (fSlow9 * state["fRec3"][0])) + (fSlow12 * state["fRec4"][0])) + (fSlow14 * state["fRec5"][0])) + (fSlow16 * state["fRec6"][0])) + (fSlow18 * state["fRec7"][0])) + (fSlow20 * state["fRec8"][0])) + (fSlow22 * state["fRec9"][0])) + (fSlow24 * state["fRec10"][0])) + (fSlow26 * state["fRec11"][0])) + (fSlow28 * state["fRec12"][0])) + (fSlow30 * state["fRec13"][0])) + (fSlow32 * state["fRec14"][0])) + (fSlow34 * state["fRec15"][0])) + (fSlow36 * state["fRec16"][0])) + (fSlow38 * state["fRec17"][0]))) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec18"][1] + state["fRec18"][2])))) 
		state["fRec18"] = state["fRec18"].at[0].set(state["fVec17"][((state["IOTA0"] - iSlow41) & 1023).astype(jnp.int32)]) 
		state["fVec18"] = state["fVec18"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec19"][1] + state["fRec19"][2])))) 
		state["fRec19"] = state["fRec19"].at[0].set(state["fVec18"][((state["IOTA0"] - iSlow43) & 2047).astype(jnp.int32)]) 
		state["fVec19"] = state["fVec19"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec20"][1] + state["fRec20"][2])))) 
		state["fRec20"] = state["fRec20"].at[0].set(state["fVec19"][((state["IOTA0"] - iSlow45) & 4095).astype(jnp.int32)]) 
		state["fVec20"] = state["fVec20"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec21"][1] + state["fRec21"][2])))) 
		state["fRec21"] = state["fRec21"].at[0].set(state["fVec20"][((state["IOTA0"] - iSlow47) & 4095).astype(jnp.int32)]) 
		state["fVec21"] = state["fVec21"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec22"][1] + state["fRec22"][2])))) 
		state["fRec22"] = state["fRec22"].at[0].set(state["fVec21"][((state["IOTA0"] - iSlow49) & 4095).astype(jnp.int32)]) 
		state["fVec22"] = state["fVec22"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec23"][1] + state["fRec23"][2])))) 
		state["fRec23"] = state["fRec23"].at[0].set(state["fVec22"][((state["IOTA0"] - iSlow51) & 4095).astype(jnp.int32)]) 
		state["fVec23"] = state["fVec23"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec24"][1] + state["fRec24"][2])))) 
		state["fRec24"] = state["fRec24"].at[0].set(state["fVec23"][((state["IOTA0"] - iSlow53) & 4095).astype(jnp.int32)]) 
		state["fVec24"] = state["fVec24"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec25"][1] + state["fRec25"][2])))) 
		state["fRec25"] = state["fRec25"].at[0].set(state["fVec24"][((state["IOTA0"] - iSlow55) & 4095).astype(jnp.int32)]) 
		state["fVec25"] = state["fVec25"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec26"][1] + state["fRec26"][2])))) 
		state["fRec26"] = state["fRec26"].at[0].set(state["fVec25"][((state["IOTA0"] - iSlow57) & 4095).astype(jnp.int32)]) 
		state["fVec26"] = state["fVec26"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec27"][1] + state["fRec27"][2])))) 
		state["fRec27"] = state["fRec27"].at[0].set(state["fVec26"][((state["IOTA0"] - iSlow59) & 4095).astype(jnp.int32)]) 
		state["fVec27"] = state["fVec27"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec28"][1] + state["fRec28"][2])))) 
		state["fRec28"] = state["fRec28"].at[0].set(state["fVec27"][((state["IOTA0"] - iSlow61) & 4095).astype(jnp.int32)]) 
		state["fVec28"] = state["fVec28"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec29"][1] + state["fRec29"][2])))) 
		state["fRec29"] = state["fRec29"].at[0].set(state["fVec28"][((state["IOTA0"] - iSlow63) & 4095).astype(jnp.int32)]) 
		state["fVec29"] = state["fVec29"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec30"][1] + state["fRec30"][2])))) 
		state["fRec30"] = state["fRec30"].at[0].set(state["fVec29"][((state["IOTA0"] - iSlow65) & 4095).astype(jnp.int32)]) 
		state["fVec30"] = state["fVec30"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec31"][1] + state["fRec31"][2])))) 
		state["fRec31"] = state["fRec31"].at[0].set(state["fVec30"][((state["IOTA0"] - iSlow67) & 4095).astype(jnp.int32)]) 
		state["fVec31"] = state["fVec31"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec32"][1] + state["fRec32"][2])))) 
		state["fRec32"] = state["fRec32"].at[0].set(state["fVec31"][((state["IOTA0"] - iSlow69) & 4095).astype(jnp.int32)]) 
		state["fVec32"] = state["fVec32"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp0 + (fSlow3 * (state["fRec33"][1] + state["fRec33"][2])))) 
		state["fRec33"] = state["fRec33"].at[0].set(state["fVec32"][((state["IOTA0"] - iSlow71) & 4095).astype(jnp.int32)]) 
		_result1 = (fSlow0 * ((((((((((((((((fSlow40 * state["fRec18"][0]) + (fSlow42 * state["fRec19"][0])) + (fSlow44 * state["fRec20"][0])) + (fSlow46 * state["fRec21"][0])) + (fSlow48 * state["fRec22"][0])) + (fSlow50 * state["fRec23"][0])) + (fSlow52 * state["fRec24"][0])) + (fSlow54 * state["fRec25"][0])) + (fSlow56 * state["fRec26"][0])) + (fSlow58 * state["fRec27"][0])) + (fSlow60 * state["fRec28"][0])) + (fSlow62 * state["fRec29"][0])) + (fSlow64 * state["fRec30"][0])) + (fSlow66 * state["fRec31"][0])) + (fSlow68 * state["fRec32"][0])) + (fSlow70 * state["fRec33"][0]))) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec0"] = jnp.roll(state["fRec0"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec10"] = jnp.roll(state["fRec10"], 1) 
		state["fRec11"] = jnp.roll(state["fRec11"], 1) 
		state["fRec12"] = jnp.roll(state["fRec12"], 1) 
		state["fRec13"] = jnp.roll(state["fRec13"], 1) 
		state["fRec14"] = jnp.roll(state["fRec14"], 1) 
		state["fRec15"] = jnp.roll(state["fRec15"], 1) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec17"] = jnp.roll(state["fRec17"], 1) 
		state["fRec18"] = jnp.roll(state["fRec18"], 1) 
		state["fRec19"] = jnp.roll(state["fRec19"], 1) 
		state["fRec20"] = jnp.roll(state["fRec20"], 1) 
		state["fRec21"] = jnp.roll(state["fRec21"], 1) 
		state["fRec22"] = jnp.roll(state["fRec22"], 1) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec24"] = jnp.roll(state["fRec24"], 1) 
		state["fRec25"] = jnp.roll(state["fRec25"], 1) 
		state["fRec26"] = jnp.roll(state["fRec26"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec28"] = jnp.roll(state["fRec28"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec30"] = jnp.roll(state["fRec30"], 1) 
		state["fRec31"] = jnp.roll(state["fRec31"], 1) 
		state["fRec32"] = jnp.roll(state["fRec32"], 1) 
		state["fRec33"] = jnp.roll(state["fRec33"], 1) 
		return state, jnp.stack([_result0,_result1]) 
		
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
