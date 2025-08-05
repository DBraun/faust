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
		# Table used in inline subcontainer but not declared globally
		ftbl0mydspSIG0 = np.zeros((65537,), dtype=np.float32)
		# Initialize waveform data
		# Process inline subcontainers for static table initialization
		# iRec16

		iRec16 = np.int32(0)
		for i1 in range(0, 65536):
			iRec16_temp = iRec16 
			iRec16 = (iRec16_temp + np.int32(1)) 
			ftbl0mydspSIG0[i1] = np.sin((np.float32(9.58738e-05) * ((iRec16 + np.int32(-1))))) 
		
		
		
		# Convert static tables and waveform data to JAX arrays
		self._ftbl0mydspSIG0 = jnp.array(ftbl0mydspSIG0)
		# Initialize UI parameters
		unnorm_funcs = {}
		ui_path = []
		ui_path.append("Modulations") 
		ui_path.append("Instrument") 
		self.add_hslider("fHslider4", ui_path, "Frequency", 3.3e+02, 1e+02, 1.2e+03, unnorm_funcs, "linear") 
		self.add_hslider("fHslider0", ui_path, "General Volume", 1.0, 0.75, 4.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider8", ui_path, "Oscillator Volume", 0.5, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "Modulating Frequency", 1.2e+03, 9e+02, 1.7e+03, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Modulations") 
		self.add_hslider("fHslider5", ui_path, "Play Modulation 0 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider6", ui_path, "Play Modulation 1 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider7", ui_path, "Play Modulation 2 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider3", ui_path, "Play Modulation 3 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Reverb") 
		self.add_hslider("fHslider1", ui_path, "Reverberation Room Size(InstrReverb)", 0.5, 0.05, 2.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider9", ui_path, "Reverberation Volume(InstrReverb)", 0.25, 0.05, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = np.cos((np.float32(37699.113) / self._fConst0)) 
		self._fConst2 = np.floor(((np.float32(0.219991) * self._fConst0) + np.float32(0.5))) 
		self._fConst3 = (self._fConst2 / self._fConst0) 
		self._fConst4 = (np.float32(3.4538777) * self._fConst3) 
		self._fConst5 = (np.float32(2.3025851) * self._fConst3) 
		self._fConst6 = (np.float32(1.0) / np.tan((np.float32(628.31854) / self._fConst0))) 
		self._fConst7 = (np.float32(1.0) / (self._fConst6 + np.float32(1.0))) 
		self._fConst8 = (np.float32(1.0) - self._fConst6) 
		self._fConst9 = np.floor(((np.float32(0.019123) * self._fConst0) + np.float32(0.5))) 
		self._iConst10 = np.int32(np.minimum(np.float32(16384.0), np.maximum(np.float32(0.0), (self._fConst2 - self._fConst9)))) 
		self._fConst11 = (np.float32(1.0) / np.tan((np.float32(6283.1855) / self._fConst0))) 
		self._fConst12 = (np.float32(1.0) / (self._fConst11 + np.float32(1.0))) 
		self._fConst13 = (np.float32(1.0) - self._fConst11) 
		self._fConst14 = (np.float32(1.0) / self._fConst0) 
		self._fConst15 = (np.float32(2.0) * self._fConst0) 
		self._fConst16 = (np.float32(3.0) * self._fConst0) 
		self._fConst17 = (np.float32(0.33333334) / self._fConst0) 
		self._fConst18 = (np.float32(0.5) / self._fConst0) 
		self._iConst19 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (np.float32(0.02) * self._fConst0)))) 
		self._iConst20 = np.int32(np.minimum(np.float32(1024.0), np.maximum(np.float32(0.0), (self._fConst9 + np.float32(-1.0))))) 
		self._fConst21 = np.floor(((np.float32(0.256891) * self._fConst0) + np.float32(0.5))) 
		self._fConst22 = (self._fConst21 / self._fConst0) 
		self._fConst23 = (np.float32(3.4538777) * self._fConst22) 
		self._fConst24 = (np.float32(2.3025851) * self._fConst22) 
		self._fConst25 = np.floor(((np.float32(0.027333) * self._fConst0) + np.float32(0.5))) 
		self._iConst26 = np.int32(np.minimum(np.float32(16384.0), np.maximum(np.float32(0.0), (self._fConst21 - self._fConst25)))) 
		self._iConst27 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst25 + np.float32(-1.0))))) 
		self._fConst28 = np.floor(((np.float32(0.192303) * self._fConst0) + np.float32(0.5))) 
		self._fConst29 = (self._fConst28 / self._fConst0) 
		self._fConst30 = (np.float32(3.4538777) * self._fConst29) 
		self._fConst31 = (np.float32(2.3025851) * self._fConst29) 
		self._fConst32 = np.floor(((np.float32(0.029291) * self._fConst0) + np.float32(0.5))) 
		self._iConst33 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst28 - self._fConst32)))) 
		self._iConst34 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst32 + np.float32(-1.0))))) 
		self._fConst35 = np.floor(((np.float32(0.210389) * self._fConst0) + np.float32(0.5))) 
		self._fConst36 = (self._fConst35 / self._fConst0) 
		self._fConst37 = (np.float32(3.4538777) * self._fConst36) 
		self._fConst38 = (np.float32(2.3025851) * self._fConst36) 
		self._fConst39 = np.floor(((np.float32(0.024421) * self._fConst0) + np.float32(0.5))) 
		self._iConst40 = np.int32(np.minimum(np.float32(16384.0), np.maximum(np.float32(0.0), (self._fConst35 - self._fConst39)))) 
		self._iConst41 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst39 + np.float32(-1.0))))) 
		self._fConst42 = np.floor(((np.float32(0.125) * self._fConst0) + np.float32(0.5))) 
		self._fConst43 = (self._fConst42 / self._fConst0) 
		self._fConst44 = (np.float32(3.4538777) * self._fConst43) 
		self._fConst45 = (np.float32(2.3025851) * self._fConst43) 
		self._fConst46 = np.floor(((np.float32(0.013458) * self._fConst0) + np.float32(0.5))) 
		self._iConst47 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst42 - self._fConst46)))) 
		self._iConst48 = np.int32(np.minimum(np.float32(1024.0), np.maximum(np.float32(0.0), (self._fConst46 + np.float32(-1.0))))) 
		self._fConst49 = np.floor(((np.float32(0.127837) * self._fConst0) + np.float32(0.5))) 
		self._fConst50 = (self._fConst49 / self._fConst0) 
		self._fConst51 = (np.float32(3.4538777) * self._fConst50) 
		self._fConst52 = (np.float32(2.3025851) * self._fConst50) 
		self._fConst53 = np.floor(((np.float32(0.031604) * self._fConst0) + np.float32(0.5))) 
		self._iConst54 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst49 - self._fConst53)))) 
		self._iConst55 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst53 + np.float32(-1.0))))) 
		self._fConst56 = np.floor(((np.float32(0.174713) * self._fConst0) + np.float32(0.5))) 
		self._fConst57 = (self._fConst56 / self._fConst0) 
		self._fConst58 = (np.float32(3.4538777) * self._fConst57) 
		self._fConst59 = (np.float32(2.3025851) * self._fConst57) 
		self._fConst60 = np.floor(((np.float32(0.022904) * self._fConst0) + np.float32(0.5))) 
		self._iConst61 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst56 - self._fConst60)))) 
		self._iConst62 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst60 + np.float32(-1.0))))) 
		self._fConst63 = np.floor(((np.float32(0.153129) * self._fConst0) + np.float32(0.5))) 
		self._fConst64 = (self._fConst63 / self._fConst0) 
		self._fConst65 = (np.float32(3.4538777) * self._fConst64) 
		self._fConst66 = (np.float32(2.3025851) * self._fConst64) 
		self._fConst67 = np.floor(((np.float32(0.020346) * self._fConst0) + np.float32(0.5))) 
		self._iConst68 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst63 - self._fConst67)))) 
		self._iConst69 = np.int32(np.minimum(np.float32(1024.0), np.maximum(np.float32(0.0), (self._fConst67 + np.float32(-1.0))))) 
		
	@property
	def num_inputs(self):
		return 0
	
	@property
	def num_outputs(self):
		return 2
	
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec20"] = np.float32(0)
		state["fRec21"] = np.float32(0)
		state["fRec22"] = np.float32(0)
		state["fRec23"] = np.float32(0)
		state["fRec24"] = np.float32(0)
		state["fRec25"] = np.float32(0)
		state["fRec26"] = np.float32(0)
		state["fRec27"] = np.float32(0)
		state["fRec28"] = np.float32(0)
		state["fRec29"] = np.float32(0)
		state["fRec30"] = np.float32(0)
		state["fRec31"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec33"] = np.float32(0)
		state["fRec34"] = np.float32(0)
		state["fRec35"] = np.float32(0)
		state["fRec36"] = np.float32(0)
		state["fRec37"] = np.float32(0)
		state["fRec38"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec40"] = np.float32(0)
		state["fRec41"] = np.float32(0)
		state["fRec42"] = np.float32(0)
		state["fRec43"] = np.float32(0)
		state["fRec44"] = np.float32(0)
		state["fRec45"] = np.float32(0)
		state["fRec46"] = np.float32(0)
		state["fRec47"] = np.float32(0)
		state["fRec48"] = np.float32(0)
		state["fRec49"] = np.float32(0)
		state["fRec50"] = np.float32(0)
		state["fRec51"] = np.float32(0)
		state["fRec52"] = np.float32(0)
		state["fRec53"] = np.float32(0)
		state["fRec54"] = np.float32(0)
		state["fRec55"] = np.float32(0)
		state["fRec56"] = np.float32(0)
		state["fRec57"] = np.float32(0)
		state["fRec58"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec62"] = np.float32(0)
		state["fRec64"] = np.float32(0)
		state["fRec65"] = np.float32(0)
		state["fRec66"] = np.float32(0)
		state["fRec68"] = np.float32(0)
		state["fRec69"] = np.float32(0)
		state["fRec70"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec73"] = np.float32(0)
		state["fRec74"] = np.float32(0)
		state["fRec76"] = np.float32(0)
		state["fRec77"] = np.float32(0)
		state["fRec78"] = np.float32(0)
		state["fRec80"] = np.float32(0)
		state["fRec81"] = np.float32(0)
		state["fRec82"] = np.float32(0)
		state["fRec84"] = np.float32(0)
		state["fRec85"] = np.float32(0)
		state["fRec9"] = np.float32(0)
		state["fVec1"] = np.float32(0)
		state["fVec2"] = np.float32(0)
		state["iRec16"] = np.int32(0)
		# Initialize array delays
		state["fVec0"] = np.zeros((32768,), dtype=np.float32)
		state["fVec3"] = np.zeros((4096,), dtype=np.float32)
		state["fVec4"] = np.zeros((2048,), dtype=np.float32)
		state["fVec5"] = np.zeros((32768,), dtype=np.float32)
		state["fVec6"] = np.zeros((4096,), dtype=np.float32)
		state["fVec7"] = np.zeros((16384,), dtype=np.float32)
		state["fVec8"] = np.zeros((4096,), dtype=np.float32)
		state["fVec9"] = np.zeros((32768,), dtype=np.float32)
		state["fVec10"] = np.zeros((4096,), dtype=np.float32)
		state["fVec11"] = np.zeros((16384,), dtype=np.float32)
		state["fVec12"] = np.zeros((2048,), dtype=np.float32)
		state["fVec13"] = np.zeros((16384,), dtype=np.float32)
		state["fVec14"] = np.zeros((4096,), dtype=np.float32)
		state["fVec15"] = np.zeros((16384,), dtype=np.float32)
		state["fVec16"] = np.zeros((4096,), dtype=np.float32)
		state["fVec17"] = np.zeros((16384,), dtype=np.float32)
		state["fVec18"] = np.zeros((2048,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize read-write tables
		state["ftbl1mydspSIG0"] = np.zeros((65537,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		# Pre-compute table initialization patterns
		# WARNING: Unknown initialization pattern for ftbl1mydspSIG0
		# Table will remain initialized to zeros
		# Process inline subcontainers for read-write table initialization
		iRec16 = np.zeros((4,), dtype=np.int32)
		
		pass 
		for i1 in range(0, 65537):
			iRec16_temp = state["iRec16"] 
			state["iRec16"] = (iRec16_temp + np.int32(1)) 
			state["ftbl1mydspSIG0"][i1] = np.sin((np.float32(9.58738e-05) * ((state["iRec16"] + np.int32(-1))))) 
		
		
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = (jnp.float32(0.001) * params["fHslider0"]) 
		fSlow1 = jnp.maximum(jnp.float32(0.05), jnp.minimum(jnp.float32(2.0), params["fHslider1"])) 
		fSlow2 = jnp.exp(-((self._fConst4 / fSlow1))) 
		fSlow3 = jnp.power(fSlow2, jnp.float32(2.0)) 
		fSlow4 = (jnp.float32(1.0) - (self._fConst1 * fSlow3)) 
		fSlow5 = (jnp.float32(1.0) - fSlow3) 
		fSlow6 = (fSlow4 / fSlow5) 
		fSlow7 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow4, jnp.float32(2.0)) / jnp.power(fSlow5, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow8 = (fSlow6 - fSlow7) 
		fSlow9 = (fSlow2 * (fSlow7 + (jnp.float32(1.0) - fSlow6))) 
		fSlow10 = ((jnp.exp(-((self._fConst5 / fSlow1))) / fSlow2) + jnp.float32(-1.0)) 
		fSlow11 = (jnp.float32(0.001) * params["fHslider2"]) 
		fSlow12 = params["fHslider3"] 
		iSlow13 = (fSlow12 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow14 = (iSlow13 > jnp.int32(0)).astype(jnp.int32) 
		fSlow15 = (iSlow13) 
		iSlow16 = ((fSlow12 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow17 = (self._fConst17 * fSlow12) 
		fSlow18 = (jnp.float32(0.001) * params["fHslider4"]) 
		fSlow19 = params["fHslider5"] 
		iSlow20 = (fSlow19 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow21 = (iSlow20 > jnp.int32(0)).astype(jnp.int32) 
		fSlow22 = (iSlow20) 
		iSlow23 = ((fSlow19 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow24 = (self._fConst17 * fSlow19) 
		fSlow25 = params["fHslider6"] 
		iSlow26 = (fSlow25 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow27 = (iSlow26 > jnp.int32(0)).astype(jnp.int32) 
		fSlow28 = (iSlow26) 
		iSlow29 = ((fSlow25 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow30 = (self._fConst17 * fSlow25) 
		fSlow31 = params["fHslider7"] 
		iSlow32 = (fSlow31 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow33 = (iSlow32 > jnp.int32(0)).astype(jnp.int32) 
		fSlow34 = (iSlow32) 
		iSlow35 = ((fSlow31 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow36 = (self._fConst17 * fSlow31) 
		fSlow37 = (jnp.float32(0.001) * jnp.power(params["fHslider8"], jnp.float32(2.0))) 
		fSlow38 = (jnp.float32(0.001) * params["fHslider9"]) 
		fSlow39 = jnp.exp(-((self._fConst23 / fSlow1))) 
		fSlow40 = jnp.power(fSlow39, jnp.float32(2.0)) 
		fSlow41 = (jnp.float32(1.0) - (self._fConst1 * fSlow40)) 
		fSlow42 = (jnp.float32(1.0) - fSlow40) 
		fSlow43 = (fSlow41 / fSlow42) 
		fSlow44 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow41, jnp.float32(2.0)) / jnp.power(fSlow42, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow45 = (fSlow43 - fSlow44) 
		fSlow46 = (fSlow39 * (fSlow44 + (jnp.float32(1.0) - fSlow43))) 
		fSlow47 = ((jnp.exp(-((self._fConst24 / fSlow1))) / fSlow39) + jnp.float32(-1.0)) 
		fSlow48 = jnp.exp(-((self._fConst30 / fSlow1))) 
		fSlow49 = jnp.power(fSlow48, jnp.float32(2.0)) 
		fSlow50 = (jnp.float32(1.0) - (self._fConst1 * fSlow49)) 
		fSlow51 = (jnp.float32(1.0) - fSlow49) 
		fSlow52 = (fSlow50 / fSlow51) 
		fSlow53 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow50, jnp.float32(2.0)) / jnp.power(fSlow51, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow54 = (fSlow52 - fSlow53) 
		fSlow55 = (fSlow48 * (fSlow53 + (jnp.float32(1.0) - fSlow52))) 
		fSlow56 = ((jnp.exp(-((self._fConst31 / fSlow1))) / fSlow48) + jnp.float32(-1.0)) 
		fSlow57 = jnp.exp(-((self._fConst37 / fSlow1))) 
		fSlow58 = jnp.power(fSlow57, jnp.float32(2.0)) 
		fSlow59 = (jnp.float32(1.0) - (self._fConst1 * fSlow58)) 
		fSlow60 = (jnp.float32(1.0) - fSlow58) 
		fSlow61 = (fSlow59 / fSlow60) 
		fSlow62 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow59, jnp.float32(2.0)) / jnp.power(fSlow60, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow63 = (fSlow61 - fSlow62) 
		fSlow64 = (fSlow57 * (fSlow62 + (jnp.float32(1.0) - fSlow61))) 
		fSlow65 = ((jnp.exp(-((self._fConst38 / fSlow1))) / fSlow57) + jnp.float32(-1.0)) 
		fSlow66 = jnp.exp(-((self._fConst44 / fSlow1))) 
		fSlow67 = jnp.power(fSlow66, jnp.float32(2.0)) 
		fSlow68 = (jnp.float32(1.0) - (self._fConst1 * fSlow67)) 
		fSlow69 = (jnp.float32(1.0) - fSlow67) 
		fSlow70 = (fSlow68 / fSlow69) 
		fSlow71 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow68, jnp.float32(2.0)) / jnp.power(fSlow69, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow72 = (fSlow70 - fSlow71) 
		fSlow73 = (fSlow66 * (fSlow71 + (jnp.float32(1.0) - fSlow70))) 
		fSlow74 = ((jnp.exp(-((self._fConst45 / fSlow1))) / fSlow66) + jnp.float32(-1.0)) 
		fSlow75 = jnp.exp(-((self._fConst51 / fSlow1))) 
		fSlow76 = jnp.power(fSlow75, jnp.float32(2.0)) 
		fSlow77 = (jnp.float32(1.0) - (self._fConst1 * fSlow76)) 
		fSlow78 = (jnp.float32(1.0) - fSlow76) 
		fSlow79 = (fSlow77 / fSlow78) 
		fSlow80 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow77, jnp.float32(2.0)) / jnp.power(fSlow78, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow81 = (fSlow79 - fSlow80) 
		fSlow82 = (fSlow75 * (fSlow80 + (jnp.float32(1.0) - fSlow79))) 
		fSlow83 = ((jnp.exp(-((self._fConst52 / fSlow1))) / fSlow75) + jnp.float32(-1.0)) 
		fSlow84 = jnp.exp(-((self._fConst58 / fSlow1))) 
		fSlow85 = jnp.power(fSlow84, jnp.float32(2.0)) 
		fSlow86 = (jnp.float32(1.0) - (self._fConst1 * fSlow85)) 
		fSlow87 = (jnp.float32(1.0) - fSlow85) 
		fSlow88 = (fSlow86 / fSlow87) 
		fSlow89 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow86, jnp.float32(2.0)) / jnp.power(fSlow87, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow90 = (fSlow88 - fSlow89) 
		fSlow91 = (fSlow84 * (fSlow89 + (jnp.float32(1.0) - fSlow88))) 
		fSlow92 = ((jnp.exp(-((self._fConst59 / fSlow1))) / fSlow84) + jnp.float32(-1.0)) 
		fSlow93 = jnp.exp(-((self._fConst65 / fSlow1))) 
		fSlow94 = jnp.power(fSlow93, jnp.float32(2.0)) 
		fSlow95 = (jnp.float32(1.0) - (self._fConst1 * fSlow94)) 
		fSlow96 = (jnp.float32(1.0) - fSlow94) 
		fSlow97 = (fSlow95 / fSlow96) 
		fSlow98 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow95, jnp.float32(2.0)) / jnp.power(fSlow96, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow99 = (fSlow97 - fSlow98) 
		fSlow100 = (fSlow93 * (fSlow98 + (jnp.float32(1.0) - fSlow97))) 
		fSlow101 = ((jnp.exp(-((self._fConst66 / fSlow1))) / fSlow93) + jnp.float32(-1.0)) 
		fRec0_temp = state["fRec0"] 
		fRec12_temp = state["fRec12"] 
		fRec11_temp = state["fRec11"] 
		fRec15_temp = state["fRec15"] 
		fRec18_temp = state["fRec18"] 
		fRec17_temp = state["fRec17"] 
		fRec19_temp = state["fRec19"] 
		fRec21_temp = state["fRec21"] 
		fRec20_temp = state["fRec20"] 
		fRec23_temp = state["fRec23"] 
		fRec22_temp = state["fRec22"] 
		fRec28_temp = state["fRec28"] 
		fRec27_temp = state["fRec27"] 
		fRec26_temp = state["fRec26"] 
		fRec25_temp = state["fRec25"] 
		fRec24_temp = state["fRec24"] 
		fRec14_temp = state["fRec14"] 
		fRec29_temp = state["fRec29"] 
		fRec31_temp = state["fRec31"] 
		fRec30_temp = state["fRec30"] 
		fRec37_temp = state["fRec37"] 
		fRec36_temp = state["fRec36"] 
		fRec35_temp = state["fRec35"] 
		fRec34_temp = state["fRec34"] 
		fRec33_temp = state["fRec33"] 
		fRec32_temp = state["fRec32"] 
		fVec1_temp = state["fVec1"] 
		fRec38_temp = state["fRec38"] 
		fRec40_temp = state["fRec40"] 
		fRec39_temp = state["fRec39"] 
		fRec46_temp = state["fRec46"] 
		fRec45_temp = state["fRec45"] 
		fRec44_temp = state["fRec44"] 
		fRec43_temp = state["fRec43"] 
		fRec42_temp = state["fRec42"] 
		fRec41_temp = state["fRec41"] 
		fRec47_temp = state["fRec47"] 
		fRec49_temp = state["fRec49"] 
		fRec48_temp = state["fRec48"] 
		fRec55_temp = state["fRec55"] 
		fRec54_temp = state["fRec54"] 
		fRec53_temp = state["fRec53"] 
		fRec52_temp = state["fRec52"] 
		fRec51_temp = state["fRec51"] 
		fRec50_temp = state["fRec50"] 
		fVec2_temp = state["fVec2"] 
		fRec13_temp = state["fRec13"] 
		fRec56_temp = state["fRec56"] 
		fRec57_temp = state["fRec57"] 
		fRec9_temp = state["fRec9"] 
		fRec61_temp = state["fRec61"] 
		fRec60_temp = state["fRec60"] 
		fRec58_temp = state["fRec58"] 
		fRec65_temp = state["fRec65"] 
		fRec64_temp = state["fRec64"] 
		fRec62_temp = state["fRec62"] 
		fRec69_temp = state["fRec69"] 
		fRec68_temp = state["fRec68"] 
		fRec66_temp = state["fRec66"] 
		fRec73_temp = state["fRec73"] 
		fRec72_temp = state["fRec72"] 
		fRec70_temp = state["fRec70"] 
		fRec77_temp = state["fRec77"] 
		fRec76_temp = state["fRec76"] 
		fRec74_temp = state["fRec74"] 
		fRec81_temp = state["fRec81"] 
		fRec80_temp = state["fRec80"] 
		fRec78_temp = state["fRec78"] 
		fRec85_temp = state["fRec85"] 
		fRec84_temp = state["fRec84"] 
		fRec82_temp = state["fRec82"] 
		state["fRec0"] = (fSlow0 + (jnp.float32(0.999) * fRec0_temp)) 
		fTemp0 = jnp.maximum(jnp.float32(0.75), jnp.minimum(jnp.float32(4.0), state["fRec0"])) 
		state["fRec12"] = -((self._fConst7 * ((self._fConst8 * fRec12_temp) - (state["fRec8"][1] + state["fRec8"][2])))) 
		state["fRec11"] = ((fSlow8 * fRec11_temp) + (fSlow9 * (state["fRec8"][1] + (fSlow10 * state["fRec12"])))) 
		state["fVec0"] = state["fVec0"].at[(state["IOTA0"] & 32767).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec11"]) + jnp.float32(1e-20))) 
		fTemp1 = ((jnp.float32(0.6) * fRec9_temp) + state["fVec0"][((state["IOTA0"] - self._iConst10) & 32767).astype(jnp.int32)]) 
		state["fRec15"] = ((jnp.float32(0.999) * fRec15_temp) + jnp.float32(0.0008)) 
		state["fRec18"] = (fSlow11 + (jnp.float32(0.999) * fRec18_temp)) 
		fTemp2 = (fRec17_temp + (self._fConst14 * state["fRec18"])) 
		state["fRec17"] = (fTemp2 - jnp.floor(fTemp2)) 
		state["fRec19"] = jnp.where((iSlow14 != 0), jnp.float32(0.0), jnp.minimum(self._fConst15, (fRec19_temp + jnp.float32(1.0)))) 
		state["fRec21"] = jnp.where((iSlow16 != 0), jnp.float32(0.0), jnp.minimum(self._fConst16, (fRec21_temp + jnp.float32(1.0)))) 
		state["fRec20"] = jnp.where((iSlow13 != 0), (fSlow15 * jnp.where(((state["fRec21"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec21"] < self._fConst16).astype(jnp.int32) != 0), (fSlow17 * state["fRec21"]), fSlow12))), fRec20_temp) 
		fTemp3 = (jnp.float32(3.1415927) * ((state["fRec15"] * self._ftbl0mydspSIG0[jnp.maximum(0, jnp.minimum(jnp.int32((jnp.float32(65536.0) * state["fRec17"])), 65535))]) * jnp.where(((state["fRec19"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec20"], jnp.where(((state["fRec19"] < self._fConst15).astype(jnp.int32) != 0), (state["fRec20"] * (jnp.float32(1.0) - (self._fConst18 * state["fRec19"]))), jnp.float32(0.0))))) 
		fTemp4 = jnp.sin(fTemp3) 
		state["fRec23"] = (fSlow18 + (jnp.float32(0.999) * fRec23_temp)) 
		fTemp5 = (fRec22_temp + (self._fConst14 * state["fRec23"])) 
		state["fRec22"] = (fTemp5 - jnp.floor(fTemp5)) 
		fTemp6 = (jnp.float32(65536.0) * state["fRec22"]) 
		iTemp7 = jnp.int32(fTemp6) 
		fTemp8 = state["ftbl1mydspSIG0"][jnp.maximum(0, jnp.minimum(iTemp7, 65536))] 
		fTemp9 = (fTemp8 + ((fTemp6 - jnp.floor(fTemp6)) * (state["ftbl1mydspSIG0"][jnp.maximum(0, jnp.minimum((iTemp7 + 1), 65536))] - fTemp8))) 
		fTemp10 = jnp.cos(fTemp3) 
		fTemp11 = ((fTemp9 * fTemp10) - (fTemp4 * fRec14_temp)) 
		fTemp12 = ((fTemp10 * fTemp11) - (fTemp4 * fRec24_temp)) 
		fTemp13 = ((fTemp10 * fTemp12) - (fTemp4 * fRec25_temp)) 
		fTemp14 = ((fTemp10 * fTemp13) - (fTemp4 * fRec26_temp)) 
		fTemp15 = ((fTemp10 * fTemp14) - (fTemp4 * fRec27_temp)) 
		state["fRec28"] = ((fTemp10 * fTemp15) - (fTemp4 * fRec28_temp)) 
		state["fRec27"] = ((fTemp4 * fTemp15) + (fTemp10 * fRec28_temp)) 
		state["fRec26"] = ((fTemp4 * fTemp14) + (fTemp10 * fRec27_temp)) 
		state["fRec25"] = ((fTemp4 * fTemp13) + (fTemp10 * fRec26_temp)) 
		state["fRec24"] = ((fTemp4 * fTemp12) + (fTemp10 * fRec25_temp)) 
		state["fRec14"] = ((fTemp4 * fTemp11) + (fTemp10 * fRec24_temp)) 
		state["fRec29"] = jnp.where((iSlow21 != 0), jnp.float32(0.0), jnp.minimum(self._fConst15, (fRec29_temp + jnp.float32(1.0)))) 
		state["fRec31"] = jnp.where((iSlow23 != 0), jnp.float32(0.0), jnp.minimum(self._fConst16, (fRec31_temp + jnp.float32(1.0)))) 
		state["fRec30"] = jnp.where((iSlow20 != 0), (fSlow22 * jnp.where(((state["fRec31"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec31"] < self._fConst16).astype(jnp.int32) != 0), (fSlow24 * state["fRec31"]), fSlow19))), fRec30_temp) 
		fTemp16 = (jnp.float32(3.1415927) * ((state["fRec15"] * fTemp9) * jnp.where(((state["fRec29"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec30"], jnp.where(((state["fRec29"] < self._fConst15).astype(jnp.int32) != 0), (state["fRec30"] * (jnp.float32(1.0) - (self._fConst18 * state["fRec29"]))), jnp.float32(0.0))))) 
		fTemp17 = jnp.sin(fTemp16) 
		fTemp18 = jnp.cos(fTemp16) 
		fTemp19 = ((fTemp9 * fTemp18) - (fTemp17 * fRec32_temp)) 
		fTemp20 = ((fTemp18 * fTemp19) - (fTemp17 * fRec33_temp)) 
		fTemp21 = ((fTemp18 * fTemp20) - (fTemp17 * fRec34_temp)) 
		fTemp22 = ((fTemp18 * fTemp21) - (fTemp17 * fRec35_temp)) 
		fTemp23 = ((fTemp18 * fTemp22) - (fTemp17 * fRec36_temp)) 
		state["fRec37"] = ((fTemp18 * fTemp23) - (fTemp17 * fRec37_temp)) 
		state["fRec36"] = ((fTemp17 * fTemp23) + (fTemp18 * fRec37_temp)) 
		state["fRec35"] = ((fTemp17 * fTemp22) + (fTemp18 * fRec36_temp)) 
		state["fRec34"] = ((fTemp17 * fTemp21) + (fTemp18 * fRec35_temp)) 
		state["fRec33"] = ((fTemp17 * fTemp20) + (fTemp18 * fRec34_temp)) 
		state["fRec32"] = ((fTemp17 * fTemp19) + (fTemp18 * fRec33_temp)) 
		fTemp24 = (jnp.float32(1.0) - state["fRec15"]) 
		fTemp25 = ((state["fRec15"] * ((fTemp9 * fTemp17) + (fRec32_temp * fTemp18))) + (fTemp24 * fTemp9)) 
		state["fVec1"] = jnp.float32(fTemp25) 
		state["fRec38"] = jnp.where((iSlow27 != 0), jnp.float32(0.0), jnp.minimum(self._fConst15, (fRec38_temp + jnp.float32(1.0)))) 
		state["fRec40"] = jnp.where((iSlow29 != 0), jnp.float32(0.0), jnp.minimum(self._fConst16, (fRec40_temp + jnp.float32(1.0)))) 
		state["fRec39"] = jnp.where((iSlow26 != 0), (fSlow28 * jnp.where(((state["fRec40"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec40"] < self._fConst16).astype(jnp.int32) != 0), (fSlow30 * state["fRec40"]), fSlow25))), fRec39_temp) 
		fTemp26 = (jnp.float32(1.5707964) * ((state["fRec15"] * jnp.where(((state["fRec38"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec39"], jnp.where(((state["fRec38"] < self._fConst15).astype(jnp.int32) != 0), (state["fRec39"] * (jnp.float32(1.0) - (self._fConst18 * state["fRec38"]))), jnp.float32(0.0)))) * (fTemp25 + fVec1_temp))) 
		fTemp27 = jnp.sin(fTemp26) 
		fTemp28 = jnp.cos(fTemp26) 
		fTemp29 = ((fTemp25 * fTemp28) - (fTemp27 * fRec41_temp)) 
		fTemp30 = ((fTemp28 * fTemp29) - (fTemp27 * fRec42_temp)) 
		fTemp31 = ((fTemp28 * fTemp30) - (fTemp27 * fRec43_temp)) 
		fTemp32 = ((fTemp28 * fTemp31) - (fTemp27 * fRec44_temp)) 
		fTemp33 = ((fTemp28 * fTemp32) - (fTemp27 * fRec45_temp)) 
		state["fRec46"] = ((fTemp28 * fTemp33) - (fTemp27 * fRec46_temp)) 
		state["fRec45"] = ((fTemp27 * fTemp33) + (fTemp28 * fRec46_temp)) 
		state["fRec44"] = ((fTemp27 * fTemp32) + (fTemp28 * fRec45_temp)) 
		state["fRec43"] = ((fTemp27 * fTemp31) + (fTemp28 * fRec44_temp)) 
		state["fRec42"] = ((fTemp27 * fTemp30) + (fTemp28 * fRec43_temp)) 
		state["fRec41"] = ((fTemp27 * fTemp29) + (fTemp28 * fRec42_temp)) 
		fTemp34 = ((state["fRec15"] * ((fTemp25 * fTemp27) + (fRec41_temp * fTemp28))) + (fTemp24 * fTemp25)) 
		state["fRec47"] = jnp.where((iSlow33 != 0), jnp.float32(0.0), jnp.minimum(self._fConst15, (fRec47_temp + jnp.float32(1.0)))) 
		state["fRec49"] = jnp.where((iSlow35 != 0), jnp.float32(0.0), jnp.minimum(self._fConst16, (fRec49_temp + jnp.float32(1.0)))) 
		state["fRec48"] = jnp.where((iSlow32 != 0), (fSlow34 * jnp.where(((state["fRec49"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec49"] < self._fConst16).astype(jnp.int32) != 0), (fSlow36 * state["fRec49"]), fSlow31))), fRec48_temp) 
		fTemp35 = (jnp.float32(3.1415927) * ((state["fRec15"] * jnp.power(fTemp34, jnp.float32(2.0))) * jnp.where(((state["fRec47"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec48"], jnp.where(((state["fRec47"] < self._fConst15).astype(jnp.int32) != 0), (state["fRec48"] * (jnp.float32(1.0) - (self._fConst18 * state["fRec47"]))), jnp.float32(0.0))))) 
		fTemp36 = jnp.sin(fTemp35) 
		fTemp37 = jnp.cos(fTemp35) 
		fTemp38 = ((fTemp34 * fTemp37) - (fTemp36 * fRec50_temp)) 
		fTemp39 = ((fTemp37 * fTemp38) - (fTemp36 * fRec51_temp)) 
		fTemp40 = ((fTemp37 * fTemp39) - (fTemp36 * fRec52_temp)) 
		fTemp41 = ((fTemp37 * fTemp40) - (fTemp36 * fRec53_temp)) 
		fTemp42 = ((fTemp37 * fTemp41) - (fTemp36 * fRec54_temp)) 
		state["fRec55"] = ((fTemp37 * fTemp42) - (fTemp36 * fRec55_temp)) 
		state["fRec54"] = ((fTemp36 * fTemp42) + (fTemp37 * fRec55_temp)) 
		state["fRec53"] = ((fTemp36 * fTemp41) + (fTemp37 * fRec54_temp)) 
		state["fRec52"] = ((fTemp36 * fTemp40) + (fTemp37 * fRec53_temp)) 
		state["fRec51"] = ((fTemp36 * fTemp39) + (fTemp37 * fRec52_temp)) 
		state["fRec50"] = ((fTemp36 * fTemp38) + (fTemp37 * fRec51_temp)) 
		fTemp43 = ((fRec14_temp * fTemp10) + (((state["fRec15"] * ((fTemp34 * fTemp36) + (fRec50_temp * fTemp37))) + (fTemp24 * fTemp34)) + (fTemp9 * fTemp4))) 
		state["fVec2"] = jnp.float32(fTemp43) 
		state["fRec13"] = -((self._fConst12 * ((self._fConst13 * fRec13_temp) - (fTemp43 + fVec2_temp)))) 
		state["fRec56"] = (fSlow37 + (jnp.float32(0.999) * fRec56_temp)) 
		fTemp44 = (state["fRec13"] * state["fRec56"]) 
		state["fRec57"] = (fSlow38 + (jnp.float32(0.999) * fRec57_temp)) 
		fTemp45 = jnp.maximum(jnp.float32(0.05), jnp.minimum(jnp.float32(1.0), state["fRec57"])) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp44 * fTemp45)) 
		fTemp46 = (jnp.float32(0.18) * state["fVec3"][((state["IOTA0"] - self._iConst19) & 4095).astype(jnp.int32)]) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set((fTemp1 - fTemp46)) 
		state["fRec9"] = state["fVec4"][((state["IOTA0"] - self._iConst20) & 2047).astype(jnp.int32)] 
		fRec10 = (jnp.float32(0.6) * (fTemp46 - fTemp1)) 
		state["fRec61"] = -((self._fConst7 * ((self._fConst8 * fRec61_temp) - (state["fRec4"][1] + state["fRec4"][2])))) 
		state["fRec60"] = ((fSlow45 * fRec60_temp) + (fSlow46 * (state["fRec4"][1] + (fSlow47 * state["fRec61"])))) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 32767).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec60"]) + jnp.float32(1e-20))) 
		fTemp47 = ((jnp.float32(0.6) * fRec58_temp) + state["fVec5"][((state["IOTA0"] - self._iConst26) & 32767).astype(jnp.int32)]) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp47 - fTemp46)) 
		state["fRec58"] = state["fVec6"][((state["IOTA0"] - self._iConst27) & 4095).astype(jnp.int32)] 
		fRec59 = (jnp.float32(0.6) * (fTemp46 - fTemp47)) 
		state["fRec65"] = -((self._fConst7 * ((self._fConst8 * fRec65_temp) - (state["fRec6"][1] + state["fRec6"][2])))) 
		state["fRec64"] = ((fSlow54 * fRec64_temp) + (fSlow55 * (state["fRec6"][1] + (fSlow56 * state["fRec65"])))) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec64"]) + jnp.float32(1e-20))) 
		fTemp48 = (state["fVec7"][((state["IOTA0"] - self._iConst33) & 16383).astype(jnp.int32)] + (fTemp46 + (jnp.float32(0.6) * fRec62_temp))) 
		state["fVec8"] = state["fVec8"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp48) 
		state["fRec62"] = state["fVec8"][((state["IOTA0"] - self._iConst34) & 4095).astype(jnp.int32)] 
		fRec63 = -((jnp.float32(0.6) * fTemp48)) 
		state["fRec69"] = -((self._fConst7 * ((self._fConst8 * fRec69_temp) - (state["fRec2"][1] + state["fRec2"][2])))) 
		state["fRec68"] = ((fSlow63 * fRec68_temp) + (fSlow64 * (state["fRec2"][1] + (fSlow65 * state["fRec69"])))) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 32767).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec68"]) + jnp.float32(1e-20))) 
		fTemp49 = (state["fVec9"][((state["IOTA0"] - self._iConst40) & 32767).astype(jnp.int32)] + (fTemp46 + (jnp.float32(0.6) * fRec66_temp))) 
		state["fVec10"] = state["fVec10"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp49) 
		state["fRec66"] = state["fVec10"][((state["IOTA0"] - self._iConst41) & 4095).astype(jnp.int32)] 
		fRec67 = -((jnp.float32(0.6) * fTemp49)) 
		state["fRec73"] = -((self._fConst7 * ((self._fConst8 * fRec73_temp) - (state["fRec7"][1] + state["fRec7"][2])))) 
		state["fRec72"] = ((fSlow72 * fRec72_temp) + (fSlow73 * (state["fRec7"][1] + (fSlow74 * state["fRec73"])))) 
		state["fVec11"] = state["fVec11"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec72"]) + jnp.float32(1e-20))) 
		fTemp50 = (state["fVec11"][((state["IOTA0"] - self._iConst47) & 16383).astype(jnp.int32)] - (fTemp46 + (jnp.float32(0.6) * fRec70_temp))) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp50) 
		state["fRec70"] = state["fVec12"][((state["IOTA0"] - self._iConst48) & 2047).astype(jnp.int32)] 
		fRec71 = (jnp.float32(0.6) * fTemp50) 
		state["fRec77"] = -((self._fConst7 * ((self._fConst8 * fRec77_temp) - (state["fRec3"][1] + state["fRec3"][2])))) 
		state["fRec76"] = ((fSlow81 * fRec76_temp) + (fSlow82 * (state["fRec3"][1] + (fSlow83 * state["fRec77"])))) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec76"]) + jnp.float32(1e-20))) 
		fTemp51 = (state["fVec13"][((state["IOTA0"] - self._iConst54) & 16383).astype(jnp.int32)] - (fTemp46 + (jnp.float32(0.6) * fRec74_temp))) 
		state["fVec14"] = state["fVec14"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp51) 
		state["fRec74"] = state["fVec14"][((state["IOTA0"] - self._iConst55) & 4095).astype(jnp.int32)] 
		fRec75 = (jnp.float32(0.6) * fTemp51) 
		state["fRec81"] = -((self._fConst7 * ((self._fConst8 * fRec81_temp) - (state["fRec5"][1] + state["fRec5"][2])))) 
		state["fRec80"] = ((fSlow90 * fRec80_temp) + (fSlow91 * (state["fRec5"][1] + (fSlow92 * state["fRec81"])))) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec80"]) + jnp.float32(1e-20))) 
		fTemp52 = ((fTemp46 + state["fVec15"][((state["IOTA0"] - self._iConst61) & 16383).astype(jnp.int32)]) - (jnp.float32(0.6) * fRec78_temp)) 
		state["fVec16"] = state["fVec16"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp52) 
		state["fRec78"] = state["fVec16"][((state["IOTA0"] - self._iConst62) & 4095).astype(jnp.int32)] 
		fRec79 = (jnp.float32(0.6) * fTemp52) 
		state["fRec85"] = -((self._fConst7 * ((self._fConst8 * fRec85_temp) - (state["fRec1"][1] + state["fRec1"][2])))) 
		state["fRec84"] = ((fSlow99 * fRec84_temp) + (fSlow100 * (state["fRec1"][1] + (fSlow101 * state["fRec85"])))) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec84"]) + jnp.float32(1e-20))) 
		fTemp53 = ((state["fVec17"][((state["IOTA0"] - self._iConst68) & 16383).astype(jnp.int32)] + fTemp46) - (jnp.float32(0.6) * fRec82_temp)) 
		state["fVec18"] = state["fVec18"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp53) 
		state["fRec82"] = state["fVec18"][((state["IOTA0"] - self._iConst69) & 2047).astype(jnp.int32)] 
		fRec83 = (jnp.float32(0.6) * fTemp53) 
		fTemp54 = (fRec83 + fRec79) 
		fTemp55 = (fRec71 + (fRec75 + fTemp54)) 
		state["fRec1"] = state["fRec1"].at[0].set((fRec9_temp + (fRec58_temp + (fRec62_temp + (fRec66_temp + (fRec70_temp + (fRec74_temp + (fRec78_temp + (fRec82_temp + (fRec10 + (fRec59 + (fRec63 + (fRec67 + fTemp55))))))))))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((fRec70_temp + (fRec74_temp + (fRec78_temp + (fRec82_temp + fTemp55)))) - (fRec9_temp + (fRec58_temp + (fRec62_temp + (fRec66_temp + (fRec10 + (fRec59 + (fRec67 + fRec63))))))))) 
		fTemp56 = (fRec75 + fRec71) 
		state["fRec3"] = state["fRec3"].at[0].set(((fRec62_temp + (fRec66_temp + (fRec78_temp + (fRec82_temp + (fRec63 + (fRec67 + fTemp54)))))) - (fRec9_temp + (fRec58_temp + (fRec70_temp + (fRec74_temp + (fRec10 + (fRec59 + fTemp56)))))))) 
		state["fRec4"] = state["fRec4"].at[0].set(((fRec9_temp + (fRec58_temp + (fRec78_temp + (fRec82_temp + (fRec10 + (fRec59 + fTemp54)))))) - (fRec62_temp + (fRec66_temp + (fRec70_temp + (fRec74_temp + (fRec63 + (fRec67 + fTemp56)))))))) 
		fTemp57 = (fRec83 + fRec75) 
		fTemp58 = (fRec79 + fRec71) 
		state["fRec5"] = state["fRec5"].at[0].set(((fRec58_temp + (fRec66_temp + (fRec74_temp + (fRec82_temp + (fRec59 + (fRec67 + fTemp57)))))) - (fRec9_temp + (fRec62_temp + (fRec70_temp + (fRec78_temp + (fRec10 + (fRec63 + fTemp58)))))))) 
		state["fRec6"] = state["fRec6"].at[0].set(((fRec9_temp + (fRec62_temp + (fRec74_temp + (fRec82_temp + (fRec10 + (fRec63 + fTemp57)))))) - (fRec58_temp + (fRec66_temp + (fRec70_temp + (fRec78_temp + (fRec59 + (fRec67 + fTemp58)))))))) 
		fTemp59 = (fRec83 + fRec71) 
		fTemp60 = (fRec79 + fRec75) 
		state["fRec7"] = state["fRec7"].at[0].set(((fRec9_temp + (fRec66_temp + (fRec70_temp + (fRec82_temp + (fRec10 + (fRec67 + fTemp59)))))) - (fRec58_temp + (fRec62_temp + (fRec74_temp + (fRec78_temp + (fRec59 + (fRec63 + fTemp60)))))))) 
		state["fRec8"] = state["fRec8"].at[0].set(((fRec58_temp + (fRec62_temp + (fRec70_temp + (fRec82_temp + (fRec59 + (fRec63 + fTemp59)))))) - (fRec9_temp + (fRec66_temp + (fRec74_temp + (fRec78_temp + (fRec10 + (fRec67 + fTemp60)))))))) 
		fTemp61 = (jnp.float32(0.6) * (fTemp44 * (jnp.float32(1.0) - fTemp45))) 
		_result0 = (fTemp0 * ((jnp.float32(0.37) * (state["fRec2"][0] + state["fRec3"][0])) + fTemp61)) 
		_result1 = (fTemp0 * (fTemp61 + (jnp.float32(0.37) * (state["fRec2"][0] - state["fRec3"][0])))) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
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
