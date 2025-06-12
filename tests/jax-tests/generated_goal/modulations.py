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
import re
from typing import Any, Dict, List, Tuple
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

try:
	import librosa
except ImportError:
	print("Warning: librosa not installed. Soundfile loading will return dummy data.")
	print("Install with: pip install librosa")
	librosa = None

# Generated code
"""
Code generated with Faust version 2.80.7
Compilation options: -a ../../architecture/jax/minimal.py -lang jax -it -ct 1 -es 1 -mcd 16 -mdd 1024 -mdy 33 -single -ftz 0 
"""

# enable single precision
FAUSTFLOAT = jnp.float32
FAUSTINT = jnp.int32

def remainder(x, y):
	"""C++ std::remainder implemented with jax numpy"""
	quo = jnp.round(x/y)
	return x - quo * y
	
class mydsp(nn.Module):
	
	sample_rate: int
	soundfile_dirs: list[str] = dataclasses.field(default_factory=list)
	
	@property
	def num_inputs(self):
		return 0
	
	@property
	def num_outputs(self):
		return 2
	
	# fmt: off
	def setup(self):
		# Initialize static tables
		# Table used in inline subcontainer but not declared globally
		ftbl0mydspSIG0 = np.zeros((65537,), dtype=np.float32)
		# Initialize waveform data
		# Process inline subcontainers for static table initialization
		# iRec6

		iRec6 = np.int32(0)
		for i1 in range(0, 65536):
			iRec6_temp = iRec6 
			iRec6 = (iRec6_temp + np.int32(1)) 
			ftbl0mydspSIG0[i1] = np.sin((np.float32(9.58738e-05) * ((iRec6 + np.int32(-1))))) 
		
		
		# Skipping loop that fills read-write table - handled in _initialize_carry
		
		# Convert static tables and waveform data to JAX arrays
		self._ftbl0mydspSIG0 = jnp.array(ftbl0mydspSIG0)
		# Initialize UI parameters
		unnorm_funcs = {}
		ui_path = []
		ui_path.append("Modulations") 
		ui_path.append("Instrument") 
		self.add_hslider("fHslider4", ui_path, "Frequency", 3.3e+02, 1e+02, 1.2e+03, unnorm_funcs, "linear") 
		self.add_hslider("fHslider9", ui_path, "General Volume", 1.0, 0.75, 4.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider1", ui_path, "Oscillator Volume", 0.5, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider3", ui_path, "Modulating Frequency", 1.2e+03, 9e+02, 1.7e+03, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Modulations") 
		self.add_hslider("fHslider5", ui_path, "Play Modulation 0 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider6", ui_path, "Play Modulation 1 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider7", ui_path, "Play Modulation 2 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "Play Modulation 3 (ASR Envelope)", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Reverb") 
		self.add_hslider("fHslider8", ui_path, "Reverberation Room Size(InstrReverb)", 0.5, 0.05, 2.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider0", ui_path, "Reverberation Volume(InstrReverb)", 0.25, 0.05, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = (np.float32(2.0) * self._fConst0) 
		self._fConst2 = (np.float32(0.5) / self._fConst0) 
		self._fConst3 = (np.float32(3.0) * self._fConst0) 
		self._fConst4 = (np.float32(0.33333334) / self._fConst0) 
		self._fConst5 = (np.float32(1.0) / self._fConst0) 
		self._fConst6 = (np.float32(1.0) / np.tan((np.float32(6283.1855) / self._fConst0))) 
		self._fConst7 = (np.float32(1.0) - self._fConst6) 
		self._fConst8 = (np.float32(1.0) / (self._fConst6 + np.float32(1.0))) 
		self._fConst9 = (np.float32(1.0) / np.tan((np.float32(628.31854) / self._fConst0))) 
		self._fConst10 = (np.float32(1.0) - self._fConst9) 
		self._fConst11 = (np.float32(1.0) / (self._fConst9 + np.float32(1.0))) 
		self._fConst12 = np.floor(((np.float32(0.174713) * self._fConst0) + np.float32(0.5))) 
		self._fConst13 = (self._fConst12 / self._fConst0) 
		self._fConst14 = (np.float32(3.4538777) * self._fConst13) 
		self._fConst15 = (np.float32(2.3025851) * self._fConst13) 
		self._fConst16 = np.cos((np.float32(37699.113) / self._fConst0)) 
		self._fConst17 = np.floor(((np.float32(0.022904) * self._fConst0) + np.float32(0.5))) 
		self._iConst18 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst12 - self._fConst17)))) 
		self._iConst19 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (np.float32(0.02) * self._fConst0)))) 
		self._iConst20 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst17 + np.float32(-1.0))))) 
		self._fConst21 = np.floor(((np.float32(0.153129) * self._fConst0) + np.float32(0.5))) 
		self._fConst22 = (self._fConst21 / self._fConst0) 
		self._fConst23 = (np.float32(3.4538777) * self._fConst22) 
		self._fConst24 = (np.float32(2.3025851) * self._fConst22) 
		self._fConst25 = np.floor(((np.float32(0.020346) * self._fConst0) + np.float32(0.5))) 
		self._iConst26 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst21 - self._fConst25)))) 
		self._iConst27 = np.int32(np.minimum(np.float32(1024.0), np.maximum(np.float32(0.0), (self._fConst25 + np.float32(-1.0))))) 
		self._fConst28 = np.floor(((np.float32(0.127837) * self._fConst0) + np.float32(0.5))) 
		self._fConst29 = (self._fConst28 / self._fConst0) 
		self._fConst30 = (np.float32(3.4538777) * self._fConst29) 
		self._fConst31 = (np.float32(2.3025851) * self._fConst29) 
		self._fConst32 = np.floor(((np.float32(0.031604) * self._fConst0) + np.float32(0.5))) 
		self._iConst33 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst28 - self._fConst32)))) 
		self._iConst34 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst32 + np.float32(-1.0))))) 
		self._fConst35 = np.floor(((np.float32(0.125) * self._fConst0) + np.float32(0.5))) 
		self._fConst36 = (self._fConst35 / self._fConst0) 
		self._fConst37 = (np.float32(3.4538777) * self._fConst36) 
		self._fConst38 = (np.float32(2.3025851) * self._fConst36) 
		self._fConst39 = np.floor(((np.float32(0.013458) * self._fConst0) + np.float32(0.5))) 
		self._iConst40 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst35 - self._fConst39)))) 
		self._iConst41 = np.int32(np.minimum(np.float32(1024.0), np.maximum(np.float32(0.0), (self._fConst39 + np.float32(-1.0))))) 
		self._fConst42 = np.floor(((np.float32(0.210389) * self._fConst0) + np.float32(0.5))) 
		self._fConst43 = (self._fConst42 / self._fConst0) 
		self._fConst44 = (np.float32(3.4538777) * self._fConst43) 
		self._fConst45 = (np.float32(2.3025851) * self._fConst43) 
		self._fConst46 = np.floor(((np.float32(0.024421) * self._fConst0) + np.float32(0.5))) 
		self._iConst47 = np.int32(np.minimum(np.float32(16384.0), np.maximum(np.float32(0.0), (self._fConst42 - self._fConst46)))) 
		self._iConst48 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst46 + np.float32(-1.0))))) 
		self._fConst49 = np.floor(((np.float32(0.192303) * self._fConst0) + np.float32(0.5))) 
		self._fConst50 = (self._fConst49 / self._fConst0) 
		self._fConst51 = (np.float32(3.4538777) * self._fConst50) 
		self._fConst52 = (np.float32(2.3025851) * self._fConst50) 
		self._fConst53 = np.floor(((np.float32(0.029291) * self._fConst0) + np.float32(0.5))) 
		self._iConst54 = np.int32(np.minimum(np.float32(8192.0), np.maximum(np.float32(0.0), (self._fConst49 - self._fConst53)))) 
		self._iConst55 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst53 + np.float32(-1.0))))) 
		self._fConst56 = np.floor(((np.float32(0.256891) * self._fConst0) + np.float32(0.5))) 
		self._fConst57 = (self._fConst56 / self._fConst0) 
		self._fConst58 = (np.float32(3.4538777) * self._fConst57) 
		self._fConst59 = (np.float32(2.3025851) * self._fConst57) 
		self._fConst60 = np.floor(((np.float32(0.027333) * self._fConst0) + np.float32(0.5))) 
		self._iConst61 = np.int32(np.minimum(np.float32(16384.0), np.maximum(np.float32(0.0), (self._fConst56 - self._fConst60)))) 
		self._iConst62 = np.int32(np.minimum(np.float32(2048.0), np.maximum(np.float32(0.0), (self._fConst60 + np.float32(-1.0))))) 
		self._fConst63 = np.floor(((np.float32(0.219991) * self._fConst0) + np.float32(0.5))) 
		self._fConst64 = (self._fConst63 / self._fConst0) 
		self._fConst65 = (np.float32(3.4538777) * self._fConst64) 
		self._fConst66 = (np.float32(2.3025851) * self._fConst64) 
		self._fConst67 = np.floor(((np.float32(0.019123) * self._fConst0) + np.float32(0.5))) 
		self._iConst68 = np.int32(np.minimum(np.float32(16384.0), np.maximum(np.float32(0.0), (self._fConst63 - self._fConst67)))) 
		self._iConst69 = np.int32(np.minimum(np.float32(1024.0), np.maximum(np.float32(0.0), (self._fConst67 + np.float32(-1.0))))) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec1"] = np.float32(0)
		state["fRec10"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec16"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec2"] = np.float32(0)
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
		state["fRec3"] = np.float32(0)
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
		state["fRec4"] = np.float32(0)
		state["fRec40"] = np.float32(0)
		state["fRec41"] = np.float32(0)
		state["fRec42"] = np.float32(0)
		state["fRec43"] = np.float32(0)
		state["fRec44"] = np.float32(0)
		state["fRec5"] = np.float32(0)
		state["fRec53"] = np.float32(0)
		state["fRec55"] = np.float32(0)
		state["fRec56"] = np.float32(0)
		state["fRec57"] = np.float32(0)
		state["fRec59"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec63"] = np.float32(0)
		state["fRec64"] = np.float32(0)
		state["fRec65"] = np.float32(0)
		state["fRec67"] = np.float32(0)
		state["fRec68"] = np.float32(0)
		state["fRec69"] = np.float32(0)
		state["fRec7"] = np.float32(0)
		state["fRec71"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec73"] = np.float32(0)
		state["fRec75"] = np.float32(0)
		state["fRec76"] = np.float32(0)
		state["fRec77"] = np.float32(0)
		state["fRec79"] = np.float32(0)
		state["fRec8"] = np.float32(0)
		state["fRec80"] = np.float32(0)
		state["fRec81"] = np.float32(0)
		state["fRec83"] = np.float32(0)
		state["fRec84"] = np.float32(0)
		state["fRec85"] = np.float32(0)
		state["fRec9"] = np.float32(0)
		state["fVec0"] = np.float32(0)
		state["fVec1"] = np.float32(0)
		state["iRec6"] = np.int32(0)
		# Initialize array delays
		state["fVec2"] = np.zeros((16384,), dtype=np.float32)
		state["fVec3"] = np.zeros((4096,), dtype=np.float32)
		state["fVec4"] = np.zeros((4096,), dtype=np.float32)
		state["fVec5"] = np.zeros((16384,), dtype=np.float32)
		state["fVec6"] = np.zeros((2048,), dtype=np.float32)
		state["fVec7"] = np.zeros((16384,), dtype=np.float32)
		state["fVec8"] = np.zeros((4096,), dtype=np.float32)
		state["fVec9"] = np.zeros((16384,), dtype=np.float32)
		state["fVec10"] = np.zeros((2048,), dtype=np.float32)
		state["fVec11"] = np.zeros((32768,), dtype=np.float32)
		state["fVec12"] = np.zeros((4096,), dtype=np.float32)
		state["fVec13"] = np.zeros((16384,), dtype=np.float32)
		state["fVec14"] = np.zeros((4096,), dtype=np.float32)
		state["fVec15"] = np.zeros((32768,), dtype=np.float32)
		state["fVec16"] = np.zeros((4096,), dtype=np.float32)
		state["fVec17"] = np.zeros((32768,), dtype=np.float32)
		state["fVec18"] = np.zeros((2048,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec46"] = np.zeros((3,), dtype=np.float32)
		state["fRec47"] = np.zeros((3,), dtype=np.float32)
		state["fRec48"] = np.zeros((3,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fRec50"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize read-write tables
		state["ftbl1mydspSIG0"] = np.zeros((65537,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		# Pre-compute table initialization patterns
		# WARNING: Unknown initialization pattern for ftbl1mydspSIG0
		# Table will remain initialized to zeros
		# Process inline subcontainers for read-write table initialization
		iRec16 = np.int32(0)
		
		iRec6 = np.zeros((4,), dtype=np.int32)
		
		# Skipping loop that fills static table - already handled in setup
		pass 
		for i1 in range(0, 65537):
			iRec6_temp = state["iRec6"] 
			state["iRec6"] = (iRec6_temp + np.int32(1)) 
			state["ftbl1mydspSIG0"][i1] = np.sin((np.float32(9.58738e-05) * ((state["iRec6"] + np.int32(-1))))) 
		
		
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = (jnp.float32(0.001) * params["fHslider0"]) 
		fSlow1 = (jnp.float32(0.001) * jnp.power(params["fHslider1"], jnp.float32(2.0))) 
		fSlow2 = params["fHslider2"] 
		iSlow3 = (fSlow2 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow4 = (iSlow3 > jnp.int32(0)).astype(jnp.int32) 
		iSlow5 = ((fSlow2 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow6 = (self._fConst4 * fSlow2) 
		fSlow7 = (iSlow3) 
		fSlow8 = (jnp.float32(0.001) * params["fHslider3"]) 
		fSlow9 = (jnp.float32(0.001) * params["fHslider4"]) 
		fSlow10 = params["fHslider5"] 
		iSlow11 = (fSlow10 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow12 = (iSlow11 > jnp.int32(0)).astype(jnp.int32) 
		iSlow13 = ((fSlow10 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow14 = (self._fConst4 * fSlow10) 
		fSlow15 = (iSlow11) 
		fSlow16 = params["fHslider6"] 
		iSlow17 = (fSlow16 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow18 = (iSlow17 > jnp.int32(0)).astype(jnp.int32) 
		iSlow19 = ((fSlow16 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow20 = (self._fConst4 * fSlow16) 
		fSlow21 = (iSlow17) 
		fSlow22 = params["fHslider7"] 
		iSlow23 = (fSlow22 > jnp.float32(0.0)).astype(jnp.int32) 
		iSlow24 = (iSlow23 > jnp.int32(0)).astype(jnp.int32) 
		iSlow25 = ((fSlow22 == jnp.float32(0.0)).astype(jnp.int32) > jnp.int32(0)).astype(jnp.int32) 
		fSlow26 = (self._fConst4 * fSlow22) 
		fSlow27 = (iSlow23) 
		fSlow28 = jnp.maximum(jnp.float32(0.05), jnp.minimum(jnp.float32(2.0), params["fHslider8"])) 
		fSlow29 = jnp.exp(-((self._fConst14 / fSlow28))) 
		fSlow30 = ((jnp.exp(-((self._fConst15 / fSlow28))) / fSlow29) + jnp.float32(-1.0)) 
		fSlow31 = jnp.power(fSlow29, jnp.float32(2.0)) 
		fSlow32 = (jnp.float32(1.0) - fSlow31) 
		fSlow33 = (jnp.float32(1.0) - (self._fConst16 * fSlow31)) 
		fSlow34 = (fSlow33 / fSlow32) 
		fSlow35 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow33, jnp.float32(2.0)) / jnp.power(fSlow32, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow36 = (fSlow29 * (fSlow35 + (jnp.float32(1.0) - fSlow34))) 
		fSlow37 = (fSlow34 - fSlow35) 
		fSlow38 = jnp.exp(-((self._fConst23 / fSlow28))) 
		fSlow39 = ((jnp.exp(-((self._fConst24 / fSlow28))) / fSlow38) + jnp.float32(-1.0)) 
		fSlow40 = jnp.power(fSlow38, jnp.float32(2.0)) 
		fSlow41 = (jnp.float32(1.0) - fSlow40) 
		fSlow42 = (jnp.float32(1.0) - (self._fConst16 * fSlow40)) 
		fSlow43 = (fSlow42 / fSlow41) 
		fSlow44 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow42, jnp.float32(2.0)) / jnp.power(fSlow41, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow45 = (fSlow38 * (fSlow44 + (jnp.float32(1.0) - fSlow43))) 
		fSlow46 = (fSlow43 - fSlow44) 
		fSlow47 = jnp.exp(-((self._fConst30 / fSlow28))) 
		fSlow48 = ((jnp.exp(-((self._fConst31 / fSlow28))) / fSlow47) + jnp.float32(-1.0)) 
		fSlow49 = jnp.power(fSlow47, jnp.float32(2.0)) 
		fSlow50 = (jnp.float32(1.0) - fSlow49) 
		fSlow51 = (jnp.float32(1.0) - (self._fConst16 * fSlow49)) 
		fSlow52 = (fSlow51 / fSlow50) 
		fSlow53 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow51, jnp.float32(2.0)) / jnp.power(fSlow50, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow54 = (fSlow47 * (fSlow53 + (jnp.float32(1.0) - fSlow52))) 
		fSlow55 = (fSlow52 - fSlow53) 
		fSlow56 = jnp.exp(-((self._fConst37 / fSlow28))) 
		fSlow57 = ((jnp.exp(-((self._fConst38 / fSlow28))) / fSlow56) + jnp.float32(-1.0)) 
		fSlow58 = jnp.power(fSlow56, jnp.float32(2.0)) 
		fSlow59 = (jnp.float32(1.0) - fSlow58) 
		fSlow60 = (jnp.float32(1.0) - (self._fConst16 * fSlow58)) 
		fSlow61 = (fSlow60 / fSlow59) 
		fSlow62 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow60, jnp.float32(2.0)) / jnp.power(fSlow59, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow63 = (fSlow56 * (fSlow62 + (jnp.float32(1.0) - fSlow61))) 
		fSlow64 = (fSlow61 - fSlow62) 
		fSlow65 = jnp.exp(-((self._fConst44 / fSlow28))) 
		fSlow66 = ((jnp.exp(-((self._fConst45 / fSlow28))) / fSlow65) + jnp.float32(-1.0)) 
		fSlow67 = jnp.power(fSlow65, jnp.float32(2.0)) 
		fSlow68 = (jnp.float32(1.0) - fSlow67) 
		fSlow69 = (jnp.float32(1.0) - (self._fConst16 * fSlow67)) 
		fSlow70 = (fSlow69 / fSlow68) 
		fSlow71 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow69, jnp.float32(2.0)) / jnp.power(fSlow68, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow72 = (fSlow65 * (fSlow71 + (jnp.float32(1.0) - fSlow70))) 
		fSlow73 = (fSlow70 - fSlow71) 
		fSlow74 = jnp.exp(-((self._fConst51 / fSlow28))) 
		fSlow75 = ((jnp.exp(-((self._fConst52 / fSlow28))) / fSlow74) + jnp.float32(-1.0)) 
		fSlow76 = jnp.power(fSlow74, jnp.float32(2.0)) 
		fSlow77 = (jnp.float32(1.0) - fSlow76) 
		fSlow78 = (jnp.float32(1.0) - (self._fConst16 * fSlow76)) 
		fSlow79 = (fSlow78 / fSlow77) 
		fSlow80 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow78, jnp.float32(2.0)) / jnp.power(fSlow77, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow81 = (fSlow74 * (fSlow80 + (jnp.float32(1.0) - fSlow79))) 
		fSlow82 = (fSlow79 - fSlow80) 
		fSlow83 = jnp.exp(-((self._fConst58 / fSlow28))) 
		fSlow84 = ((jnp.exp(-((self._fConst59 / fSlow28))) / fSlow83) + jnp.float32(-1.0)) 
		fSlow85 = jnp.power(fSlow83, jnp.float32(2.0)) 
		fSlow86 = (jnp.float32(1.0) - fSlow85) 
		fSlow87 = (jnp.float32(1.0) - (self._fConst16 * fSlow85)) 
		fSlow88 = (fSlow87 / fSlow86) 
		fSlow89 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow87, jnp.float32(2.0)) / jnp.power(fSlow86, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow90 = (fSlow83 * (fSlow89 + (jnp.float32(1.0) - fSlow88))) 
		fSlow91 = (fSlow88 - fSlow89) 
		fSlow92 = jnp.exp(-((self._fConst65 / fSlow28))) 
		fSlow93 = ((jnp.exp(-((self._fConst66 / fSlow28))) / fSlow92) + jnp.float32(-1.0)) 
		fSlow94 = jnp.power(fSlow92, jnp.float32(2.0)) 
		fSlow95 = (jnp.float32(1.0) - fSlow94) 
		fSlow96 = (jnp.float32(1.0) - (self._fConst16 * fSlow94)) 
		fSlow97 = (fSlow96 / fSlow95) 
		fSlow98 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow96, jnp.float32(2.0)) / jnp.power(fSlow95, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow99 = (fSlow92 * (fSlow98 + (jnp.float32(1.0) - fSlow97))) 
		fSlow100 = (fSlow97 - fSlow98) 
		fSlow101 = (jnp.float32(0.001) * params["fHslider9"]) 
		fRec0_temp = state["fRec0"] 
		fRec1_temp = state["fRec1"] 
		fRec3_temp = state["fRec3"] 
		fRec5_temp = state["fRec5"] 
		fRec4_temp = state["fRec4"] 
		fRec8_temp = state["fRec8"] 
		fRec7_temp = state["fRec7"] 
		fRec9_temp = state["fRec9"] 
		fRec11_temp = state["fRec11"] 
		fRec10_temp = state["fRec10"] 
		fRec12_temp = state["fRec12"] 
		fRec14_temp = state["fRec14"] 
		fRec13_temp = state["fRec13"] 
		fRec20_temp = state["fRec20"] 
		fRec19_temp = state["fRec19"] 
		fRec18_temp = state["fRec18"] 
		fRec17_temp = state["fRec17"] 
		fRec16_temp = state["fRec16"] 
		fRec15_temp = state["fRec15"] 
		fVec0_temp = state["fVec0"] 
		fRec21_temp = state["fRec21"] 
		fRec23_temp = state["fRec23"] 
		fRec22_temp = state["fRec22"] 
		fRec29_temp = state["fRec29"] 
		fRec28_temp = state["fRec28"] 
		fRec27_temp = state["fRec27"] 
		fRec26_temp = state["fRec26"] 
		fRec25_temp = state["fRec25"] 
		fRec24_temp = state["fRec24"] 
		fRec30_temp = state["fRec30"] 
		fRec32_temp = state["fRec32"] 
		fRec31_temp = state["fRec31"] 
		fRec38_temp = state["fRec38"] 
		fRec37_temp = state["fRec37"] 
		fRec36_temp = state["fRec36"] 
		fRec35_temp = state["fRec35"] 
		fRec34_temp = state["fRec34"] 
		fRec33_temp = state["fRec33"] 
		fRec44_temp = state["fRec44"] 
		fRec43_temp = state["fRec43"] 
		fRec42_temp = state["fRec42"] 
		fRec41_temp = state["fRec41"] 
		fRec40_temp = state["fRec40"] 
		fRec39_temp = state["fRec39"] 
		fVec1_temp = state["fVec1"] 
		fRec2_temp = state["fRec2"] 
		fRec56_temp = state["fRec56"] 
		fRec55_temp = state["fRec55"] 
		fRec53_temp = state["fRec53"] 
		fRec60_temp = state["fRec60"] 
		fRec59_temp = state["fRec59"] 
		fRec57_temp = state["fRec57"] 
		fRec64_temp = state["fRec64"] 
		fRec63_temp = state["fRec63"] 
		fRec61_temp = state["fRec61"] 
		fRec68_temp = state["fRec68"] 
		fRec67_temp = state["fRec67"] 
		fRec65_temp = state["fRec65"] 
		fRec72_temp = state["fRec72"] 
		fRec71_temp = state["fRec71"] 
		fRec69_temp = state["fRec69"] 
		fRec76_temp = state["fRec76"] 
		fRec75_temp = state["fRec75"] 
		fRec73_temp = state["fRec73"] 
		fRec80_temp = state["fRec80"] 
		fRec79_temp = state["fRec79"] 
		fRec77_temp = state["fRec77"] 
		fRec84_temp = state["fRec84"] 
		fRec83_temp = state["fRec83"] 
		fRec81_temp = state["fRec81"] 
		fRec85_temp = state["fRec85"] 
		state["fRec0"] = (fSlow0 + (jnp.float32(0.999) * fRec0_temp)) 
		fTemp0 = jnp.maximum(jnp.float32(0.05), jnp.minimum(jnp.float32(1.0), state["fRec0"])) 
		state["fRec1"] = (fSlow1 + (jnp.float32(0.999) * fRec1_temp)) 
		state["fRec3"] = jnp.where((iSlow4 != 0), jnp.float32(0.0), jnp.minimum(self._fConst1, (fRec3_temp + jnp.float32(1.0)))) 
		state["fRec5"] = jnp.where((iSlow5 != 0), jnp.float32(0.0), jnp.minimum(self._fConst3, (fRec5_temp + jnp.float32(1.0)))) 
		state["fRec4"] = jnp.where((iSlow3 != 0), (fSlow7 * jnp.where(((state["fRec5"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec5"] < self._fConst3).astype(jnp.int32) != 0), (fSlow6 * state["fRec5"]), fSlow2))), fRec4_temp) 
		state["fRec8"] = (fSlow8 + (jnp.float32(0.999) * fRec8_temp)) 
		fTemp1 = (fRec7_temp + (self._fConst5 * state["fRec8"])) 
		state["fRec7"] = (fTemp1 - jnp.floor(fTemp1)) 
		state["fRec9"] = ((jnp.float32(0.999) * fRec9_temp) + jnp.float32(0.0008)) 
		fTemp2 = (jnp.float32(3.1415927) * ((state["fRec9"] * self._ftbl0mydspSIG0[jnp.maximum(0, jnp.minimum(jnp.int32((jnp.float32(65536.0) * state["fRec7"])), 65535))]) * jnp.where(((state["fRec3"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec4"], jnp.where(((state["fRec3"] < self._fConst1).astype(jnp.int32) != 0), (state["fRec4"] * (jnp.float32(1.0) - (self._fConst2 * state["fRec3"]))), jnp.float32(0.0))))) 
		fTemp3 = jnp.sin(fTemp2) 
		state["fRec11"] = (fSlow9 + (jnp.float32(0.999) * fRec11_temp)) 
		fTemp4 = (fRec10_temp + (self._fConst5 * state["fRec11"])) 
		state["fRec10"] = (fTemp4 - jnp.floor(fTemp4)) 
		fTemp5 = (jnp.float32(65536.0) * state["fRec10"]) 
		iTemp6 = jnp.int32(fTemp5) 
		fTemp7 = state["ftbl1mydspSIG0"][jnp.maximum(0, jnp.minimum(iTemp6, 65536))] 
		fTemp8 = (fTemp7 + ((fTemp5 - jnp.floor(fTemp5)) * (state["ftbl1mydspSIG0"][jnp.maximum(0, jnp.minimum((iTemp6 + 1), 65536))] - fTemp7))) 
		fTemp9 = (jnp.float32(1.0) - state["fRec9"]) 
		state["fRec12"] = jnp.where((iSlow12 != 0), jnp.float32(0.0), jnp.minimum(self._fConst1, (fRec12_temp + jnp.float32(1.0)))) 
		state["fRec14"] = jnp.where((iSlow13 != 0), jnp.float32(0.0), jnp.minimum(self._fConst3, (fRec14_temp + jnp.float32(1.0)))) 
		state["fRec13"] = jnp.where((iSlow11 != 0), (fSlow15 * jnp.where(((state["fRec14"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec14"] < self._fConst3).astype(jnp.int32) != 0), (fSlow14 * state["fRec14"]), fSlow10))), fRec13_temp) 
		fTemp10 = (jnp.float32(3.1415927) * ((state["fRec9"] * fTemp8) * jnp.where(((state["fRec12"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec13"], jnp.where(((state["fRec12"] < self._fConst1).astype(jnp.int32) != 0), (state["fRec13"] * (jnp.float32(1.0) - (self._fConst2 * state["fRec12"]))), jnp.float32(0.0))))) 
		fTemp11 = jnp.cos(fTemp10) 
		fTemp12 = jnp.sin(fTemp10) 
		fTemp13 = ((fTemp8 * fTemp11) - (fTemp12 * fRec15_temp)) 
		fTemp14 = ((fTemp11 * fTemp13) - (fTemp12 * fRec16_temp)) 
		fTemp15 = ((fTemp11 * fTemp14) - (fTemp12 * fRec17_temp)) 
		fTemp16 = ((fTemp11 * fTemp15) - (fTemp12 * fRec18_temp)) 
		fTemp17 = ((fTemp11 * fTemp16) - (fTemp12 * fRec19_temp)) 
		state["fRec20"] = ((fTemp11 * fTemp17) - (fTemp12 * fRec20_temp)) 
		state["fRec19"] = ((fTemp12 * fTemp17) + (fTemp11 * fRec20_temp)) 
		state["fRec18"] = ((fTemp12 * fTemp16) + (fTemp11 * fRec19_temp)) 
		state["fRec17"] = ((fTemp12 * fTemp15) + (fTemp11 * fRec18_temp)) 
		state["fRec16"] = ((fTemp12 * fTemp14) + (fTemp11 * fRec17_temp)) 
		state["fRec15"] = ((fTemp12 * fTemp13) + (fTemp11 * fRec16_temp)) 
		fTemp18 = ((state["fRec9"] * ((fTemp8 * fTemp12) + (fRec15_temp * fTemp11))) + (fTemp9 * fTemp8)) 
		state["fVec0"] = jnp.float32(fTemp18) 
		state["fRec21"] = jnp.where((iSlow18 != 0), jnp.float32(0.0), jnp.minimum(self._fConst1, (fRec21_temp + jnp.float32(1.0)))) 
		state["fRec23"] = jnp.where((iSlow19 != 0), jnp.float32(0.0), jnp.minimum(self._fConst3, (fRec23_temp + jnp.float32(1.0)))) 
		state["fRec22"] = jnp.where((iSlow17 != 0), (fSlow21 * jnp.where(((state["fRec23"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec23"] < self._fConst3).astype(jnp.int32) != 0), (fSlow20 * state["fRec23"]), fSlow16))), fRec22_temp) 
		fTemp19 = (jnp.float32(1.5707964) * ((state["fRec9"] * jnp.where(((state["fRec21"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec22"], jnp.where(((state["fRec21"] < self._fConst1).astype(jnp.int32) != 0), (state["fRec22"] * (jnp.float32(1.0) - (self._fConst2 * state["fRec21"]))), jnp.float32(0.0)))) * (fTemp18 + fVec0_temp))) 
		fTemp20 = jnp.cos(fTemp19) 
		fTemp21 = jnp.sin(fTemp19) 
		fTemp22 = ((fTemp18 * fTemp20) - (fTemp21 * fRec24_temp)) 
		fTemp23 = ((fTemp20 * fTemp22) - (fTemp21 * fRec25_temp)) 
		fTemp24 = ((fTemp20 * fTemp23) - (fTemp21 * fRec26_temp)) 
		fTemp25 = ((fTemp20 * fTemp24) - (fTemp21 * fRec27_temp)) 
		fTemp26 = ((fTemp20 * fTemp25) - (fTemp21 * fRec28_temp)) 
		state["fRec29"] = ((fTemp20 * fTemp26) - (fTemp21 * fRec29_temp)) 
		state["fRec28"] = ((fTemp21 * fTemp26) + (fTemp20 * fRec29_temp)) 
		state["fRec27"] = ((fTemp21 * fTemp25) + (fTemp20 * fRec28_temp)) 
		state["fRec26"] = ((fTemp21 * fTemp24) + (fTemp20 * fRec27_temp)) 
		state["fRec25"] = ((fTemp21 * fTemp23) + (fTemp20 * fRec26_temp)) 
		state["fRec24"] = ((fTemp21 * fTemp22) + (fTemp20 * fRec25_temp)) 
		fTemp27 = ((state["fRec9"] * ((fTemp18 * fTemp21) + (fRec24_temp * fTemp20))) + (fTemp9 * fTemp18)) 
		state["fRec30"] = jnp.where((iSlow24 != 0), jnp.float32(0.0), jnp.minimum(self._fConst1, (fRec30_temp + jnp.float32(1.0)))) 
		state["fRec32"] = jnp.where((iSlow25 != 0), jnp.float32(0.0), jnp.minimum(self._fConst3, (fRec32_temp + jnp.float32(1.0)))) 
		state["fRec31"] = jnp.where((iSlow23 != 0), (fSlow27 * jnp.where(((state["fRec32"] < jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.float32(0.0), jnp.where(((state["fRec32"] < self._fConst3).astype(jnp.int32) != 0), (fSlow26 * state["fRec32"]), fSlow22))), fRec31_temp) 
		fTemp28 = (jnp.float32(3.1415927) * ((state["fRec9"] * jnp.power(fTemp27, jnp.float32(2.0))) * jnp.where(((state["fRec30"] < jnp.float32(0.0)).astype(jnp.int32) != 0), state["fRec31"], jnp.where(((state["fRec30"] < self._fConst1).astype(jnp.int32) != 0), (state["fRec31"] * (jnp.float32(1.0) - (self._fConst2 * state["fRec30"]))), jnp.float32(0.0))))) 
		fTemp29 = jnp.cos(fTemp28) 
		fTemp30 = jnp.sin(fTemp28) 
		fTemp31 = ((fTemp27 * fTemp29) - (fTemp30 * fRec33_temp)) 
		fTemp32 = ((fTemp29 * fTemp31) - (fTemp30 * fRec34_temp)) 
		fTemp33 = ((fTemp29 * fTemp32) - (fTemp30 * fRec35_temp)) 
		fTemp34 = ((fTemp29 * fTemp33) - (fTemp30 * fRec36_temp)) 
		fTemp35 = ((fTemp29 * fTemp34) - (fTemp30 * fRec37_temp)) 
		state["fRec38"] = ((fTemp29 * fTemp35) - (fTemp30 * fRec38_temp)) 
		state["fRec37"] = ((fTemp30 * fTemp35) + (fTemp29 * fRec38_temp)) 
		state["fRec36"] = ((fTemp30 * fTemp34) + (fTemp29 * fRec37_temp)) 
		state["fRec35"] = ((fTemp30 * fTemp33) + (fTemp29 * fRec36_temp)) 
		state["fRec34"] = ((fTemp30 * fTemp32) + (fTemp29 * fRec35_temp)) 
		state["fRec33"] = ((fTemp30 * fTemp31) + (fTemp29 * fRec34_temp)) 
		fTemp36 = jnp.cos(fTemp2) 
		fTemp37 = ((fTemp8 * fTemp36) - (fTemp3 * fRec39_temp)) 
		fTemp38 = ((fTemp36 * fTemp37) - (fTemp3 * fRec40_temp)) 
		fTemp39 = ((fTemp36 * fTemp38) - (fTemp3 * fRec41_temp)) 
		fTemp40 = ((fTemp36 * fTemp39) - (fTemp3 * fRec42_temp)) 
		fTemp41 = ((fTemp36 * fTemp40) - (fTemp3 * fRec43_temp)) 
		state["fRec44"] = ((fTemp36 * fTemp41) - (fTemp3 * fRec44_temp)) 
		state["fRec43"] = ((fTemp3 * fTemp41) + (fTemp36 * fRec44_temp)) 
		state["fRec42"] = ((fTemp3 * fTemp40) + (fTemp36 * fRec43_temp)) 
		state["fRec41"] = ((fTemp3 * fTemp39) + (fTemp36 * fRec42_temp)) 
		state["fRec40"] = ((fTemp3 * fTemp38) + (fTemp36 * fRec41_temp)) 
		state["fRec39"] = ((fTemp3 * fTemp37) + (fTemp36 * fRec40_temp)) 
		fTemp42 = ((fRec39_temp * fTemp36) + (((state["fRec9"] * ((fTemp27 * fTemp30) + (fRec33_temp * fTemp29))) + (fTemp9 * fTemp27)) + (fTemp8 * fTemp3))) 
		state["fVec1"] = jnp.float32(fTemp42) 
		state["fRec2"] = -((self._fConst8 * ((self._fConst7 * fRec2_temp) - (fTemp42 + fVec1_temp)))) 
		fTemp43 = (state["fRec2"] * state["fRec1"]) 
		fTemp44 = (jnp.float32(0.6) * (fTemp43 * (jnp.float32(1.0) - fTemp0))) 
		state["fRec56"] = -((self._fConst11 * ((self._fConst10 * fRec56_temp) - (state["fRec49"][1] + state["fRec49"][2])))) 
		state["fRec55"] = ((fSlow37 * fRec55_temp) + (fSlow36 * (state["fRec49"][1] + (fSlow30 * state["fRec56"])))) 
		state["fVec2"] = state["fVec2"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec55"]) + jnp.float32(1e-20))) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp43 * fTemp0)) 
		fTemp45 = (jnp.float32(0.18) * state["fVec3"][((state["IOTA0"] - self._iConst19) & 4095).astype(jnp.int32)]) 
		fTemp46 = ((fTemp45 + state["fVec2"][((state["IOTA0"] - self._iConst18) & 16383).astype(jnp.int32)]) - (jnp.float32(0.6) * fRec53_temp)) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp46) 
		state["fRec53"] = state["fVec4"][((state["IOTA0"] - self._iConst20) & 4095).astype(jnp.int32)] 
		fRec54 = (jnp.float32(0.6) * fTemp46) 
		state["fRec60"] = -((self._fConst11 * ((self._fConst10 * fRec60_temp) - (state["fRec45"][1] + state["fRec45"][2])))) 
		state["fRec59"] = ((fSlow46 * fRec59_temp) + (fSlow45 * (state["fRec45"][1] + (fSlow39 * state["fRec60"])))) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec59"]) + jnp.float32(1e-20))) 
		fTemp47 = ((state["fVec5"][((state["IOTA0"] - self._iConst26) & 16383).astype(jnp.int32)] + fTemp45) - (jnp.float32(0.6) * fRec57_temp)) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp47) 
		state["fRec57"] = state["fVec6"][((state["IOTA0"] - self._iConst27) & 2047).astype(jnp.int32)] 
		fRec58 = (jnp.float32(0.6) * fTemp47) 
		fTemp48 = (fRec58 + fRec54) 
		state["fRec64"] = -((self._fConst11 * ((self._fConst10 * fRec64_temp) - (state["fRec47"][1] + state["fRec47"][2])))) 
		state["fRec63"] = ((fSlow55 * fRec63_temp) + (fSlow54 * (state["fRec47"][1] + (fSlow48 * state["fRec64"])))) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec63"]) + jnp.float32(1e-20))) 
		fTemp49 = (state["fVec7"][((state["IOTA0"] - self._iConst33) & 16383).astype(jnp.int32)] - (fTemp45 + (jnp.float32(0.6) * fRec61_temp))) 
		state["fVec8"] = state["fVec8"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp49) 
		state["fRec61"] = state["fVec8"][((state["IOTA0"] - self._iConst34) & 4095).astype(jnp.int32)] 
		fRec62 = (jnp.float32(0.6) * fTemp49) 
		state["fRec68"] = -((self._fConst11 * ((self._fConst10 * fRec68_temp) - (state["fRec51"][1] + state["fRec51"][2])))) 
		state["fRec67"] = ((fSlow64 * fRec67_temp) + (fSlow63 * (state["fRec51"][1] + (fSlow57 * state["fRec68"])))) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec67"]) + jnp.float32(1e-20))) 
		fTemp50 = (state["fVec9"][((state["IOTA0"] - self._iConst40) & 16383).astype(jnp.int32)] - (fTemp45 + (jnp.float32(0.6) * fRec65_temp))) 
		state["fVec10"] = state["fVec10"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp50) 
		state["fRec65"] = state["fVec10"][((state["IOTA0"] - self._iConst41) & 2047).astype(jnp.int32)] 
		fRec66 = (jnp.float32(0.6) * fTemp50) 
		fTemp51 = (fRec66 + (fRec62 + fTemp48)) 
		state["fRec72"] = -((self._fConst11 * ((self._fConst10 * fRec72_temp) - (state["fRec46"][1] + state["fRec46"][2])))) 
		state["fRec71"] = ((fSlow73 * fRec71_temp) + (fSlow72 * (state["fRec46"][1] + (fSlow66 * state["fRec72"])))) 
		state["fVec11"] = state["fVec11"].at[(state["IOTA0"] & 32767).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec71"]) + jnp.float32(1e-20))) 
		fTemp52 = (state["fVec11"][((state["IOTA0"] - self._iConst47) & 32767).astype(jnp.int32)] + (fTemp45 + (jnp.float32(0.6) * fRec69_temp))) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp52) 
		state["fRec69"] = state["fVec12"][((state["IOTA0"] - self._iConst48) & 4095).astype(jnp.int32)] 
		fRec70 = -((jnp.float32(0.6) * fTemp52)) 
		state["fRec76"] = -((self._fConst11 * ((self._fConst10 * fRec76_temp) - (state["fRec50"][1] + state["fRec50"][2])))) 
		state["fRec75"] = ((fSlow82 * fRec75_temp) + (fSlow81 * (state["fRec50"][1] + (fSlow75 * state["fRec76"])))) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec75"]) + jnp.float32(1e-20))) 
		fTemp53 = (state["fVec13"][((state["IOTA0"] - self._iConst54) & 16383).astype(jnp.int32)] + (fTemp45 + (jnp.float32(0.6) * fRec73_temp))) 
		state["fVec14"] = state["fVec14"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp53) 
		state["fRec73"] = state["fVec14"][((state["IOTA0"] - self._iConst55) & 4095).astype(jnp.int32)] 
		fRec74 = -((jnp.float32(0.6) * fTemp53)) 
		state["fRec80"] = -((self._fConst11 * ((self._fConst10 * fRec80_temp) - (state["fRec48"][1] + state["fRec48"][2])))) 
		state["fRec79"] = ((fSlow91 * fRec79_temp) + (fSlow90 * (state["fRec48"][1] + (fSlow84 * state["fRec80"])))) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 32767).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec79"]) + jnp.float32(1e-20))) 
		fTemp54 = ((jnp.float32(0.6) * fRec77_temp) + state["fVec15"][((state["IOTA0"] - self._iConst61) & 32767).astype(jnp.int32)]) 
		state["fVec16"] = state["fVec16"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set((fTemp54 - fTemp45)) 
		state["fRec77"] = state["fVec16"][((state["IOTA0"] - self._iConst62) & 4095).astype(jnp.int32)] 
		fRec78 = (jnp.float32(0.6) * (fTemp45 - fTemp54)) 
		state["fRec84"] = -((self._fConst11 * ((self._fConst10 * fRec84_temp) - (state["fRec52"][1] + state["fRec52"][2])))) 
		state["fRec83"] = ((fSlow100 * fRec83_temp) + (fSlow99 * (state["fRec52"][1] + (fSlow93 * state["fRec84"])))) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 32767).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec83"]) + jnp.float32(1e-20))) 
		fTemp55 = ((jnp.float32(0.6) * fRec81_temp) + state["fVec17"][((state["IOTA0"] - self._iConst68) & 32767).astype(jnp.int32)]) 
		state["fVec18"] = state["fVec18"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set((fTemp55 - fTemp45)) 
		state["fRec81"] = state["fVec18"][((state["IOTA0"] - self._iConst69) & 2047).astype(jnp.int32)] 
		fRec82 = (jnp.float32(0.6) * (fTemp45 - fTemp55)) 
		state["fRec45"] = state["fRec45"].at[0].set((fRec81_temp + (fRec77_temp + (fRec73_temp + (fRec69_temp + (fRec65_temp + (fRec61_temp + (fRec53_temp + (fRec57_temp + (fRec82 + (fRec78 + (fRec74 + (fRec70 + fTemp51))))))))))))) 
		state["fRec46"] = state["fRec46"].at[0].set(((fRec65_temp + (fRec61_temp + (fRec53_temp + (fRec57_temp + fTemp51)))) - (fRec81_temp + (fRec77_temp + (fRec73_temp + (fRec69_temp + (fRec82 + (fRec78 + (fRec70 + fRec74))))))))) 
		fTemp56 = (fRec62 + fRec66) 
		state["fRec47"] = state["fRec47"].at[0].set(((fRec73_temp + (fRec69_temp + (fRec53_temp + (fRec57_temp + (fRec74 + (fRec70 + fTemp48)))))) - (fRec81_temp + (fRec77_temp + (fRec65_temp + (fRec61_temp + (fRec82 + (fRec78 + fTemp56)))))))) 
		state["fRec48"] = state["fRec48"].at[0].set(((fRec81_temp + (fRec77_temp + (fRec53_temp + (fRec57_temp + (fRec82 + (fRec78 + fTemp48)))))) - (fRec73_temp + (fRec69_temp + (fRec65_temp + (fRec61_temp + (fRec74 + (fRec70 + fTemp56)))))))) 
		fTemp57 = (fRec54 + fRec66) 
		fTemp58 = (fRec58 + fRec62) 
		state["fRec49"] = state["fRec49"].at[0].set(((fRec77_temp + (fRec69_temp + (fRec61_temp + (fRec57_temp + (fRec78 + (fRec70 + fTemp58)))))) - (fRec81_temp + (fRec73_temp + (fRec65_temp + (fRec53_temp + (fRec82 + (fRec74 + fTemp57)))))))) 
		state["fRec50"] = state["fRec50"].at[0].set(((fRec81_temp + (fRec73_temp + (fRec61_temp + (fRec57_temp + (fRec82 + (fRec74 + fTemp58)))))) - (fRec77_temp + (fRec69_temp + (fRec65_temp + (fRec53_temp + (fRec78 + (fRec70 + fTemp57)))))))) 
		fTemp59 = (fRec54 + fRec62) 
		fTemp60 = (fRec58 + fRec66) 
		state["fRec51"] = state["fRec51"].at[0].set(((fRec81_temp + (fRec69_temp + (fRec65_temp + (fRec57_temp + (fRec82 + (fRec70 + fTemp60)))))) - (fRec77_temp + (fRec73_temp + (fRec61_temp + (fRec53_temp + (fRec78 + (fRec74 + fTemp59)))))))) 
		state["fRec52"] = state["fRec52"].at[0].set(((fRec77_temp + (fRec73_temp + (fRec65_temp + (fRec57_temp + (fRec78 + (fRec74 + fTemp60)))))) - (fRec81_temp + (fRec69_temp + (fRec61_temp + (fRec53_temp + (fRec82 + (fRec70 + fTemp59)))))))) 
		state["fRec85"] = (fSlow101 + (jnp.float32(0.999) * fRec85_temp)) 
		fTemp61 = jnp.maximum(jnp.float32(0.75), jnp.minimum(jnp.float32(4.0), state["fRec85"])) 
		_result0 = (fTemp61 * ((jnp.float32(0.37) * (state["fRec46"][0] + state["fRec47"][0])) + fTemp44)) 
		_result1 = (fTemp61 * (fTemp44 + (jnp.float32(0.37) * (state["fRec46"][0] - state["fRec47"][0])))) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec46"] = jnp.roll(state["fRec46"], 1) 
		state["fRec47"] = jnp.roll(state["fRec47"], 1) 
		state["fRec48"] = jnp.roll(state["fRec48"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec50"] = jnp.roll(state["fRec50"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		return state, jnp.stack([_result0,_result1]) 
		
	# fmt: on	
	def load_soundfile(self, filepath: str):
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
		
		# If none of the paths worked, return the default silence array and sample rate
		return np.zeros((1, 1024)), self.sample_rate
	
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
			y = jnp.array(y)
			fLength.append(y.shape[1])
			fOffset.append(offset)
			fBuffers = fBuffers.at[:y.shape[0],offset:offset+y.shape[1]].set(y)
			offset += y.shape[1]
		if label.startswith("param:"):
			label = label[6:]  # remove param:
			label = "/".join(ui_path+[label])
			fBuffers = self.param("_"+label, (lambda key, shape: fBuffers), None)
			unnorm_funcs[zone] = (zone, lambda x: x)
		else:
			label = "/".join(ui_path+[label])

		setattr(self, zone, {"fLength": fLength, "fOffset": fOffset, "fBuffers": fBuffers, "fSR": fSR})
	
	def add_button(self, zone: str, ui_path: list[str], label: str, unnorm_funcs: dict):
		label = "/".join(ui_path+[label])
		setattr(self, zone, self.param(label, nn.initializers.constant(0., dtype=FAUSTFLOAT), ()))
		unnorm_funcs[label] = (zone, lambda x: x)
	
	def add_checkbox(self, zone: str, ui_path: list[str], label: str, unnorm_funcs: dict):
		self.add_button(zone, ui_path, label, unnorm_funcs)
	
	def add_nentry(
		self, zone: str, ui_path: List[str], label: str,
		init: float, a_min: float, a_max: float, step_size: float,
		unnorm_funcs: dict, scale_mode: str = "linear",
	):
		"""
		Gumbel-Softmax version of a FAUST nentry:
			* logits param  (num_steps,)
			* learnable temperature τ
			* optional Gumbel noise from self.make_rng("gumbel")
		Returns a *soft* value in the physical range.

		todo: this implementation may be problematic for custom value nentry like:
		`foo = nentry("foo[style:menu{'low':0;'mid':5;'high':7}]",0,0,7,1)`
		"""
		# ---------- set up grid ----------
		label = "/".join(ui_path + [label])
		num_steps  = int(round((a_max - a_min) / step_size)) + 1
		init_step  = int(round((init - a_min) / step_size))
		step_values = jnp.arange(num_steps, dtype=FAUSTFLOAT) * FAUSTFLOAT(step_size) + FAUSTFLOAT(a_min)

		# ---------- parameters ----------
		# (1) logits, initialised to favour the initial step
		def init_logits(key, shape):
			logits = jnp.zeros(shape, dtype=FAUSTFLOAT)
			return logits.at[init_step].set(FAUSTFLOAT(5.0))        # bias ≈ exp(5) ≈ 148
		logits_zone = zone + "_logits"
		logits_label = label + ":logits"
		setattr(self, logits_zone, self.param(logits_label, init_logits, (num_steps,)))

		# temperature (optional learnable scalar)
		# tau = self.param(f"{zone}_tau", nn.initializers.constant(1.0), ())
		tau = 1.0  # TODO: user should be able to configure via UI Label metadata:
		# https://faustdoc.grame.fr/manual/syntax/#ui-label-metadata

		# Store nentry metadata as attributes. TODO: necessary?
		setattr(self, f"_{zone}_step_values", step_values)
		setattr(self, f"_{zone}_tau", tau)
		setattr(self, f"_{zone}_logits_zone", logits_zone)
		
		# Add unnormalization lambda for nentry
		def make_nentry_unnorm(zone, logits_zone, tau, step_values):
			def unnorm_nentry(module):
				logits = getattr(module, logits_zone)
				# Gumbel-softmax computation
				if module.has_rng("gumbel"):  # training
					gumbel_noise = random.gumbel(module.make_rng("gumbel"), logits.shape, dtype=FAUSTFLOAT)
					logits_with_noise = logits + gumbel_noise
					probs = nn.softmax(logits_with_noise / tau, axis=-1)
					return jnp.dot(probs, step_values)
				else:  # inference
					index = jnp.argmax(logits, axis=-1)
					return step_values[index]
			return unnorm_nentry
		
		unnorm_funcs[label] = (zone, make_nentry_unnorm(zone, logits_zone, tau, step_values))
	
	def normalize_value(self, value: float, a_min: float, a_max: float, scale_mode: str) -> float:
		"""Normalize a value from [a_min, a_max] to [-1, 1] based on scale mode."""
		if scale_mode == "linear":
			return jnp.interp(value, jnp.array([a_min, a_max], dtype=FAUSTFLOAT), 
							 jnp.array([FAUSTFLOAT(-1), FAUSTFLOAT(1)], dtype=FAUSTFLOAT))
		elif scale_mode == "exp":
			# Map to [1, e], take log, then map to [-1, 1]
			value_exp = jnp.interp(value, jnp.array([a_min, a_max], dtype=FAUSTFLOAT), 
								  jnp.array([FAUSTFLOAT(1), jnp.e], dtype=FAUSTFLOAT))
			value_log = jnp.log(value_exp)
			return jnp.interp(value_log, jnp.array([FAUSTFLOAT(0), FAUSTFLOAT(1)], dtype=FAUSTFLOAT), 
							 jnp.array([FAUSTFLOAT(-1), FAUSTFLOAT(1)], dtype=FAUSTFLOAT))
		elif scale_mode == "log":
			# Map to [-4, 0], apply 10^x, then map to [-1, 1]
			value_log10 = jnp.interp(value, jnp.array([a_min, a_max], dtype=FAUSTFLOAT), 
									jnp.array([FAUSTFLOAT(-4), FAUSTFLOAT(0)], dtype=FAUSTFLOAT))
			value_pow = jnp.power(FAUSTFLOAT(10), value_log10)
			return jnp.interp(value_pow, jnp.array([FAUSTFLOAT(10**-4), FAUSTFLOAT(1)], dtype=FAUSTFLOAT), 
							 jnp.array([FAUSTFLOAT(-1), FAUSTFLOAT(1)], dtype=FAUSTFLOAT))
		else:
			raise ValueError(f"Unknown scale mode: {scale_mode}")
	
	def create_unnormalize_func(self, a_min: float, a_max: float, scale_mode: str):
		"""Create an unnormalization function for the given scale mode."""
		if scale_mode == "linear":
			return lambda normalized: jnp.interp(
				jnp.clip(normalized, FAUSTFLOAT(-1), FAUSTFLOAT(1)),
				jnp.array([FAUSTFLOAT(-1), FAUSTFLOAT(1)], dtype=FAUSTFLOAT),
				jnp.array([a_min, a_max], dtype=FAUSTFLOAT)
			)
		elif scale_mode == "exp":
			return lambda normalized: jnp.interp(
				jnp.exp(jnp.interp(
					jnp.clip(normalized, FAUSTFLOAT(-1), FAUSTFLOAT(1)),
					jnp.array([FAUSTFLOAT(-1), FAUSTFLOAT(1)], dtype=FAUSTFLOAT),
					jnp.array([FAUSTFLOAT(0), FAUSTFLOAT(1)], dtype=FAUSTFLOAT)
				)), 
				jnp.array([FAUSTFLOAT(1), jnp.e], dtype=FAUSTFLOAT), 
				jnp.array([a_min, a_max], dtype=FAUSTFLOAT)
			)
		elif scale_mode == "log":
			return lambda normalized: jnp.interp(
				jnp.log10(jnp.interp(
					jnp.clip(normalized, FAUSTFLOAT(-1), FAUSTFLOAT(1)),
					jnp.array([FAUSTFLOAT(-1), FAUSTFLOAT(1)], dtype=FAUSTFLOAT),
					jnp.array([FAUSTFLOAT(10**-4), FAUSTFLOAT(1)], dtype=FAUSTFLOAT)
				)), 
				jnp.array([FAUSTFLOAT(-4), FAUSTFLOAT(0)], dtype=FAUSTFLOAT), 
				jnp.array([a_min, a_max], dtype=FAUSTFLOAT)
			)
		else:
			raise ValueError(f"Unknown scale mode: {scale_mode}")
	
	def add_slider(self, zone: str, ui_path: list[str], label: str, init: float, a_min: float, a_max: float, unnorm_funcs: dict, scale_mode="linear"):
		"""Add a slider UI element with the specified parameters."""
		label = "/".join(ui_path + [label])
		init, a_min, a_max = FAUSTFLOAT(init), FAUSTFLOAT(a_min), FAUSTFLOAT(a_max)
		
		# Normalize init value to [-1, 1] based on scale mode
		normalized_init = self.normalize_value(init, a_min, a_max, scale_mode)
		
		# Create the normalized parameter with label as name
		setattr(self, zone, self.param(label, nn.initializers.constant(normalized_init, dtype=FAUSTFLOAT), ()))
		
		# Create and store the unnormalization function
		unnorm_func = self.create_unnormalize_func(a_min, a_max, scale_mode)
		unnorm_funcs[label] = (zone, unnorm_func)
	
	def add_hslider(self, zone: str, ui_path: list[str], label: str, init: float, a_min: float, a_max: float, unnorm_funcs: dict, scale_mode: str):
		self.add_slider(zone, ui_path, label, init, a_min, a_max, unnorm_funcs, scale_mode)
	
	def add_vslider(self, zone: str, ui_path: list[str], label: str, init: float, a_min: float, a_max: float, unnorm_funcs: dict, scale_mode: str):
		self.add_slider(zone, ui_path, label, init, a_min, a_max, unnorm_funcs, scale_mode)
	
	def add_hbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float, unnorm_funcs: dict):
		# Bargraphs are output-only, no parameters needed
		pass
	
	def add_vbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float, unnorm_funcs: dict):
		# Bargraphs are output-only, no parameters needed
		pass

	def random_uniform(self):
		"""
		Generate a random uniform value in the range [-1, 1] using JAX's PRNG.
		This method is called by foreign functions declared in Faust code.
		"""
		return random.uniform(self.make_rng("rng_stream"), shape=(), minval=-1, maxval=1, dtype=FAUSTFLOAT)

	def unnormalize(self) -> Dict[str, jnp.array]:
		"""
		Unnormalize all UI parameters from [-1, 1] to their original ranges.
		
		Returns:
			Dictionary mapping zones to unnormalized parameter values
		"""
		params = {}
		
		# Simply use the stored unnormalization functions
		for label, (zone, unnorm_func) in self._unnorm_funcs.items():
			# Check if it's a nentry (needs module as arg)
			if hasattr(self, f"_{zone}_logits_zone"):
				params[zone] = unnorm_func(self)
			elif hasattr(self, zone):
				# Regular parameter
				normalized_value = getattr(self, zone)
				params[zone] = unnorm_func(normalized_value)
			else:
				raise ValueError(f"Zone not found: {zone}")
			self.sow("intermediates", label, params[zone])
		
		return params

	def initialize_carry(self) -> Dict[str, jnp.array]:
		"""
		Initialize the carry state for real-time processing.
			
		Returns:
			Dictionary containing all stateful components (delays, filter states, etc.)
		"""
		# Create dummy input for initialization
		dummy_x = jnp.zeros((self.num_inputs, 1), dtype=FAUSTFLOAT)
		
		# Initialize the full state using fast numpy
		state = self._initialize_carry(dummy_x, 1)
		
		# Convert numpy to JAX numpy arrays
		state = jax.tree.map(jnp.array, state)
		
		return state
	
	def process_block(self, carry: Dict[str, jnp.array], inputs: jnp.array = None, length: int = None, unroll: int = 1) -> Tuple[jnp.array, Dict[str, jnp.array]]:
		"""
		Process one block of audio and return updated state.
		
		Args:
			carry: State dictionary from previous block
			inputs: Input audio block of shape (num_inputs, block_size)
			length (int): block size of generated output
			unroll (int): 
			
		Returns:
			Tuple of (output_block, new_carry) where:
			- output_block has shape (num_outputs, block_size)
			- new_carry is the updated state dictionary
		"""
		if length is None and inputs is not None and hasattr(inputs, "shape"):
			length = inputs.shape[-1]

		# Transpose for scan: (block_size, num_inputs)
		if inputs is not None:
			inputs = jnp.transpose(inputs, axes=(1, 0))

		# Unnormalize parameters once before the scan
		params = self.unnormalize()
		
		def tick(module, carry, *xs):
			return module.tick(params, carry, *xs)
		
		scan_fn = nn.scan(tick,
			variable_broadcast="params",
			split_rngs={"rng_stream": True},
			length=length,
			unroll=unroll,
		)
		new_carry, outputs = scan_fn(self, carry, inputs)
		
		# Transpose back: (num_outputs, block_size)
		outputs_t = jnp.transpose(outputs, axes=(1, 0))

		return outputs_t, new_carry
	
	def __call__(self, x: jnp.array, length: int = None, unroll: int = 1) -> jnp.array:

		if length is None and x is not None:
			length = x.shape[-1]

		# Handle generators (no input case)
		if x is None:
			x = jnp.zeros((self.num_inputs, length), dtype=FAUSTFLOAT)

		carry = self.initialize_carry()
		
		# Unnormalize parameters once before the scan
		params = self.unnormalize()
		
		def tick(module, carry, *xs):
			return module.tick(params, carry, *xs)

		scan_fn = nn.scan(tick,
			variable_broadcast="params",
			split_rngs={"rng_stream": True},
			length=length,
			unroll=unroll,
		)
		new_carry, outputs = scan_fn(self, carry, jnp.transpose(x, axes=(1, 0)))
		
		return jnp.transpose(outputs, axes=(1,0))


def test(args):

	import logging

	model = mydsp(sample_rate=args.sample_rate)

	log_level = getattr(logging, args.log_level.upper())
	logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

	logger = logging.getLogger(__name__)

	logger.info(f"Number of input channels: {model.num_inputs}")
	logger.info(f"Number of output channels: {model.num_outputs}")

	# json_obj = model.json_metadata
	# logger.debug(f"JSON info: {json_obj}")

	key = random.key(args.seed)

	if args.input is not None:
		input_audio, _ = librosa.load(args.input, mono=False, sr=args.sample_rate, duration=args.duration)
		if input_audio.ndim == 1:
			input_audio = input_audio.unsqueeze(0)

		N_SAMPLES = input_audio.shape[1]
		N_CHANNELS = input_audio.shape[0]
		assert N_CHANNELS == model.num_inputs

		input_audio = FAUSTFLOAT(input_audio)
	else:
		duration_sec = args.duration or 1.  # default to 1 second when making noise.

		N_SAMPLES = int(duration_sec*args.sample_rate)
		if isinstance(args.unroll, int):
			N_SAMPLES = (N_SAMPLES//int(args.unroll))*int(args.unroll)
		N_CHANNELS = model.num_inputs

		if args.random:
			input_audio = -1.+2.*random.uniform(key, shape=(N_CHANNELS, N_SAMPLES), dtype=FAUSTFLOAT)
		else:
			input_audio = jnp.zeros((N_CHANNELS, N_SAMPLES), dtype=FAUSTFLOAT)
			input_audio = input_audio.at[:,0].set(1.)

	variables = model.init({"params": key, "rng_stream": key}, input_audio, length=N_SAMPLES, unroll=args.unroll)
	if args.verbose:
		print("variables:", variables)

	def forward(variables, x: jnp.ndarray):
		y = model.apply(variables, x, length=N_SAMPLES, unroll=args.unroll, rngs={"rng_stream": key})
		return y
	
	if args.jit:
		forward = jax.jit(forward)

	if args.benchmark:
		import tqdm
		for _ in range(3):
			y = forward(variables, input_audio).block_until_ready()
		for _ in tqdm.trange(args.benchmark):
			y = forward(variables, input_audio).block_until_ready()

	y = forward(variables, input_audio)

	_, mod_vars = model.apply(variables, mutable="intermediates", rngs={"rng_stream": key}, method="unnormalize")
	if args.verbose:
		print("mod_vars", mod_vars)

	assert y.ndim == 2
	assert y.shape[0] == model.num_outputs
	assert y.shape[1] == input_audio.shape[1]
	assert y.shape[1] == N_SAMPLES

	if args.output is not None:
		from scipy.io import wavfile
		output_audio = np.array(y).T
		wavfile.write(args.output, args.sample_rate, output_audio)

	logger.info("All done!")


def realtime_audio_example(unroll: int = 1):
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
	
	# Audio settings
	SAMPLE_RATE = 48000
	BLOCK_SIZE = 512
	
	# Initialize model
	model = mydsp(sample_rate=SAMPLE_RATE)
	key = random.key(0)
	
	# Initialize parameters
	if model.num_inputs > 0:
		dummy_input = jnp.zeros((model.num_inputs, 1))
	else:
		dummy_input = None
	
	variables = model.init({"params": key, "rng_stream": key}, dummy_input, length=1)
	
	# Initialize carry state
	carry = model.apply(variables, method="initialize_carry")
	
	# JIT compile the process method
	@jax.jit
	def process_block_jit(carry, inputs: jnp.ndarray, rng: jax.Array):
		return model.apply(variables, carry, inputs, length=BLOCK_SIZE, unroll=unroll, method="process_block", rngs={"rng_stream": rng})
	
	# Create a generator for audio blocks
	def audio_generator():
		nonlocal carry
		key = random.key(0)
		while True:
			# For generators, create empty input
			if model.num_inputs == 0:
				inputs = jnp.zeros((0, BLOCK_SIZE))
			else:
				# For processors, you would get input from sounddevice
				# For this example, we'll use zeros
				inputs = jnp.zeros((model.num_inputs, BLOCK_SIZE))
			
			# Process block
			key, subkey = random.split(key)
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
	print(f"▶ Streaming audio at {SAMPLE_RATE}Hz, {BLOCK_SIZE} samples/block")
	print(f"  Model: {model.num_inputs} inputs → {model.num_outputs} outputs")
	print("  Press Ctrl+C to stop...")
	
	try:
		with sd.OutputStream(
			channels=model.num_outputs,
			samplerate=SAMPLE_RATE,
			blocksize=BLOCK_SIZE,
			dtype='float32',
			callback=callback
		):
			while True:
				time.sleep(1)
	except KeyboardInterrupt:
		print("\n⏹ Stopped.")
		

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Run a JAX/Flax model converted from Faust code")
	parser.add_argument("-sr", "--sample-rate", type=int, default=44100, help="Sample rate (such as 44100)")
	parser.add_argument("-d", "--duration", type=float, default=None, help="Output duration in seconds")
	parser.add_argument("--unroll", type=int, default=1, help="Unroll size (default is 1)")
	parser.add_argument("--random", default=False, action=argparse.BooleanOptionalAction,
		help="Whether the default audio is random. By default it\"s an impulse.")
	parser.add_argument("--seed", default=0, type=int, help="Seed for random number generator (default: 0)")
	parser.add_argument("-i", "--input", type=str, default=None, help="Filepath for input audio WAV")
	parser.add_argument("-o", "--output", type=str, default=None, help="Filepath for output audio WAV")
	parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
						help="Set the logger level (default: INFO)")
	parser.add_argument("--jit", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to use JIT.")
	parser.add_argument("--benchmark", type=int, default=0, help="Number of loops for a speed benchmark with tqdm (default=0).")
	parser.add_argument("--platform", default="cpu", choices=["cpu", "gpu", "metal", "tpu"])
	parser.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction,
						help="Whether to print the variables of the DSP")
	parser.add_argument("--realtime", default=False, action=argparse.BooleanOptionalAction,
						help="Run the DSP with silent input and send the output to an audio device in real-time.")

	args = parser.parse_args()
	
	# Global flag to set a specific platform, must be used at startup.
	jax.config.update("jax_platform_name", args.platform)

	if args.realtime:
		realtime_audio_example(args.unroll)
	else:
		test(args)
