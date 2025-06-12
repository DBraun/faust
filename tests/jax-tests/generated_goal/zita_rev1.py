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
		return 2
	
	@property
	def num_outputs(self):
		return 2
	
	# fmt: off
	def setup(self):
		# Initialize static tables
		# Initialize waveform data
		# Convert static tables and waveform data to JAX arrays
		# Initialize UI parameters
		unnorm_funcs = {}
		ui_path = []
		ui_path.append("Zita_Rev1") 
		ui_path.append("Input") 
		self.add_vslider("fVslider9", ui_path, "In Delay", 6e+01, 2e+01, 1e+02, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Decay Times in Bands (see tooltips)") 
		self.add_vslider("fVslider5", ui_path, "LF X", 2e+02, 5e+01, 1e+03, unnorm_funcs, "log") 
		self.add_vslider("fVslider7", ui_path, "Low RT60", 3.0, 1.0, 8.0, unnorm_funcs, "log") 
		self.add_vslider("fVslider6", ui_path, "Mid RT60", 2.0, 1.0, 8.0, unnorm_funcs, "log") 
		self.add_vslider("fVslider8", ui_path, "HF Damping", 6e+03, 1.5e+03, 2.352e+04, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("RM Peaking Equalizer 1") 
		self.add_vslider("fVslider4", ui_path, "Eq1 Freq", 315.0, 4e+01, 2.5e+03, unnorm_funcs, "log") 
		self.add_vslider("fVslider3", ui_path, "Eq1 Level", 0.0, -15.0, 15.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("RM Peaking Equalizer 2") 
		self.add_vslider("fVslider2", ui_path, "Eq2 Freq", 1.5e+03, 1.6e+02, 1e+04, unnorm_funcs, "log") 
		self.add_vslider("fVslider1", ui_path, "Eq2 Level", 0.0, -15.0, 15.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Output") 
		self.add_vslider("fVslider0", ui_path, "Dry/Wet Mix", 0.4492, -1.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider10", ui_path, "Level", 16.79, -7e+01, 4e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = (np.float32(6.2831855) / self._fConst0) 
		self._fConst2 = (np.float32(3.1415927) / self._fConst0) 
		self._fConst3 = np.floor(((np.float32(0.174713) * self._fConst0) + np.float32(0.5))) 
		self._fConst4 = (np.float32(6.9077554) * (self._fConst3 / self._fConst0)) 
		self._fConst5 = np.floor(((np.float32(0.022904) * self._fConst0) + np.float32(0.5))) 
		self._iConst6 = (np.int32((self._fConst3 - self._fConst5)) & np.int32(8191)).astype(jnp.int32) 
		self._fConst7 = (np.float32(0.001) * self._fConst0) 
		self._iConst8 = (np.int32((self._fConst5 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst9 = np.floor(((np.float32(0.153129) * self._fConst0) + np.float32(0.5))) 
		self._fConst10 = (np.float32(6.9077554) * (self._fConst9 / self._fConst0)) 
		self._fConst11 = np.floor(((np.float32(0.020346) * self._fConst0) + np.float32(0.5))) 
		self._iConst12 = (np.int32((self._fConst9 - self._fConst11)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst13 = (np.int32((self._fConst11 + np.float32(-1.0))) & np.int32(1023)).astype(jnp.int32) 
		self._fConst14 = np.floor(((np.float32(0.127837) * self._fConst0) + np.float32(0.5))) 
		self._fConst15 = (np.float32(6.9077554) * (self._fConst14 / self._fConst0)) 
		self._fConst16 = np.floor(((np.float32(0.031604) * self._fConst0) + np.float32(0.5))) 
		self._iConst17 = (np.int32((self._fConst14 - self._fConst16)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst18 = (np.int32((self._fConst16 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst19 = np.floor(((np.float32(0.125) * self._fConst0) + np.float32(0.5))) 
		self._fConst20 = (np.float32(6.9077554) * (self._fConst19 / self._fConst0)) 
		self._fConst21 = np.floor(((np.float32(0.013458) * self._fConst0) + np.float32(0.5))) 
		self._iConst22 = (np.int32((self._fConst19 - self._fConst21)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst23 = (np.int32((self._fConst21 + np.float32(-1.0))) & np.int32(1023)).astype(jnp.int32) 
		self._fConst24 = np.floor(((np.float32(0.210389) * self._fConst0) + np.float32(0.5))) 
		self._fConst25 = (np.float32(6.9077554) * (self._fConst24 / self._fConst0)) 
		self._fConst26 = np.floor(((np.float32(0.024421) * self._fConst0) + np.float32(0.5))) 
		self._iConst27 = (np.int32((self._fConst24 - self._fConst26)) & np.int32(16383)).astype(jnp.int32) 
		self._iConst28 = (np.int32((self._fConst26 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst29 = np.floor(((np.float32(0.192303) * self._fConst0) + np.float32(0.5))) 
		self._fConst30 = (np.float32(6.9077554) * (self._fConst29 / self._fConst0)) 
		self._fConst31 = np.floor(((np.float32(0.029291) * self._fConst0) + np.float32(0.5))) 
		self._iConst32 = (np.int32((self._fConst29 - self._fConst31)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst33 = (np.int32((self._fConst31 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst34 = np.floor(((np.float32(0.256891) * self._fConst0) + np.float32(0.5))) 
		self._fConst35 = (np.float32(6.9077554) * (self._fConst34 / self._fConst0)) 
		self._fConst36 = np.floor(((np.float32(0.027333) * self._fConst0) + np.float32(0.5))) 
		self._iConst37 = (np.int32((self._fConst34 - self._fConst36)) & np.int32(16383)).astype(jnp.int32) 
		self._iConst38 = (np.int32((self._fConst36 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst39 = np.floor(((np.float32(0.219991) * self._fConst0) + np.float32(0.5))) 
		self._fConst40 = (np.float32(6.9077554) * (self._fConst39 / self._fConst0)) 
		self._fConst41 = np.floor(((np.float32(0.019123) * self._fConst0) + np.float32(0.5))) 
		self._iConst42 = (np.int32((self._fConst39 - self._fConst41)) & np.int32(16383)).astype(jnp.int32) 
		self._iConst43 = (np.int32((self._fConst41 + np.float32(-1.0))) & np.int32(1023)).astype(jnp.int32) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec21"] = np.float32(0)
		state["fRec22"] = np.float32(0)
		state["fRec23"] = np.float32(0)
		state["fRec25"] = np.float32(0)
		state["fRec26"] = np.float32(0)
		state["fRec27"] = np.float32(0)
		state["fRec29"] = np.float32(0)
		state["fRec30"] = np.float32(0)
		state["fRec31"] = np.float32(0)
		state["fRec33"] = np.float32(0)
		state["fRec34"] = np.float32(0)
		state["fRec35"] = np.float32(0)
		state["fRec37"] = np.float32(0)
		state["fRec38"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec41"] = np.float32(0)
		state["fRec42"] = np.float32(0)
		state["fRec43"] = np.float32(0)
		# Initialize array delays
		state["fVec0"] = np.zeros((8192,), dtype=np.float32)
		state["fVec1"] = np.zeros((8192,), dtype=np.float32)
		state["fVec2"] = np.zeros((2048,), dtype=np.float32)
		state["fVec3"] = np.zeros((8192,), dtype=np.float32)
		state["fVec4"] = np.zeros((1024,), dtype=np.float32)
		state["fVec5"] = np.zeros((8192,), dtype=np.float32)
		state["fVec6"] = np.zeros((2048,), dtype=np.float32)
		state["fVec7"] = np.zeros((8192,), dtype=np.float32)
		state["fVec8"] = np.zeros((1024,), dtype=np.float32)
		state["fVec9"] = np.zeros((16384,), dtype=np.float32)
		state["fVec10"] = np.zeros((8192,), dtype=np.float32)
		state["fVec11"] = np.zeros((2048,), dtype=np.float32)
		state["fVec12"] = np.zeros((8192,), dtype=np.float32)
		state["fVec13"] = np.zeros((2048,), dtype=np.float32)
		state["fVec14"] = np.zeros((16384,), dtype=np.float32)
		state["fVec15"] = np.zeros((2048,), dtype=np.float32)
		state["fVec16"] = np.zeros((16384,), dtype=np.float32)
		state["fVec17"] = np.zeros((1024,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = (jnp.float32(0.001) * params["fVslider0"]) 
		fSlow1 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider1"])) 
		fSlow2 = params["fVslider2"] 
		fSlow3 = (self._fConst1 * (fSlow2 / jnp.sqrt(jnp.maximum(jnp.float32(0.0), fSlow1)))) 
		fSlow4 = ((jnp.float32(1.0) - fSlow3) / (fSlow3 + jnp.float32(1.0))) 
		fSlow5 = (jnp.cos((self._fConst1 * fSlow2)) * (fSlow4 + jnp.float32(1.0))) 
		fSlow6 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider3"])) 
		fSlow7 = params["fVslider4"] 
		fSlow8 = (self._fConst1 * (fSlow7 / jnp.sqrt(jnp.maximum(jnp.float32(0.0), fSlow6)))) 
		fSlow9 = ((jnp.float32(1.0) - fSlow8) / (fSlow8 + jnp.float32(1.0))) 
		fSlow10 = (jnp.cos((self._fConst1 * fSlow7)) * (fSlow9 + jnp.float32(1.0))) 
		fSlow11 = (jnp.float32(1.0) / jnp.tan((self._fConst2 * params["fVslider5"]))) 
		fSlow12 = (jnp.float32(1.0) - fSlow11) 
		fSlow13 = (jnp.float32(1.0) / (fSlow11 + jnp.float32(1.0))) 
		fSlow14 = params["fVslider6"] 
		fSlow15 = jnp.exp(-((self._fConst4 / fSlow14))) 
		fSlow16 = params["fVslider7"] 
		fSlow17 = ((jnp.exp(-((self._fConst4 / fSlow16))) / fSlow15) + jnp.float32(-1.0)) 
		fSlow18 = jnp.power(fSlow15, jnp.float32(2.0)) 
		fSlow19 = (jnp.float32(1.0) - fSlow18) 
		fSlow20 = jnp.cos((self._fConst1 * params["fVslider8"])) 
		fSlow21 = (jnp.float32(1.0) - (fSlow20 * fSlow18)) 
		fSlow22 = (fSlow21 / fSlow19) 
		fSlow23 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow21, jnp.float32(2.0)) / jnp.power(fSlow19, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow24 = (fSlow15 * (fSlow23 + (jnp.float32(1.0) - fSlow22))) 
		fSlow25 = (fSlow22 - fSlow23) 
		iSlow26 = (jnp.int32((self._fConst7 * params["fVslider9"])) & jnp.int32(8191)).astype(jnp.int32) 
		fSlow27 = jnp.exp(-((self._fConst10 / fSlow14))) 
		fSlow28 = ((jnp.exp(-((self._fConst10 / fSlow16))) / fSlow27) + jnp.float32(-1.0)) 
		fSlow29 = jnp.power(fSlow27, jnp.float32(2.0)) 
		fSlow30 = (jnp.float32(1.0) - fSlow29) 
		fSlow31 = (jnp.float32(1.0) - (fSlow29 * fSlow20)) 
		fSlow32 = (fSlow31 / fSlow30) 
		fSlow33 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow31, jnp.float32(2.0)) / jnp.power(fSlow30, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow34 = (fSlow27 * (fSlow33 + (jnp.float32(1.0) - fSlow32))) 
		fSlow35 = (fSlow32 - fSlow33) 
		fSlow36 = jnp.exp(-((self._fConst15 / fSlow14))) 
		fSlow37 = ((jnp.exp(-((self._fConst15 / fSlow16))) / fSlow36) + jnp.float32(-1.0)) 
		fSlow38 = jnp.power(fSlow36, jnp.float32(2.0)) 
		fSlow39 = (jnp.float32(1.0) - fSlow38) 
		fSlow40 = (jnp.float32(1.0) - (fSlow20 * fSlow38)) 
		fSlow41 = (fSlow40 / fSlow39) 
		fSlow42 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow40, jnp.float32(2.0)) / jnp.power(fSlow39, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow43 = (fSlow36 * (fSlow42 + (jnp.float32(1.0) - fSlow41))) 
		fSlow44 = (fSlow41 - fSlow42) 
		fSlow45 = jnp.exp(-((self._fConst20 / fSlow14))) 
		fSlow46 = ((jnp.exp(-((self._fConst20 / fSlow16))) / fSlow45) + jnp.float32(-1.0)) 
		fSlow47 = jnp.power(fSlow45, jnp.float32(2.0)) 
		fSlow48 = (jnp.float32(1.0) - fSlow47) 
		fSlow49 = (jnp.float32(1.0) - (fSlow20 * fSlow47)) 
		fSlow50 = (fSlow49 / fSlow48) 
		fSlow51 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow49, jnp.float32(2.0)) / jnp.power(fSlow48, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow52 = (fSlow45 * (fSlow51 + (jnp.float32(1.0) - fSlow50))) 
		fSlow53 = (fSlow50 - fSlow51) 
		fSlow54 = jnp.exp(-((self._fConst25 / fSlow14))) 
		fSlow55 = ((jnp.exp(-((self._fConst25 / fSlow16))) / fSlow54) + jnp.float32(-1.0)) 
		fSlow56 = jnp.power(fSlow54, jnp.float32(2.0)) 
		fSlow57 = (jnp.float32(1.0) - fSlow56) 
		fSlow58 = (jnp.float32(1.0) - (fSlow20 * fSlow56)) 
		fSlow59 = (fSlow58 / fSlow57) 
		fSlow60 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow58, jnp.float32(2.0)) / jnp.power(fSlow57, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow61 = (fSlow54 * (fSlow60 + (jnp.float32(1.0) - fSlow59))) 
		fSlow62 = (fSlow59 - fSlow60) 
		fSlow63 = jnp.exp(-((self._fConst30 / fSlow14))) 
		fSlow64 = ((jnp.exp(-((self._fConst30 / fSlow16))) / fSlow63) + jnp.float32(-1.0)) 
		fSlow65 = jnp.power(fSlow63, jnp.float32(2.0)) 
		fSlow66 = (jnp.float32(1.0) - fSlow65) 
		fSlow67 = (jnp.float32(1.0) - (fSlow20 * fSlow65)) 
		fSlow68 = (fSlow67 / fSlow66) 
		fSlow69 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow67, jnp.float32(2.0)) / jnp.power(fSlow66, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow70 = (fSlow63 * (fSlow69 + (jnp.float32(1.0) - fSlow68))) 
		fSlow71 = (fSlow68 - fSlow69) 
		fSlow72 = jnp.exp(-((self._fConst35 / fSlow14))) 
		fSlow73 = ((jnp.exp(-((self._fConst35 / fSlow16))) / fSlow72) + jnp.float32(-1.0)) 
		fSlow74 = jnp.power(fSlow72, jnp.float32(2.0)) 
		fSlow75 = (jnp.float32(1.0) - fSlow74) 
		fSlow76 = (jnp.float32(1.0) - (fSlow20 * fSlow74)) 
		fSlow77 = (fSlow76 / fSlow75) 
		fSlow78 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow76, jnp.float32(2.0)) / jnp.power(fSlow75, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow79 = (fSlow72 * (fSlow78 + (jnp.float32(1.0) - fSlow77))) 
		fSlow80 = (fSlow77 - fSlow78) 
		fSlow81 = jnp.exp(-((self._fConst40 / fSlow14))) 
		fSlow82 = ((jnp.exp(-((self._fConst40 / fSlow16))) / fSlow81) + jnp.float32(-1.0)) 
		fSlow83 = jnp.power(fSlow81, jnp.float32(2.0)) 
		fSlow84 = (jnp.float32(1.0) - fSlow83) 
		fSlow85 = (jnp.float32(1.0) - (fSlow20 * fSlow83)) 
		fSlow86 = (fSlow85 / fSlow84) 
		fSlow87 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow85, jnp.float32(2.0)) / jnp.power(fSlow84, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow88 = (fSlow81 * (fSlow87 + (jnp.float32(1.0) - fSlow86))) 
		fSlow89 = (fSlow86 - fSlow87) 
		fSlow90 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider10"]))) 
		fRec0_temp = state["fRec0"] 
		fRec14_temp = state["fRec14"] 
		fRec13_temp = state["fRec13"] 
		fRec11_temp = state["fRec11"] 
		fRec18_temp = state["fRec18"] 
		fRec17_temp = state["fRec17"] 
		fRec15_temp = state["fRec15"] 
		fRec22_temp = state["fRec22"] 
		fRec21_temp = state["fRec21"] 
		fRec19_temp = state["fRec19"] 
		fRec26_temp = state["fRec26"] 
		fRec25_temp = state["fRec25"] 
		fRec23_temp = state["fRec23"] 
		fRec30_temp = state["fRec30"] 
		fRec29_temp = state["fRec29"] 
		fRec27_temp = state["fRec27"] 
		fRec34_temp = state["fRec34"] 
		fRec33_temp = state["fRec33"] 
		fRec31_temp = state["fRec31"] 
		fRec38_temp = state["fRec38"] 
		fRec37_temp = state["fRec37"] 
		fRec35_temp = state["fRec35"] 
		fRec42_temp = state["fRec42"] 
		fRec41_temp = state["fRec41"] 
		fRec39_temp = state["fRec39"] 
		fRec43_temp = state["fRec43"] 
		state["fRec0"] = (fSlow0 + (jnp.float32(0.999) * fRec0_temp)) 
		fTemp0 = (state["fRec0"] + jnp.float32(1.0)) 
		fTemp1 = (jnp.float32(1.0) - (jnp.float32(0.5) * fTemp0)) 
		fTemp2 = inputs[0] 
		state["fVec0"] = state["fVec0"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(fTemp2) 
		fTemp3 = (fSlow5 * state["fRec1"][1]) 
		fTemp4 = (fSlow10 * state["fRec2"][1]) 
		state["fRec14"] = -((fSlow13 * ((fSlow12 * fRec14_temp) - (state["fRec7"][1] + state["fRec7"][2])))) 
		state["fRec13"] = ((fSlow25 * fRec13_temp) + (fSlow24 * (state["fRec7"][1] + (fSlow17 * state["fRec14"])))) 
		state["fVec1"] = state["fVec1"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec13"]) + jnp.float32(1e-20))) 
		fTemp5 = (jnp.float32(0.3) * state["fVec0"][((state["IOTA0"] - iSlow26) & 8191).astype(jnp.int32)]) 
		fTemp6 = ((fTemp5 + state["fVec1"][((state["IOTA0"] - self._iConst6) & 8191).astype(jnp.int32)]) - (jnp.float32(0.6) * fRec11_temp)) 
		state["fVec2"] = state["fVec2"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp6) 
		state["fRec11"] = state["fVec2"][((state["IOTA0"] - self._iConst8) & 2047).astype(jnp.int32)] 
		fRec12 = (jnp.float32(0.6) * fTemp6) 
		state["fRec18"] = -((fSlow13 * ((fSlow12 * fRec18_temp) - (state["fRec3"][1] + state["fRec3"][2])))) 
		state["fRec17"] = ((fSlow35 * fRec17_temp) + (fSlow34 * (state["fRec3"][1] + (fSlow28 * state["fRec18"])))) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec17"]) + jnp.float32(1e-20))) 
		fTemp7 = ((state["fVec3"][((state["IOTA0"] - self._iConst12) & 8191).astype(jnp.int32)] + fTemp5) - (jnp.float32(0.6) * fRec15_temp)) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(fTemp7) 
		state["fRec15"] = state["fVec4"][((state["IOTA0"] - self._iConst13) & 1023).astype(jnp.int32)] 
		fRec16 = (jnp.float32(0.6) * fTemp7) 
		fTemp8 = (fRec16 + fRec12) 
		state["fRec22"] = -((fSlow13 * ((fSlow12 * fRec22_temp) - (state["fRec5"][1] + state["fRec5"][2])))) 
		state["fRec21"] = ((fSlow44 * fRec21_temp) + (fSlow43 * (state["fRec5"][1] + (fSlow37 * state["fRec22"])))) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec21"]) + jnp.float32(1e-20))) 
		fTemp9 = (state["fVec5"][((state["IOTA0"] - self._iConst17) & 8191).astype(jnp.int32)] - (fTemp5 + (jnp.float32(0.6) * fRec19_temp))) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp9) 
		state["fRec19"] = state["fVec6"][((state["IOTA0"] - self._iConst18) & 2047).astype(jnp.int32)] 
		fRec20 = (jnp.float32(0.6) * fTemp9) 
		state["fRec26"] = -((fSlow13 * ((fSlow12 * fRec26_temp) - (state["fRec9"][1] + state["fRec9"][2])))) 
		state["fRec25"] = ((fSlow53 * fRec25_temp) + (fSlow52 * (state["fRec9"][1] + (fSlow46 * state["fRec26"])))) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec25"]) + jnp.float32(1e-20))) 
		fTemp10 = (state["fVec7"][((state["IOTA0"] - self._iConst22) & 8191).astype(jnp.int32)] - (fTemp5 + (jnp.float32(0.6) * fRec23_temp))) 
		state["fVec8"] = state["fVec8"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(fTemp10) 
		state["fRec23"] = state["fVec8"][((state["IOTA0"] - self._iConst23) & 1023).astype(jnp.int32)] 
		fRec24 = (jnp.float32(0.6) * fTemp10) 
		fTemp11 = (fRec24 + (fRec20 + fTemp8)) 
		state["fRec30"] = -((fSlow13 * ((fSlow12 * fRec30_temp) - (state["fRec4"][1] + state["fRec4"][2])))) 
		state["fRec29"] = ((fSlow62 * fRec29_temp) + (fSlow61 * (state["fRec4"][1] + (fSlow55 * state["fRec30"])))) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec29"]) + jnp.float32(1e-20))) 
		fTemp12 = inputs[1] 
		state["fVec10"] = state["fVec10"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(fTemp12) 
		fTemp13 = (jnp.float32(0.3) * state["fVec10"][((state["IOTA0"] - iSlow26) & 8191).astype(jnp.int32)]) 
		fTemp14 = (fTemp13 + ((jnp.float32(0.6) * fRec27_temp) + state["fVec9"][((state["IOTA0"] - self._iConst27) & 16383).astype(jnp.int32)])) 
		state["fVec11"] = state["fVec11"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp14) 
		state["fRec27"] = state["fVec11"][((state["IOTA0"] - self._iConst28) & 2047).astype(jnp.int32)] 
		fRec28 = -((jnp.float32(0.6) * fTemp14)) 
		state["fRec34"] = -((fSlow13 * ((fSlow12 * fRec34_temp) - (state["fRec8"][1] + state["fRec8"][2])))) 
		state["fRec33"] = ((fSlow71 * fRec33_temp) + (fSlow70 * (state["fRec8"][1] + (fSlow64 * state["fRec34"])))) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec33"]) + jnp.float32(1e-20))) 
		fTemp15 = (state["fVec12"][((state["IOTA0"] - self._iConst32) & 8191).astype(jnp.int32)] + (fTemp13 + (jnp.float32(0.6) * fRec31_temp))) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp15) 
		state["fRec31"] = state["fVec13"][((state["IOTA0"] - self._iConst33) & 2047).astype(jnp.int32)] 
		fRec32 = -((jnp.float32(0.6) * fTemp15)) 
		state["fRec38"] = -((fSlow13 * ((fSlow12 * fRec38_temp) - (state["fRec6"][1] + state["fRec6"][2])))) 
		state["fRec37"] = ((fSlow80 * fRec37_temp) + (fSlow79 * (state["fRec6"][1] + (fSlow73 * state["fRec38"])))) 
		state["fVec14"] = state["fVec14"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec37"]) + jnp.float32(1e-20))) 
		fTemp16 = ((jnp.float32(0.6) * fRec35_temp) + state["fVec14"][((state["IOTA0"] - self._iConst37) & 16383).astype(jnp.int32)]) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set((fTemp16 - fTemp13)) 
		state["fRec35"] = state["fVec15"][((state["IOTA0"] - self._iConst38) & 2047).astype(jnp.int32)] 
		fRec36 = (jnp.float32(0.6) * (fTemp13 - fTemp16)) 
		state["fRec42"] = -((fSlow13 * ((fSlow12 * fRec42_temp) - (state["fRec10"][1] + state["fRec10"][2])))) 
		state["fRec41"] = ((fSlow89 * fRec41_temp) + (fSlow88 * (state["fRec10"][1] + (fSlow82 * state["fRec42"])))) 
		state["fVec16"] = state["fVec16"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec41"]) + jnp.float32(1e-20))) 
		fTemp17 = ((jnp.float32(0.6) * fRec39_temp) + state["fVec16"][((state["IOTA0"] - self._iConst42) & 16383).astype(jnp.int32)]) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp17 - fTemp13)) 
		state["fRec39"] = state["fVec17"][((state["IOTA0"] - self._iConst43) & 1023).astype(jnp.int32)] 
		fRec40 = (jnp.float32(0.6) * (fTemp13 - fTemp17)) 
		state["fRec3"] = state["fRec3"].at[0].set((fRec39_temp + (fRec35_temp + (fRec31_temp + (fRec27_temp + (fRec23_temp + (fRec19_temp + (fRec11_temp + (fRec15_temp + (fRec40 + (fRec36 + (fRec32 + (fRec28 + fTemp11))))))))))))) 
		state["fRec4"] = state["fRec4"].at[0].set(((fRec23_temp + (fRec19_temp + (fRec11_temp + (fRec15_temp + fTemp11)))) - (fRec39_temp + (fRec35_temp + (fRec31_temp + (fRec27_temp + (fRec40 + (fRec36 + (fRec28 + fRec32))))))))) 
		fTemp18 = (fRec20 + fRec24) 
		state["fRec5"] = state["fRec5"].at[0].set(((fRec31_temp + (fRec27_temp + (fRec11_temp + (fRec15_temp + (fRec32 + (fRec28 + fTemp8)))))) - (fRec39_temp + (fRec35_temp + (fRec23_temp + (fRec19_temp + (fRec40 + (fRec36 + fTemp18)))))))) 
		state["fRec6"] = state["fRec6"].at[0].set(((fRec39_temp + (fRec35_temp + (fRec11_temp + (fRec15_temp + (fRec40 + (fRec36 + fTemp8)))))) - (fRec31_temp + (fRec27_temp + (fRec23_temp + (fRec19_temp + (fRec32 + (fRec28 + fTemp18)))))))) 
		fTemp19 = (fRec12 + fRec24) 
		fTemp20 = (fRec16 + fRec20) 
		state["fRec7"] = state["fRec7"].at[0].set(((fRec35_temp + (fRec27_temp + (fRec19_temp + (fRec15_temp + (fRec36 + (fRec28 + fTemp20)))))) - (fRec39_temp + (fRec31_temp + (fRec23_temp + (fRec11_temp + (fRec40 + (fRec32 + fTemp19)))))))) 
		state["fRec8"] = state["fRec8"].at[0].set(((fRec39_temp + (fRec31_temp + (fRec19_temp + (fRec15_temp + (fRec40 + (fRec32 + fTemp20)))))) - (fRec35_temp + (fRec27_temp + (fRec23_temp + (fRec11_temp + (fRec36 + (fRec28 + fTemp19)))))))) 
		fTemp21 = (fRec12 + fRec20) 
		fTemp22 = (fRec16 + fRec24) 
		state["fRec9"] = state["fRec9"].at[0].set(((fRec39_temp + (fRec27_temp + (fRec23_temp + (fRec15_temp + (fRec40 + (fRec28 + fTemp22)))))) - (fRec35_temp + (fRec31_temp + (fRec19_temp + (fRec11_temp + (fRec36 + (fRec32 + fTemp21)))))))) 
		state["fRec10"] = state["fRec10"].at[0].set(((fRec35_temp + (fRec31_temp + (fRec23_temp + (fRec15_temp + (fRec36 + (fRec32 + fTemp22)))))) - (fRec39_temp + (fRec27_temp + (fRec19_temp + (fRec11_temp + (fRec40 + (fRec28 + fTemp21)))))))) 
		fTemp23 = (jnp.float32(0.37) * (state["fRec4"][0] + state["fRec5"][0])) 
		fTemp24 = (fTemp23 + fTemp4) 
		state["fRec2"] = state["fRec2"].at[0].set((fTemp24 - (fSlow9 * state["fRec2"][2]))) 
		fTemp25 = (fSlow9 * state["fRec2"][0]) 
		fTemp26 = (jnp.float32(0.5) * (((fTemp25 + (fTemp23 + state["fRec2"][2])) - fTemp4) + (fSlow6 * ((state["fRec2"][2] + fTemp25) - fTemp24)))) 
		fTemp27 = (fTemp26 + fTemp3) 
		state["fRec1"] = state["fRec1"].at[0].set((fTemp27 - (fSlow4 * state["fRec1"][2]))) 
		fTemp28 = (fSlow4 * state["fRec1"][0]) 
		state["fRec43"] = (fSlow90 + (jnp.float32(0.999) * fRec43_temp)) 
		_result0 = (state["fRec43"] * ((jnp.float32(0.25) * (fTemp0 * (((fTemp28 + (fTemp26 + state["fRec1"][2])) - fTemp3) + (fSlow1 * ((state["fRec1"][2] + fTemp28) - fTemp27))))) + (fTemp2 * fTemp1))) 
		fTemp29 = (fSlow5 * state["fRec44"][1]) 
		fTemp30 = (fSlow10 * state["fRec45"][1]) 
		fTemp31 = (jnp.float32(0.37) * (state["fRec4"][0] - state["fRec5"][0])) 
		fTemp32 = (fTemp31 + fTemp30) 
		state["fRec45"] = state["fRec45"].at[0].set((fTemp32 - (fSlow9 * state["fRec45"][2]))) 
		fTemp33 = (fSlow9 * state["fRec45"][0]) 
		fTemp34 = (jnp.float32(0.5) * (((fTemp33 + (fTemp31 + state["fRec45"][2])) - fTemp30) + (fSlow6 * ((state["fRec45"][2] + fTemp33) - fTemp32)))) 
		fTemp35 = (fTemp34 + fTemp29) 
		state["fRec44"] = state["fRec44"].at[0].set((fTemp35 - (fSlow4 * state["fRec44"][2]))) 
		fTemp36 = (fSlow4 * state["fRec44"][0]) 
		_result1 = (state["fRec43"] * ((jnp.float32(0.25) * (fTemp0 * (((fTemp36 + (fTemp34 + state["fRec44"][2])) - fTemp29) + (fSlow1 * ((state["fRec44"][2] + fTemp36) - fTemp35))))) + (fTemp12 * fTemp1))) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec10"] = jnp.roll(state["fRec10"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
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
