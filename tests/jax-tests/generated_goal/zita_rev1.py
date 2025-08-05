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
		ui_path.append("Zita_Rev1") 
		ui_path.append("Input") 
		self.add_vslider("fVslider10", ui_path, "In Delay", 6e+01, 2e+01, 1e+02, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Decay Times in Bands (see tooltips)") 
		self.add_vslider("fVslider9", ui_path, "LF X", 2e+02, 5e+01, 1e+03, unnorm_funcs, "log") 
		self.add_vslider("fVslider8", ui_path, "Low RT60", 3.0, 1.0, 8.0, unnorm_funcs, "log") 
		self.add_vslider("fVslider7", ui_path, "Mid RT60", 2.0, 1.0, 8.0, unnorm_funcs, "log") 
		self.add_vslider("fVslider6", ui_path, "HF Damping", 6e+03, 1.5e+03, 2.352e+04, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("RM Peaking Equalizer 1") 
		self.add_vslider("fVslider4", ui_path, "Eq1 Freq", 315.0, 4e+01, 2.5e+03, unnorm_funcs, "log") 
		self.add_vslider("fVslider5", ui_path, "Eq1 Level", 0.0, -15.0, 15.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("RM Peaking Equalizer 2") 
		self.add_vslider("fVslider2", ui_path, "Eq2 Freq", 1.5e+03, 1.6e+02, 1e+04, unnorm_funcs, "log") 
		self.add_vslider("fVslider3", ui_path, "Eq2 Level", 0.0, -15.0, 15.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Output") 
		self.add_vslider("fVslider1", ui_path, "Dry/Wet Mix", 0.4492, -1.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider0", ui_path, "Level", 16.79, -7e+01, 4e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = (np.float32(6.2831855) / self._fConst0) 
		self._fConst2 = np.floor(((np.float32(0.219991) * self._fConst0) + np.float32(0.5))) 
		self._fConst3 = (np.float32(6.9077554) * (self._fConst2 / self._fConst0)) 
		self._fConst4 = (np.float32(3.1415927) / self._fConst0) 
		self._fConst5 = np.floor(((np.float32(0.019123) * self._fConst0) + np.float32(0.5))) 
		self._iConst6 = (np.int32((self._fConst2 - self._fConst5)) & np.int32(16383)).astype(jnp.int32) 
		self._fConst7 = (np.float32(0.001) * self._fConst0) 
		self._iConst8 = (np.int32((self._fConst5 + np.float32(-1.0))) & np.int32(1023)).astype(jnp.int32) 
		self._fConst9 = np.floor(((np.float32(0.256891) * self._fConst0) + np.float32(0.5))) 
		self._fConst10 = (np.float32(6.9077554) * (self._fConst9 / self._fConst0)) 
		self._fConst11 = np.floor(((np.float32(0.027333) * self._fConst0) + np.float32(0.5))) 
		self._iConst12 = (np.int32((self._fConst9 - self._fConst11)) & np.int32(16383)).astype(jnp.int32) 
		self._iConst13 = (np.int32((self._fConst11 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst14 = np.floor(((np.float32(0.192303) * self._fConst0) + np.float32(0.5))) 
		self._fConst15 = (np.float32(6.9077554) * (self._fConst14 / self._fConst0)) 
		self._fConst16 = np.floor(((np.float32(0.029291) * self._fConst0) + np.float32(0.5))) 
		self._iConst17 = (np.int32((self._fConst14 - self._fConst16)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst18 = (np.int32((self._fConst16 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst19 = np.floor(((np.float32(0.210389) * self._fConst0) + np.float32(0.5))) 
		self._fConst20 = (np.float32(6.9077554) * (self._fConst19 / self._fConst0)) 
		self._fConst21 = np.floor(((np.float32(0.024421) * self._fConst0) + np.float32(0.5))) 
		self._iConst22 = (np.int32((self._fConst19 - self._fConst21)) & np.int32(16383)).astype(jnp.int32) 
		self._iConst23 = (np.int32((self._fConst21 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst24 = np.floor(((np.float32(0.125) * self._fConst0) + np.float32(0.5))) 
		self._fConst25 = (np.float32(6.9077554) * (self._fConst24 / self._fConst0)) 
		self._fConst26 = np.floor(((np.float32(0.013458) * self._fConst0) + np.float32(0.5))) 
		self._iConst27 = (np.int32((self._fConst24 - self._fConst26)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst28 = (np.int32((self._fConst26 + np.float32(-1.0))) & np.int32(1023)).astype(jnp.int32) 
		self._fConst29 = np.floor(((np.float32(0.127837) * self._fConst0) + np.float32(0.5))) 
		self._fConst30 = (np.float32(6.9077554) * (self._fConst29 / self._fConst0)) 
		self._fConst31 = np.floor(((np.float32(0.031604) * self._fConst0) + np.float32(0.5))) 
		self._iConst32 = (np.int32((self._fConst29 - self._fConst31)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst33 = (np.int32((self._fConst31 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst34 = np.floor(((np.float32(0.174713) * self._fConst0) + np.float32(0.5))) 
		self._fConst35 = (np.float32(6.9077554) * (self._fConst34 / self._fConst0)) 
		self._fConst36 = np.floor(((np.float32(0.022904) * self._fConst0) + np.float32(0.5))) 
		self._iConst37 = (np.int32((self._fConst34 - self._fConst36)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst38 = (np.int32((self._fConst36 + np.float32(-1.0))) & np.int32(2047)).astype(jnp.int32) 
		self._fConst39 = np.floor(((np.float32(0.153129) * self._fConst0) + np.float32(0.5))) 
		self._fConst40 = (np.float32(6.9077554) * (self._fConst39 / self._fConst0)) 
		self._fConst41 = np.floor(((np.float32(0.020346) * self._fConst0) + np.float32(0.5))) 
		self._iConst42 = (np.int32((self._fConst39 - self._fConst41)) & np.int32(8191)).astype(jnp.int32) 
		self._iConst43 = (np.int32((self._fConst41 + np.float32(-1.0))) & np.int32(1023)).astype(jnp.int32) 
		
	@property
	def num_inputs(self):
		return 2
	
	@property
	def num_outputs(self):
		return 2
	
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec1"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec16"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec20"] = np.float32(0)
		state["fRec22"] = np.float32(0)
		state["fRec23"] = np.float32(0)
		state["fRec24"] = np.float32(0)
		state["fRec26"] = np.float32(0)
		state["fRec27"] = np.float32(0)
		state["fRec28"] = np.float32(0)
		state["fRec30"] = np.float32(0)
		state["fRec31"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec34"] = np.float32(0)
		state["fRec35"] = np.float32(0)
		state["fRec36"] = np.float32(0)
		state["fRec38"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec40"] = np.float32(0)
		state["fRec42"] = np.float32(0)
		state["fRec43"] = np.float32(0)
		# Initialize array delays
		state["fVec0"] = np.zeros((16384,), dtype=np.float32)
		state["fVec1"] = np.zeros((8192,), dtype=np.float32)
		state["fVec2"] = np.zeros((1024,), dtype=np.float32)
		state["fVec3"] = np.zeros((16384,), dtype=np.float32)
		state["fVec4"] = np.zeros((2048,), dtype=np.float32)
		state["fVec5"] = np.zeros((8192,), dtype=np.float32)
		state["fVec6"] = np.zeros((2048,), dtype=np.float32)
		state["fVec7"] = np.zeros((16384,), dtype=np.float32)
		state["fVec8"] = np.zeros((2048,), dtype=np.float32)
		state["fVec9"] = np.zeros((8192,), dtype=np.float32)
		state["fVec10"] = np.zeros((8192,), dtype=np.float32)
		state["fVec11"] = np.zeros((1024,), dtype=np.float32)
		state["fVec12"] = np.zeros((8192,), dtype=np.float32)
		state["fVec13"] = np.zeros((2048,), dtype=np.float32)
		state["fVec14"] = np.zeros((8192,), dtype=np.float32)
		state["fVec15"] = np.zeros((2048,), dtype=np.float32)
		state["fVec16"] = np.zeros((8192,), dtype=np.float32)
		state["fVec17"] = np.zeros((1024,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fRec11"] = np.zeros((3,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider0"]))) 
		fSlow1 = (jnp.float32(0.001) * params["fVslider1"]) 
		fSlow2 = params["fVslider2"] 
		fSlow3 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider3"])) 
		fSlow4 = (self._fConst1 * (fSlow2 / jnp.sqrt(jnp.maximum(jnp.float32(0.0), fSlow3)))) 
		fSlow5 = ((jnp.float32(1.0) - fSlow4) / (fSlow4 + jnp.float32(1.0))) 
		fSlow6 = params["fVslider4"] 
		fSlow7 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider5"])) 
		fSlow8 = (self._fConst1 * (fSlow6 / jnp.sqrt(jnp.maximum(jnp.float32(0.0), fSlow7)))) 
		fSlow9 = ((jnp.float32(1.0) - fSlow8) / (fSlow8 + jnp.float32(1.0))) 
		fSlow10 = jnp.cos((self._fConst1 * params["fVslider6"])) 
		fSlow11 = params["fVslider7"] 
		fSlow12 = jnp.exp(-((self._fConst3 / fSlow11))) 
		fSlow13 = jnp.power(fSlow12, jnp.float32(2.0)) 
		fSlow14 = (jnp.float32(1.0) - (fSlow10 * fSlow13)) 
		fSlow15 = (jnp.float32(1.0) - fSlow13) 
		fSlow16 = (fSlow14 / fSlow15) 
		fSlow17 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow14, jnp.float32(2.0)) / jnp.power(fSlow15, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow18 = (fSlow16 - fSlow17) 
		fSlow19 = (fSlow12 * (fSlow17 + (jnp.float32(1.0) - fSlow16))) 
		fSlow20 = params["fVslider8"] 
		fSlow21 = ((jnp.exp(-((self._fConst3 / fSlow20))) / fSlow12) + jnp.float32(-1.0)) 
		fSlow22 = (jnp.float32(1.0) / jnp.tan((self._fConst4 * params["fVslider9"]))) 
		fSlow23 = (jnp.float32(1.0) / (fSlow22 + jnp.float32(1.0))) 
		fSlow24 = (jnp.float32(1.0) - fSlow22) 
		iSlow25 = (jnp.int32((self._fConst7 * params["fVslider10"])) & jnp.int32(8191)).astype(jnp.int32) 
		fSlow26 = jnp.exp(-((self._fConst10 / fSlow11))) 
		fSlow27 = jnp.power(fSlow26, jnp.float32(2.0)) 
		fSlow28 = (jnp.float32(1.0) - (fSlow10 * fSlow27)) 
		fSlow29 = (jnp.float32(1.0) - fSlow27) 
		fSlow30 = (fSlow28 / fSlow29) 
		fSlow31 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow28, jnp.float32(2.0)) / jnp.power(fSlow29, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow32 = (fSlow30 - fSlow31) 
		fSlow33 = (fSlow26 * (fSlow31 + (jnp.float32(1.0) - fSlow30))) 
		fSlow34 = ((jnp.exp(-((self._fConst10 / fSlow20))) / fSlow26) + jnp.float32(-1.0)) 
		fSlow35 = jnp.exp(-((self._fConst15 / fSlow11))) 
		fSlow36 = jnp.power(fSlow35, jnp.float32(2.0)) 
		fSlow37 = (jnp.float32(1.0) - (fSlow10 * fSlow36)) 
		fSlow38 = (jnp.float32(1.0) - fSlow36) 
		fSlow39 = (fSlow37 / fSlow38) 
		fSlow40 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow37, jnp.float32(2.0)) / jnp.power(fSlow38, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow41 = (fSlow39 - fSlow40) 
		fSlow42 = (fSlow35 * (fSlow40 + (jnp.float32(1.0) - fSlow39))) 
		fSlow43 = ((jnp.exp(-((self._fConst15 / fSlow20))) / fSlow35) + jnp.float32(-1.0)) 
		fSlow44 = jnp.exp(-((self._fConst20 / fSlow11))) 
		fSlow45 = jnp.power(fSlow44, jnp.float32(2.0)) 
		fSlow46 = (jnp.float32(1.0) - (fSlow10 * fSlow45)) 
		fSlow47 = (jnp.float32(1.0) - fSlow45) 
		fSlow48 = (fSlow46 / fSlow47) 
		fSlow49 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow46, jnp.float32(2.0)) / jnp.power(fSlow47, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow50 = (fSlow48 - fSlow49) 
		fSlow51 = (fSlow44 * (fSlow49 + (jnp.float32(1.0) - fSlow48))) 
		fSlow52 = ((jnp.exp(-((self._fConst20 / fSlow20))) / fSlow44) + jnp.float32(-1.0)) 
		fSlow53 = jnp.exp(-((self._fConst25 / fSlow11))) 
		fSlow54 = jnp.power(fSlow53, jnp.float32(2.0)) 
		fSlow55 = (jnp.float32(1.0) - (fSlow10 * fSlow54)) 
		fSlow56 = (jnp.float32(1.0) - fSlow54) 
		fSlow57 = (fSlow55 / fSlow56) 
		fSlow58 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow55, jnp.float32(2.0)) / jnp.power(fSlow56, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow59 = (fSlow57 - fSlow58) 
		fSlow60 = (fSlow53 * (fSlow58 + (jnp.float32(1.0) - fSlow57))) 
		fSlow61 = ((jnp.exp(-((self._fConst25 / fSlow20))) / fSlow53) + jnp.float32(-1.0)) 
		fSlow62 = jnp.exp(-((self._fConst30 / fSlow11))) 
		fSlow63 = jnp.power(fSlow62, jnp.float32(2.0)) 
		fSlow64 = (jnp.float32(1.0) - (fSlow10 * fSlow63)) 
		fSlow65 = (jnp.float32(1.0) - fSlow63) 
		fSlow66 = (fSlow64 / fSlow65) 
		fSlow67 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow64, jnp.float32(2.0)) / jnp.power(fSlow65, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow68 = (fSlow66 - fSlow67) 
		fSlow69 = (fSlow62 * (fSlow67 + (jnp.float32(1.0) - fSlow66))) 
		fSlow70 = ((jnp.exp(-((self._fConst30 / fSlow20))) / fSlow62) + jnp.float32(-1.0)) 
		fSlow71 = jnp.exp(-((self._fConst35 / fSlow11))) 
		fSlow72 = jnp.power(fSlow71, jnp.float32(2.0)) 
		fSlow73 = (jnp.float32(1.0) - (fSlow10 * fSlow72)) 
		fSlow74 = (jnp.float32(1.0) - fSlow72) 
		fSlow75 = (fSlow73 / fSlow74) 
		fSlow76 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow73, jnp.float32(2.0)) / jnp.power(fSlow74, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow77 = (fSlow75 - fSlow76) 
		fSlow78 = (fSlow71 * (fSlow76 + (jnp.float32(1.0) - fSlow75))) 
		fSlow79 = ((jnp.exp(-((self._fConst35 / fSlow20))) / fSlow71) + jnp.float32(-1.0)) 
		fSlow80 = jnp.exp(-((self._fConst40 / fSlow11))) 
		fSlow81 = jnp.power(fSlow80, jnp.float32(2.0)) 
		fSlow82 = (jnp.float32(1.0) - (fSlow81 * fSlow10)) 
		fSlow83 = (jnp.float32(1.0) - fSlow81) 
		fSlow84 = (fSlow82 / fSlow83) 
		fSlow85 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), ((jnp.power(fSlow82, jnp.float32(2.0)) / jnp.power(fSlow83, jnp.float32(2.0))) + jnp.float32(-1.0)))) 
		fSlow86 = (fSlow84 - fSlow85) 
		fSlow87 = (fSlow80 * (fSlow85 + (jnp.float32(1.0) - fSlow84))) 
		fSlow88 = ((jnp.exp(-((self._fConst40 / fSlow20))) / fSlow80) + jnp.float32(-1.0)) 
		fSlow89 = (jnp.cos((self._fConst1 * fSlow6)) * (fSlow9 + jnp.float32(1.0))) 
		fSlow90 = (jnp.cos((self._fConst1 * fSlow2)) * (fSlow5 + jnp.float32(1.0))) 
		fRec0_temp = state["fRec0"] 
		fRec1_temp = state["fRec1"] 
		fRec15_temp = state["fRec15"] 
		fRec14_temp = state["fRec14"] 
		fRec12_temp = state["fRec12"] 
		fRec19_temp = state["fRec19"] 
		fRec18_temp = state["fRec18"] 
		fRec16_temp = state["fRec16"] 
		fRec23_temp = state["fRec23"] 
		fRec22_temp = state["fRec22"] 
		fRec20_temp = state["fRec20"] 
		fRec27_temp = state["fRec27"] 
		fRec26_temp = state["fRec26"] 
		fRec24_temp = state["fRec24"] 
		fRec31_temp = state["fRec31"] 
		fRec30_temp = state["fRec30"] 
		fRec28_temp = state["fRec28"] 
		fRec35_temp = state["fRec35"] 
		fRec34_temp = state["fRec34"] 
		fRec32_temp = state["fRec32"] 
		fRec39_temp = state["fRec39"] 
		fRec38_temp = state["fRec38"] 
		fRec36_temp = state["fRec36"] 
		fRec43_temp = state["fRec43"] 
		fRec42_temp = state["fRec42"] 
		fRec40_temp = state["fRec40"] 
		state["fRec0"] = (fSlow0 + (jnp.float32(0.999) * fRec0_temp)) 
		state["fRec1"] = (fSlow1 + (jnp.float32(0.999) * fRec1_temp)) 
		fTemp0 = (state["fRec1"] + jnp.float32(1.0)) 
		state["fRec15"] = -((fSlow23 * ((fSlow24 * fRec15_temp) - (state["fRec11"][1] + state["fRec11"][2])))) 
		state["fRec14"] = ((fSlow18 * fRec14_temp) + (fSlow19 * (state["fRec11"][1] + (fSlow21 * state["fRec15"])))) 
		state["fVec0"] = state["fVec0"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec14"]) + jnp.float32(1e-20))) 
		fTemp1 = ((jnp.float32(0.6) * fRec12_temp) + state["fVec0"][((state["IOTA0"] - self._iConst6) & 16383).astype(jnp.int32)]) 
		fTemp2 = inputs[1] 
		state["fVec1"] = state["fVec1"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(fTemp2) 
		fTemp3 = (jnp.float32(0.3) * state["fVec1"][((state["IOTA0"] - iSlow25) & 8191).astype(jnp.int32)]) 
		state["fVec2"] = state["fVec2"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp1 - fTemp3)) 
		state["fRec12"] = state["fVec2"][((state["IOTA0"] - self._iConst8) & 1023).astype(jnp.int32)] 
		fRec13 = (jnp.float32(0.6) * (fTemp3 - fTemp1)) 
		state["fRec19"] = -((fSlow23 * ((fSlow24 * fRec19_temp) - (state["fRec7"][1] + state["fRec7"][2])))) 
		state["fRec18"] = ((fSlow32 * fRec18_temp) + (fSlow33 * (state["fRec7"][1] + (fSlow34 * state["fRec19"])))) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec18"]) + jnp.float32(1e-20))) 
		fTemp4 = ((jnp.float32(0.6) * fRec16_temp) + state["fVec3"][((state["IOTA0"] - self._iConst12) & 16383).astype(jnp.int32)]) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set((fTemp4 - fTemp3)) 
		state["fRec16"] = state["fVec4"][((state["IOTA0"] - self._iConst13) & 2047).astype(jnp.int32)] 
		fRec17 = (jnp.float32(0.6) * (fTemp3 - fTemp4)) 
		state["fRec23"] = -((fSlow23 * ((fSlow24 * fRec23_temp) - (state["fRec9"][1] + state["fRec9"][2])))) 
		state["fRec22"] = ((fSlow41 * fRec22_temp) + (fSlow42 * (state["fRec9"][1] + (fSlow43 * state["fRec23"])))) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec22"]) + jnp.float32(1e-20))) 
		fTemp5 = (state["fVec5"][((state["IOTA0"] - self._iConst17) & 8191).astype(jnp.int32)] + (fTemp3 + (jnp.float32(0.6) * fRec20_temp))) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp5) 
		state["fRec20"] = state["fVec6"][((state["IOTA0"] - self._iConst18) & 2047).astype(jnp.int32)] 
		fRec21 = -((jnp.float32(0.6) * fTemp5)) 
		state["fRec27"] = -((fSlow23 * ((fSlow24 * fRec27_temp) - (state["fRec5"][1] + state["fRec5"][2])))) 
		state["fRec26"] = ((fSlow50 * fRec26_temp) + (fSlow51 * (state["fRec5"][1] + (fSlow52 * state["fRec27"])))) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 16383).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec26"]) + jnp.float32(1e-20))) 
		fTemp6 = (fTemp3 + ((jnp.float32(0.6) * fRec24_temp) + state["fVec7"][((state["IOTA0"] - self._iConst22) & 16383).astype(jnp.int32)])) 
		state["fVec8"] = state["fVec8"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp6) 
		state["fRec24"] = state["fVec8"][((state["IOTA0"] - self._iConst23) & 2047).astype(jnp.int32)] 
		fRec25 = -((jnp.float32(0.6) * fTemp6)) 
		state["fRec31"] = -((fSlow23 * ((fSlow24 * fRec31_temp) - (state["fRec10"][1] + state["fRec10"][2])))) 
		state["fRec30"] = ((fSlow59 * fRec30_temp) + (fSlow60 * (state["fRec10"][1] + (fSlow61 * state["fRec31"])))) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec30"]) + jnp.float32(1e-20))) 
		fTemp7 = inputs[0] 
		state["fVec10"] = state["fVec10"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(fTemp7) 
		fTemp8 = (jnp.float32(0.3) * state["fVec10"][((state["IOTA0"] - iSlow25) & 8191).astype(jnp.int32)]) 
		fTemp9 = (state["fVec9"][((state["IOTA0"] - self._iConst27) & 8191).astype(jnp.int32)] - (fTemp8 + (jnp.float32(0.6) * fRec28_temp))) 
		state["fVec11"] = state["fVec11"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(fTemp9) 
		state["fRec28"] = state["fVec11"][((state["IOTA0"] - self._iConst28) & 1023).astype(jnp.int32)] 
		fRec29 = (jnp.float32(0.6) * fTemp9) 
		state["fRec35"] = -((fSlow23 * ((fSlow24 * fRec35_temp) - (state["fRec6"][1] + state["fRec6"][2])))) 
		state["fRec34"] = ((fSlow68 * fRec34_temp) + (fSlow69 * (state["fRec6"][1] + (fSlow70 * state["fRec35"])))) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec34"]) + jnp.float32(1e-20))) 
		fTemp10 = (state["fVec12"][((state["IOTA0"] - self._iConst32) & 8191).astype(jnp.int32)] - (fTemp8 + (jnp.float32(0.6) * fRec32_temp))) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp10) 
		state["fRec32"] = state["fVec13"][((state["IOTA0"] - self._iConst33) & 2047).astype(jnp.int32)] 
		fRec33 = (jnp.float32(0.6) * fTemp10) 
		state["fRec39"] = -((fSlow23 * ((fSlow24 * fRec39_temp) - (state["fRec8"][1] + state["fRec8"][2])))) 
		state["fRec38"] = ((fSlow77 * fRec38_temp) + (fSlow78 * (state["fRec8"][1] + (fSlow79 * state["fRec39"])))) 
		state["fVec14"] = state["fVec14"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec38"]) + jnp.float32(1e-20))) 
		fTemp11 = ((fTemp8 + state["fVec14"][((state["IOTA0"] - self._iConst37) & 8191).astype(jnp.int32)]) - (jnp.float32(0.6) * fRec36_temp)) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 2047).astype(jnp.int32)].set(fTemp11) 
		state["fRec36"] = state["fVec15"][((state["IOTA0"] - self._iConst38) & 2047).astype(jnp.int32)] 
		fRec37 = (jnp.float32(0.6) * fTemp11) 
		state["fRec43"] = -((fSlow23 * ((fSlow24 * fRec43_temp) - (state["fRec4"][1] + state["fRec4"][2])))) 
		state["fRec42"] = ((fSlow86 * fRec42_temp) + (fSlow87 * (state["fRec4"][1] + (fSlow88 * state["fRec43"])))) 
		state["fVec16"] = state["fVec16"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set(((jnp.float32(0.35355338) * state["fRec42"]) + jnp.float32(1e-20))) 
		fTemp12 = ((state["fVec16"][((state["IOTA0"] - self._iConst42) & 8191).astype(jnp.int32)] + fTemp8) - (jnp.float32(0.6) * fRec40_temp)) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(fTemp12) 
		state["fRec40"] = state["fVec17"][((state["IOTA0"] - self._iConst43) & 1023).astype(jnp.int32)] 
		fRec41 = (jnp.float32(0.6) * fTemp12) 
		fTemp13 = (fRec41 + fRec37) 
		fTemp14 = (fRec29 + (fRec33 + fTemp13)) 
		state["fRec4"] = state["fRec4"].at[0].set((fRec12_temp + (fRec16_temp + (fRec20_temp + (fRec24_temp + (fRec28_temp + (fRec32_temp + (fRec36_temp + (fRec40_temp + (fRec13 + (fRec17 + (fRec21 + (fRec25 + fTemp14))))))))))))) 
		state["fRec5"] = state["fRec5"].at[0].set(((fRec28_temp + (fRec32_temp + (fRec36_temp + (fRec40_temp + fTemp14)))) - (fRec12_temp + (fRec16_temp + (fRec20_temp + (fRec24_temp + (fRec13 + (fRec17 + (fRec25 + fRec21))))))))) 
		fTemp15 = (fRec33 + fRec29) 
		state["fRec6"] = state["fRec6"].at[0].set(((fRec20_temp + (fRec24_temp + (fRec36_temp + (fRec40_temp + (fRec21 + (fRec25 + fTemp13)))))) - (fRec12_temp + (fRec16_temp + (fRec28_temp + (fRec32_temp + (fRec13 + (fRec17 + fTemp15)))))))) 
		state["fRec7"] = state["fRec7"].at[0].set(((fRec12_temp + (fRec16_temp + (fRec36_temp + (fRec40_temp + (fRec13 + (fRec17 + fTemp13)))))) - (fRec20_temp + (fRec24_temp + (fRec28_temp + (fRec32_temp + (fRec21 + (fRec25 + fTemp15)))))))) 
		fTemp16 = (fRec41 + fRec33) 
		fTemp17 = (fRec37 + fRec29) 
		state["fRec8"] = state["fRec8"].at[0].set(((fRec16_temp + (fRec24_temp + (fRec32_temp + (fRec40_temp + (fRec17 + (fRec25 + fTemp16)))))) - (fRec12_temp + (fRec20_temp + (fRec28_temp + (fRec36_temp + (fRec13 + (fRec21 + fTemp17)))))))) 
		state["fRec9"] = state["fRec9"].at[0].set(((fRec12_temp + (fRec20_temp + (fRec32_temp + (fRec40_temp + (fRec13 + (fRec21 + fTemp16)))))) - (fRec16_temp + (fRec24_temp + (fRec28_temp + (fRec36_temp + (fRec17 + (fRec25 + fTemp17)))))))) 
		fTemp18 = (fRec41 + fRec29) 
		fTemp19 = (fRec37 + fRec33) 
		state["fRec10"] = state["fRec10"].at[0].set(((fRec12_temp + (fRec24_temp + (fRec28_temp + (fRec40_temp + (fRec13 + (fRec25 + fTemp18)))))) - (fRec16_temp + (fRec20_temp + (fRec32_temp + (fRec36_temp + (fRec17 + (fRec21 + fTemp19)))))))) 
		state["fRec11"] = state["fRec11"].at[0].set(((fRec16_temp + (fRec20_temp + (fRec28_temp + (fRec40_temp + (fRec17 + (fRec21 + fTemp18)))))) - (fRec12_temp + (fRec24_temp + (fRec32_temp + (fRec36_temp + (fRec13 + (fRec25 + fTemp19)))))))) 
		fTemp20 = (jnp.float32(0.37) * (state["fRec5"][0] + state["fRec6"][0])) 
		fTemp21 = (fSlow89 * state["fRec3"][1]) 
		fTemp22 = (fTemp20 + fTemp21) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp22 - (fSlow9 * state["fRec3"][2]))) 
		fTemp23 = (fSlow9 * state["fRec3"][0]) 
		fTemp24 = (jnp.float32(0.5) * (((fTemp23 + (fTemp20 + state["fRec3"][2])) - fTemp21) + (fSlow7 * ((state["fRec3"][2] + fTemp23) - fTemp22)))) 
		fTemp25 = (fSlow90 * state["fRec2"][1]) 
		fTemp26 = (fTemp24 + fTemp25) 
		state["fRec2"] = state["fRec2"].at[0].set((fTemp26 - (fSlow5 * state["fRec2"][2]))) 
		fTemp27 = (fSlow5 * state["fRec2"][0]) 
		fTemp28 = (jnp.float32(1.0) - (jnp.float32(0.5) * fTemp0)) 
		_result0 = (state["fRec0"] * ((jnp.float32(0.25) * (fTemp0 * (((fTemp27 + (fTemp24 + state["fRec2"][2])) - fTemp25) + (fSlow3 * ((state["fRec2"][2] + fTemp27) - fTemp26))))) + (fTemp7 * fTemp28))) 
		fTemp29 = (jnp.float32(0.37) * (state["fRec5"][0] - state["fRec6"][0])) 
		fTemp30 = (fSlow89 * state["fRec45"][1]) 
		fTemp31 = (fTemp29 + fTemp30) 
		state["fRec45"] = state["fRec45"].at[0].set((fTemp31 - (fSlow9 * state["fRec45"][2]))) 
		fTemp32 = (fSlow9 * state["fRec45"][0]) 
		fTemp33 = (jnp.float32(0.5) * (((fTemp32 + (fTemp29 + state["fRec45"][2])) - fTemp30) + (fSlow7 * ((state["fRec45"][2] + fTemp32) - fTemp31)))) 
		fTemp34 = (fSlow90 * state["fRec44"][1]) 
		fTemp35 = (fTemp33 + fTemp34) 
		state["fRec44"] = state["fRec44"].at[0].set((fTemp35 - (fSlow5 * state["fRec44"][2]))) 
		fTemp36 = (fSlow5 * state["fRec44"][0]) 
		_result1 = (state["fRec0"] * ((jnp.float32(0.25) * (fTemp0 * (((fTemp36 + (fTemp33 + state["fRec44"][2])) - fTemp34) + (fSlow3 * ((state["fRec44"][2] + fTemp36) - fTemp35))))) + (fTemp2 * fTemp28))) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec10"] = jnp.roll(state["fRec10"], 1) 
		state["fRec11"] = jnp.roll(state["fRec11"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
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
