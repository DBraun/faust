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
		ui_path.append("reverb_designer") 
		ui_path.append("FEEDBACK DELAY NETWORK (FDN) REVERBERATOR, ORDER 16") 
		ui_path.append("Band Crossover Frequencies") 
		self.add_hslider("fHslider5", ui_path, "Band 0 upper edge in Hz", 5e+02, 1e+02, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider4", ui_path, "Band 1 upper edge in Hz", 1e+03, 1e+02, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider3", ui_path, "Band 2 upper edge in Hz", 2e+03, 1e+02, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider2", ui_path, "Band 3 upper edge in Hz", 4e+03, 1e+02, 1e+04, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Band Decay Times (T60)") 
		self.add_vslider("fVslider4", ui_path, "0", 8.4, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider3", ui_path, "1", 6.5, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider2", ui_path, "2", 5.0, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider1", ui_path, "3", 3.8, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider0", ui_path, "4", 2.7, 0.1, 1e+02, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Room Dimensions") 
		self.add_hslider("fHslider1", ui_path, "min acoustic ray length", 46.0, 0.1, 63.0, unnorm_funcs, "log") 
		self.add_hslider("fHslider6", ui_path, "max acoustic ray length", 63.0, 0.1, 63.0, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Input Controls") 
		ui_path.append("Input Config") 
		self.add_button("fCheckbox1", ui_path, "Mute Ext Inputs", unnorm_funcs) 
		self.add_button("fCheckbox0", ui_path, "Pink Noise", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Impulse Selection") 
		self.add_button("fButton0", ui_path, "Left", unnorm_funcs) 
		self.add_button("fButton1", ui_path, "Center", unnorm_funcs) 
		self.add_button("fButton3", ui_path, "Right", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Reverb State") 
		self.add_button("fButton2", ui_path, "Quench", unnorm_funcs) 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		self.add_hslider("fHslider0", ui_path, "Output Level (dB)", -4e+01, -7e+01, 2e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = (np.float32(6.9077554) / self._fConst0) 
		self._fConst2 = (np.float32(0.002915452) * self._fConst0) 
		self._fConst3 = (np.float32(3.1415927) / self._fConst0) 
		
	@property
	def num_inputs(self):
		return 2
	
	@property
	def num_outputs(self):
		return 2
	
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec101"] = np.float32(0)
		state["fRec103"] = np.float32(0)
		state["fRec105"] = np.float32(0)
		state["fRec110"] = np.float32(0)
		state["fRec114"] = np.float32(0)
		state["fRec116"] = np.float32(0)
		state["fRec119"] = np.float32(0)
		state["fRec121"] = np.float32(0)
		state["fRec123"] = np.float32(0)
		state["fRec125"] = np.float32(0)
		state["fRec127"] = np.float32(0)
		state["fRec132"] = np.float32(0)
		state["fRec136"] = np.float32(0)
		state["fRec138"] = np.float32(0)
		state["fRec141"] = np.float32(0)
		state["fRec143"] = np.float32(0)
		state["fRec145"] = np.float32(0)
		state["fRec147"] = np.float32(0)
		state["fRec149"] = np.float32(0)
		state["fRec154"] = np.float32(0)
		state["fRec158"] = np.float32(0)
		state["fRec160"] = np.float32(0)
		state["fRec163"] = np.float32(0)
		state["fRec165"] = np.float32(0)
		state["fRec167"] = np.float32(0)
		state["fRec169"] = np.float32(0)
		state["fRec171"] = np.float32(0)
		state["fRec176"] = np.float32(0)
		state["fRec180"] = np.float32(0)
		state["fRec182"] = np.float32(0)
		state["fRec185"] = np.float32(0)
		state["fRec187"] = np.float32(0)
		state["fRec189"] = np.float32(0)
		state["fRec191"] = np.float32(0)
		state["fRec193"] = np.float32(0)
		state["fRec198"] = np.float32(0)
		state["fRec202"] = np.float32(0)
		state["fRec204"] = np.float32(0)
		state["fRec207"] = np.float32(0)
		state["fRec209"] = np.float32(0)
		state["fRec211"] = np.float32(0)
		state["fRec213"] = np.float32(0)
		state["fRec215"] = np.float32(0)
		state["fRec22"] = np.float32(0)
		state["fRec220"] = np.float32(0)
		state["fRec224"] = np.float32(0)
		state["fRec226"] = np.float32(0)
		state["fRec229"] = np.float32(0)
		state["fRec231"] = np.float32(0)
		state["fRec233"] = np.float32(0)
		state["fRec235"] = np.float32(0)
		state["fRec237"] = np.float32(0)
		state["fRec242"] = np.float32(0)
		state["fRec246"] = np.float32(0)
		state["fRec248"] = np.float32(0)
		state["fRec251"] = np.float32(0)
		state["fRec253"] = np.float32(0)
		state["fRec255"] = np.float32(0)
		state["fRec257"] = np.float32(0)
		state["fRec259"] = np.float32(0)
		state["fRec26"] = np.float32(0)
		state["fRec264"] = np.float32(0)
		state["fRec268"] = np.float32(0)
		state["fRec270"] = np.float32(0)
		state["fRec273"] = np.float32(0)
		state["fRec275"] = np.float32(0)
		state["fRec277"] = np.float32(0)
		state["fRec279"] = np.float32(0)
		state["fRec28"] = np.float32(0)
		state["fRec281"] = np.float32(0)
		state["fRec286"] = np.float32(0)
		state["fRec290"] = np.float32(0)
		state["fRec292"] = np.float32(0)
		state["fRec295"] = np.float32(0)
		state["fRec297"] = np.float32(0)
		state["fRec299"] = np.float32(0)
		state["fRec301"] = np.float32(0)
		state["fRec303"] = np.float32(0)
		state["fRec308"] = np.float32(0)
		state["fRec31"] = np.float32(0)
		state["fRec312"] = np.float32(0)
		state["fRec314"] = np.float32(0)
		state["fRec317"] = np.float32(0)
		state["fRec319"] = np.float32(0)
		state["fRec321"] = np.float32(0)
		state["fRec323"] = np.float32(0)
		state["fRec325"] = np.float32(0)
		state["fRec33"] = np.float32(0)
		state["fRec330"] = np.float32(0)
		state["fRec334"] = np.float32(0)
		state["fRec336"] = np.float32(0)
		state["fRec339"] = np.float32(0)
		state["fRec341"] = np.float32(0)
		state["fRec343"] = np.float32(0)
		state["fRec345"] = np.float32(0)
		state["fRec347"] = np.float32(0)
		state["fRec35"] = np.float32(0)
		state["fRec352"] = np.float32(0)
		state["fRec356"] = np.float32(0)
		state["fRec358"] = np.float32(0)
		state["fRec361"] = np.float32(0)
		state["fRec363"] = np.float32(0)
		state["fRec365"] = np.float32(0)
		state["fRec367"] = np.float32(0)
		state["fRec369"] = np.float32(0)
		state["fRec37"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec44"] = np.float32(0)
		state["fRec48"] = np.float32(0)
		state["fRec50"] = np.float32(0)
		state["fRec53"] = np.float32(0)
		state["fRec55"] = np.float32(0)
		state["fRec57"] = np.float32(0)
		state["fRec59"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec66"] = np.float32(0)
		state["fRec70"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec75"] = np.float32(0)
		state["fRec77"] = np.float32(0)
		state["fRec79"] = np.float32(0)
		state["fRec81"] = np.float32(0)
		state["fRec83"] = np.float32(0)
		state["fRec88"] = np.float32(0)
		state["fRec92"] = np.float32(0)
		state["fRec94"] = np.float32(0)
		state["fRec97"] = np.float32(0)
		state["fRec99"] = np.float32(0)
		state["fVec0"] = np.float32(0)
		state["fVec1"] = np.float32(0)
		state["fVec10"] = np.float32(0)
		state["fVec11"] = np.float32(0)
		state["fVec12"] = np.float32(0)
		state["fVec13"] = np.float32(0)
		state["fVec14"] = np.float32(0)
		state["fVec15"] = np.float32(0)
		state["fVec16"] = np.float32(0)
		state["fVec17"] = np.float32(0)
		state["fVec18"] = np.float32(0)
		state["fVec19"] = np.float32(0)
		state["fVec2"] = np.float32(0)
		state["fVec20"] = np.float32(0)
		state["fVec21"] = np.float32(0)
		state["fVec22"] = np.float32(0)
		state["fVec23"] = np.float32(0)
		state["fVec24"] = np.float32(0)
		state["fVec25"] = np.float32(0)
		state["fVec26"] = np.float32(0)
		state["fVec27"] = np.float32(0)
		state["fVec28"] = np.float32(0)
		state["fVec29"] = np.float32(0)
		state["fVec3"] = np.float32(0)
		state["fVec30"] = np.float32(0)
		state["fVec31"] = np.float32(0)
		state["fVec32"] = np.float32(0)
		state["fVec33"] = np.float32(0)
		state["fVec34"] = np.float32(0)
		state["fVec35"] = np.float32(0)
		state["fVec36"] = np.float32(0)
		state["fVec37"] = np.float32(0)
		state["fVec38"] = np.float32(0)
		state["fVec39"] = np.float32(0)
		state["fVec4"] = np.float32(0)
		state["fVec40"] = np.float32(0)
		state["fVec41"] = np.float32(0)
		state["fVec42"] = np.float32(0)
		state["fVec43"] = np.float32(0)
		state["fVec44"] = np.float32(0)
		state["fVec45"] = np.float32(0)
		state["fVec46"] = np.float32(0)
		state["fVec47"] = np.float32(0)
		state["fVec48"] = np.float32(0)
		state["fVec49"] = np.float32(0)
		state["fVec5"] = np.float32(0)
		state["fVec51"] = np.float32(0)
		state["fVec6"] = np.float32(0)
		state["fVec7"] = np.float32(0)
		state["fVec8"] = np.float32(0)
		state["fVec9"] = np.float32(0)
		state["iRec17"] = np.int32(0)
		# Initialize array delays
		state["fRec16"] = np.zeros((4,), dtype=np.float32)
		state["fRec21"] = np.zeros((3,), dtype=np.float32)
		state["fRec20"] = np.zeros((3,), dtype=np.float32)
		state["fRec19"] = np.zeros((3,), dtype=np.float32)
		state["fRec18"] = np.zeros((3,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fRec25"] = np.zeros((3,), dtype=np.float32)
		state["fRec24"] = np.zeros((3,), dtype=np.float32)
		state["fRec23"] = np.zeros((3,), dtype=np.float32)
		state["fRec32"] = np.zeros((3,), dtype=np.float32)
		state["fRec30"] = np.zeros((3,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fRec36"] = np.zeros((3,), dtype=np.float32)
		state["fRec34"] = np.zeros((3,), dtype=np.float32)
		state["fRec38"] = np.zeros((3,), dtype=np.float32)
		state["fRec43"] = np.zeros((3,), dtype=np.float32)
		state["fRec42"] = np.zeros((3,), dtype=np.float32)
		state["fRec41"] = np.zeros((3,), dtype=np.float32)
		state["fRec40"] = np.zeros((3,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fRec47"] = np.zeros((3,), dtype=np.float32)
		state["fRec46"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec54"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec58"] = np.zeros((3,), dtype=np.float32)
		state["fRec56"] = np.zeros((3,), dtype=np.float32)
		state["fRec60"] = np.zeros((3,), dtype=np.float32)
		state["fRec65"] = np.zeros((3,), dtype=np.float32)
		state["fRec64"] = np.zeros((3,), dtype=np.float32)
		state["fRec63"] = np.zeros((3,), dtype=np.float32)
		state["fRec62"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec69"] = np.zeros((3,), dtype=np.float32)
		state["fRec68"] = np.zeros((3,), dtype=np.float32)
		state["fRec67"] = np.zeros((3,), dtype=np.float32)
		state["fRec76"] = np.zeros((3,), dtype=np.float32)
		state["fRec74"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec82"] = np.zeros((3,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec84"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec91"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fRec98"] = np.zeros((3,), dtype=np.float32)
		state["fRec96"] = np.zeros((3,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fRec102"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec112"] = np.zeros((3,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fRec120"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec124"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec126"] = np.zeros((3,), dtype=np.float32)
		state["fRec131"] = np.zeros((3,), dtype=np.float32)
		state["fRec130"] = np.zeros((3,), dtype=np.float32)
		state["fRec129"] = np.zeros((3,), dtype=np.float32)
		state["fRec128"] = np.zeros((3,), dtype=np.float32)
		state["fRec137"] = np.zeros((3,), dtype=np.float32)
		state["fRec135"] = np.zeros((3,), dtype=np.float32)
		state["fRec134"] = np.zeros((3,), dtype=np.float32)
		state["fRec133"] = np.zeros((3,), dtype=np.float32)
		state["fRec142"] = np.zeros((3,), dtype=np.float32)
		state["fRec140"] = np.zeros((3,), dtype=np.float32)
		state["fRec139"] = np.zeros((3,), dtype=np.float32)
		state["fRec146"] = np.zeros((3,), dtype=np.float32)
		state["fRec144"] = np.zeros((3,), dtype=np.float32)
		state["fRec148"] = np.zeros((3,), dtype=np.float32)
		state["fRec153"] = np.zeros((3,), dtype=np.float32)
		state["fRec152"] = np.zeros((3,), dtype=np.float32)
		state["fRec151"] = np.zeros((3,), dtype=np.float32)
		state["fRec150"] = np.zeros((3,), dtype=np.float32)
		state["fRec159"] = np.zeros((3,), dtype=np.float32)
		state["fRec157"] = np.zeros((3,), dtype=np.float32)
		state["fRec156"] = np.zeros((3,), dtype=np.float32)
		state["fRec155"] = np.zeros((3,), dtype=np.float32)
		state["fRec164"] = np.zeros((3,), dtype=np.float32)
		state["fRec162"] = np.zeros((3,), dtype=np.float32)
		state["fRec161"] = np.zeros((3,), dtype=np.float32)
		state["fRec168"] = np.zeros((3,), dtype=np.float32)
		state["fRec166"] = np.zeros((3,), dtype=np.float32)
		state["fRec170"] = np.zeros((3,), dtype=np.float32)
		state["fRec175"] = np.zeros((3,), dtype=np.float32)
		state["fRec174"] = np.zeros((3,), dtype=np.float32)
		state["fRec173"] = np.zeros((3,), dtype=np.float32)
		state["fRec172"] = np.zeros((3,), dtype=np.float32)
		state["fRec181"] = np.zeros((3,), dtype=np.float32)
		state["fRec179"] = np.zeros((3,), dtype=np.float32)
		state["fRec178"] = np.zeros((3,), dtype=np.float32)
		state["fRec177"] = np.zeros((3,), dtype=np.float32)
		state["fRec186"] = np.zeros((3,), dtype=np.float32)
		state["fRec184"] = np.zeros((3,), dtype=np.float32)
		state["fRec183"] = np.zeros((3,), dtype=np.float32)
		state["fRec190"] = np.zeros((3,), dtype=np.float32)
		state["fRec188"] = np.zeros((3,), dtype=np.float32)
		state["fRec192"] = np.zeros((3,), dtype=np.float32)
		state["fRec197"] = np.zeros((3,), dtype=np.float32)
		state["fRec196"] = np.zeros((3,), dtype=np.float32)
		state["fRec195"] = np.zeros((3,), dtype=np.float32)
		state["fRec194"] = np.zeros((3,), dtype=np.float32)
		state["fRec203"] = np.zeros((3,), dtype=np.float32)
		state["fRec201"] = np.zeros((3,), dtype=np.float32)
		state["fRec200"] = np.zeros((3,), dtype=np.float32)
		state["fRec199"] = np.zeros((3,), dtype=np.float32)
		state["fRec208"] = np.zeros((3,), dtype=np.float32)
		state["fRec206"] = np.zeros((3,), dtype=np.float32)
		state["fRec205"] = np.zeros((3,), dtype=np.float32)
		state["fRec212"] = np.zeros((3,), dtype=np.float32)
		state["fRec210"] = np.zeros((3,), dtype=np.float32)
		state["fRec214"] = np.zeros((3,), dtype=np.float32)
		state["fRec219"] = np.zeros((3,), dtype=np.float32)
		state["fRec218"] = np.zeros((3,), dtype=np.float32)
		state["fRec217"] = np.zeros((3,), dtype=np.float32)
		state["fRec216"] = np.zeros((3,), dtype=np.float32)
		state["fRec225"] = np.zeros((3,), dtype=np.float32)
		state["fRec223"] = np.zeros((3,), dtype=np.float32)
		state["fRec222"] = np.zeros((3,), dtype=np.float32)
		state["fRec221"] = np.zeros((3,), dtype=np.float32)
		state["fRec230"] = np.zeros((3,), dtype=np.float32)
		state["fRec228"] = np.zeros((3,), dtype=np.float32)
		state["fRec227"] = np.zeros((3,), dtype=np.float32)
		state["fRec234"] = np.zeros((3,), dtype=np.float32)
		state["fRec232"] = np.zeros((3,), dtype=np.float32)
		state["fRec236"] = np.zeros((3,), dtype=np.float32)
		state["fRec241"] = np.zeros((3,), dtype=np.float32)
		state["fRec240"] = np.zeros((3,), dtype=np.float32)
		state["fRec239"] = np.zeros((3,), dtype=np.float32)
		state["fRec238"] = np.zeros((3,), dtype=np.float32)
		state["fRec247"] = np.zeros((3,), dtype=np.float32)
		state["fRec245"] = np.zeros((3,), dtype=np.float32)
		state["fRec244"] = np.zeros((3,), dtype=np.float32)
		state["fRec243"] = np.zeros((3,), dtype=np.float32)
		state["fRec252"] = np.zeros((3,), dtype=np.float32)
		state["fRec250"] = np.zeros((3,), dtype=np.float32)
		state["fRec249"] = np.zeros((3,), dtype=np.float32)
		state["fRec256"] = np.zeros((3,), dtype=np.float32)
		state["fRec254"] = np.zeros((3,), dtype=np.float32)
		state["fRec258"] = np.zeros((3,), dtype=np.float32)
		state["fRec263"] = np.zeros((3,), dtype=np.float32)
		state["fRec262"] = np.zeros((3,), dtype=np.float32)
		state["fRec261"] = np.zeros((3,), dtype=np.float32)
		state["fRec260"] = np.zeros((3,), dtype=np.float32)
		state["fRec269"] = np.zeros((3,), dtype=np.float32)
		state["fRec267"] = np.zeros((3,), dtype=np.float32)
		state["fRec266"] = np.zeros((3,), dtype=np.float32)
		state["fRec265"] = np.zeros((3,), dtype=np.float32)
		state["fRec274"] = np.zeros((3,), dtype=np.float32)
		state["fRec272"] = np.zeros((3,), dtype=np.float32)
		state["fRec271"] = np.zeros((3,), dtype=np.float32)
		state["fRec278"] = np.zeros((3,), dtype=np.float32)
		state["fRec276"] = np.zeros((3,), dtype=np.float32)
		state["fRec280"] = np.zeros((3,), dtype=np.float32)
		state["fRec285"] = np.zeros((3,), dtype=np.float32)
		state["fRec284"] = np.zeros((3,), dtype=np.float32)
		state["fRec283"] = np.zeros((3,), dtype=np.float32)
		state["fRec282"] = np.zeros((3,), dtype=np.float32)
		state["fRec291"] = np.zeros((3,), dtype=np.float32)
		state["fRec289"] = np.zeros((3,), dtype=np.float32)
		state["fRec288"] = np.zeros((3,), dtype=np.float32)
		state["fRec287"] = np.zeros((3,), dtype=np.float32)
		state["fRec296"] = np.zeros((3,), dtype=np.float32)
		state["fRec294"] = np.zeros((3,), dtype=np.float32)
		state["fRec293"] = np.zeros((3,), dtype=np.float32)
		state["fRec300"] = np.zeros((3,), dtype=np.float32)
		state["fRec298"] = np.zeros((3,), dtype=np.float32)
		state["fRec302"] = np.zeros((3,), dtype=np.float32)
		state["fRec307"] = np.zeros((3,), dtype=np.float32)
		state["fRec306"] = np.zeros((3,), dtype=np.float32)
		state["fRec305"] = np.zeros((3,), dtype=np.float32)
		state["fRec304"] = np.zeros((3,), dtype=np.float32)
		state["fRec313"] = np.zeros((3,), dtype=np.float32)
		state["fRec311"] = np.zeros((3,), dtype=np.float32)
		state["fRec310"] = np.zeros((3,), dtype=np.float32)
		state["fRec309"] = np.zeros((3,), dtype=np.float32)
		state["fRec318"] = np.zeros((3,), dtype=np.float32)
		state["fRec316"] = np.zeros((3,), dtype=np.float32)
		state["fRec315"] = np.zeros((3,), dtype=np.float32)
		state["fRec322"] = np.zeros((3,), dtype=np.float32)
		state["fRec320"] = np.zeros((3,), dtype=np.float32)
		state["fRec324"] = np.zeros((3,), dtype=np.float32)
		state["fRec329"] = np.zeros((3,), dtype=np.float32)
		state["fRec328"] = np.zeros((3,), dtype=np.float32)
		state["fRec327"] = np.zeros((3,), dtype=np.float32)
		state["fRec326"] = np.zeros((3,), dtype=np.float32)
		state["fRec335"] = np.zeros((3,), dtype=np.float32)
		state["fRec333"] = np.zeros((3,), dtype=np.float32)
		state["fRec332"] = np.zeros((3,), dtype=np.float32)
		state["fRec331"] = np.zeros((3,), dtype=np.float32)
		state["fRec340"] = np.zeros((3,), dtype=np.float32)
		state["fRec338"] = np.zeros((3,), dtype=np.float32)
		state["fRec337"] = np.zeros((3,), dtype=np.float32)
		state["fRec344"] = np.zeros((3,), dtype=np.float32)
		state["fRec342"] = np.zeros((3,), dtype=np.float32)
		state["fRec346"] = np.zeros((3,), dtype=np.float32)
		state["fRec351"] = np.zeros((3,), dtype=np.float32)
		state["fRec350"] = np.zeros((3,), dtype=np.float32)
		state["fRec349"] = np.zeros((3,), dtype=np.float32)
		state["fRec348"] = np.zeros((3,), dtype=np.float32)
		state["fRec357"] = np.zeros((3,), dtype=np.float32)
		state["fRec355"] = np.zeros((3,), dtype=np.float32)
		state["fRec354"] = np.zeros((3,), dtype=np.float32)
		state["fRec353"] = np.zeros((3,), dtype=np.float32)
		state["fRec362"] = np.zeros((3,), dtype=np.float32)
		state["fRec360"] = np.zeros((3,), dtype=np.float32)
		state["fRec359"] = np.zeros((3,), dtype=np.float32)
		state["fRec366"] = np.zeros((3,), dtype=np.float32)
		state["fRec364"] = np.zeros((3,), dtype=np.float32)
		state["fRec368"] = np.zeros((3,), dtype=np.float32)
		state["fVec50"] = np.zeros((8192,), dtype=np.float32)
		state["fRec0"] = np.zeros((3,), dtype=np.float32)
		state["fVec52"] = np.zeros((8192,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fVec53"] = np.zeros((8192,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fVec54"] = np.zeros((8192,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fVec55"] = np.zeros((8192,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fVec56"] = np.zeros((8192,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fVec57"] = np.zeros((8192,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fVec58"] = np.zeros((8192,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fVec59"] = np.zeros((8192,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fVec60"] = np.zeros((8192,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fVec61"] = np.zeros((8192,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fVec62"] = np.zeros((8192,), dtype=np.float32)
		state["fRec11"] = np.zeros((3,), dtype=np.float32)
		state["fVec63"] = np.zeros((8192,), dtype=np.float32)
		state["fRec12"] = np.zeros((3,), dtype=np.float32)
		state["fVec64"] = np.zeros((8192,), dtype=np.float32)
		state["fRec13"] = np.zeros((3,), dtype=np.float32)
		state["fVec65"] = np.zeros((8192,), dtype=np.float32)
		state["fRec14"] = np.zeros((3,), dtype=np.float32)
		state["fVec66"] = np.zeros((8192,), dtype=np.float32)
		state["fRec15"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider0"])) 
		fSlow1 = (jnp.float32(0.1) * params["fCheckbox0"]) 
		fSlow2 = params["fButton0"] 
		fSlow3 = params["fButton1"] 
		fSlow4 = (jnp.float32(0.25) * (jnp.float32(1.0) - (jnp.float32(0.5) * params["fButton2"]))) 
		fSlow5 = params["fHslider1"] 
		fSlow6 = jnp.power(jnp.float32(2.0), jnp.floor(((jnp.float32(1.442695) * jnp.log((self._fConst2 * fSlow5))) + jnp.float32(0.5)))) 
		fSlow7 = params["fVslider0"] 
		fSlow8 = jnp.exp(-((self._fConst1 * (fSlow6 / fSlow7)))) 
		fSlow9 = jnp.tan((self._fConst3 * params["fHslider2"])) 
		fSlow10 = jnp.power(fSlow9, jnp.float32(2.0)) 
		fSlow11 = (jnp.float32(1.0) / fSlow9) 
		fSlow12 = (((fSlow11 + jnp.float32(1.0)) / fSlow9) + jnp.float32(1.0)) 
		fSlow13 = (jnp.float32(1.0) / (fSlow10 * fSlow12)) 
		fSlow14 = (jnp.float32(1.0) / (fSlow11 + jnp.float32(1.0))) 
		fSlow15 = (jnp.float32(1.0) - fSlow11) 
		fSlow16 = (jnp.float32(1.0) / fSlow12) 
		fSlow17 = (((fSlow11 + jnp.float32(-1.0)) / fSlow9) + jnp.float32(1.0)) 
		fSlow18 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow10))) 
		fSlow19 = jnp.tan((self._fConst3 * params["fHslider3"])) 
		fSlow20 = (jnp.float32(1.0) / fSlow19) 
		fSlow21 = (fSlow20 + jnp.float32(1.0)) 
		fSlow22 = (jnp.float32(1.0) / ((fSlow21 / fSlow19) + jnp.float32(1.0))) 
		fSlow23 = (jnp.float32(1.0) - fSlow20) 
		fSlow24 = (jnp.float32(1.0) - (fSlow23 / fSlow19)) 
		fSlow25 = jnp.power(fSlow19, jnp.float32(2.0)) 
		fSlow26 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow25))) 
		fSlow27 = jnp.tan((self._fConst3 * params["fHslider4"])) 
		fSlow28 = (jnp.float32(1.0) / fSlow27) 
		fSlow29 = (fSlow28 + jnp.float32(1.0)) 
		fSlow30 = (jnp.float32(1.0) / ((fSlow29 / fSlow27) + jnp.float32(1.0))) 
		fSlow31 = (jnp.float32(1.0) - fSlow28) 
		fSlow32 = (jnp.float32(1.0) - (fSlow31 / fSlow27)) 
		fSlow33 = jnp.power(fSlow27, jnp.float32(2.0)) 
		fSlow34 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow33))) 
		fSlow35 = jnp.tan((self._fConst3 * params["fHslider5"])) 
		fSlow36 = (jnp.float32(1.0) / fSlow35) 
		fSlow37 = (fSlow36 + jnp.float32(1.0)) 
		fSlow38 = (jnp.float32(1.0) / ((fSlow37 / fSlow35) + jnp.float32(1.0))) 
		fSlow39 = (jnp.float32(1.0) - fSlow36) 
		fSlow40 = (jnp.float32(1.0) - (fSlow39 / fSlow35)) 
		fSlow41 = jnp.power(fSlow35, jnp.float32(2.0)) 
		fSlow42 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow41))) 
		fSlow43 = params["fVslider1"] 
		fSlow44 = jnp.exp(-((self._fConst1 * (fSlow6 / fSlow43)))) 
		fSlow45 = (((fSlow20 + jnp.float32(1.0)) / fSlow19) + jnp.float32(1.0)) 
		fSlow46 = (jnp.float32(1.0) / (fSlow25 * fSlow45)) 
		fSlow47 = (jnp.float32(1.0) / fSlow21) 
		fSlow48 = (jnp.float32(1.0) / fSlow45) 
		fSlow49 = (((fSlow20 + jnp.float32(-1.0)) / fSlow19) + jnp.float32(1.0)) 
		fSlow50 = params["fVslider2"] 
		fSlow51 = jnp.exp(-((self._fConst1 * (fSlow6 / fSlow50)))) 
		fSlow52 = (((fSlow28 + jnp.float32(1.0)) / fSlow27) + jnp.float32(1.0)) 
		fSlow53 = (jnp.float32(1.0) / (fSlow33 * fSlow52)) 
		fSlow54 = (jnp.float32(1.0) / fSlow29) 
		fSlow55 = (jnp.float32(1.0) / fSlow52) 
		fSlow56 = (((fSlow28 + jnp.float32(-1.0)) / fSlow27) + jnp.float32(1.0)) 
		fSlow57 = (jnp.float32(1.0) / (((fSlow36 + jnp.float32(1.0)) / fSlow35) + jnp.float32(1.0))) 
		fSlow58 = params["fVslider3"] 
		fSlow59 = (jnp.exp(-((self._fConst1 * (fSlow6 / fSlow58)))) / fSlow41) 
		fSlow60 = (jnp.float32(1.0) / fSlow37) 
		fSlow61 = (((fSlow36 + jnp.float32(-1.0)) / fSlow35) + jnp.float32(1.0)) 
		fSlow62 = params["fVslider4"] 
		fSlow63 = jnp.exp(-((self._fConst1 * (fSlow6 / fSlow62)))) 
		fSlow64 = params["fHslider6"] 
		fSlow65 = (fSlow64 / fSlow5) 
		fSlow66 = jnp.power(jnp.float32(23.0), jnp.floor(((jnp.float32(0.318929) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.53333336)))))) + jnp.float32(0.5)))) 
		fSlow67 = jnp.exp(-((self._fConst1 * (fSlow66 / fSlow7)))) 
		fSlow68 = jnp.exp(-((self._fConst1 * (fSlow66 / fSlow43)))) 
		fSlow69 = jnp.exp(-((self._fConst1 * (fSlow66 / fSlow50)))) 
		fSlow70 = (jnp.exp(-((self._fConst1 * (fSlow66 / fSlow58)))) / fSlow41) 
		fSlow71 = jnp.exp(-((self._fConst1 * (fSlow66 / fSlow62)))) 
		fSlow72 = jnp.power(jnp.float32(11.0), jnp.floor(((jnp.float32(0.4170324) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.26666668)))))) + jnp.float32(0.5)))) 
		fSlow73 = jnp.exp(-((self._fConst1 * (fSlow72 / fSlow7)))) 
		fSlow74 = jnp.exp(-((self._fConst1 * (fSlow72 / fSlow43)))) 
		fSlow75 = jnp.exp(-((self._fConst1 * (fSlow72 / fSlow50)))) 
		fSlow76 = (jnp.exp(-((self._fConst1 * (fSlow72 / fSlow58)))) / fSlow41) 
		fSlow77 = jnp.exp(-((self._fConst1 * (fSlow72 / fSlow62)))) 
		fSlow78 = jnp.power(jnp.float32(41.0), jnp.floor(((jnp.float32(0.26928252) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.8)))))) + jnp.float32(0.5)))) 
		fSlow79 = jnp.exp(-((self._fConst1 * (fSlow78 / fSlow7)))) 
		fSlow80 = jnp.exp(-((self._fConst1 * (fSlow78 / fSlow43)))) 
		fSlow81 = jnp.exp(-((self._fConst1 * (fSlow78 / fSlow50)))) 
		fSlow82 = (jnp.exp(-((self._fConst1 * (fSlow78 / fSlow58)))) / fSlow41) 
		fSlow83 = jnp.exp(-((self._fConst1 * (fSlow78 / fSlow62)))) 
		fSlow84 = jnp.power(jnp.float32(5.0), jnp.floor(((jnp.float32(0.6213349) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.13333334)))))) + jnp.float32(0.5)))) 
		fSlow85 = jnp.exp(-((self._fConst1 * (fSlow84 / fSlow7)))) 
		fSlow86 = jnp.exp(-((self._fConst1 * (fSlow84 / fSlow43)))) 
		fSlow87 = jnp.exp(-((self._fConst1 * (fSlow84 / fSlow50)))) 
		fSlow88 = (jnp.exp(-((self._fConst1 * (fSlow84 / fSlow58)))) / fSlow41) 
		fSlow89 = jnp.exp(-((self._fConst1 * (fSlow84 / fSlow62)))) 
		fSlow90 = jnp.power(jnp.float32(31.0), jnp.floor(((jnp.float32(0.2912067) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.6666667)))))) + jnp.float32(0.5)))) 
		fSlow91 = jnp.exp(-((self._fConst1 * (fSlow90 / fSlow7)))) 
		fSlow92 = jnp.exp(-((self._fConst1 * (fSlow90 / fSlow43)))) 
		fSlow93 = jnp.exp(-((self._fConst1 * (fSlow90 / fSlow50)))) 
		fSlow94 = (jnp.exp(-((self._fConst1 * (fSlow90 / fSlow58)))) / fSlow41) 
		fSlow95 = jnp.exp(-((self._fConst1 * (fSlow90 / fSlow62)))) 
		fSlow96 = jnp.power(jnp.float32(17.0), jnp.floor(((jnp.float32(0.35295612) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.4)))))) + jnp.float32(0.5)))) 
		fSlow97 = jnp.exp(-((self._fConst1 * (fSlow96 / fSlow7)))) 
		fSlow98 = jnp.exp(-((self._fConst1 * (fSlow96 / fSlow43)))) 
		fSlow99 = jnp.exp(-((self._fConst1 * (fSlow96 / fSlow50)))) 
		fSlow100 = (jnp.exp(-((self._fConst1 * (fSlow96 / fSlow58)))) / fSlow41) 
		fSlow101 = jnp.exp(-((self._fConst1 * (fSlow96 / fSlow62)))) 
		fSlow102 = jnp.power(jnp.float32(47.0), jnp.floor(((jnp.float32(0.2597303) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.93333334)))))) + jnp.float32(0.5)))) 
		fSlow103 = jnp.exp(-((self._fConst1 * (fSlow102 / fSlow7)))) 
		fSlow104 = jnp.exp(-((self._fConst1 * (fSlow102 / fSlow43)))) 
		fSlow105 = jnp.exp(-((self._fConst1 * (fSlow102 / fSlow50)))) 
		fSlow106 = (jnp.exp(-((self._fConst1 * (fSlow102 / fSlow58)))) / fSlow41) 
		fSlow107 = jnp.exp(-((self._fConst1 * (fSlow102 / fSlow62)))) 
		fSlow108 = jnp.power(jnp.float32(3.0), jnp.floor(((jnp.float32(0.9102392) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.06666667)))))) + jnp.float32(0.5)))) 
		fSlow109 = jnp.exp(-((self._fConst1 * (fSlow108 / fSlow7)))) 
		fSlow110 = jnp.exp(-((self._fConst1 * (fSlow108 / fSlow43)))) 
		fSlow111 = jnp.exp(-((self._fConst1 * (fSlow108 / fSlow50)))) 
		fSlow112 = (jnp.exp(-((self._fConst1 * (fSlow108 / fSlow58)))) / fSlow41) 
		fSlow113 = jnp.exp(-((self._fConst1 * (fSlow108 / fSlow62)))) 
		fSlow114 = jnp.power(jnp.float32(29.0), jnp.floor(((jnp.float32(0.2969742) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.6)))))) + jnp.float32(0.5)))) 
		fSlow115 = jnp.exp(-((self._fConst1 * (fSlow114 / fSlow7)))) 
		fSlow116 = jnp.exp(-((self._fConst1 * (fSlow114 / fSlow43)))) 
		fSlow117 = jnp.exp(-((self._fConst1 * (fSlow114 / fSlow50)))) 
		fSlow118 = (jnp.exp(-((self._fConst1 * (fSlow114 / fSlow58)))) / fSlow41) 
		fSlow119 = jnp.exp(-((self._fConst1 * (fSlow114 / fSlow62)))) 
		fSlow120 = jnp.power(jnp.float32(13.0), jnp.floor(((jnp.float32(0.38987124) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.33333334)))))) + jnp.float32(0.5)))) 
		fSlow121 = jnp.exp(-((self._fConst1 * (fSlow120 / fSlow7)))) 
		fSlow122 = jnp.exp(-((self._fConst1 * (fSlow120 / fSlow43)))) 
		fSlow123 = jnp.exp(-((self._fConst1 * (fSlow120 / fSlow50)))) 
		fSlow124 = (jnp.exp(-((self._fConst1 * (fSlow120 / fSlow58)))) / fSlow41) 
		fSlow125 = jnp.exp(-((self._fConst1 * (fSlow120 / fSlow62)))) 
		fSlow126 = jnp.power(jnp.float32(43.0), jnp.floor(((jnp.float32(0.2658726) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.8666667)))))) + jnp.float32(0.5)))) 
		fSlow127 = jnp.exp(-((self._fConst1 * (fSlow126 / fSlow7)))) 
		fSlow128 = jnp.exp(-((self._fConst1 * (fSlow126 / fSlow43)))) 
		fSlow129 = jnp.exp(-((self._fConst1 * (fSlow126 / fSlow50)))) 
		fSlow130 = (jnp.exp(-((self._fConst1 * (fSlow126 / fSlow58)))) / fSlow41) 
		fSlow131 = jnp.exp(-((self._fConst1 * (fSlow126 / fSlow62)))) 
		fSlow132 = jnp.power(jnp.float32(7.0), jnp.floor(((jnp.float32(0.5138983) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.2)))))) + jnp.float32(0.5)))) 
		fSlow133 = jnp.exp(-((self._fConst1 * (fSlow132 / fSlow7)))) 
		fSlow134 = jnp.exp(-((self._fConst1 * (fSlow132 / fSlow43)))) 
		fSlow135 = jnp.exp(-((self._fConst1 * (fSlow132 / fSlow50)))) 
		fSlow136 = (jnp.exp(-((self._fConst1 * (fSlow132 / fSlow58)))) / fSlow41) 
		fSlow137 = jnp.exp(-((self._fConst1 * (fSlow132 / fSlow62)))) 
		fSlow138 = jnp.power(jnp.float32(37.0), jnp.floor(((jnp.float32(0.2769379) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.73333335)))))) + jnp.float32(0.5)))) 
		fSlow139 = jnp.exp(-((self._fConst1 * (fSlow138 / fSlow7)))) 
		fSlow140 = jnp.exp(-((self._fConst1 * (fSlow138 / fSlow43)))) 
		fSlow141 = jnp.exp(-((self._fConst1 * (fSlow138 / fSlow50)))) 
		fSlow142 = (jnp.exp(-((self._fConst1 * (fSlow138 / fSlow58)))) / fSlow41) 
		fSlow143 = jnp.exp(-((self._fConst1 * (fSlow138 / fSlow62)))) 
		fSlow144 = jnp.power(jnp.float32(19.0), jnp.floor(((jnp.float32(0.33962327) * jnp.log((self._fConst2 * (fSlow5 * jnp.power(fSlow65, jnp.float32(0.46666667)))))) + jnp.float32(0.5)))) 
		fSlow145 = jnp.exp(-((self._fConst1 * (fSlow144 / fSlow7)))) 
		fSlow146 = jnp.exp(-((self._fConst1 * (fSlow144 / fSlow43)))) 
		fSlow147 = jnp.exp(-((self._fConst1 * (fSlow144 / fSlow50)))) 
		fSlow148 = (jnp.exp(-((self._fConst1 * (fSlow144 / fSlow58)))) / fSlow41) 
		fSlow149 = jnp.exp(-((self._fConst1 * (fSlow144 / fSlow62)))) 
		fSlow150 = jnp.power(jnp.float32(53.0), jnp.floor(((jnp.float32(0.25187066) * jnp.log((self._fConst2 * fSlow64))) + jnp.float32(0.5)))) 
		fSlow151 = jnp.exp(-((self._fConst1 * (fSlow150 / fSlow7)))) 
		fSlow152 = jnp.exp(-((self._fConst1 * (fSlow150 / fSlow43)))) 
		fSlow153 = jnp.exp(-((self._fConst1 * (fSlow150 / fSlow50)))) 
		fSlow154 = (jnp.exp(-((self._fConst1 * (fSlow150 / fSlow58)))) / fSlow41) 
		fSlow155 = jnp.exp(-((self._fConst1 * (fSlow150 / fSlow62)))) 
		fSlow156 = (jnp.float32(1.0) - params["fCheckbox1"]) 
		iSlow157 = (jnp.int32((fSlow6 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		fSlow158 = params["fButton3"] 
		iSlow159 = (jnp.int32((fSlow108 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow160 = (jnp.int32((fSlow84 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow161 = (jnp.int32((fSlow132 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow162 = (jnp.int32((fSlow72 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow163 = (jnp.int32((fSlow120 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow164 = (jnp.int32((fSlow96 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow165 = (jnp.int32((fSlow144 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow166 = (jnp.int32((fSlow66 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow167 = (jnp.int32((fSlow114 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow168 = (jnp.int32((fSlow90 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow169 = (jnp.int32((fSlow138 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow170 = (jnp.int32((fSlow78 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow171 = (jnp.int32((fSlow126 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow172 = (jnp.int32((fSlow102 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow173 = (jnp.int32((fSlow150 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iRec17_temp = state["iRec17"] 
		fVec0_temp = state["fVec0"] 
		fVec1_temp = state["fVec1"] 
		fRec22_temp = state["fRec22"] 
		fRec28_temp = state["fRec28"] 
		fVec2_temp = state["fVec2"] 
		fRec26_temp = state["fRec26"] 
		fRec33_temp = state["fRec33"] 
		fVec3_temp = state["fVec3"] 
		fRec31_temp = state["fRec31"] 
		fRec37_temp = state["fRec37"] 
		fVec4_temp = state["fVec4"] 
		fRec35_temp = state["fRec35"] 
		fRec39_temp = state["fRec39"] 
		fRec44_temp = state["fRec44"] 
		fRec50_temp = state["fRec50"] 
		fVec5_temp = state["fVec5"] 
		fRec48_temp = state["fRec48"] 
		fRec55_temp = state["fRec55"] 
		fVec6_temp = state["fVec6"] 
		fRec53_temp = state["fRec53"] 
		fRec59_temp = state["fRec59"] 
		fVec7_temp = state["fVec7"] 
		fRec57_temp = state["fRec57"] 
		fRec61_temp = state["fRec61"] 
		fRec66_temp = state["fRec66"] 
		fRec72_temp = state["fRec72"] 
		fVec8_temp = state["fVec8"] 
		fRec70_temp = state["fRec70"] 
		fRec77_temp = state["fRec77"] 
		fVec9_temp = state["fVec9"] 
		fRec75_temp = state["fRec75"] 
		fRec81_temp = state["fRec81"] 
		fVec10_temp = state["fVec10"] 
		fRec79_temp = state["fRec79"] 
		fRec83_temp = state["fRec83"] 
		fRec88_temp = state["fRec88"] 
		fRec94_temp = state["fRec94"] 
		fVec11_temp = state["fVec11"] 
		fRec92_temp = state["fRec92"] 
		fRec99_temp = state["fRec99"] 
		fVec12_temp = state["fVec12"] 
		fRec97_temp = state["fRec97"] 
		fRec103_temp = state["fRec103"] 
		fVec13_temp = state["fVec13"] 
		fRec101_temp = state["fRec101"] 
		fRec105_temp = state["fRec105"] 
		fRec110_temp = state["fRec110"] 
		fRec116_temp = state["fRec116"] 
		fVec14_temp = state["fVec14"] 
		fRec114_temp = state["fRec114"] 
		fRec121_temp = state["fRec121"] 
		fVec15_temp = state["fVec15"] 
		fRec119_temp = state["fRec119"] 
		fRec125_temp = state["fRec125"] 
		fVec16_temp = state["fVec16"] 
		fRec123_temp = state["fRec123"] 
		fRec127_temp = state["fRec127"] 
		fRec132_temp = state["fRec132"] 
		fRec138_temp = state["fRec138"] 
		fVec17_temp = state["fVec17"] 
		fRec136_temp = state["fRec136"] 
		fRec143_temp = state["fRec143"] 
		fVec18_temp = state["fVec18"] 
		fRec141_temp = state["fRec141"] 
		fRec147_temp = state["fRec147"] 
		fVec19_temp = state["fVec19"] 
		fRec145_temp = state["fRec145"] 
		fRec149_temp = state["fRec149"] 
		fRec154_temp = state["fRec154"] 
		fRec160_temp = state["fRec160"] 
		fVec20_temp = state["fVec20"] 
		fRec158_temp = state["fRec158"] 
		fRec165_temp = state["fRec165"] 
		fVec21_temp = state["fVec21"] 
		fRec163_temp = state["fRec163"] 
		fRec169_temp = state["fRec169"] 
		fVec22_temp = state["fVec22"] 
		fRec167_temp = state["fRec167"] 
		fRec171_temp = state["fRec171"] 
		fRec176_temp = state["fRec176"] 
		fRec182_temp = state["fRec182"] 
		fVec23_temp = state["fVec23"] 
		fRec180_temp = state["fRec180"] 
		fRec187_temp = state["fRec187"] 
		fVec24_temp = state["fVec24"] 
		fRec185_temp = state["fRec185"] 
		fRec191_temp = state["fRec191"] 
		fVec25_temp = state["fVec25"] 
		fRec189_temp = state["fRec189"] 
		fRec193_temp = state["fRec193"] 
		fRec198_temp = state["fRec198"] 
		fRec204_temp = state["fRec204"] 
		fVec26_temp = state["fVec26"] 
		fRec202_temp = state["fRec202"] 
		fRec209_temp = state["fRec209"] 
		fVec27_temp = state["fVec27"] 
		fRec207_temp = state["fRec207"] 
		fRec213_temp = state["fRec213"] 
		fVec28_temp = state["fVec28"] 
		fRec211_temp = state["fRec211"] 
		fRec215_temp = state["fRec215"] 
		fRec220_temp = state["fRec220"] 
		fRec226_temp = state["fRec226"] 
		fVec29_temp = state["fVec29"] 
		fRec224_temp = state["fRec224"] 
		fRec231_temp = state["fRec231"] 
		fVec30_temp = state["fVec30"] 
		fRec229_temp = state["fRec229"] 
		fRec235_temp = state["fRec235"] 
		fVec31_temp = state["fVec31"] 
		fRec233_temp = state["fRec233"] 
		fRec237_temp = state["fRec237"] 
		fRec242_temp = state["fRec242"] 
		fRec248_temp = state["fRec248"] 
		fVec32_temp = state["fVec32"] 
		fRec246_temp = state["fRec246"] 
		fRec253_temp = state["fRec253"] 
		fVec33_temp = state["fVec33"] 
		fRec251_temp = state["fRec251"] 
		fRec257_temp = state["fRec257"] 
		fVec34_temp = state["fVec34"] 
		fRec255_temp = state["fRec255"] 
		fRec259_temp = state["fRec259"] 
		fRec264_temp = state["fRec264"] 
		fRec270_temp = state["fRec270"] 
		fVec35_temp = state["fVec35"] 
		fRec268_temp = state["fRec268"] 
		fRec275_temp = state["fRec275"] 
		fVec36_temp = state["fVec36"] 
		fRec273_temp = state["fRec273"] 
		fRec279_temp = state["fRec279"] 
		fVec37_temp = state["fVec37"] 
		fRec277_temp = state["fRec277"] 
		fRec281_temp = state["fRec281"] 
		fRec286_temp = state["fRec286"] 
		fRec292_temp = state["fRec292"] 
		fVec38_temp = state["fVec38"] 
		fRec290_temp = state["fRec290"] 
		fRec297_temp = state["fRec297"] 
		fVec39_temp = state["fVec39"] 
		fRec295_temp = state["fRec295"] 
		fRec301_temp = state["fRec301"] 
		fVec40_temp = state["fVec40"] 
		fRec299_temp = state["fRec299"] 
		fRec303_temp = state["fRec303"] 
		fRec308_temp = state["fRec308"] 
		fRec314_temp = state["fRec314"] 
		fVec41_temp = state["fVec41"] 
		fRec312_temp = state["fRec312"] 
		fRec319_temp = state["fRec319"] 
		fVec42_temp = state["fVec42"] 
		fRec317_temp = state["fRec317"] 
		fRec323_temp = state["fRec323"] 
		fVec43_temp = state["fVec43"] 
		fRec321_temp = state["fRec321"] 
		fRec325_temp = state["fRec325"] 
		fRec330_temp = state["fRec330"] 
		fRec336_temp = state["fRec336"] 
		fVec44_temp = state["fVec44"] 
		fRec334_temp = state["fRec334"] 
		fRec341_temp = state["fRec341"] 
		fVec45_temp = state["fVec45"] 
		fRec339_temp = state["fRec339"] 
		fRec345_temp = state["fRec345"] 
		fVec46_temp = state["fVec46"] 
		fRec343_temp = state["fRec343"] 
		fRec347_temp = state["fRec347"] 
		fRec352_temp = state["fRec352"] 
		fRec358_temp = state["fRec358"] 
		fVec47_temp = state["fVec47"] 
		fRec356_temp = state["fRec356"] 
		fRec363_temp = state["fRec363"] 
		fVec48_temp = state["fVec48"] 
		fRec361_temp = state["fRec361"] 
		fRec367_temp = state["fRec367"] 
		fVec49_temp = state["fVec49"] 
		fRec365_temp = state["fRec365"] 
		fRec369_temp = state["fRec369"] 
		fVec51_temp = state["fVec51"] 
		state["iRec17"] = ((jnp.int32(1103515245) * iRec17_temp) + jnp.int32(12345)) 
		state["fRec16"] = state["fRec16"].at[0].set((((jnp.float32(0.5221894) * state["fRec16"][3]) + ((jnp.float32(4.656613e-10) * (state["iRec17"])) + (jnp.float32(2.494956) * state["fRec16"][1]))) - (jnp.float32(2.0172658) * state["fRec16"][2]))) 
		fTemp0 = (fSlow1 * (((jnp.float32(0.049922034) * state["fRec16"][0]) + (jnp.float32(0.0506127) * state["fRec16"][2])) - ((jnp.float32(0.095993534) * state["fRec16"][1]) + (jnp.float32(0.004408786) * state["fRec16"][3])))) 
		state["fVec0"] = fSlow2 
		state["fVec1"] = fSlow3 
		iTemp1 = ((fSlow3 - fVec1_temp) > jnp.float32(0.0)).astype(jnp.int32) 
		fTemp2 = ((((fSlow2 - fVec0_temp) > jnp.float32(0.0)).astype(jnp.int32) + iTemp1)) 
		state["fRec22"] = -((fSlow14 * ((fSlow15 * fRec22_temp) - (fSlow11 * (state["fRec0"][1] - state["fRec0"][2]))))) 
		state["fRec21"] = state["fRec21"].at[0].set((state["fRec22"] - (fSlow16 * ((fSlow17 * state["fRec21"][2]) + (fSlow18 * state["fRec21"][1]))))) 
		fTemp3 = (fSlow26 * state["fRec20"][1]) 
		state["fRec20"] = state["fRec20"].at[0].set(((fSlow13 * (state["fRec21"][2] + (state["fRec21"][0] - (jnp.float32(2.0) * state["fRec21"][1])))) - (fSlow22 * ((fSlow24 * state["fRec20"][2]) + fTemp3)))) 
		fTemp4 = (fSlow34 * state["fRec19"][1]) 
		state["fRec19"] = state["fRec19"].at[0].set(((state["fRec20"][2] + (fSlow22 * (fTemp3 + (fSlow24 * state["fRec20"][0])))) - (fSlow30 * ((fSlow32 * state["fRec19"][2]) + fTemp4)))) 
		fTemp5 = (fSlow42 * state["fRec18"][1]) 
		state["fRec18"] = state["fRec18"].at[0].set(((state["fRec19"][2] + (fSlow30 * (fTemp4 + (fSlow32 * state["fRec19"][0])))) - (fSlow38 * ((fSlow40 * state["fRec18"][2]) + fTemp5)))) 
		state["fRec28"] = -((fSlow14 * ((fSlow15 * fRec28_temp) - (state["fRec0"][1] + state["fRec0"][2])))) 
		state["fRec27"] = state["fRec27"].at[0].set((state["fRec28"] - (fSlow16 * ((fSlow17 * state["fRec27"][2]) + (fSlow18 * state["fRec27"][1]))))) 
		fTemp6 = (fSlow16 * (state["fRec27"][2] + (state["fRec27"][0] + (jnp.float32(2.0) * state["fRec27"][1])))) 
		state["fVec2"] = jnp.float32(fTemp6) 
		state["fRec26"] = -((fSlow47 * ((fSlow23 * fRec26_temp) - (fSlow20 * (fTemp6 - fVec2_temp))))) 
		state["fRec25"] = state["fRec25"].at[0].set((state["fRec26"] - (fSlow48 * ((fSlow49 * state["fRec25"][2]) + (fSlow26 * state["fRec25"][1]))))) 
		fTemp7 = (fSlow34 * state["fRec24"][1]) 
		state["fRec24"] = state["fRec24"].at[0].set(((fSlow46 * (state["fRec25"][2] + (state["fRec25"][0] - (jnp.float32(2.0) * state["fRec25"][1])))) - (fSlow30 * ((fSlow32 * state["fRec24"][2]) + fTemp7)))) 
		fTemp8 = (fSlow42 * state["fRec23"][1]) 
		state["fRec23"] = state["fRec23"].at[0].set(((state["fRec24"][2] + (fSlow30 * (fTemp7 + (fSlow32 * state["fRec24"][0])))) - (fSlow38 * ((fSlow40 * state["fRec23"][2]) + fTemp8)))) 
		state["fRec33"] = -((fSlow47 * ((fSlow23 * fRec33_temp) - (fTemp6 + fVec2_temp)))) 
		state["fRec32"] = state["fRec32"].at[0].set((state["fRec33"] - (fSlow48 * ((fSlow49 * state["fRec32"][2]) + (fSlow26 * state["fRec32"][1]))))) 
		fTemp9 = (fSlow48 * (state["fRec32"][2] + (state["fRec32"][0] + (jnp.float32(2.0) * state["fRec32"][1])))) 
		state["fVec3"] = jnp.float32(fTemp9) 
		state["fRec31"] = -((fSlow54 * ((fSlow31 * fRec31_temp) - (fSlow28 * (fTemp9 - fVec3_temp))))) 
		state["fRec30"] = state["fRec30"].at[0].set((state["fRec31"] - (fSlow55 * ((fSlow56 * state["fRec30"][2]) + (fSlow34 * state["fRec30"][1]))))) 
		fTemp10 = (fSlow42 * state["fRec29"][1]) 
		state["fRec29"] = state["fRec29"].at[0].set(((fSlow53 * (state["fRec30"][2] + (state["fRec30"][0] - (jnp.float32(2.0) * state["fRec30"][1])))) - (fSlow38 * ((fSlow40 * state["fRec29"][2]) + fTemp10)))) 
		state["fRec37"] = -((fSlow54 * ((fSlow31 * fRec37_temp) - (fTemp9 + fVec3_temp)))) 
		state["fRec36"] = state["fRec36"].at[0].set((state["fRec37"] - (fSlow55 * ((fSlow56 * state["fRec36"][2]) + (fSlow34 * state["fRec36"][1]))))) 
		fTemp11 = (fSlow55 * (state["fRec36"][2] + (state["fRec36"][0] + (jnp.float32(2.0) * state["fRec36"][1])))) 
		state["fVec4"] = jnp.float32(fTemp11) 
		state["fRec35"] = -((fSlow60 * ((fSlow39 * fRec35_temp) - (fSlow36 * (fTemp11 - fVec4_temp))))) 
		state["fRec34"] = state["fRec34"].at[0].set((state["fRec35"] - (fSlow57 * ((fSlow61 * state["fRec34"][2]) + (fSlow42 * state["fRec34"][1]))))) 
		state["fRec39"] = -((fSlow60 * ((fSlow39 * fRec39_temp) - (fTemp11 + fVec4_temp)))) 
		state["fRec38"] = state["fRec38"].at[0].set((state["fRec39"] - (fSlow57 * ((fSlow61 * state["fRec38"][2]) + (fSlow42 * state["fRec38"][1]))))) 
		fTemp12 = ((((fSlow8 * (state["fRec18"][2] + (fSlow38 * (fTemp5 + (fSlow40 * state["fRec18"][0]))))) + (fSlow44 * (state["fRec23"][2] + (fSlow38 * (fTemp8 + (fSlow40 * state["fRec23"][0])))))) + (fSlow51 * (state["fRec29"][2] + (fSlow38 * (fTemp10 + (fSlow40 * state["fRec29"][0])))))) + (fSlow57 * ((fSlow59 * (state["fRec34"][2] + (state["fRec34"][0] - (jnp.float32(2.0) * state["fRec34"][1])))) + (fSlow63 * (state["fRec38"][2] + (state["fRec38"][0] + (jnp.float32(2.0) * state["fRec38"][1]))))))) 
		state["fRec44"] = -((fSlow14 * ((fSlow15 * fRec44_temp) - (fSlow11 * (state["fRec8"][1] - state["fRec8"][2]))))) 
		state["fRec43"] = state["fRec43"].at[0].set((state["fRec44"] - (fSlow16 * ((fSlow17 * state["fRec43"][2]) + (fSlow18 * state["fRec43"][1]))))) 
		fTemp13 = (fSlow26 * state["fRec42"][1]) 
		state["fRec42"] = state["fRec42"].at[0].set(((fSlow13 * (state["fRec43"][2] + (state["fRec43"][0] - (jnp.float32(2.0) * state["fRec43"][1])))) - (fSlow22 * ((fSlow24 * state["fRec42"][2]) + fTemp13)))) 
		fTemp14 = (fSlow34 * state["fRec41"][1]) 
		state["fRec41"] = state["fRec41"].at[0].set(((state["fRec42"][2] + (fSlow22 * (fTemp13 + (fSlow24 * state["fRec42"][0])))) - (fSlow30 * ((fSlow32 * state["fRec41"][2]) + fTemp14)))) 
		fTemp15 = (fSlow42 * state["fRec40"][1]) 
		state["fRec40"] = state["fRec40"].at[0].set(((state["fRec41"][2] + (fSlow30 * (fTemp14 + (fSlow32 * state["fRec41"][0])))) - (fSlow38 * ((fSlow40 * state["fRec40"][2]) + fTemp15)))) 
		state["fRec50"] = -((fSlow14 * ((fSlow15 * fRec50_temp) - (state["fRec8"][1] + state["fRec8"][2])))) 
		state["fRec49"] = state["fRec49"].at[0].set((state["fRec50"] - (fSlow16 * ((fSlow17 * state["fRec49"][2]) + (fSlow18 * state["fRec49"][1]))))) 
		fTemp16 = (fSlow16 * (state["fRec49"][2] + (state["fRec49"][0] + (jnp.float32(2.0) * state["fRec49"][1])))) 
		state["fVec5"] = jnp.float32(fTemp16) 
		state["fRec48"] = -((fSlow47 * ((fSlow23 * fRec48_temp) - (fSlow20 * (fTemp16 - fVec5_temp))))) 
		state["fRec47"] = state["fRec47"].at[0].set((state["fRec48"] - (fSlow48 * ((fSlow49 * state["fRec47"][2]) + (fSlow26 * state["fRec47"][1]))))) 
		fTemp17 = (fSlow34 * state["fRec46"][1]) 
		state["fRec46"] = state["fRec46"].at[0].set(((fSlow46 * (state["fRec47"][2] + (state["fRec47"][0] - (jnp.float32(2.0) * state["fRec47"][1])))) - (fSlow30 * ((fSlow32 * state["fRec46"][2]) + fTemp17)))) 
		fTemp18 = (fSlow42 * state["fRec45"][1]) 
		state["fRec45"] = state["fRec45"].at[0].set(((state["fRec46"][2] + (fSlow30 * (fTemp17 + (fSlow32 * state["fRec46"][0])))) - (fSlow38 * ((fSlow40 * state["fRec45"][2]) + fTemp18)))) 
		state["fRec55"] = -((fSlow47 * ((fSlow23 * fRec55_temp) - (fTemp16 + fVec5_temp)))) 
		state["fRec54"] = state["fRec54"].at[0].set((state["fRec55"] - (fSlow48 * ((fSlow49 * state["fRec54"][2]) + (fSlow26 * state["fRec54"][1]))))) 
		fTemp19 = (fSlow48 * (state["fRec54"][2] + (state["fRec54"][0] + (jnp.float32(2.0) * state["fRec54"][1])))) 
		state["fVec6"] = jnp.float32(fTemp19) 
		state["fRec53"] = -((fSlow54 * ((fSlow31 * fRec53_temp) - (fSlow28 * (fTemp19 - fVec6_temp))))) 
		state["fRec52"] = state["fRec52"].at[0].set((state["fRec53"] - (fSlow55 * ((fSlow56 * state["fRec52"][2]) + (fSlow34 * state["fRec52"][1]))))) 
		fTemp20 = (fSlow42 * state["fRec51"][1]) 
		state["fRec51"] = state["fRec51"].at[0].set(((fSlow53 * (state["fRec52"][2] + (state["fRec52"][0] - (jnp.float32(2.0) * state["fRec52"][1])))) - (fSlow38 * ((fSlow40 * state["fRec51"][2]) + fTemp20)))) 
		state["fRec59"] = -((fSlow54 * ((fSlow31 * fRec59_temp) - (fTemp19 + fVec6_temp)))) 
		state["fRec58"] = state["fRec58"].at[0].set((state["fRec59"] - (fSlow55 * ((fSlow56 * state["fRec58"][2]) + (fSlow34 * state["fRec58"][1]))))) 
		fTemp21 = (fSlow55 * (state["fRec58"][2] + (state["fRec58"][0] + (jnp.float32(2.0) * state["fRec58"][1])))) 
		state["fVec7"] = jnp.float32(fTemp21) 
		state["fRec57"] = -((fSlow60 * ((fSlow39 * fRec57_temp) - (fSlow36 * (fTemp21 - fVec7_temp))))) 
		state["fRec56"] = state["fRec56"].at[0].set((state["fRec57"] - (fSlow57 * ((fSlow61 * state["fRec56"][2]) + (fSlow42 * state["fRec56"][1]))))) 
		state["fRec61"] = -((fSlow60 * ((fSlow39 * fRec61_temp) - (fTemp21 + fVec7_temp)))) 
		state["fRec60"] = state["fRec60"].at[0].set((state["fRec61"] - (fSlow57 * ((fSlow61 * state["fRec60"][2]) + (fSlow42 * state["fRec60"][1]))))) 
		fTemp22 = ((((fSlow67 * (state["fRec40"][2] + (fSlow38 * (fTemp15 + (fSlow40 * state["fRec40"][0]))))) + (fSlow68 * (state["fRec45"][2] + (fSlow38 * (fTemp18 + (fSlow40 * state["fRec45"][0])))))) + (fSlow69 * (state["fRec51"][2] + (fSlow38 * (fTemp20 + (fSlow40 * state["fRec51"][0])))))) + (fSlow57 * ((fSlow70 * (state["fRec56"][2] + (state["fRec56"][0] - (jnp.float32(2.0) * state["fRec56"][1])))) + (fSlow71 * (state["fRec60"][2] + (state["fRec60"][0] + (jnp.float32(2.0) * state["fRec60"][1]))))))) 
		fTemp23 = (fTemp12 + fTemp22) 
		state["fRec66"] = -((fSlow14 * ((fSlow15 * fRec66_temp) - (fSlow11 * (state["fRec4"][1] - state["fRec4"][2]))))) 
		state["fRec65"] = state["fRec65"].at[0].set((state["fRec66"] - (fSlow16 * ((fSlow17 * state["fRec65"][2]) + (fSlow18 * state["fRec65"][1]))))) 
		fTemp24 = (fSlow26 * state["fRec64"][1]) 
		state["fRec64"] = state["fRec64"].at[0].set(((fSlow13 * (state["fRec65"][2] + (state["fRec65"][0] - (jnp.float32(2.0) * state["fRec65"][1])))) - (fSlow22 * ((fSlow24 * state["fRec64"][2]) + fTemp24)))) 
		fTemp25 = (fSlow34 * state["fRec63"][1]) 
		state["fRec63"] = state["fRec63"].at[0].set(((state["fRec64"][2] + (fSlow22 * (fTemp24 + (fSlow24 * state["fRec64"][0])))) - (fSlow30 * ((fSlow32 * state["fRec63"][2]) + fTemp25)))) 
		fTemp26 = (fSlow42 * state["fRec62"][1]) 
		state["fRec62"] = state["fRec62"].at[0].set(((state["fRec63"][2] + (fSlow30 * (fTemp25 + (fSlow32 * state["fRec63"][0])))) - (fSlow38 * ((fSlow40 * state["fRec62"][2]) + fTemp26)))) 
		state["fRec72"] = -((fSlow14 * ((fSlow15 * fRec72_temp) - (state["fRec4"][1] + state["fRec4"][2])))) 
		state["fRec71"] = state["fRec71"].at[0].set((state["fRec72"] - (fSlow16 * ((fSlow17 * state["fRec71"][2]) + (fSlow18 * state["fRec71"][1]))))) 
		fTemp27 = (fSlow16 * (state["fRec71"][2] + (state["fRec71"][0] + (jnp.float32(2.0) * state["fRec71"][1])))) 
		state["fVec8"] = jnp.float32(fTemp27) 
		state["fRec70"] = -((fSlow47 * ((fSlow23 * fRec70_temp) - (fSlow20 * (fTemp27 - fVec8_temp))))) 
		state["fRec69"] = state["fRec69"].at[0].set((state["fRec70"] - (fSlow48 * ((fSlow49 * state["fRec69"][2]) + (fSlow26 * state["fRec69"][1]))))) 
		fTemp28 = (fSlow34 * state["fRec68"][1]) 
		state["fRec68"] = state["fRec68"].at[0].set(((fSlow46 * (state["fRec69"][2] + (state["fRec69"][0] - (jnp.float32(2.0) * state["fRec69"][1])))) - (fSlow30 * ((fSlow32 * state["fRec68"][2]) + fTemp28)))) 
		fTemp29 = (fSlow42 * state["fRec67"][1]) 
		state["fRec67"] = state["fRec67"].at[0].set(((state["fRec68"][2] + (fSlow30 * (fTemp28 + (fSlow32 * state["fRec68"][0])))) - (fSlow38 * ((fSlow40 * state["fRec67"][2]) + fTemp29)))) 
		state["fRec77"] = -((fSlow47 * ((fSlow23 * fRec77_temp) - (fTemp27 + fVec8_temp)))) 
		state["fRec76"] = state["fRec76"].at[0].set((state["fRec77"] - (fSlow48 * ((fSlow49 * state["fRec76"][2]) + (fSlow26 * state["fRec76"][1]))))) 
		fTemp30 = (fSlow48 * (state["fRec76"][2] + (state["fRec76"][0] + (jnp.float32(2.0) * state["fRec76"][1])))) 
		state["fVec9"] = jnp.float32(fTemp30) 
		state["fRec75"] = -((fSlow54 * ((fSlow31 * fRec75_temp) - (fSlow28 * (fTemp30 - fVec9_temp))))) 
		state["fRec74"] = state["fRec74"].at[0].set((state["fRec75"] - (fSlow55 * ((fSlow56 * state["fRec74"][2]) + (fSlow34 * state["fRec74"][1]))))) 
		fTemp31 = (fSlow42 * state["fRec73"][1]) 
		state["fRec73"] = state["fRec73"].at[0].set(((fSlow53 * (state["fRec74"][2] + (state["fRec74"][0] - (jnp.float32(2.0) * state["fRec74"][1])))) - (fSlow38 * ((fSlow40 * state["fRec73"][2]) + fTemp31)))) 
		state["fRec81"] = -((fSlow54 * ((fSlow31 * fRec81_temp) - (fTemp30 + fVec9_temp)))) 
		state["fRec80"] = state["fRec80"].at[0].set((state["fRec81"] - (fSlow55 * ((fSlow56 * state["fRec80"][2]) + (fSlow34 * state["fRec80"][1]))))) 
		fTemp32 = (fSlow55 * (state["fRec80"][2] + (state["fRec80"][0] + (jnp.float32(2.0) * state["fRec80"][1])))) 
		state["fVec10"] = jnp.float32(fTemp32) 
		state["fRec79"] = -((fSlow60 * ((fSlow39 * fRec79_temp) - (fSlow36 * (fTemp32 - fVec10_temp))))) 
		state["fRec78"] = state["fRec78"].at[0].set((state["fRec79"] - (fSlow57 * ((fSlow61 * state["fRec78"][2]) + (fSlow42 * state["fRec78"][1]))))) 
		state["fRec83"] = -((fSlow60 * ((fSlow39 * fRec83_temp) - (fTemp32 + fVec10_temp)))) 
		state["fRec82"] = state["fRec82"].at[0].set((state["fRec83"] - (fSlow57 * ((fSlow61 * state["fRec82"][2]) + (fSlow42 * state["fRec82"][1]))))) 
		fTemp33 = ((((fSlow73 * (state["fRec62"][2] + (fSlow38 * (fTemp26 + (fSlow40 * state["fRec62"][0]))))) + (fSlow74 * (state["fRec67"][2] + (fSlow38 * (fTemp29 + (fSlow40 * state["fRec67"][0])))))) + (fSlow75 * (state["fRec73"][2] + (fSlow38 * (fTemp31 + (fSlow40 * state["fRec73"][0])))))) + (fSlow57 * ((fSlow76 * (state["fRec78"][2] + (state["fRec78"][0] - (jnp.float32(2.0) * state["fRec78"][1])))) + (fSlow77 * (state["fRec82"][2] + (state["fRec82"][0] + (jnp.float32(2.0) * state["fRec82"][1]))))))) 
		state["fRec88"] = -((fSlow14 * ((fSlow15 * fRec88_temp) - (fSlow11 * (state["fRec12"][1] - state["fRec12"][2]))))) 
		state["fRec87"] = state["fRec87"].at[0].set((state["fRec88"] - (fSlow16 * ((fSlow17 * state["fRec87"][2]) + (fSlow18 * state["fRec87"][1]))))) 
		fTemp34 = (fSlow26 * state["fRec86"][1]) 
		state["fRec86"] = state["fRec86"].at[0].set(((fSlow13 * (state["fRec87"][2] + (state["fRec87"][0] - (jnp.float32(2.0) * state["fRec87"][1])))) - (fSlow22 * ((fSlow24 * state["fRec86"][2]) + fTemp34)))) 
		fTemp35 = (fSlow34 * state["fRec85"][1]) 
		state["fRec85"] = state["fRec85"].at[0].set(((state["fRec86"][2] + (fSlow22 * (fTemp34 + (fSlow24 * state["fRec86"][0])))) - (fSlow30 * ((fSlow32 * state["fRec85"][2]) + fTemp35)))) 
		fTemp36 = (fSlow42 * state["fRec84"][1]) 
		state["fRec84"] = state["fRec84"].at[0].set(((state["fRec85"][2] + (fSlow30 * (fTemp35 + (fSlow32 * state["fRec85"][0])))) - (fSlow38 * ((fSlow40 * state["fRec84"][2]) + fTemp36)))) 
		state["fRec94"] = -((fSlow14 * ((fSlow15 * fRec94_temp) - (state["fRec12"][1] + state["fRec12"][2])))) 
		state["fRec93"] = state["fRec93"].at[0].set((state["fRec94"] - (fSlow16 * ((fSlow17 * state["fRec93"][2]) + (fSlow18 * state["fRec93"][1]))))) 
		fTemp37 = (fSlow16 * (state["fRec93"][2] + (state["fRec93"][0] + (jnp.float32(2.0) * state["fRec93"][1])))) 
		state["fVec11"] = jnp.float32(fTemp37) 
		state["fRec92"] = -((fSlow47 * ((fSlow23 * fRec92_temp) - (fSlow20 * (fTemp37 - fVec11_temp))))) 
		state["fRec91"] = state["fRec91"].at[0].set((state["fRec92"] - (fSlow48 * ((fSlow49 * state["fRec91"][2]) + (fSlow26 * state["fRec91"][1]))))) 
		fTemp38 = (fSlow34 * state["fRec90"][1]) 
		state["fRec90"] = state["fRec90"].at[0].set(((fSlow46 * (state["fRec91"][2] + (state["fRec91"][0] - (jnp.float32(2.0) * state["fRec91"][1])))) - (fSlow30 * ((fSlow32 * state["fRec90"][2]) + fTemp38)))) 
		fTemp39 = (fSlow42 * state["fRec89"][1]) 
		state["fRec89"] = state["fRec89"].at[0].set(((state["fRec90"][2] + (fSlow30 * (fTemp38 + (fSlow32 * state["fRec90"][0])))) - (fSlow38 * ((fSlow40 * state["fRec89"][2]) + fTemp39)))) 
		state["fRec99"] = -((fSlow47 * ((fSlow23 * fRec99_temp) - (fTemp37 + fVec11_temp)))) 
		state["fRec98"] = state["fRec98"].at[0].set((state["fRec99"] - (fSlow48 * ((fSlow49 * state["fRec98"][2]) + (fSlow26 * state["fRec98"][1]))))) 
		fTemp40 = (fSlow48 * (state["fRec98"][2] + (state["fRec98"][0] + (jnp.float32(2.0) * state["fRec98"][1])))) 
		state["fVec12"] = jnp.float32(fTemp40) 
		state["fRec97"] = -((fSlow54 * ((fSlow31 * fRec97_temp) - (fSlow28 * (fTemp40 - fVec12_temp))))) 
		state["fRec96"] = state["fRec96"].at[0].set((state["fRec97"] - (fSlow55 * ((fSlow56 * state["fRec96"][2]) + (fSlow34 * state["fRec96"][1]))))) 
		fTemp41 = (fSlow42 * state["fRec95"][1]) 
		state["fRec95"] = state["fRec95"].at[0].set(((fSlow53 * (state["fRec96"][2] + (state["fRec96"][0] - (jnp.float32(2.0) * state["fRec96"][1])))) - (fSlow38 * ((fSlow40 * state["fRec95"][2]) + fTemp41)))) 
		state["fRec103"] = -((fSlow54 * ((fSlow31 * fRec103_temp) - (fTemp40 + fVec12_temp)))) 
		state["fRec102"] = state["fRec102"].at[0].set((state["fRec103"] - (fSlow55 * ((fSlow56 * state["fRec102"][2]) + (fSlow34 * state["fRec102"][1]))))) 
		fTemp42 = (fSlow55 * (state["fRec102"][2] + (state["fRec102"][0] + (jnp.float32(2.0) * state["fRec102"][1])))) 
		state["fVec13"] = jnp.float32(fTemp42) 
		state["fRec101"] = -((fSlow60 * ((fSlow39 * fRec101_temp) - (fSlow36 * (fTemp42 - fVec13_temp))))) 
		state["fRec100"] = state["fRec100"].at[0].set((state["fRec101"] - (fSlow57 * ((fSlow61 * state["fRec100"][2]) + (fSlow42 * state["fRec100"][1]))))) 
		state["fRec105"] = -((fSlow60 * ((fSlow39 * fRec105_temp) - (fTemp42 + fVec13_temp)))) 
		state["fRec104"] = state["fRec104"].at[0].set((state["fRec105"] - (fSlow57 * ((fSlow61 * state["fRec104"][2]) + (fSlow42 * state["fRec104"][1]))))) 
		fTemp43 = ((((fSlow79 * (state["fRec84"][2] + (fSlow38 * (fTemp36 + (fSlow40 * state["fRec84"][0]))))) + (fSlow80 * (state["fRec89"][2] + (fSlow38 * (fTemp39 + (fSlow40 * state["fRec89"][0])))))) + (fSlow81 * (state["fRec95"][2] + (fSlow38 * (fTemp41 + (fSlow40 * state["fRec95"][0])))))) + (fSlow57 * ((fSlow82 * (state["fRec100"][2] + (state["fRec100"][0] - (jnp.float32(2.0) * state["fRec100"][1])))) + (fSlow83 * (state["fRec104"][2] + (state["fRec104"][0] + (jnp.float32(2.0) * state["fRec104"][1]))))))) 
		fTemp44 = (fTemp33 + fTemp43) 
		fTemp45 = (fTemp23 + fTemp44) 
		state["fRec110"] = -((fSlow14 * ((fSlow15 * fRec110_temp) - (fSlow11 * (state["fRec2"][1] - state["fRec2"][2]))))) 
		state["fRec109"] = state["fRec109"].at[0].set((state["fRec110"] - (fSlow16 * ((fSlow17 * state["fRec109"][2]) + (fSlow18 * state["fRec109"][1]))))) 
		fTemp46 = (fSlow26 * state["fRec108"][1]) 
		state["fRec108"] = state["fRec108"].at[0].set(((fSlow13 * (state["fRec109"][2] + (state["fRec109"][0] - (jnp.float32(2.0) * state["fRec109"][1])))) - (fSlow22 * ((fSlow24 * state["fRec108"][2]) + fTemp46)))) 
		fTemp47 = (fSlow34 * state["fRec107"][1]) 
		state["fRec107"] = state["fRec107"].at[0].set(((state["fRec108"][2] + (fSlow22 * (fTemp46 + (fSlow24 * state["fRec108"][0])))) - (fSlow30 * ((fSlow32 * state["fRec107"][2]) + fTemp47)))) 
		fTemp48 = (fSlow42 * state["fRec106"][1]) 
		state["fRec106"] = state["fRec106"].at[0].set(((state["fRec107"][2] + (fSlow30 * (fTemp47 + (fSlow32 * state["fRec107"][0])))) - (fSlow38 * ((fSlow40 * state["fRec106"][2]) + fTemp48)))) 
		state["fRec116"] = -((fSlow14 * ((fSlow15 * fRec116_temp) - (state["fRec2"][1] + state["fRec2"][2])))) 
		state["fRec115"] = state["fRec115"].at[0].set((state["fRec116"] - (fSlow16 * ((fSlow17 * state["fRec115"][2]) + (fSlow18 * state["fRec115"][1]))))) 
		fTemp49 = (fSlow16 * (state["fRec115"][2] + (state["fRec115"][0] + (jnp.float32(2.0) * state["fRec115"][1])))) 
		state["fVec14"] = jnp.float32(fTemp49) 
		state["fRec114"] = -((fSlow47 * ((fSlow23 * fRec114_temp) - (fSlow20 * (fTemp49 - fVec14_temp))))) 
		state["fRec113"] = state["fRec113"].at[0].set((state["fRec114"] - (fSlow48 * ((fSlow49 * state["fRec113"][2]) + (fSlow26 * state["fRec113"][1]))))) 
		fTemp50 = (fSlow34 * state["fRec112"][1]) 
		state["fRec112"] = state["fRec112"].at[0].set(((fSlow46 * (state["fRec113"][2] + (state["fRec113"][0] - (jnp.float32(2.0) * state["fRec113"][1])))) - (fSlow30 * ((fSlow32 * state["fRec112"][2]) + fTemp50)))) 
		fTemp51 = (fSlow42 * state["fRec111"][1]) 
		state["fRec111"] = state["fRec111"].at[0].set(((state["fRec112"][2] + (fSlow30 * (fTemp50 + (fSlow32 * state["fRec112"][0])))) - (fSlow38 * ((fSlow40 * state["fRec111"][2]) + fTemp51)))) 
		state["fRec121"] = -((fSlow47 * ((fSlow23 * fRec121_temp) - (fTemp49 + fVec14_temp)))) 
		state["fRec120"] = state["fRec120"].at[0].set((state["fRec121"] - (fSlow48 * ((fSlow49 * state["fRec120"][2]) + (fSlow26 * state["fRec120"][1]))))) 
		fTemp52 = (fSlow48 * (state["fRec120"][2] + (state["fRec120"][0] + (jnp.float32(2.0) * state["fRec120"][1])))) 
		state["fVec15"] = jnp.float32(fTemp52) 
		state["fRec119"] = -((fSlow54 * ((fSlow31 * fRec119_temp) - (fSlow28 * (fTemp52 - fVec15_temp))))) 
		state["fRec118"] = state["fRec118"].at[0].set((state["fRec119"] - (fSlow55 * ((fSlow56 * state["fRec118"][2]) + (fSlow34 * state["fRec118"][1]))))) 
		fTemp53 = (fSlow42 * state["fRec117"][1]) 
		state["fRec117"] = state["fRec117"].at[0].set(((fSlow53 * (state["fRec118"][2] + (state["fRec118"][0] - (jnp.float32(2.0) * state["fRec118"][1])))) - (fSlow38 * ((fSlow40 * state["fRec117"][2]) + fTemp53)))) 
		state["fRec125"] = -((fSlow54 * ((fSlow31 * fRec125_temp) - (fTemp52 + fVec15_temp)))) 
		state["fRec124"] = state["fRec124"].at[0].set((state["fRec125"] - (fSlow55 * ((fSlow56 * state["fRec124"][2]) + (fSlow34 * state["fRec124"][1]))))) 
		fTemp54 = (fSlow55 * (state["fRec124"][2] + (state["fRec124"][0] + (jnp.float32(2.0) * state["fRec124"][1])))) 
		state["fVec16"] = jnp.float32(fTemp54) 
		state["fRec123"] = -((fSlow60 * ((fSlow39 * fRec123_temp) - (fSlow36 * (fTemp54 - fVec16_temp))))) 
		state["fRec122"] = state["fRec122"].at[0].set((state["fRec123"] - (fSlow57 * ((fSlow61 * state["fRec122"][2]) + (fSlow42 * state["fRec122"][1]))))) 
		state["fRec127"] = -((fSlow60 * ((fSlow39 * fRec127_temp) - (fTemp54 + fVec16_temp)))) 
		state["fRec126"] = state["fRec126"].at[0].set((state["fRec127"] - (fSlow57 * ((fSlow61 * state["fRec126"][2]) + (fSlow42 * state["fRec126"][1]))))) 
		fTemp55 = ((((fSlow85 * (state["fRec106"][2] + (fSlow38 * (fTemp48 + (fSlow40 * state["fRec106"][0]))))) + (fSlow86 * (state["fRec111"][2] + (fSlow38 * (fTemp51 + (fSlow40 * state["fRec111"][0])))))) + (fSlow87 * (state["fRec117"][2] + (fSlow38 * (fTemp53 + (fSlow40 * state["fRec117"][0])))))) + (fSlow57 * ((fSlow88 * (state["fRec122"][2] + (state["fRec122"][0] - (jnp.float32(2.0) * state["fRec122"][1])))) + (fSlow89 * (state["fRec126"][2] + (state["fRec126"][0] + (jnp.float32(2.0) * state["fRec126"][1]))))))) 
		state["fRec132"] = -((fSlow14 * ((fSlow15 * fRec132_temp) - (fSlow11 * (state["fRec10"][1] - state["fRec10"][2]))))) 
		state["fRec131"] = state["fRec131"].at[0].set((state["fRec132"] - (fSlow16 * ((fSlow17 * state["fRec131"][2]) + (fSlow18 * state["fRec131"][1]))))) 
		fTemp56 = (fSlow26 * state["fRec130"][1]) 
		state["fRec130"] = state["fRec130"].at[0].set(((fSlow13 * (state["fRec131"][2] + (state["fRec131"][0] - (jnp.float32(2.0) * state["fRec131"][1])))) - (fSlow22 * ((fSlow24 * state["fRec130"][2]) + fTemp56)))) 
		fTemp57 = (fSlow34 * state["fRec129"][1]) 
		state["fRec129"] = state["fRec129"].at[0].set(((state["fRec130"][2] + (fSlow22 * (fTemp56 + (fSlow24 * state["fRec130"][0])))) - (fSlow30 * ((fSlow32 * state["fRec129"][2]) + fTemp57)))) 
		fTemp58 = (fSlow42 * state["fRec128"][1]) 
		state["fRec128"] = state["fRec128"].at[0].set(((state["fRec129"][2] + (fSlow30 * (fTemp57 + (fSlow32 * state["fRec129"][0])))) - (fSlow38 * ((fSlow40 * state["fRec128"][2]) + fTemp58)))) 
		state["fRec138"] = -((fSlow14 * ((fSlow15 * fRec138_temp) - (state["fRec10"][1] + state["fRec10"][2])))) 
		state["fRec137"] = state["fRec137"].at[0].set((state["fRec138"] - (fSlow16 * ((fSlow17 * state["fRec137"][2]) + (fSlow18 * state["fRec137"][1]))))) 
		fTemp59 = (fSlow16 * (state["fRec137"][2] + (state["fRec137"][0] + (jnp.float32(2.0) * state["fRec137"][1])))) 
		state["fVec17"] = jnp.float32(fTemp59) 
		state["fRec136"] = -((fSlow47 * ((fSlow23 * fRec136_temp) - (fSlow20 * (fTemp59 - fVec17_temp))))) 
		state["fRec135"] = state["fRec135"].at[0].set((state["fRec136"] - (fSlow48 * ((fSlow49 * state["fRec135"][2]) + (fSlow26 * state["fRec135"][1]))))) 
		fTemp60 = (fSlow34 * state["fRec134"][1]) 
		state["fRec134"] = state["fRec134"].at[0].set(((fSlow46 * (state["fRec135"][2] + (state["fRec135"][0] - (jnp.float32(2.0) * state["fRec135"][1])))) - (fSlow30 * ((fSlow32 * state["fRec134"][2]) + fTemp60)))) 
		fTemp61 = (fSlow42 * state["fRec133"][1]) 
		state["fRec133"] = state["fRec133"].at[0].set(((state["fRec134"][2] + (fSlow30 * (fTemp60 + (fSlow32 * state["fRec134"][0])))) - (fSlow38 * ((fSlow40 * state["fRec133"][2]) + fTemp61)))) 
		state["fRec143"] = -((fSlow47 * ((fSlow23 * fRec143_temp) - (fTemp59 + fVec17_temp)))) 
		state["fRec142"] = state["fRec142"].at[0].set((state["fRec143"] - (fSlow48 * ((fSlow49 * state["fRec142"][2]) + (fSlow26 * state["fRec142"][1]))))) 
		fTemp62 = (fSlow48 * (state["fRec142"][2] + (state["fRec142"][0] + (jnp.float32(2.0) * state["fRec142"][1])))) 
		state["fVec18"] = jnp.float32(fTemp62) 
		state["fRec141"] = -((fSlow54 * ((fSlow31 * fRec141_temp) - (fSlow28 * (fTemp62 - fVec18_temp))))) 
		state["fRec140"] = state["fRec140"].at[0].set((state["fRec141"] - (fSlow55 * ((fSlow56 * state["fRec140"][2]) + (fSlow34 * state["fRec140"][1]))))) 
		fTemp63 = (fSlow42 * state["fRec139"][1]) 
		state["fRec139"] = state["fRec139"].at[0].set(((fSlow53 * (state["fRec140"][2] + (state["fRec140"][0] - (jnp.float32(2.0) * state["fRec140"][1])))) - (fSlow38 * ((fSlow40 * state["fRec139"][2]) + fTemp63)))) 
		state["fRec147"] = -((fSlow54 * ((fSlow31 * fRec147_temp) - (fTemp62 + fVec18_temp)))) 
		state["fRec146"] = state["fRec146"].at[0].set((state["fRec147"] - (fSlow55 * ((fSlow56 * state["fRec146"][2]) + (fSlow34 * state["fRec146"][1]))))) 
		fTemp64 = (fSlow55 * (state["fRec146"][2] + (state["fRec146"][0] + (jnp.float32(2.0) * state["fRec146"][1])))) 
		state["fVec19"] = jnp.float32(fTemp64) 
		state["fRec145"] = -((fSlow60 * ((fSlow39 * fRec145_temp) - (fSlow36 * (fTemp64 - fVec19_temp))))) 
		state["fRec144"] = state["fRec144"].at[0].set((state["fRec145"] - (fSlow57 * ((fSlow61 * state["fRec144"][2]) + (fSlow42 * state["fRec144"][1]))))) 
		state["fRec149"] = -((fSlow60 * ((fSlow39 * fRec149_temp) - (fTemp64 + fVec19_temp)))) 
		state["fRec148"] = state["fRec148"].at[0].set((state["fRec149"] - (fSlow57 * ((fSlow61 * state["fRec148"][2]) + (fSlow42 * state["fRec148"][1]))))) 
		fTemp65 = ((((fSlow91 * (state["fRec128"][2] + (fSlow38 * (fTemp58 + (fSlow40 * state["fRec128"][0]))))) + (fSlow92 * (state["fRec133"][2] + (fSlow38 * (fTemp61 + (fSlow40 * state["fRec133"][0])))))) + (fSlow93 * (state["fRec139"][2] + (fSlow38 * (fTemp63 + (fSlow40 * state["fRec139"][0])))))) + (fSlow57 * ((fSlow94 * (state["fRec144"][2] + (state["fRec144"][0] - (jnp.float32(2.0) * state["fRec144"][1])))) + (fSlow95 * (state["fRec148"][2] + (state["fRec148"][0] + (jnp.float32(2.0) * state["fRec148"][1]))))))) 
		fTemp66 = (fTemp55 + fTemp65) 
		state["fRec154"] = -((fSlow14 * ((fSlow15 * fRec154_temp) - (fSlow11 * (state["fRec6"][1] - state["fRec6"][2]))))) 
		state["fRec153"] = state["fRec153"].at[0].set((state["fRec154"] - (fSlow16 * ((fSlow17 * state["fRec153"][2]) + (fSlow18 * state["fRec153"][1]))))) 
		fTemp67 = (fSlow26 * state["fRec152"][1]) 
		state["fRec152"] = state["fRec152"].at[0].set(((fSlow13 * (state["fRec153"][2] + (state["fRec153"][0] - (jnp.float32(2.0) * state["fRec153"][1])))) - (fSlow22 * ((fSlow24 * state["fRec152"][2]) + fTemp67)))) 
		fTemp68 = (fSlow34 * state["fRec151"][1]) 
		state["fRec151"] = state["fRec151"].at[0].set(((state["fRec152"][2] + (fSlow22 * (fTemp67 + (fSlow24 * state["fRec152"][0])))) - (fSlow30 * ((fSlow32 * state["fRec151"][2]) + fTemp68)))) 
		fTemp69 = (fSlow42 * state["fRec150"][1]) 
		state["fRec150"] = state["fRec150"].at[0].set(((state["fRec151"][2] + (fSlow30 * (fTemp68 + (fSlow32 * state["fRec151"][0])))) - (fSlow38 * ((fSlow40 * state["fRec150"][2]) + fTemp69)))) 
		state["fRec160"] = -((fSlow14 * ((fSlow15 * fRec160_temp) - (state["fRec6"][1] + state["fRec6"][2])))) 
		state["fRec159"] = state["fRec159"].at[0].set((state["fRec160"] - (fSlow16 * ((fSlow17 * state["fRec159"][2]) + (fSlow18 * state["fRec159"][1]))))) 
		fTemp70 = (fSlow16 * (state["fRec159"][2] + (state["fRec159"][0] + (jnp.float32(2.0) * state["fRec159"][1])))) 
		state["fVec20"] = jnp.float32(fTemp70) 
		state["fRec158"] = -((fSlow47 * ((fSlow23 * fRec158_temp) - (fSlow20 * (fTemp70 - fVec20_temp))))) 
		state["fRec157"] = state["fRec157"].at[0].set((state["fRec158"] - (fSlow48 * ((fSlow49 * state["fRec157"][2]) + (fSlow26 * state["fRec157"][1]))))) 
		fTemp71 = (fSlow34 * state["fRec156"][1]) 
		state["fRec156"] = state["fRec156"].at[0].set(((fSlow46 * (state["fRec157"][2] + (state["fRec157"][0] - (jnp.float32(2.0) * state["fRec157"][1])))) - (fSlow30 * ((fSlow32 * state["fRec156"][2]) + fTemp71)))) 
		fTemp72 = (fSlow42 * state["fRec155"][1]) 
		state["fRec155"] = state["fRec155"].at[0].set(((state["fRec156"][2] + (fSlow30 * (fTemp71 + (fSlow32 * state["fRec156"][0])))) - (fSlow38 * ((fSlow40 * state["fRec155"][2]) + fTemp72)))) 
		state["fRec165"] = -((fSlow47 * ((fSlow23 * fRec165_temp) - (fTemp70 + fVec20_temp)))) 
		state["fRec164"] = state["fRec164"].at[0].set((state["fRec165"] - (fSlow48 * ((fSlow49 * state["fRec164"][2]) + (fSlow26 * state["fRec164"][1]))))) 
		fTemp73 = (fSlow48 * (state["fRec164"][2] + (state["fRec164"][0] + (jnp.float32(2.0) * state["fRec164"][1])))) 
		state["fVec21"] = jnp.float32(fTemp73) 
		state["fRec163"] = -((fSlow54 * ((fSlow31 * fRec163_temp) - (fSlow28 * (fTemp73 - fVec21_temp))))) 
		state["fRec162"] = state["fRec162"].at[0].set((state["fRec163"] - (fSlow55 * ((fSlow56 * state["fRec162"][2]) + (fSlow34 * state["fRec162"][1]))))) 
		fTemp74 = (fSlow42 * state["fRec161"][1]) 
		state["fRec161"] = state["fRec161"].at[0].set(((fSlow53 * (state["fRec162"][2] + (state["fRec162"][0] - (jnp.float32(2.0) * state["fRec162"][1])))) - (fSlow38 * ((fSlow40 * state["fRec161"][2]) + fTemp74)))) 
		state["fRec169"] = -((fSlow54 * ((fSlow31 * fRec169_temp) - (fTemp73 + fVec21_temp)))) 
		state["fRec168"] = state["fRec168"].at[0].set((state["fRec169"] - (fSlow55 * ((fSlow56 * state["fRec168"][2]) + (fSlow34 * state["fRec168"][1]))))) 
		fTemp75 = (fSlow55 * (state["fRec168"][2] + (state["fRec168"][0] + (jnp.float32(2.0) * state["fRec168"][1])))) 
		state["fVec22"] = jnp.float32(fTemp75) 
		state["fRec167"] = -((fSlow60 * ((fSlow39 * fRec167_temp) - (fSlow36 * (fTemp75 - fVec22_temp))))) 
		state["fRec166"] = state["fRec166"].at[0].set((state["fRec167"] - (fSlow57 * ((fSlow61 * state["fRec166"][2]) + (fSlow42 * state["fRec166"][1]))))) 
		state["fRec171"] = -((fSlow60 * ((fSlow39 * fRec171_temp) - (fTemp75 + fVec22_temp)))) 
		state["fRec170"] = state["fRec170"].at[0].set((state["fRec171"] - (fSlow57 * ((fSlow61 * state["fRec170"][2]) + (fSlow42 * state["fRec170"][1]))))) 
		fTemp76 = ((((fSlow97 * (state["fRec150"][2] + (fSlow38 * (fTemp69 + (fSlow40 * state["fRec150"][0]))))) + (fSlow98 * (state["fRec155"][2] + (fSlow38 * (fTemp72 + (fSlow40 * state["fRec155"][0])))))) + (fSlow99 * (state["fRec161"][2] + (fSlow38 * (fTemp74 + (fSlow40 * state["fRec161"][0])))))) + (fSlow57 * ((fSlow100 * (state["fRec166"][2] + (state["fRec166"][0] - (jnp.float32(2.0) * state["fRec166"][1])))) + (fSlow101 * (state["fRec170"][2] + (state["fRec170"][0] + (jnp.float32(2.0) * state["fRec170"][1]))))))) 
		state["fRec176"] = -((fSlow14 * ((fSlow15 * fRec176_temp) - (fSlow11 * (state["fRec14"][1] - state["fRec14"][2]))))) 
		state["fRec175"] = state["fRec175"].at[0].set((state["fRec176"] - (fSlow16 * ((fSlow17 * state["fRec175"][2]) + (fSlow18 * state["fRec175"][1]))))) 
		fTemp77 = (fSlow26 * state["fRec174"][1]) 
		state["fRec174"] = state["fRec174"].at[0].set(((fSlow13 * (state["fRec175"][2] + (state["fRec175"][0] - (jnp.float32(2.0) * state["fRec175"][1])))) - (fSlow22 * ((fSlow24 * state["fRec174"][2]) + fTemp77)))) 
		fTemp78 = (fSlow34 * state["fRec173"][1]) 
		state["fRec173"] = state["fRec173"].at[0].set(((state["fRec174"][2] + (fSlow22 * (fTemp77 + (fSlow24 * state["fRec174"][0])))) - (fSlow30 * ((fSlow32 * state["fRec173"][2]) + fTemp78)))) 
		fTemp79 = (fSlow42 * state["fRec172"][1]) 
		state["fRec172"] = state["fRec172"].at[0].set(((state["fRec173"][2] + (fSlow30 * (fTemp78 + (fSlow32 * state["fRec173"][0])))) - (fSlow38 * ((fSlow40 * state["fRec172"][2]) + fTemp79)))) 
		state["fRec182"] = -((fSlow14 * ((fSlow15 * fRec182_temp) - (state["fRec14"][1] + state["fRec14"][2])))) 
		state["fRec181"] = state["fRec181"].at[0].set((state["fRec182"] - (fSlow16 * ((fSlow17 * state["fRec181"][2]) + (fSlow18 * state["fRec181"][1]))))) 
		fTemp80 = (fSlow16 * (state["fRec181"][2] + (state["fRec181"][0] + (jnp.float32(2.0) * state["fRec181"][1])))) 
		state["fVec23"] = jnp.float32(fTemp80) 
		state["fRec180"] = -((fSlow47 * ((fSlow23 * fRec180_temp) - (fSlow20 * (fTemp80 - fVec23_temp))))) 
		state["fRec179"] = state["fRec179"].at[0].set((state["fRec180"] - (fSlow48 * ((fSlow49 * state["fRec179"][2]) + (fSlow26 * state["fRec179"][1]))))) 
		fTemp81 = (fSlow34 * state["fRec178"][1]) 
		state["fRec178"] = state["fRec178"].at[0].set(((fSlow46 * (state["fRec179"][2] + (state["fRec179"][0] - (jnp.float32(2.0) * state["fRec179"][1])))) - (fSlow30 * ((fSlow32 * state["fRec178"][2]) + fTemp81)))) 
		fTemp82 = (fSlow42 * state["fRec177"][1]) 
		state["fRec177"] = state["fRec177"].at[0].set(((state["fRec178"][2] + (fSlow30 * (fTemp81 + (fSlow32 * state["fRec178"][0])))) - (fSlow38 * ((fSlow40 * state["fRec177"][2]) + fTemp82)))) 
		state["fRec187"] = -((fSlow47 * ((fSlow23 * fRec187_temp) - (fTemp80 + fVec23_temp)))) 
		state["fRec186"] = state["fRec186"].at[0].set((state["fRec187"] - (fSlow48 * ((fSlow49 * state["fRec186"][2]) + (fSlow26 * state["fRec186"][1]))))) 
		fTemp83 = (fSlow48 * (state["fRec186"][2] + (state["fRec186"][0] + (jnp.float32(2.0) * state["fRec186"][1])))) 
		state["fVec24"] = jnp.float32(fTemp83) 
		state["fRec185"] = -((fSlow54 * ((fSlow31 * fRec185_temp) - (fSlow28 * (fTemp83 - fVec24_temp))))) 
		state["fRec184"] = state["fRec184"].at[0].set((state["fRec185"] - (fSlow55 * ((fSlow56 * state["fRec184"][2]) + (fSlow34 * state["fRec184"][1]))))) 
		fTemp84 = (fSlow42 * state["fRec183"][1]) 
		state["fRec183"] = state["fRec183"].at[0].set(((fSlow53 * (state["fRec184"][2] + (state["fRec184"][0] - (jnp.float32(2.0) * state["fRec184"][1])))) - (fSlow38 * ((fSlow40 * state["fRec183"][2]) + fTemp84)))) 
		state["fRec191"] = -((fSlow54 * ((fSlow31 * fRec191_temp) - (fTemp83 + fVec24_temp)))) 
		state["fRec190"] = state["fRec190"].at[0].set((state["fRec191"] - (fSlow55 * ((fSlow56 * state["fRec190"][2]) + (fSlow34 * state["fRec190"][1]))))) 
		fTemp85 = (fSlow55 * (state["fRec190"][2] + (state["fRec190"][0] + (jnp.float32(2.0) * state["fRec190"][1])))) 
		state["fVec25"] = jnp.float32(fTemp85) 
		state["fRec189"] = -((fSlow60 * ((fSlow39 * fRec189_temp) - (fSlow36 * (fTemp85 - fVec25_temp))))) 
		state["fRec188"] = state["fRec188"].at[0].set((state["fRec189"] - (fSlow57 * ((fSlow61 * state["fRec188"][2]) + (fSlow42 * state["fRec188"][1]))))) 
		state["fRec193"] = -((fSlow60 * ((fSlow39 * fRec193_temp) - (fTemp85 + fVec25_temp)))) 
		state["fRec192"] = state["fRec192"].at[0].set((state["fRec193"] - (fSlow57 * ((fSlow61 * state["fRec192"][2]) + (fSlow42 * state["fRec192"][1]))))) 
		fTemp86 = ((((fSlow103 * (state["fRec172"][2] + (fSlow38 * (fTemp79 + (fSlow40 * state["fRec172"][0]))))) + (fSlow104 * (state["fRec177"][2] + (fSlow38 * (fTemp82 + (fSlow40 * state["fRec177"][0])))))) + (fSlow105 * (state["fRec183"][2] + (fSlow38 * (fTemp84 + (fSlow40 * state["fRec183"][0])))))) + (fSlow57 * ((fSlow106 * (state["fRec188"][2] + (state["fRec188"][0] - (jnp.float32(2.0) * state["fRec188"][1])))) + (fSlow107 * (state["fRec192"][2] + (state["fRec192"][0] + (jnp.float32(2.0) * state["fRec192"][1]))))))) 
		fTemp87 = (fTemp76 + fTemp86) 
		fTemp88 = (fTemp66 + fTemp87) 
		fTemp89 = (fTemp45 + fTemp88) 
		state["fRec198"] = -((fSlow14 * ((fSlow15 * fRec198_temp) - (fSlow11 * (state["fRec1"][1] - state["fRec1"][2]))))) 
		state["fRec197"] = state["fRec197"].at[0].set((state["fRec198"] - (fSlow16 * ((fSlow17 * state["fRec197"][2]) + (fSlow18 * state["fRec197"][1]))))) 
		fTemp90 = (fSlow26 * state["fRec196"][1]) 
		state["fRec196"] = state["fRec196"].at[0].set(((fSlow13 * (state["fRec197"][2] + (state["fRec197"][0] - (jnp.float32(2.0) * state["fRec197"][1])))) - (fSlow22 * ((fSlow24 * state["fRec196"][2]) + fTemp90)))) 
		fTemp91 = (fSlow34 * state["fRec195"][1]) 
		state["fRec195"] = state["fRec195"].at[0].set(((state["fRec196"][2] + (fSlow22 * (fTemp90 + (fSlow24 * state["fRec196"][0])))) - (fSlow30 * ((fSlow32 * state["fRec195"][2]) + fTemp91)))) 
		fTemp92 = (fSlow42 * state["fRec194"][1]) 
		state["fRec194"] = state["fRec194"].at[0].set(((state["fRec195"][2] + (fSlow30 * (fTemp91 + (fSlow32 * state["fRec195"][0])))) - (fSlow38 * ((fSlow40 * state["fRec194"][2]) + fTemp92)))) 
		state["fRec204"] = -((fSlow14 * ((fSlow15 * fRec204_temp) - (state["fRec1"][1] + state["fRec1"][2])))) 
		state["fRec203"] = state["fRec203"].at[0].set((state["fRec204"] - (fSlow16 * ((fSlow17 * state["fRec203"][2]) + (fSlow18 * state["fRec203"][1]))))) 
		fTemp93 = (fSlow16 * (state["fRec203"][2] + (state["fRec203"][0] + (jnp.float32(2.0) * state["fRec203"][1])))) 
		state["fVec26"] = jnp.float32(fTemp93) 
		state["fRec202"] = -((fSlow47 * ((fSlow23 * fRec202_temp) - (fSlow20 * (fTemp93 - fVec26_temp))))) 
		state["fRec201"] = state["fRec201"].at[0].set((state["fRec202"] - (fSlow48 * ((fSlow49 * state["fRec201"][2]) + (fSlow26 * state["fRec201"][1]))))) 
		fTemp94 = (fSlow34 * state["fRec200"][1]) 
		state["fRec200"] = state["fRec200"].at[0].set(((fSlow46 * (state["fRec201"][2] + (state["fRec201"][0] - (jnp.float32(2.0) * state["fRec201"][1])))) - (fSlow30 * ((fSlow32 * state["fRec200"][2]) + fTemp94)))) 
		fTemp95 = (fSlow42 * state["fRec199"][1]) 
		state["fRec199"] = state["fRec199"].at[0].set(((state["fRec200"][2] + (fSlow30 * (fTemp94 + (fSlow32 * state["fRec200"][0])))) - (fSlow38 * ((fSlow40 * state["fRec199"][2]) + fTemp95)))) 
		state["fRec209"] = -((fSlow47 * ((fSlow23 * fRec209_temp) - (fTemp93 + fVec26_temp)))) 
		state["fRec208"] = state["fRec208"].at[0].set((state["fRec209"] - (fSlow48 * ((fSlow49 * state["fRec208"][2]) + (fSlow26 * state["fRec208"][1]))))) 
		fTemp96 = (fSlow48 * (state["fRec208"][2] + (state["fRec208"][0] + (jnp.float32(2.0) * state["fRec208"][1])))) 
		state["fVec27"] = jnp.float32(fTemp96) 
		state["fRec207"] = -((fSlow54 * ((fSlow31 * fRec207_temp) - (fSlow28 * (fTemp96 - fVec27_temp))))) 
		state["fRec206"] = state["fRec206"].at[0].set((state["fRec207"] - (fSlow55 * ((fSlow56 * state["fRec206"][2]) + (fSlow34 * state["fRec206"][1]))))) 
		fTemp97 = (fSlow42 * state["fRec205"][1]) 
		state["fRec205"] = state["fRec205"].at[0].set(((fSlow53 * (state["fRec206"][2] + (state["fRec206"][0] - (jnp.float32(2.0) * state["fRec206"][1])))) - (fSlow38 * ((fSlow40 * state["fRec205"][2]) + fTemp97)))) 
		state["fRec213"] = -((fSlow54 * ((fSlow31 * fRec213_temp) - (fTemp96 + fVec27_temp)))) 
		state["fRec212"] = state["fRec212"].at[0].set((state["fRec213"] - (fSlow55 * ((fSlow56 * state["fRec212"][2]) + (fSlow34 * state["fRec212"][1]))))) 
		fTemp98 = (fSlow55 * (state["fRec212"][2] + (state["fRec212"][0] + (jnp.float32(2.0) * state["fRec212"][1])))) 
		state["fVec28"] = jnp.float32(fTemp98) 
		state["fRec211"] = -((fSlow60 * ((fSlow39 * fRec211_temp) - (fSlow36 * (fTemp98 - fVec28_temp))))) 
		state["fRec210"] = state["fRec210"].at[0].set((state["fRec211"] - (fSlow57 * ((fSlow61 * state["fRec210"][2]) + (fSlow42 * state["fRec210"][1]))))) 
		state["fRec215"] = -((fSlow60 * ((fSlow39 * fRec215_temp) - (fTemp98 + fVec28_temp)))) 
		state["fRec214"] = state["fRec214"].at[0].set((state["fRec215"] - (fSlow57 * ((fSlow61 * state["fRec214"][2]) + (fSlow42 * state["fRec214"][1]))))) 
		fTemp99 = ((((fSlow109 * (state["fRec194"][2] + (fSlow38 * (fTemp92 + (fSlow40 * state["fRec194"][0]))))) + (fSlow110 * (state["fRec199"][2] + (fSlow38 * (fTemp95 + (fSlow40 * state["fRec199"][0])))))) + (fSlow111 * (state["fRec205"][2] + (fSlow38 * (fTemp97 + (fSlow40 * state["fRec205"][0])))))) + (fSlow57 * ((fSlow112 * (state["fRec210"][2] + (state["fRec210"][0] - (jnp.float32(2.0) * state["fRec210"][1])))) + (fSlow113 * (state["fRec214"][2] + (state["fRec214"][0] + (jnp.float32(2.0) * state["fRec214"][1]))))))) 
		state["fRec220"] = -((fSlow14 * ((fSlow15 * fRec220_temp) - (fSlow11 * (state["fRec9"][1] - state["fRec9"][2]))))) 
		state["fRec219"] = state["fRec219"].at[0].set((state["fRec220"] - (fSlow16 * ((fSlow17 * state["fRec219"][2]) + (fSlow18 * state["fRec219"][1]))))) 
		fTemp100 = (fSlow26 * state["fRec218"][1]) 
		state["fRec218"] = state["fRec218"].at[0].set(((fSlow13 * (state["fRec219"][2] + (state["fRec219"][0] - (jnp.float32(2.0) * state["fRec219"][1])))) - (fSlow22 * ((fSlow24 * state["fRec218"][2]) + fTemp100)))) 
		fTemp101 = (fSlow34 * state["fRec217"][1]) 
		state["fRec217"] = state["fRec217"].at[0].set(((state["fRec218"][2] + (fSlow22 * (fTemp100 + (fSlow24 * state["fRec218"][0])))) - (fSlow30 * ((fSlow32 * state["fRec217"][2]) + fTemp101)))) 
		fTemp102 = (fSlow42 * state["fRec216"][1]) 
		state["fRec216"] = state["fRec216"].at[0].set(((state["fRec217"][2] + (fSlow30 * (fTemp101 + (fSlow32 * state["fRec217"][0])))) - (fSlow38 * ((fSlow40 * state["fRec216"][2]) + fTemp102)))) 
		state["fRec226"] = -((fSlow14 * ((fSlow15 * fRec226_temp) - (state["fRec9"][1] + state["fRec9"][2])))) 
		state["fRec225"] = state["fRec225"].at[0].set((state["fRec226"] - (fSlow16 * ((fSlow17 * state["fRec225"][2]) + (fSlow18 * state["fRec225"][1]))))) 
		fTemp103 = (fSlow16 * (state["fRec225"][2] + (state["fRec225"][0] + (jnp.float32(2.0) * state["fRec225"][1])))) 
		state["fVec29"] = jnp.float32(fTemp103) 
		state["fRec224"] = -((fSlow47 * ((fSlow23 * fRec224_temp) - (fSlow20 * (fTemp103 - fVec29_temp))))) 
		state["fRec223"] = state["fRec223"].at[0].set((state["fRec224"] - (fSlow48 * ((fSlow49 * state["fRec223"][2]) + (fSlow26 * state["fRec223"][1]))))) 
		fTemp104 = (fSlow34 * state["fRec222"][1]) 
		state["fRec222"] = state["fRec222"].at[0].set(((fSlow46 * (state["fRec223"][2] + (state["fRec223"][0] - (jnp.float32(2.0) * state["fRec223"][1])))) - (fSlow30 * ((fSlow32 * state["fRec222"][2]) + fTemp104)))) 
		fTemp105 = (fSlow42 * state["fRec221"][1]) 
		state["fRec221"] = state["fRec221"].at[0].set(((state["fRec222"][2] + (fSlow30 * (fTemp104 + (fSlow32 * state["fRec222"][0])))) - (fSlow38 * ((fSlow40 * state["fRec221"][2]) + fTemp105)))) 
		state["fRec231"] = -((fSlow47 * ((fSlow23 * fRec231_temp) - (fTemp103 + fVec29_temp)))) 
		state["fRec230"] = state["fRec230"].at[0].set((state["fRec231"] - (fSlow48 * ((fSlow49 * state["fRec230"][2]) + (fSlow26 * state["fRec230"][1]))))) 
		fTemp106 = (fSlow48 * (state["fRec230"][2] + (state["fRec230"][0] + (jnp.float32(2.0) * state["fRec230"][1])))) 
		state["fVec30"] = jnp.float32(fTemp106) 
		state["fRec229"] = -((fSlow54 * ((fSlow31 * fRec229_temp) - (fSlow28 * (fTemp106 - fVec30_temp))))) 
		state["fRec228"] = state["fRec228"].at[0].set((state["fRec229"] - (fSlow55 * ((fSlow56 * state["fRec228"][2]) + (fSlow34 * state["fRec228"][1]))))) 
		fTemp107 = (fSlow42 * state["fRec227"][1]) 
		state["fRec227"] = state["fRec227"].at[0].set(((fSlow53 * (state["fRec228"][2] + (state["fRec228"][0] - (jnp.float32(2.0) * state["fRec228"][1])))) - (fSlow38 * ((fSlow40 * state["fRec227"][2]) + fTemp107)))) 
		state["fRec235"] = -((fSlow54 * ((fSlow31 * fRec235_temp) - (fTemp106 + fVec30_temp)))) 
		state["fRec234"] = state["fRec234"].at[0].set((state["fRec235"] - (fSlow55 * ((fSlow56 * state["fRec234"][2]) + (fSlow34 * state["fRec234"][1]))))) 
		fTemp108 = (fSlow55 * (state["fRec234"][2] + (state["fRec234"][0] + (jnp.float32(2.0) * state["fRec234"][1])))) 
		state["fVec31"] = jnp.float32(fTemp108) 
		state["fRec233"] = -((fSlow60 * ((fSlow39 * fRec233_temp) - (fSlow36 * (fTemp108 - fVec31_temp))))) 
		state["fRec232"] = state["fRec232"].at[0].set((state["fRec233"] - (fSlow57 * ((fSlow61 * state["fRec232"][2]) + (fSlow42 * state["fRec232"][1]))))) 
		state["fRec237"] = -((fSlow60 * ((fSlow39 * fRec237_temp) - (fTemp108 + fVec31_temp)))) 
		state["fRec236"] = state["fRec236"].at[0].set((state["fRec237"] - (fSlow57 * ((fSlow61 * state["fRec236"][2]) + (fSlow42 * state["fRec236"][1]))))) 
		fTemp109 = ((((fSlow115 * (state["fRec216"][2] + (fSlow38 * (fTemp102 + (fSlow40 * state["fRec216"][0]))))) + (fSlow116 * (state["fRec221"][2] + (fSlow38 * (fTemp105 + (fSlow40 * state["fRec221"][0])))))) + (fSlow117 * (state["fRec227"][2] + (fSlow38 * (fTemp107 + (fSlow40 * state["fRec227"][0])))))) + (fSlow57 * ((fSlow118 * (state["fRec232"][2] + (state["fRec232"][0] - (jnp.float32(2.0) * state["fRec232"][1])))) + (fSlow119 * (state["fRec236"][2] + (state["fRec236"][0] + (jnp.float32(2.0) * state["fRec236"][1]))))))) 
		fTemp110 = (fTemp99 + fTemp109) 
		state["fRec242"] = -((fSlow14 * ((fSlow15 * fRec242_temp) - (fSlow11 * (state["fRec5"][1] - state["fRec5"][2]))))) 
		state["fRec241"] = state["fRec241"].at[0].set((state["fRec242"] - (fSlow16 * ((fSlow17 * state["fRec241"][2]) + (fSlow18 * state["fRec241"][1]))))) 
		fTemp111 = (fSlow26 * state["fRec240"][1]) 
		state["fRec240"] = state["fRec240"].at[0].set(((fSlow13 * (state["fRec241"][2] + (state["fRec241"][0] - (jnp.float32(2.0) * state["fRec241"][1])))) - (fSlow22 * ((fSlow24 * state["fRec240"][2]) + fTemp111)))) 
		fTemp112 = (fSlow34 * state["fRec239"][1]) 
		state["fRec239"] = state["fRec239"].at[0].set(((state["fRec240"][2] + (fSlow22 * (fTemp111 + (fSlow24 * state["fRec240"][0])))) - (fSlow30 * ((fSlow32 * state["fRec239"][2]) + fTemp112)))) 
		fTemp113 = (fSlow42 * state["fRec238"][1]) 
		state["fRec238"] = state["fRec238"].at[0].set(((state["fRec239"][2] + (fSlow30 * (fTemp112 + (fSlow32 * state["fRec239"][0])))) - (fSlow38 * ((fSlow40 * state["fRec238"][2]) + fTemp113)))) 
		state["fRec248"] = -((fSlow14 * ((fSlow15 * fRec248_temp) - (state["fRec5"][1] + state["fRec5"][2])))) 
		state["fRec247"] = state["fRec247"].at[0].set((state["fRec248"] - (fSlow16 * ((fSlow17 * state["fRec247"][2]) + (fSlow18 * state["fRec247"][1]))))) 
		fTemp114 = (fSlow16 * (state["fRec247"][2] + (state["fRec247"][0] + (jnp.float32(2.0) * state["fRec247"][1])))) 
		state["fVec32"] = jnp.float32(fTemp114) 
		state["fRec246"] = -((fSlow47 * ((fSlow23 * fRec246_temp) - (fSlow20 * (fTemp114 - fVec32_temp))))) 
		state["fRec245"] = state["fRec245"].at[0].set((state["fRec246"] - (fSlow48 * ((fSlow49 * state["fRec245"][2]) + (fSlow26 * state["fRec245"][1]))))) 
		fTemp115 = (fSlow34 * state["fRec244"][1]) 
		state["fRec244"] = state["fRec244"].at[0].set(((fSlow46 * (state["fRec245"][2] + (state["fRec245"][0] - (jnp.float32(2.0) * state["fRec245"][1])))) - (fSlow30 * ((fSlow32 * state["fRec244"][2]) + fTemp115)))) 
		fTemp116 = (fSlow42 * state["fRec243"][1]) 
		state["fRec243"] = state["fRec243"].at[0].set(((state["fRec244"][2] + (fSlow30 * (fTemp115 + (fSlow32 * state["fRec244"][0])))) - (fSlow38 * ((fSlow40 * state["fRec243"][2]) + fTemp116)))) 
		state["fRec253"] = -((fSlow47 * ((fSlow23 * fRec253_temp) - (fTemp114 + fVec32_temp)))) 
		state["fRec252"] = state["fRec252"].at[0].set((state["fRec253"] - (fSlow48 * ((fSlow49 * state["fRec252"][2]) + (fSlow26 * state["fRec252"][1]))))) 
		fTemp117 = (fSlow48 * (state["fRec252"][2] + (state["fRec252"][0] + (jnp.float32(2.0) * state["fRec252"][1])))) 
		state["fVec33"] = jnp.float32(fTemp117) 
		state["fRec251"] = -((fSlow54 * ((fSlow31 * fRec251_temp) - (fSlow28 * (fTemp117 - fVec33_temp))))) 
		state["fRec250"] = state["fRec250"].at[0].set((state["fRec251"] - (fSlow55 * ((fSlow56 * state["fRec250"][2]) + (fSlow34 * state["fRec250"][1]))))) 
		fTemp118 = (fSlow42 * state["fRec249"][1]) 
		state["fRec249"] = state["fRec249"].at[0].set(((fSlow53 * (state["fRec250"][2] + (state["fRec250"][0] - (jnp.float32(2.0) * state["fRec250"][1])))) - (fSlow38 * ((fSlow40 * state["fRec249"][2]) + fTemp118)))) 
		state["fRec257"] = -((fSlow54 * ((fSlow31 * fRec257_temp) - (fTemp117 + fVec33_temp)))) 
		state["fRec256"] = state["fRec256"].at[0].set((state["fRec257"] - (fSlow55 * ((fSlow56 * state["fRec256"][2]) + (fSlow34 * state["fRec256"][1]))))) 
		fTemp119 = (fSlow55 * (state["fRec256"][2] + (state["fRec256"][0] + (jnp.float32(2.0) * state["fRec256"][1])))) 
		state["fVec34"] = jnp.float32(fTemp119) 
		state["fRec255"] = -((fSlow60 * ((fSlow39 * fRec255_temp) - (fSlow36 * (fTemp119 - fVec34_temp))))) 
		state["fRec254"] = state["fRec254"].at[0].set((state["fRec255"] - (fSlow57 * ((fSlow61 * state["fRec254"][2]) + (fSlow42 * state["fRec254"][1]))))) 
		state["fRec259"] = -((fSlow60 * ((fSlow39 * fRec259_temp) - (fTemp119 + fVec34_temp)))) 
		state["fRec258"] = state["fRec258"].at[0].set((state["fRec259"] - (fSlow57 * ((fSlow61 * state["fRec258"][2]) + (fSlow42 * state["fRec258"][1]))))) 
		fTemp120 = ((((fSlow121 * (state["fRec238"][2] + (fSlow38 * (fTemp113 + (fSlow40 * state["fRec238"][0]))))) + (fSlow122 * (state["fRec243"][2] + (fSlow38 * (fTemp116 + (fSlow40 * state["fRec243"][0])))))) + (fSlow123 * (state["fRec249"][2] + (fSlow38 * (fTemp118 + (fSlow40 * state["fRec249"][0])))))) + (fSlow57 * ((fSlow124 * (state["fRec254"][2] + (state["fRec254"][0] - (jnp.float32(2.0) * state["fRec254"][1])))) + (fSlow125 * (state["fRec258"][2] + (state["fRec258"][0] + (jnp.float32(2.0) * state["fRec258"][1]))))))) 
		state["fRec264"] = -((fSlow14 * ((fSlow15 * fRec264_temp) - (fSlow11 * (state["fRec13"][1] - state["fRec13"][2]))))) 
		state["fRec263"] = state["fRec263"].at[0].set((state["fRec264"] - (fSlow16 * ((fSlow17 * state["fRec263"][2]) + (fSlow18 * state["fRec263"][1]))))) 
		fTemp121 = (fSlow26 * state["fRec262"][1]) 
		state["fRec262"] = state["fRec262"].at[0].set(((fSlow13 * (state["fRec263"][2] + (state["fRec263"][0] - (jnp.float32(2.0) * state["fRec263"][1])))) - (fSlow22 * ((fSlow24 * state["fRec262"][2]) + fTemp121)))) 
		fTemp122 = (fSlow34 * state["fRec261"][1]) 
		state["fRec261"] = state["fRec261"].at[0].set(((state["fRec262"][2] + (fSlow22 * (fTemp121 + (fSlow24 * state["fRec262"][0])))) - (fSlow30 * ((fSlow32 * state["fRec261"][2]) + fTemp122)))) 
		fTemp123 = (fSlow42 * state["fRec260"][1]) 
		state["fRec260"] = state["fRec260"].at[0].set(((state["fRec261"][2] + (fSlow30 * (fTemp122 + (fSlow32 * state["fRec261"][0])))) - (fSlow38 * ((fSlow40 * state["fRec260"][2]) + fTemp123)))) 
		state["fRec270"] = -((fSlow14 * ((fSlow15 * fRec270_temp) - (state["fRec13"][1] + state["fRec13"][2])))) 
		state["fRec269"] = state["fRec269"].at[0].set((state["fRec270"] - (fSlow16 * ((fSlow17 * state["fRec269"][2]) + (fSlow18 * state["fRec269"][1]))))) 
		fTemp124 = (fSlow16 * (state["fRec269"][2] + (state["fRec269"][0] + (jnp.float32(2.0) * state["fRec269"][1])))) 
		state["fVec35"] = jnp.float32(fTemp124) 
		state["fRec268"] = -((fSlow47 * ((fSlow23 * fRec268_temp) - (fSlow20 * (fTemp124 - fVec35_temp))))) 
		state["fRec267"] = state["fRec267"].at[0].set((state["fRec268"] - (fSlow48 * ((fSlow49 * state["fRec267"][2]) + (fSlow26 * state["fRec267"][1]))))) 
		fTemp125 = (fSlow34 * state["fRec266"][1]) 
		state["fRec266"] = state["fRec266"].at[0].set(((fSlow46 * (state["fRec267"][2] + (state["fRec267"][0] - (jnp.float32(2.0) * state["fRec267"][1])))) - (fSlow30 * ((fSlow32 * state["fRec266"][2]) + fTemp125)))) 
		fTemp126 = (fSlow42 * state["fRec265"][1]) 
		state["fRec265"] = state["fRec265"].at[0].set(((state["fRec266"][2] + (fSlow30 * (fTemp125 + (fSlow32 * state["fRec266"][0])))) - (fSlow38 * ((fSlow40 * state["fRec265"][2]) + fTemp126)))) 
		state["fRec275"] = -((fSlow47 * ((fSlow23 * fRec275_temp) - (fTemp124 + fVec35_temp)))) 
		state["fRec274"] = state["fRec274"].at[0].set((state["fRec275"] - (fSlow48 * ((fSlow49 * state["fRec274"][2]) + (fSlow26 * state["fRec274"][1]))))) 
		fTemp127 = (fSlow48 * (state["fRec274"][2] + (state["fRec274"][0] + (jnp.float32(2.0) * state["fRec274"][1])))) 
		state["fVec36"] = jnp.float32(fTemp127) 
		state["fRec273"] = -((fSlow54 * ((fSlow31 * fRec273_temp) - (fSlow28 * (fTemp127 - fVec36_temp))))) 
		state["fRec272"] = state["fRec272"].at[0].set((state["fRec273"] - (fSlow55 * ((fSlow56 * state["fRec272"][2]) + (fSlow34 * state["fRec272"][1]))))) 
		fTemp128 = (fSlow42 * state["fRec271"][1]) 
		state["fRec271"] = state["fRec271"].at[0].set(((fSlow53 * (state["fRec272"][2] + (state["fRec272"][0] - (jnp.float32(2.0) * state["fRec272"][1])))) - (fSlow38 * ((fSlow40 * state["fRec271"][2]) + fTemp128)))) 
		state["fRec279"] = -((fSlow54 * ((fSlow31 * fRec279_temp) - (fTemp127 + fVec36_temp)))) 
		state["fRec278"] = state["fRec278"].at[0].set((state["fRec279"] - (fSlow55 * ((fSlow56 * state["fRec278"][2]) + (fSlow34 * state["fRec278"][1]))))) 
		fTemp129 = (fSlow55 * (state["fRec278"][2] + (state["fRec278"][0] + (jnp.float32(2.0) * state["fRec278"][1])))) 
		state["fVec37"] = jnp.float32(fTemp129) 
		state["fRec277"] = -((fSlow60 * ((fSlow39 * fRec277_temp) - (fSlow36 * (fTemp129 - fVec37_temp))))) 
		state["fRec276"] = state["fRec276"].at[0].set((state["fRec277"] - (fSlow57 * ((fSlow61 * state["fRec276"][2]) + (fSlow42 * state["fRec276"][1]))))) 
		state["fRec281"] = -((fSlow60 * ((fSlow39 * fRec281_temp) - (fTemp129 + fVec37_temp)))) 
		state["fRec280"] = state["fRec280"].at[0].set((state["fRec281"] - (fSlow57 * ((fSlow61 * state["fRec280"][2]) + (fSlow42 * state["fRec280"][1]))))) 
		fTemp130 = ((((fSlow127 * (state["fRec260"][2] + (fSlow38 * (fTemp123 + (fSlow40 * state["fRec260"][0]))))) + (fSlow128 * (state["fRec265"][2] + (fSlow38 * (fTemp126 + (fSlow40 * state["fRec265"][0])))))) + (fSlow129 * (state["fRec271"][2] + (fSlow38 * (fTemp128 + (fSlow40 * state["fRec271"][0])))))) + (fSlow57 * ((fSlow130 * (state["fRec276"][2] + (state["fRec276"][0] - (jnp.float32(2.0) * state["fRec276"][1])))) + (fSlow131 * (state["fRec280"][2] + (state["fRec280"][0] + (jnp.float32(2.0) * state["fRec280"][1]))))))) 
		fTemp131 = (fTemp120 + fTemp130) 
		fTemp132 = (fTemp110 + fTemp131) 
		state["fRec286"] = -((fSlow14 * ((fSlow15 * fRec286_temp) - (fSlow11 * (state["fRec3"][1] - state["fRec3"][2]))))) 
		state["fRec285"] = state["fRec285"].at[0].set((state["fRec286"] - (fSlow16 * ((fSlow17 * state["fRec285"][2]) + (fSlow18 * state["fRec285"][1]))))) 
		fTemp133 = (fSlow26 * state["fRec284"][1]) 
		state["fRec284"] = state["fRec284"].at[0].set(((fSlow13 * (state["fRec285"][2] + (state["fRec285"][0] - (jnp.float32(2.0) * state["fRec285"][1])))) - (fSlow22 * ((fSlow24 * state["fRec284"][2]) + fTemp133)))) 
		fTemp134 = (fSlow34 * state["fRec283"][1]) 
		state["fRec283"] = state["fRec283"].at[0].set(((state["fRec284"][2] + (fSlow22 * (fTemp133 + (fSlow24 * state["fRec284"][0])))) - (fSlow30 * ((fSlow32 * state["fRec283"][2]) + fTemp134)))) 
		fTemp135 = (fSlow42 * state["fRec282"][1]) 
		state["fRec282"] = state["fRec282"].at[0].set(((state["fRec283"][2] + (fSlow30 * (fTemp134 + (fSlow32 * state["fRec283"][0])))) - (fSlow38 * ((fSlow40 * state["fRec282"][2]) + fTemp135)))) 
		state["fRec292"] = -((fSlow14 * ((fSlow15 * fRec292_temp) - (state["fRec3"][1] + state["fRec3"][2])))) 
		state["fRec291"] = state["fRec291"].at[0].set((state["fRec292"] - (fSlow16 * ((fSlow17 * state["fRec291"][2]) + (fSlow18 * state["fRec291"][1]))))) 
		fTemp136 = (fSlow16 * (state["fRec291"][2] + (state["fRec291"][0] + (jnp.float32(2.0) * state["fRec291"][1])))) 
		state["fVec38"] = jnp.float32(fTemp136) 
		state["fRec290"] = -((fSlow47 * ((fSlow23 * fRec290_temp) - (fSlow20 * (fTemp136 - fVec38_temp))))) 
		state["fRec289"] = state["fRec289"].at[0].set((state["fRec290"] - (fSlow48 * ((fSlow49 * state["fRec289"][2]) + (fSlow26 * state["fRec289"][1]))))) 
		fTemp137 = (fSlow34 * state["fRec288"][1]) 
		state["fRec288"] = state["fRec288"].at[0].set(((fSlow46 * (state["fRec289"][2] + (state["fRec289"][0] - (jnp.float32(2.0) * state["fRec289"][1])))) - (fSlow30 * ((fSlow32 * state["fRec288"][2]) + fTemp137)))) 
		fTemp138 = (fSlow42 * state["fRec287"][1]) 
		state["fRec287"] = state["fRec287"].at[0].set(((state["fRec288"][2] + (fSlow30 * (fTemp137 + (fSlow32 * state["fRec288"][0])))) - (fSlow38 * ((fSlow40 * state["fRec287"][2]) + fTemp138)))) 
		state["fRec297"] = -((fSlow47 * ((fSlow23 * fRec297_temp) - (fTemp136 + fVec38_temp)))) 
		state["fRec296"] = state["fRec296"].at[0].set((state["fRec297"] - (fSlow48 * ((fSlow49 * state["fRec296"][2]) + (fSlow26 * state["fRec296"][1]))))) 
		fTemp139 = (fSlow48 * (state["fRec296"][2] + (state["fRec296"][0] + (jnp.float32(2.0) * state["fRec296"][1])))) 
		state["fVec39"] = jnp.float32(fTemp139) 
		state["fRec295"] = -((fSlow54 * ((fSlow31 * fRec295_temp) - (fSlow28 * (fTemp139 - fVec39_temp))))) 
		state["fRec294"] = state["fRec294"].at[0].set((state["fRec295"] - (fSlow55 * ((fSlow56 * state["fRec294"][2]) + (fSlow34 * state["fRec294"][1]))))) 
		fTemp140 = (fSlow42 * state["fRec293"][1]) 
		state["fRec293"] = state["fRec293"].at[0].set(((fSlow53 * (state["fRec294"][2] + (state["fRec294"][0] - (jnp.float32(2.0) * state["fRec294"][1])))) - (fSlow38 * ((fSlow40 * state["fRec293"][2]) + fTemp140)))) 
		state["fRec301"] = -((fSlow54 * ((fSlow31 * fRec301_temp) - (fTemp139 + fVec39_temp)))) 
		state["fRec300"] = state["fRec300"].at[0].set((state["fRec301"] - (fSlow55 * ((fSlow56 * state["fRec300"][2]) + (fSlow34 * state["fRec300"][1]))))) 
		fTemp141 = (fSlow55 * (state["fRec300"][2] + (state["fRec300"][0] + (jnp.float32(2.0) * state["fRec300"][1])))) 
		state["fVec40"] = jnp.float32(fTemp141) 
		state["fRec299"] = -((fSlow60 * ((fSlow39 * fRec299_temp) - (fSlow36 * (fTemp141 - fVec40_temp))))) 
		state["fRec298"] = state["fRec298"].at[0].set((state["fRec299"] - (fSlow57 * ((fSlow61 * state["fRec298"][2]) + (fSlow42 * state["fRec298"][1]))))) 
		state["fRec303"] = -((fSlow60 * ((fSlow39 * fRec303_temp) - (fTemp141 + fVec40_temp)))) 
		state["fRec302"] = state["fRec302"].at[0].set((state["fRec303"] - (fSlow57 * ((fSlow61 * state["fRec302"][2]) + (fSlow42 * state["fRec302"][1]))))) 
		fTemp142 = ((((fSlow133 * (state["fRec282"][2] + (fSlow38 * (fTemp135 + (fSlow40 * state["fRec282"][0]))))) + (fSlow134 * (state["fRec287"][2] + (fSlow38 * (fTemp138 + (fSlow40 * state["fRec287"][0])))))) + (fSlow135 * (state["fRec293"][2] + (fSlow38 * (fTemp140 + (fSlow40 * state["fRec293"][0])))))) + (fSlow57 * ((fSlow136 * (state["fRec298"][2] + (state["fRec298"][0] - (jnp.float32(2.0) * state["fRec298"][1])))) + (fSlow137 * (state["fRec302"][2] + (state["fRec302"][0] + (jnp.float32(2.0) * state["fRec302"][1]))))))) 
		state["fRec308"] = -((fSlow14 * ((fSlow15 * fRec308_temp) - (fSlow11 * (state["fRec11"][1] - state["fRec11"][2]))))) 
		state["fRec307"] = state["fRec307"].at[0].set((state["fRec308"] - (fSlow16 * ((fSlow17 * state["fRec307"][2]) + (fSlow18 * state["fRec307"][1]))))) 
		fTemp143 = (fSlow26 * state["fRec306"][1]) 
		state["fRec306"] = state["fRec306"].at[0].set(((fSlow13 * (state["fRec307"][2] + (state["fRec307"][0] - (jnp.float32(2.0) * state["fRec307"][1])))) - (fSlow22 * ((fSlow24 * state["fRec306"][2]) + fTemp143)))) 
		fTemp144 = (fSlow34 * state["fRec305"][1]) 
		state["fRec305"] = state["fRec305"].at[0].set(((state["fRec306"][2] + (fSlow22 * (fTemp143 + (fSlow24 * state["fRec306"][0])))) - (fSlow30 * ((fSlow32 * state["fRec305"][2]) + fTemp144)))) 
		fTemp145 = (fSlow42 * state["fRec304"][1]) 
		state["fRec304"] = state["fRec304"].at[0].set(((state["fRec305"][2] + (fSlow30 * (fTemp144 + (fSlow32 * state["fRec305"][0])))) - (fSlow38 * ((fSlow40 * state["fRec304"][2]) + fTemp145)))) 
		state["fRec314"] = -((fSlow14 * ((fSlow15 * fRec314_temp) - (state["fRec11"][1] + state["fRec11"][2])))) 
		state["fRec313"] = state["fRec313"].at[0].set((state["fRec314"] - (fSlow16 * ((fSlow17 * state["fRec313"][2]) + (fSlow18 * state["fRec313"][1]))))) 
		fTemp146 = (fSlow16 * (state["fRec313"][2] + (state["fRec313"][0] + (jnp.float32(2.0) * state["fRec313"][1])))) 
		state["fVec41"] = jnp.float32(fTemp146) 
		state["fRec312"] = -((fSlow47 * ((fSlow23 * fRec312_temp) - (fSlow20 * (fTemp146 - fVec41_temp))))) 
		state["fRec311"] = state["fRec311"].at[0].set((state["fRec312"] - (fSlow48 * ((fSlow49 * state["fRec311"][2]) + (fSlow26 * state["fRec311"][1]))))) 
		fTemp147 = (fSlow34 * state["fRec310"][1]) 
		state["fRec310"] = state["fRec310"].at[0].set(((fSlow46 * (state["fRec311"][2] + (state["fRec311"][0] - (jnp.float32(2.0) * state["fRec311"][1])))) - (fSlow30 * ((fSlow32 * state["fRec310"][2]) + fTemp147)))) 
		fTemp148 = (fSlow42 * state["fRec309"][1]) 
		state["fRec309"] = state["fRec309"].at[0].set(((state["fRec310"][2] + (fSlow30 * (fTemp147 + (fSlow32 * state["fRec310"][0])))) - (fSlow38 * ((fSlow40 * state["fRec309"][2]) + fTemp148)))) 
		state["fRec319"] = -((fSlow47 * ((fSlow23 * fRec319_temp) - (fTemp146 + fVec41_temp)))) 
		state["fRec318"] = state["fRec318"].at[0].set((state["fRec319"] - (fSlow48 * ((fSlow49 * state["fRec318"][2]) + (fSlow26 * state["fRec318"][1]))))) 
		fTemp149 = (fSlow48 * (state["fRec318"][2] + (state["fRec318"][0] + (jnp.float32(2.0) * state["fRec318"][1])))) 
		state["fVec42"] = jnp.float32(fTemp149) 
		state["fRec317"] = -((fSlow54 * ((fSlow31 * fRec317_temp) - (fSlow28 * (fTemp149 - fVec42_temp))))) 
		state["fRec316"] = state["fRec316"].at[0].set((state["fRec317"] - (fSlow55 * ((fSlow56 * state["fRec316"][2]) + (fSlow34 * state["fRec316"][1]))))) 
		fTemp150 = (fSlow42 * state["fRec315"][1]) 
		state["fRec315"] = state["fRec315"].at[0].set(((fSlow53 * (state["fRec316"][2] + (state["fRec316"][0] - (jnp.float32(2.0) * state["fRec316"][1])))) - (fSlow38 * ((fSlow40 * state["fRec315"][2]) + fTemp150)))) 
		state["fRec323"] = -((fSlow54 * ((fSlow31 * fRec323_temp) - (fTemp149 + fVec42_temp)))) 
		state["fRec322"] = state["fRec322"].at[0].set((state["fRec323"] - (fSlow55 * ((fSlow56 * state["fRec322"][2]) + (fSlow34 * state["fRec322"][1]))))) 
		fTemp151 = (fSlow55 * (state["fRec322"][2] + (state["fRec322"][0] + (jnp.float32(2.0) * state["fRec322"][1])))) 
		state["fVec43"] = jnp.float32(fTemp151) 
		state["fRec321"] = -((fSlow60 * ((fSlow39 * fRec321_temp) - (fSlow36 * (fTemp151 - fVec43_temp))))) 
		state["fRec320"] = state["fRec320"].at[0].set((state["fRec321"] - (fSlow57 * ((fSlow61 * state["fRec320"][2]) + (fSlow42 * state["fRec320"][1]))))) 
		state["fRec325"] = -((fSlow60 * ((fSlow39 * fRec325_temp) - (fTemp151 + fVec43_temp)))) 
		state["fRec324"] = state["fRec324"].at[0].set((state["fRec325"] - (fSlow57 * ((fSlow61 * state["fRec324"][2]) + (fSlow42 * state["fRec324"][1]))))) 
		fTemp152 = ((((fSlow139 * (state["fRec304"][2] + (fSlow38 * (fTemp145 + (fSlow40 * state["fRec304"][0]))))) + (fSlow140 * (state["fRec309"][2] + (fSlow38 * (fTemp148 + (fSlow40 * state["fRec309"][0])))))) + (fSlow141 * (state["fRec315"][2] + (fSlow38 * (fTemp150 + (fSlow40 * state["fRec315"][0])))))) + (fSlow57 * ((fSlow142 * (state["fRec320"][2] + (state["fRec320"][0] - (jnp.float32(2.0) * state["fRec320"][1])))) + (fSlow143 * (state["fRec324"][2] + (state["fRec324"][0] + (jnp.float32(2.0) * state["fRec324"][1]))))))) 
		fTemp153 = (fTemp142 + fTemp152) 
		state["fRec330"] = -((fSlow14 * ((fSlow15 * fRec330_temp) - (fSlow11 * (state["fRec7"][1] - state["fRec7"][2]))))) 
		state["fRec329"] = state["fRec329"].at[0].set((state["fRec330"] - (fSlow16 * ((fSlow17 * state["fRec329"][2]) + (fSlow18 * state["fRec329"][1]))))) 
		fTemp154 = (fSlow26 * state["fRec328"][1]) 
		state["fRec328"] = state["fRec328"].at[0].set(((fSlow13 * (state["fRec329"][2] + (state["fRec329"][0] - (jnp.float32(2.0) * state["fRec329"][1])))) - (fSlow22 * ((fSlow24 * state["fRec328"][2]) + fTemp154)))) 
		fTemp155 = (fSlow34 * state["fRec327"][1]) 
		state["fRec327"] = state["fRec327"].at[0].set(((state["fRec328"][2] + (fSlow22 * (fTemp154 + (fSlow24 * state["fRec328"][0])))) - (fSlow30 * ((fSlow32 * state["fRec327"][2]) + fTemp155)))) 
		fTemp156 = (fSlow42 * state["fRec326"][1]) 
		state["fRec326"] = state["fRec326"].at[0].set(((state["fRec327"][2] + (fSlow30 * (fTemp155 + (fSlow32 * state["fRec327"][0])))) - (fSlow38 * ((fSlow40 * state["fRec326"][2]) + fTemp156)))) 
		state["fRec336"] = -((fSlow14 * ((fSlow15 * fRec336_temp) - (state["fRec7"][1] + state["fRec7"][2])))) 
		state["fRec335"] = state["fRec335"].at[0].set((state["fRec336"] - (fSlow16 * ((fSlow17 * state["fRec335"][2]) + (fSlow18 * state["fRec335"][1]))))) 
		fTemp157 = (fSlow16 * (state["fRec335"][2] + (state["fRec335"][0] + (jnp.float32(2.0) * state["fRec335"][1])))) 
		state["fVec44"] = jnp.float32(fTemp157) 
		state["fRec334"] = -((fSlow47 * ((fSlow23 * fRec334_temp) - (fSlow20 * (fTemp157 - fVec44_temp))))) 
		state["fRec333"] = state["fRec333"].at[0].set((state["fRec334"] - (fSlow48 * ((fSlow49 * state["fRec333"][2]) + (fSlow26 * state["fRec333"][1]))))) 
		fTemp158 = (fSlow34 * state["fRec332"][1]) 
		state["fRec332"] = state["fRec332"].at[0].set(((fSlow46 * (state["fRec333"][2] + (state["fRec333"][0] - (jnp.float32(2.0) * state["fRec333"][1])))) - (fSlow30 * ((fSlow32 * state["fRec332"][2]) + fTemp158)))) 
		fTemp159 = (fSlow42 * state["fRec331"][1]) 
		state["fRec331"] = state["fRec331"].at[0].set(((state["fRec332"][2] + (fSlow30 * (fTemp158 + (fSlow32 * state["fRec332"][0])))) - (fSlow38 * ((fSlow40 * state["fRec331"][2]) + fTemp159)))) 
		state["fRec341"] = -((fSlow47 * ((fSlow23 * fRec341_temp) - (fTemp157 + fVec44_temp)))) 
		state["fRec340"] = state["fRec340"].at[0].set((state["fRec341"] - (fSlow48 * ((fSlow49 * state["fRec340"][2]) + (fSlow26 * state["fRec340"][1]))))) 
		fTemp160 = (fSlow48 * (state["fRec340"][2] + (state["fRec340"][0] + (jnp.float32(2.0) * state["fRec340"][1])))) 
		state["fVec45"] = jnp.float32(fTemp160) 
		state["fRec339"] = -((fSlow54 * ((fSlow31 * fRec339_temp) - (fSlow28 * (fTemp160 - fVec45_temp))))) 
		state["fRec338"] = state["fRec338"].at[0].set((state["fRec339"] - (fSlow55 * ((fSlow56 * state["fRec338"][2]) + (fSlow34 * state["fRec338"][1]))))) 
		fTemp161 = (fSlow42 * state["fRec337"][1]) 
		state["fRec337"] = state["fRec337"].at[0].set(((fSlow53 * (state["fRec338"][2] + (state["fRec338"][0] - (jnp.float32(2.0) * state["fRec338"][1])))) - (fSlow38 * ((fSlow40 * state["fRec337"][2]) + fTemp161)))) 
		state["fRec345"] = -((fSlow54 * ((fSlow31 * fRec345_temp) - (fTemp160 + fVec45_temp)))) 
		state["fRec344"] = state["fRec344"].at[0].set((state["fRec345"] - (fSlow55 * ((fSlow56 * state["fRec344"][2]) + (fSlow34 * state["fRec344"][1]))))) 
		fTemp162 = (fSlow55 * (state["fRec344"][2] + (state["fRec344"][0] + (jnp.float32(2.0) * state["fRec344"][1])))) 
		state["fVec46"] = jnp.float32(fTemp162) 
		state["fRec343"] = -((fSlow60 * ((fSlow39 * fRec343_temp) - (fSlow36 * (fTemp162 - fVec46_temp))))) 
		state["fRec342"] = state["fRec342"].at[0].set((state["fRec343"] - (fSlow57 * ((fSlow61 * state["fRec342"][2]) + (fSlow42 * state["fRec342"][1]))))) 
		state["fRec347"] = -((fSlow60 * ((fSlow39 * fRec347_temp) - (fTemp162 + fVec46_temp)))) 
		state["fRec346"] = state["fRec346"].at[0].set((state["fRec347"] - (fSlow57 * ((fSlow61 * state["fRec346"][2]) + (fSlow42 * state["fRec346"][1]))))) 
		fTemp163 = ((((fSlow145 * (state["fRec326"][2] + (fSlow38 * (fTemp156 + (fSlow40 * state["fRec326"][0]))))) + (fSlow146 * (state["fRec331"][2] + (fSlow38 * (fTemp159 + (fSlow40 * state["fRec331"][0])))))) + (fSlow147 * (state["fRec337"][2] + (fSlow38 * (fTemp161 + (fSlow40 * state["fRec337"][0])))))) + (fSlow57 * ((fSlow148 * (state["fRec342"][2] + (state["fRec342"][0] - (jnp.float32(2.0) * state["fRec342"][1])))) + (fSlow149 * (state["fRec346"][2] + (state["fRec346"][0] + (jnp.float32(2.0) * state["fRec346"][1]))))))) 
		state["fRec352"] = -((fSlow14 * ((fSlow15 * fRec352_temp) - (fSlow11 * (state["fRec15"][1] - state["fRec15"][2]))))) 
		state["fRec351"] = state["fRec351"].at[0].set((state["fRec352"] - (fSlow16 * ((fSlow17 * state["fRec351"][2]) + (fSlow18 * state["fRec351"][1]))))) 
		fTemp164 = (fSlow26 * state["fRec350"][1]) 
		state["fRec350"] = state["fRec350"].at[0].set(((fSlow13 * (state["fRec351"][2] + (state["fRec351"][0] - (jnp.float32(2.0) * state["fRec351"][1])))) - (fSlow22 * ((fSlow24 * state["fRec350"][2]) + fTemp164)))) 
		fTemp165 = (fSlow34 * state["fRec349"][1]) 
		state["fRec349"] = state["fRec349"].at[0].set(((state["fRec350"][2] + (fSlow22 * (fTemp164 + (fSlow24 * state["fRec350"][0])))) - (fSlow30 * ((fSlow32 * state["fRec349"][2]) + fTemp165)))) 
		fTemp166 = (fSlow42 * state["fRec348"][1]) 
		state["fRec348"] = state["fRec348"].at[0].set(((state["fRec349"][2] + (fSlow30 * (fTemp165 + (fSlow32 * state["fRec349"][0])))) - (fSlow38 * ((fSlow40 * state["fRec348"][2]) + fTemp166)))) 
		state["fRec358"] = -((fSlow14 * ((fSlow15 * fRec358_temp) - (state["fRec15"][1] + state["fRec15"][2])))) 
		state["fRec357"] = state["fRec357"].at[0].set((state["fRec358"] - (fSlow16 * ((fSlow17 * state["fRec357"][2]) + (fSlow18 * state["fRec357"][1]))))) 
		fTemp167 = (fSlow16 * (state["fRec357"][2] + (state["fRec357"][0] + (jnp.float32(2.0) * state["fRec357"][1])))) 
		state["fVec47"] = jnp.float32(fTemp167) 
		state["fRec356"] = -((fSlow47 * ((fSlow23 * fRec356_temp) - (fSlow20 * (fTemp167 - fVec47_temp))))) 
		state["fRec355"] = state["fRec355"].at[0].set((state["fRec356"] - (fSlow48 * ((fSlow49 * state["fRec355"][2]) + (fSlow26 * state["fRec355"][1]))))) 
		fTemp168 = (fSlow34 * state["fRec354"][1]) 
		state["fRec354"] = state["fRec354"].at[0].set(((fSlow46 * (state["fRec355"][2] + (state["fRec355"][0] - (jnp.float32(2.0) * state["fRec355"][1])))) - (fSlow30 * ((fSlow32 * state["fRec354"][2]) + fTemp168)))) 
		fTemp169 = (fSlow42 * state["fRec353"][1]) 
		state["fRec353"] = state["fRec353"].at[0].set(((state["fRec354"][2] + (fSlow30 * (fTemp168 + (fSlow32 * state["fRec354"][0])))) - (fSlow38 * ((fSlow40 * state["fRec353"][2]) + fTemp169)))) 
		state["fRec363"] = -((fSlow47 * ((fSlow23 * fRec363_temp) - (fTemp167 + fVec47_temp)))) 
		state["fRec362"] = state["fRec362"].at[0].set((state["fRec363"] - (fSlow48 * ((fSlow49 * state["fRec362"][2]) + (fSlow26 * state["fRec362"][1]))))) 
		fTemp170 = (fSlow48 * (state["fRec362"][2] + (state["fRec362"][0] + (jnp.float32(2.0) * state["fRec362"][1])))) 
		state["fVec48"] = jnp.float32(fTemp170) 
		state["fRec361"] = -((fSlow54 * ((fSlow31 * fRec361_temp) - (fSlow28 * (fTemp170 - fVec48_temp))))) 
		state["fRec360"] = state["fRec360"].at[0].set((state["fRec361"] - (fSlow55 * ((fSlow56 * state["fRec360"][2]) + (fSlow34 * state["fRec360"][1]))))) 
		fTemp171 = (fSlow42 * state["fRec359"][1]) 
		state["fRec359"] = state["fRec359"].at[0].set(((fSlow53 * (state["fRec360"][2] + (state["fRec360"][0] - (jnp.float32(2.0) * state["fRec360"][1])))) - (fSlow38 * ((fSlow40 * state["fRec359"][2]) + fTemp171)))) 
		state["fRec367"] = -((fSlow54 * ((fSlow31 * fRec367_temp) - (fTemp170 + fVec48_temp)))) 
		state["fRec366"] = state["fRec366"].at[0].set((state["fRec367"] - (fSlow55 * ((fSlow56 * state["fRec366"][2]) + (fSlow34 * state["fRec366"][1]))))) 
		fTemp172 = (fSlow55 * (state["fRec366"][2] + (state["fRec366"][0] + (jnp.float32(2.0) * state["fRec366"][1])))) 
		state["fVec49"] = jnp.float32(fTemp172) 
		state["fRec365"] = -((fSlow60 * ((fSlow39 * fRec365_temp) - (fSlow36 * (fTemp172 - fVec49_temp))))) 
		state["fRec364"] = state["fRec364"].at[0].set((state["fRec365"] - (fSlow57 * ((fSlow61 * state["fRec364"][2]) + (fSlow42 * state["fRec364"][1]))))) 
		state["fRec369"] = -((fSlow60 * ((fSlow39 * fRec369_temp) - (fTemp172 + fVec49_temp)))) 
		state["fRec368"] = state["fRec368"].at[0].set((state["fRec369"] - (fSlow57 * ((fSlow61 * state["fRec368"][2]) + (fSlow42 * state["fRec368"][1]))))) 
		fTemp173 = ((((fSlow151 * (state["fRec348"][2] + (fSlow38 * (fTemp166 + (fSlow40 * state["fRec348"][0]))))) + (fSlow152 * (state["fRec353"][2] + (fSlow38 * (fTemp169 + (fSlow40 * state["fRec353"][0])))))) + (fSlow153 * (state["fRec359"][2] + (fSlow38 * (fTemp171 + (fSlow40 * state["fRec359"][0])))))) + (fSlow57 * ((fSlow154 * (state["fRec364"][2] + (state["fRec364"][0] - (jnp.float32(2.0) * state["fRec364"][1])))) + (fSlow155 * (state["fRec368"][2] + (state["fRec368"][0] + (jnp.float32(2.0) * state["fRec368"][1]))))))) 
		fTemp174 = (fTemp163 + fTemp173) 
		fTemp175 = (fTemp153 + fTemp174) 
		fTemp176 = (fTemp132 + fTemp175) 
		fTemp177 = (fSlow156 * inputs[0]) 
		state["fVec50"] = state["fVec50"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp0 + (fTemp2 + ((fSlow4 * (fTemp89 + fTemp176)) + fTemp177)))) 
		state["fRec0"] = state["fRec0"].at[0].set(state["fVec50"][((state["IOTA0"] - iSlow157) & 8191).astype(jnp.int32)]) 
		state["fVec51"] = fSlow158 
		fTemp178 = ((iTemp1 + ((fSlow158 - fVec51_temp) > jnp.float32(0.0)).astype(jnp.int32))) 
		fTemp179 = (fSlow156 * inputs[1]) 
		state["fVec52"] = state["fVec52"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp178 + (fTemp179 + (fTemp0 + (fSlow4 * (fTemp89 - fTemp176)))))) 
		state["fRec1"] = state["fRec1"].at[0].set(state["fVec52"][((state["IOTA0"] - iSlow159) & 8191).astype(jnp.int32)]) 
		fTemp180 = ((fTemp177 + fTemp2) + fTemp0) 
		fTemp181 = (fTemp45 - fTemp88) 
		fTemp182 = (fTemp132 - fTemp175) 
		state["fVec53"] = state["fVec53"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp181 + fTemp182)))) 
		state["fRec2"] = state["fRec2"].at[0].set(state["fVec53"][((state["IOTA0"] - iSlow160) & 8191).astype(jnp.int32)]) 
		fTemp183 = (fTemp178 + (fTemp0 + fTemp179)) 
		state["fVec54"] = state["fVec54"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp181 - fTemp182)))) 
		state["fRec3"] = state["fRec3"].at[0].set(state["fVec54"][((state["IOTA0"] - iSlow161) & 8191).astype(jnp.int32)]) 
		fTemp184 = (fTemp23 - fTemp44) 
		fTemp185 = (fTemp66 - fTemp87) 
		fTemp186 = (fTemp184 + fTemp185) 
		fTemp187 = (fTemp110 - fTemp131) 
		fTemp188 = (fTemp153 - fTemp174) 
		fTemp189 = (fTemp187 + fTemp188) 
		state["fVec55"] = state["fVec55"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp186 + fTemp189)))) 
		state["fRec4"] = state["fRec4"].at[0].set(state["fVec55"][((state["IOTA0"] - iSlow162) & 8191).astype(jnp.int32)]) 
		state["fVec56"] = state["fVec56"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp186 - fTemp189)))) 
		state["fRec5"] = state["fRec5"].at[0].set(state["fVec56"][((state["IOTA0"] - iSlow163) & 8191).astype(jnp.int32)]) 
		fTemp190 = (fTemp184 - fTemp185) 
		fTemp191 = (fTemp187 - fTemp188) 
		state["fVec57"] = state["fVec57"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp190 + fTemp191)))) 
		state["fRec6"] = state["fRec6"].at[0].set(state["fVec57"][((state["IOTA0"] - iSlow164) & 8191).astype(jnp.int32)]) 
		state["fVec58"] = state["fVec58"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp190 - fTemp191)))) 
		state["fRec7"] = state["fRec7"].at[0].set(state["fVec58"][((state["IOTA0"] - iSlow165) & 8191).astype(jnp.int32)]) 
		fTemp192 = (fTemp12 - fTemp22) 
		fTemp193 = (fTemp33 - fTemp43) 
		fTemp194 = (fTemp192 + fTemp193) 
		fTemp195 = (fTemp55 - fTemp65) 
		fTemp196 = (fTemp76 - fTemp86) 
		fTemp197 = (fTemp195 + fTemp196) 
		fTemp198 = (fTemp194 + fTemp197) 
		fTemp199 = (fTemp99 - fTemp109) 
		fTemp200 = (fTemp120 - fTemp130) 
		fTemp201 = (fTemp199 + fTemp200) 
		fTemp202 = (fTemp142 - fTemp152) 
		fTemp203 = (fTemp163 - fTemp173) 
		fTemp204 = (fTemp202 + fTemp203) 
		fTemp205 = (fTemp201 + fTemp204) 
		state["fVec59"] = state["fVec59"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp198 + fTemp205)))) 
		state["fRec8"] = state["fRec8"].at[0].set(state["fVec59"][((state["IOTA0"] - iSlow166) & 8191).astype(jnp.int32)]) 
		state["fVec60"] = state["fVec60"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp198 - fTemp205)))) 
		state["fRec9"] = state["fRec9"].at[0].set(state["fVec60"][((state["IOTA0"] - iSlow167) & 8191).astype(jnp.int32)]) 
		fTemp206 = (fTemp194 - fTemp197) 
		fTemp207 = (fTemp201 - fTemp204) 
		state["fVec61"] = state["fVec61"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp206 + fTemp207)))) 
		state["fRec10"] = state["fRec10"].at[0].set(state["fVec61"][((state["IOTA0"] - iSlow168) & 8191).astype(jnp.int32)]) 
		state["fVec62"] = state["fVec62"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp206 - fTemp207)))) 
		state["fRec11"] = state["fRec11"].at[0].set(state["fVec62"][((state["IOTA0"] - iSlow169) & 8191).astype(jnp.int32)]) 
		fTemp208 = (fTemp192 - fTemp193) 
		fTemp209 = (fTemp195 - fTemp196) 
		fTemp210 = (fTemp208 + fTemp209) 
		fTemp211 = (fTemp199 - fTemp200) 
		fTemp212 = (fTemp202 - fTemp203) 
		fTemp213 = (fTemp211 + fTemp212) 
		state["fVec63"] = state["fVec63"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp210 + fTemp213)))) 
		state["fRec12"] = state["fRec12"].at[0].set(state["fVec63"][((state["IOTA0"] - iSlow170) & 8191).astype(jnp.int32)]) 
		state["fVec64"] = state["fVec64"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp210 - fTemp213)))) 
		state["fRec13"] = state["fRec13"].at[0].set(state["fVec64"][((state["IOTA0"] - iSlow171) & 8191).astype(jnp.int32)]) 
		fTemp214 = (fTemp208 - fTemp209) 
		fTemp215 = (fTemp211 - fTemp212) 
		state["fVec65"] = state["fVec65"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp180 + (fSlow4 * (fTemp214 + fTemp215)))) 
		state["fRec14"] = state["fRec14"].at[0].set(state["fVec65"][((state["IOTA0"] - iSlow172) & 8191).astype(jnp.int32)]) 
		state["fVec66"] = state["fVec66"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow4 * (fTemp214 - fTemp215)))) 
		state["fRec15"] = state["fRec15"].at[0].set(state["fVec66"][((state["IOTA0"] - iSlow173) & 8191).astype(jnp.int32)]) 
		_result0 = (fSlow0 * (((((((state["fRec0"][0] + state["fRec2"][0]) + state["fRec4"][0]) + state["fRec6"][0]) + state["fRec8"][0]) + state["fRec10"][0]) + state["fRec12"][0]) + state["fRec14"][0])) 
		_result1 = (fSlow0 * (((((((state["fRec1"][0] + state["fRec3"][0]) + state["fRec5"][0]) + state["fRec7"][0]) + state["fRec9"][0]) + state["fRec11"][0]) + state["fRec13"][0]) + state["fRec15"][0])) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec21"] = jnp.roll(state["fRec21"], 1) 
		state["fRec20"] = jnp.roll(state["fRec20"], 1) 
		state["fRec19"] = jnp.roll(state["fRec19"], 1) 
		state["fRec18"] = jnp.roll(state["fRec18"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec25"] = jnp.roll(state["fRec25"], 1) 
		state["fRec24"] = jnp.roll(state["fRec24"], 1) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec32"] = jnp.roll(state["fRec32"], 1) 
		state["fRec30"] = jnp.roll(state["fRec30"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec36"] = jnp.roll(state["fRec36"], 1) 
		state["fRec34"] = jnp.roll(state["fRec34"], 1) 
		state["fRec38"] = jnp.roll(state["fRec38"], 1) 
		state["fRec43"] = jnp.roll(state["fRec43"], 1) 
		state["fRec42"] = jnp.roll(state["fRec42"], 1) 
		state["fRec41"] = jnp.roll(state["fRec41"], 1) 
		state["fRec40"] = jnp.roll(state["fRec40"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec47"] = jnp.roll(state["fRec47"], 1) 
		state["fRec46"] = jnp.roll(state["fRec46"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec54"] = jnp.roll(state["fRec54"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec58"] = jnp.roll(state["fRec58"], 1) 
		state["fRec56"] = jnp.roll(state["fRec56"], 1) 
		state["fRec60"] = jnp.roll(state["fRec60"], 1) 
		state["fRec65"] = jnp.roll(state["fRec65"], 1) 
		state["fRec64"] = jnp.roll(state["fRec64"], 1) 
		state["fRec63"] = jnp.roll(state["fRec63"], 1) 
		state["fRec62"] = jnp.roll(state["fRec62"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec69"] = jnp.roll(state["fRec69"], 1) 
		state["fRec68"] = jnp.roll(state["fRec68"], 1) 
		state["fRec67"] = jnp.roll(state["fRec67"], 1) 
		state["fRec76"] = jnp.roll(state["fRec76"], 1) 
		state["fRec74"] = jnp.roll(state["fRec74"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec82"] = jnp.roll(state["fRec82"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec84"] = jnp.roll(state["fRec84"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec91"] = jnp.roll(state["fRec91"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec98"] = jnp.roll(state["fRec98"], 1) 
		state["fRec96"] = jnp.roll(state["fRec96"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec102"] = jnp.roll(state["fRec102"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec112"] = jnp.roll(state["fRec112"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec120"] = jnp.roll(state["fRec120"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec124"] = jnp.roll(state["fRec124"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec126"] = jnp.roll(state["fRec126"], 1) 
		state["fRec131"] = jnp.roll(state["fRec131"], 1) 
		state["fRec130"] = jnp.roll(state["fRec130"], 1) 
		state["fRec129"] = jnp.roll(state["fRec129"], 1) 
		state["fRec128"] = jnp.roll(state["fRec128"], 1) 
		state["fRec137"] = jnp.roll(state["fRec137"], 1) 
		state["fRec135"] = jnp.roll(state["fRec135"], 1) 
		state["fRec134"] = jnp.roll(state["fRec134"], 1) 
		state["fRec133"] = jnp.roll(state["fRec133"], 1) 
		state["fRec142"] = jnp.roll(state["fRec142"], 1) 
		state["fRec140"] = jnp.roll(state["fRec140"], 1) 
		state["fRec139"] = jnp.roll(state["fRec139"], 1) 
		state["fRec146"] = jnp.roll(state["fRec146"], 1) 
		state["fRec144"] = jnp.roll(state["fRec144"], 1) 
		state["fRec148"] = jnp.roll(state["fRec148"], 1) 
		state["fRec153"] = jnp.roll(state["fRec153"], 1) 
		state["fRec152"] = jnp.roll(state["fRec152"], 1) 
		state["fRec151"] = jnp.roll(state["fRec151"], 1) 
		state["fRec150"] = jnp.roll(state["fRec150"], 1) 
		state["fRec159"] = jnp.roll(state["fRec159"], 1) 
		state["fRec157"] = jnp.roll(state["fRec157"], 1) 
		state["fRec156"] = jnp.roll(state["fRec156"], 1) 
		state["fRec155"] = jnp.roll(state["fRec155"], 1) 
		state["fRec164"] = jnp.roll(state["fRec164"], 1) 
		state["fRec162"] = jnp.roll(state["fRec162"], 1) 
		state["fRec161"] = jnp.roll(state["fRec161"], 1) 
		state["fRec168"] = jnp.roll(state["fRec168"], 1) 
		state["fRec166"] = jnp.roll(state["fRec166"], 1) 
		state["fRec170"] = jnp.roll(state["fRec170"], 1) 
		state["fRec175"] = jnp.roll(state["fRec175"], 1) 
		state["fRec174"] = jnp.roll(state["fRec174"], 1) 
		state["fRec173"] = jnp.roll(state["fRec173"], 1) 
		state["fRec172"] = jnp.roll(state["fRec172"], 1) 
		state["fRec181"] = jnp.roll(state["fRec181"], 1) 
		state["fRec179"] = jnp.roll(state["fRec179"], 1) 
		state["fRec178"] = jnp.roll(state["fRec178"], 1) 
		state["fRec177"] = jnp.roll(state["fRec177"], 1) 
		state["fRec186"] = jnp.roll(state["fRec186"], 1) 
		state["fRec184"] = jnp.roll(state["fRec184"], 1) 
		state["fRec183"] = jnp.roll(state["fRec183"], 1) 
		state["fRec190"] = jnp.roll(state["fRec190"], 1) 
		state["fRec188"] = jnp.roll(state["fRec188"], 1) 
		state["fRec192"] = jnp.roll(state["fRec192"], 1) 
		state["fRec197"] = jnp.roll(state["fRec197"], 1) 
		state["fRec196"] = jnp.roll(state["fRec196"], 1) 
		state["fRec195"] = jnp.roll(state["fRec195"], 1) 
		state["fRec194"] = jnp.roll(state["fRec194"], 1) 
		state["fRec203"] = jnp.roll(state["fRec203"], 1) 
		state["fRec201"] = jnp.roll(state["fRec201"], 1) 
		state["fRec200"] = jnp.roll(state["fRec200"], 1) 
		state["fRec199"] = jnp.roll(state["fRec199"], 1) 
		state["fRec208"] = jnp.roll(state["fRec208"], 1) 
		state["fRec206"] = jnp.roll(state["fRec206"], 1) 
		state["fRec205"] = jnp.roll(state["fRec205"], 1) 
		state["fRec212"] = jnp.roll(state["fRec212"], 1) 
		state["fRec210"] = jnp.roll(state["fRec210"], 1) 
		state["fRec214"] = jnp.roll(state["fRec214"], 1) 
		state["fRec219"] = jnp.roll(state["fRec219"], 1) 
		state["fRec218"] = jnp.roll(state["fRec218"], 1) 
		state["fRec217"] = jnp.roll(state["fRec217"], 1) 
		state["fRec216"] = jnp.roll(state["fRec216"], 1) 
		state["fRec225"] = jnp.roll(state["fRec225"], 1) 
		state["fRec223"] = jnp.roll(state["fRec223"], 1) 
		state["fRec222"] = jnp.roll(state["fRec222"], 1) 
		state["fRec221"] = jnp.roll(state["fRec221"], 1) 
		state["fRec230"] = jnp.roll(state["fRec230"], 1) 
		state["fRec228"] = jnp.roll(state["fRec228"], 1) 
		state["fRec227"] = jnp.roll(state["fRec227"], 1) 
		state["fRec234"] = jnp.roll(state["fRec234"], 1) 
		state["fRec232"] = jnp.roll(state["fRec232"], 1) 
		state["fRec236"] = jnp.roll(state["fRec236"], 1) 
		state["fRec241"] = jnp.roll(state["fRec241"], 1) 
		state["fRec240"] = jnp.roll(state["fRec240"], 1) 
		state["fRec239"] = jnp.roll(state["fRec239"], 1) 
		state["fRec238"] = jnp.roll(state["fRec238"], 1) 
		state["fRec247"] = jnp.roll(state["fRec247"], 1) 
		state["fRec245"] = jnp.roll(state["fRec245"], 1) 
		state["fRec244"] = jnp.roll(state["fRec244"], 1) 
		state["fRec243"] = jnp.roll(state["fRec243"], 1) 
		state["fRec252"] = jnp.roll(state["fRec252"], 1) 
		state["fRec250"] = jnp.roll(state["fRec250"], 1) 
		state["fRec249"] = jnp.roll(state["fRec249"], 1) 
		state["fRec256"] = jnp.roll(state["fRec256"], 1) 
		state["fRec254"] = jnp.roll(state["fRec254"], 1) 
		state["fRec258"] = jnp.roll(state["fRec258"], 1) 
		state["fRec263"] = jnp.roll(state["fRec263"], 1) 
		state["fRec262"] = jnp.roll(state["fRec262"], 1) 
		state["fRec261"] = jnp.roll(state["fRec261"], 1) 
		state["fRec260"] = jnp.roll(state["fRec260"], 1) 
		state["fRec269"] = jnp.roll(state["fRec269"], 1) 
		state["fRec267"] = jnp.roll(state["fRec267"], 1) 
		state["fRec266"] = jnp.roll(state["fRec266"], 1) 
		state["fRec265"] = jnp.roll(state["fRec265"], 1) 
		state["fRec274"] = jnp.roll(state["fRec274"], 1) 
		state["fRec272"] = jnp.roll(state["fRec272"], 1) 
		state["fRec271"] = jnp.roll(state["fRec271"], 1) 
		state["fRec278"] = jnp.roll(state["fRec278"], 1) 
		state["fRec276"] = jnp.roll(state["fRec276"], 1) 
		state["fRec280"] = jnp.roll(state["fRec280"], 1) 
		state["fRec285"] = jnp.roll(state["fRec285"], 1) 
		state["fRec284"] = jnp.roll(state["fRec284"], 1) 
		state["fRec283"] = jnp.roll(state["fRec283"], 1) 
		state["fRec282"] = jnp.roll(state["fRec282"], 1) 
		state["fRec291"] = jnp.roll(state["fRec291"], 1) 
		state["fRec289"] = jnp.roll(state["fRec289"], 1) 
		state["fRec288"] = jnp.roll(state["fRec288"], 1) 
		state["fRec287"] = jnp.roll(state["fRec287"], 1) 
		state["fRec296"] = jnp.roll(state["fRec296"], 1) 
		state["fRec294"] = jnp.roll(state["fRec294"], 1) 
		state["fRec293"] = jnp.roll(state["fRec293"], 1) 
		state["fRec300"] = jnp.roll(state["fRec300"], 1) 
		state["fRec298"] = jnp.roll(state["fRec298"], 1) 
		state["fRec302"] = jnp.roll(state["fRec302"], 1) 
		state["fRec307"] = jnp.roll(state["fRec307"], 1) 
		state["fRec306"] = jnp.roll(state["fRec306"], 1) 
		state["fRec305"] = jnp.roll(state["fRec305"], 1) 
		state["fRec304"] = jnp.roll(state["fRec304"], 1) 
		state["fRec313"] = jnp.roll(state["fRec313"], 1) 
		state["fRec311"] = jnp.roll(state["fRec311"], 1) 
		state["fRec310"] = jnp.roll(state["fRec310"], 1) 
		state["fRec309"] = jnp.roll(state["fRec309"], 1) 
		state["fRec318"] = jnp.roll(state["fRec318"], 1) 
		state["fRec316"] = jnp.roll(state["fRec316"], 1) 
		state["fRec315"] = jnp.roll(state["fRec315"], 1) 
		state["fRec322"] = jnp.roll(state["fRec322"], 1) 
		state["fRec320"] = jnp.roll(state["fRec320"], 1) 
		state["fRec324"] = jnp.roll(state["fRec324"], 1) 
		state["fRec329"] = jnp.roll(state["fRec329"], 1) 
		state["fRec328"] = jnp.roll(state["fRec328"], 1) 
		state["fRec327"] = jnp.roll(state["fRec327"], 1) 
		state["fRec326"] = jnp.roll(state["fRec326"], 1) 
		state["fRec335"] = jnp.roll(state["fRec335"], 1) 
		state["fRec333"] = jnp.roll(state["fRec333"], 1) 
		state["fRec332"] = jnp.roll(state["fRec332"], 1) 
		state["fRec331"] = jnp.roll(state["fRec331"], 1) 
		state["fRec340"] = jnp.roll(state["fRec340"], 1) 
		state["fRec338"] = jnp.roll(state["fRec338"], 1) 
		state["fRec337"] = jnp.roll(state["fRec337"], 1) 
		state["fRec344"] = jnp.roll(state["fRec344"], 1) 
		state["fRec342"] = jnp.roll(state["fRec342"], 1) 
		state["fRec346"] = jnp.roll(state["fRec346"], 1) 
		state["fRec351"] = jnp.roll(state["fRec351"], 1) 
		state["fRec350"] = jnp.roll(state["fRec350"], 1) 
		state["fRec349"] = jnp.roll(state["fRec349"], 1) 
		state["fRec348"] = jnp.roll(state["fRec348"], 1) 
		state["fRec357"] = jnp.roll(state["fRec357"], 1) 
		state["fRec355"] = jnp.roll(state["fRec355"], 1) 
		state["fRec354"] = jnp.roll(state["fRec354"], 1) 
		state["fRec353"] = jnp.roll(state["fRec353"], 1) 
		state["fRec362"] = jnp.roll(state["fRec362"], 1) 
		state["fRec360"] = jnp.roll(state["fRec360"], 1) 
		state["fRec359"] = jnp.roll(state["fRec359"], 1) 
		state["fRec366"] = jnp.roll(state["fRec366"], 1) 
		state["fRec364"] = jnp.roll(state["fRec364"], 1) 
		state["fRec368"] = jnp.roll(state["fRec368"], 1) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec0"] = jnp.roll(state["fRec0"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
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
