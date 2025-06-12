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
		return 1
	
	# fmt: off
	def setup(self):
		# Initialize static tables
		# Initialize waveform data
		# Convert static tables and waveform data to JAX arrays
		# Initialize UI parameters
		unnorm_funcs = {}
		ui_path = []
		ui_path.append("bells") 
		self.add_nentry("fEntry1", ui_path, "feedback", 0.989, 0.0, 1.0, 0.001, unnorm_funcs, "linear") 
		self.add_nentry("fEntry0", ui_path, "pitch0", 6e+01, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry2", ui_path, "pitch1", 61.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry3", ui_path, "pitch2", 62.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry4", ui_path, "pitch3", 63.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry5", ui_path, "pitch4", 64.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry6", ui_path, "pitch5", 65.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry7", ui_path, "pitch6", 66.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry8", ui_path, "pitch7", 67.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry9", ui_path, "pitch8", 68.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_nentry("fEntry10", ui_path, "pitch9", 69.0, 6e+01, 1.6e+02, 0.01, unnorm_funcs, "linear") 
		self.add_button("fButton0", ui_path, "play0", unnorm_funcs) 
		self.add_button("fButton1", ui_path, "play1", unnorm_funcs) 
		self.add_button("fButton2", ui_path, "play2", unnorm_funcs) 
		self.add_button("fButton3", ui_path, "play3", unnorm_funcs) 
		self.add_button("fButton4", ui_path, "play4", unnorm_funcs) 
		self.add_button("fButton5", ui_path, "play5", unnorm_funcs) 
		self.add_button("fButton6", ui_path, "play6", unnorm_funcs) 
		self.add_button("fButton7", ui_path, "play7", unnorm_funcs) 
		self.add_button("fButton8", ui_path, "play8", unnorm_funcs) 
		self.add_button("fButton9", ui_path, "play9", unnorm_funcs) 
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = (np.float32(100.53097) / self._fConst0) 
		self._fConst2 = np.power((np.float32(1.0) - self._fConst1), np.float32(2.0)) 
		self._fConst3 = (np.float32(24876.438) / self._fConst0) 
		self._fConst4 = (np.float32(2.0) * (self._fConst1 - np.float32(1.0))) 
		self._fConst5 = (np.float32(0.00025257576) * self._fConst0) 
		self._fConst6 = (np.float32(0.5) * self._fConst0) 
		self._fConst7 = (np.float32(0.5) * (np.float32(1.0) - self._fConst2)) 
		self._fConst8 = (np.float32(8275.275) / self._fConst0) 
		self._fConst9 = (np.float32(0.0007592721) * self._fConst0) 
		self._fConst10 = (np.float32(8236.24) / self._fConst0) 
		self._fConst11 = (np.float32(0.0007628706) * self._fConst0) 
		self._fConst12 = (np.float32(2775.3606) / self._fConst0) 
		self._fConst13 = (np.float32(0.002263917) * self._fConst0) 
		self._fConst14 = (np.float32(2753.8428) / self._fConst0) 
		self._fConst15 = (np.float32(0.0022816064) * self._fConst0) 
		self._fConst16 = (np.float32(24924.258) / self._fConst0) 
		self._fConst17 = (np.float32(0.00025209118) * self._fConst0) 
		self._fConst18 = (np.float32(35478.215) / self._fConst0) 
		self._fConst19 = (np.float32(0.00017709982) * self._fConst0) 
		self._fConst20 = (np.float32(35407.31) / self._fConst0) 
		self._fConst21 = (np.float32(0.00017745448) * self._fConst0) 
		self._fConst22 = (np.float32(47774.59) / self._fConst0) 
		self._fConst23 = (np.float32(0.0001315173) * self._fConst0) 
		self._fConst24 = (np.float32(60754.96) / self._fConst0) 
		self._fConst25 = (np.float32(0.000103418475) * self._fConst0) 
		self._fConst26 = (np.float32(15770.537) / self._fConst0) 
		self._fConst27 = (np.float32(0.0003984129) * self._fConst0) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec10"] = np.float32(0)
		state["fRec100"] = np.float32(0)
		state["fRec102"] = np.float32(0)
		state["fRec104"] = np.float32(0)
		state["fRec106"] = np.float32(0)
		state["fRec108"] = np.float32(0)
		state["fRec110"] = np.float32(0)
		state["fRec112"] = np.float32(0)
		state["fRec114"] = np.float32(0)
		state["fRec116"] = np.float32(0)
		state["fRec118"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec120"] = np.float32(0)
		state["fRec122"] = np.float32(0)
		state["fRec124"] = np.float32(0)
		state["fRec126"] = np.float32(0)
		state["fRec128"] = np.float32(0)
		state["fRec130"] = np.float32(0)
		state["fRec132"] = np.float32(0)
		state["fRec134"] = np.float32(0)
		state["fRec136"] = np.float32(0)
		state["fRec138"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec140"] = np.float32(0)
		state["fRec142"] = np.float32(0)
		state["fRec144"] = np.float32(0)
		state["fRec146"] = np.float32(0)
		state["fRec148"] = np.float32(0)
		state["fRec150"] = np.float32(0)
		state["fRec152"] = np.float32(0)
		state["fRec154"] = np.float32(0)
		state["fRec156"] = np.float32(0)
		state["fRec158"] = np.float32(0)
		state["fRec16"] = np.float32(0)
		state["fRec160"] = np.float32(0)
		state["fRec162"] = np.float32(0)
		state["fRec164"] = np.float32(0)
		state["fRec166"] = np.float32(0)
		state["fRec168"] = np.float32(0)
		state["fRec170"] = np.float32(0)
		state["fRec172"] = np.float32(0)
		state["fRec174"] = np.float32(0)
		state["fRec176"] = np.float32(0)
		state["fRec178"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec180"] = np.float32(0)
		state["fRec182"] = np.float32(0)
		state["fRec184"] = np.float32(0)
		state["fRec186"] = np.float32(0)
		state["fRec188"] = np.float32(0)
		state["fRec190"] = np.float32(0)
		state["fRec192"] = np.float32(0)
		state["fRec194"] = np.float32(0)
		state["fRec196"] = np.float32(0)
		state["fRec198"] = np.float32(0)
		state["fRec2"] = np.float32(0)
		state["fRec20"] = np.float32(0)
		state["fRec200"] = np.float32(0)
		state["fRec202"] = np.float32(0)
		state["fRec204"] = np.float32(0)
		state["fRec206"] = np.float32(0)
		state["fRec208"] = np.float32(0)
		state["fRec210"] = np.float32(0)
		state["fRec212"] = np.float32(0)
		state["fRec214"] = np.float32(0)
		state["fRec216"] = np.float32(0)
		state["fRec218"] = np.float32(0)
		state["fRec22"] = np.float32(0)
		state["fRec24"] = np.float32(0)
		state["fRec26"] = np.float32(0)
		state["fRec28"] = np.float32(0)
		state["fRec30"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec34"] = np.float32(0)
		state["fRec36"] = np.float32(0)
		state["fRec38"] = np.float32(0)
		state["fRec4"] = np.float32(0)
		state["fRec40"] = np.float32(0)
		state["fRec42"] = np.float32(0)
		state["fRec44"] = np.float32(0)
		state["fRec46"] = np.float32(0)
		state["fRec48"] = np.float32(0)
		state["fRec50"] = np.float32(0)
		state["fRec52"] = np.float32(0)
		state["fRec54"] = np.float32(0)
		state["fRec56"] = np.float32(0)
		state["fRec58"] = np.float32(0)
		state["fRec6"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec62"] = np.float32(0)
		state["fRec64"] = np.float32(0)
		state["fRec66"] = np.float32(0)
		state["fRec68"] = np.float32(0)
		state["fRec70"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec74"] = np.float32(0)
		state["fRec76"] = np.float32(0)
		state["fRec78"] = np.float32(0)
		state["fRec8"] = np.float32(0)
		state["fRec80"] = np.float32(0)
		state["fRec82"] = np.float32(0)
		state["fRec84"] = np.float32(0)
		state["fRec86"] = np.float32(0)
		state["fRec88"] = np.float32(0)
		state["fRec90"] = np.float32(0)
		state["fRec92"] = np.float32(0)
		state["fRec94"] = np.float32(0)
		state["fRec96"] = np.float32(0)
		state["fRec98"] = np.float32(0)
		state["fVec0"] = np.float32(0)
		state["fVec11"] = np.float32(0)
		state["fVec22"] = np.float32(0)
		state["fVec33"] = np.float32(0)
		state["fVec44"] = np.float32(0)
		state["fVec55"] = np.float32(0)
		state["fVec66"] = np.float32(0)
		state["fVec77"] = np.float32(0)
		state["fVec88"] = np.float32(0)
		state["fVec99"] = np.float32(0)
		# Initialize array delays
		state["fVec1"] = np.zeros((128,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fVec2"] = np.zeros((256,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fVec3"] = np.zeros((256,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fVec4"] = np.zeros((1024,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fVec5"] = np.zeros((1024,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fVec6"] = np.zeros((128,), dtype=np.float32)
		state["fRec11"] = np.zeros((3,), dtype=np.float32)
		state["fVec7"] = np.zeros((64,), dtype=np.float32)
		state["fRec13"] = np.zeros((3,), dtype=np.float32)
		state["fVec8"] = np.zeros((64,), dtype=np.float32)
		state["fRec15"] = np.zeros((3,), dtype=np.float32)
		state["fVec9"] = np.zeros((64,), dtype=np.float32)
		state["fRec17"] = np.zeros((3,), dtype=np.float32)
		state["fVec10"] = np.zeros((64,), dtype=np.float32)
		state["fRec19"] = np.zeros((3,), dtype=np.float32)
		state["fVec12"] = np.zeros((1024,), dtype=np.float32)
		state["fRec21"] = np.zeros((3,), dtype=np.float32)
		state["fVec13"] = np.zeros((1024,), dtype=np.float32)
		state["fRec23"] = np.zeros((3,), dtype=np.float32)
		state["fVec14"] = np.zeros((256,), dtype=np.float32)
		state["fRec25"] = np.zeros((3,), dtype=np.float32)
		state["fVec15"] = np.zeros((256,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fVec16"] = np.zeros((128,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fVec17"] = np.zeros((128,), dtype=np.float32)
		state["fRec31"] = np.zeros((3,), dtype=np.float32)
		state["fVec18"] = np.zeros((64,), dtype=np.float32)
		state["fRec33"] = np.zeros((3,), dtype=np.float32)
		state["fVec19"] = np.zeros((64,), dtype=np.float32)
		state["fRec35"] = np.zeros((3,), dtype=np.float32)
		state["fVec20"] = np.zeros((64,), dtype=np.float32)
		state["fRec37"] = np.zeros((3,), dtype=np.float32)
		state["fVec21"] = np.zeros((64,), dtype=np.float32)
		state["fRec39"] = np.zeros((3,), dtype=np.float32)
		state["fVec23"] = np.zeros((1024,), dtype=np.float32)
		state["fRec41"] = np.zeros((3,), dtype=np.float32)
		state["fVec24"] = np.zeros((1024,), dtype=np.float32)
		state["fRec43"] = np.zeros((3,), dtype=np.float32)
		state["fVec25"] = np.zeros((256,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fVec26"] = np.zeros((256,), dtype=np.float32)
		state["fRec47"] = np.zeros((3,), dtype=np.float32)
		state["fVec27"] = np.zeros((128,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fVec28"] = np.zeros((128,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fVec29"] = np.zeros((64,), dtype=np.float32)
		state["fRec53"] = np.zeros((3,), dtype=np.float32)
		state["fVec30"] = np.zeros((64,), dtype=np.float32)
		state["fRec55"] = np.zeros((3,), dtype=np.float32)
		state["fVec31"] = np.zeros((64,), dtype=np.float32)
		state["fRec57"] = np.zeros((3,), dtype=np.float32)
		state["fVec32"] = np.zeros((64,), dtype=np.float32)
		state["fRec59"] = np.zeros((3,), dtype=np.float32)
		state["fVec34"] = np.zeros((1024,), dtype=np.float32)
		state["fRec61"] = np.zeros((3,), dtype=np.float32)
		state["fVec35"] = np.zeros((1024,), dtype=np.float32)
		state["fRec63"] = np.zeros((3,), dtype=np.float32)
		state["fVec36"] = np.zeros((256,), dtype=np.float32)
		state["fRec65"] = np.zeros((3,), dtype=np.float32)
		state["fVec37"] = np.zeros((256,), dtype=np.float32)
		state["fRec67"] = np.zeros((3,), dtype=np.float32)
		state["fVec38"] = np.zeros((128,), dtype=np.float32)
		state["fRec69"] = np.zeros((3,), dtype=np.float32)
		state["fVec39"] = np.zeros((128,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fVec40"] = np.zeros((64,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fVec41"] = np.zeros((64,), dtype=np.float32)
		state["fRec75"] = np.zeros((3,), dtype=np.float32)
		state["fVec42"] = np.zeros((64,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fVec43"] = np.zeros((64,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fVec45"] = np.zeros((1024,), dtype=np.float32)
		state["fRec81"] = np.zeros((3,), dtype=np.float32)
		state["fVec46"] = np.zeros((1024,), dtype=np.float32)
		state["fRec83"] = np.zeros((3,), dtype=np.float32)
		state["fVec47"] = np.zeros((256,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fVec48"] = np.zeros((256,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fVec49"] = np.zeros((128,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fVec50"] = np.zeros((128,), dtype=np.float32)
		state["fRec91"] = np.zeros((3,), dtype=np.float32)
		state["fVec51"] = np.zeros((64,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fVec52"] = np.zeros((64,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fVec53"] = np.zeros((64,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fVec54"] = np.zeros((64,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fVec56"] = np.zeros((1024,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fVec57"] = np.zeros((1024,), dtype=np.float32)
		state["fRec103"] = np.zeros((3,), dtype=np.float32)
		state["fVec58"] = np.zeros((256,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fVec59"] = np.zeros((256,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fVec60"] = np.zeros((128,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fVec61"] = np.zeros((128,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fVec62"] = np.zeros((64,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fVec63"] = np.zeros((64,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fVec64"] = np.zeros((64,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fVec65"] = np.zeros((64,), dtype=np.float32)
		state["fRec119"] = np.zeros((3,), dtype=np.float32)
		state["fVec67"] = np.zeros((1024,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fVec68"] = np.zeros((1024,), dtype=np.float32)
		state["fRec123"] = np.zeros((3,), dtype=np.float32)
		state["fVec69"] = np.zeros((256,), dtype=np.float32)
		state["fRec125"] = np.zeros((3,), dtype=np.float32)
		state["fVec70"] = np.zeros((256,), dtype=np.float32)
		state["fRec127"] = np.zeros((3,), dtype=np.float32)
		state["fVec71"] = np.zeros((128,), dtype=np.float32)
		state["fRec129"] = np.zeros((3,), dtype=np.float32)
		state["fVec72"] = np.zeros((128,), dtype=np.float32)
		state["fRec131"] = np.zeros((3,), dtype=np.float32)
		state["fVec73"] = np.zeros((64,), dtype=np.float32)
		state["fRec133"] = np.zeros((3,), dtype=np.float32)
		state["fVec74"] = np.zeros((64,), dtype=np.float32)
		state["fRec135"] = np.zeros((3,), dtype=np.float32)
		state["fVec75"] = np.zeros((64,), dtype=np.float32)
		state["fRec137"] = np.zeros((3,), dtype=np.float32)
		state["fVec76"] = np.zeros((64,), dtype=np.float32)
		state["fRec139"] = np.zeros((3,), dtype=np.float32)
		state["fVec78"] = np.zeros((1024,), dtype=np.float32)
		state["fRec141"] = np.zeros((3,), dtype=np.float32)
		state["fVec79"] = np.zeros((1024,), dtype=np.float32)
		state["fRec143"] = np.zeros((3,), dtype=np.float32)
		state["fVec80"] = np.zeros((256,), dtype=np.float32)
		state["fRec145"] = np.zeros((3,), dtype=np.float32)
		state["fVec81"] = np.zeros((256,), dtype=np.float32)
		state["fRec147"] = np.zeros((3,), dtype=np.float32)
		state["fVec82"] = np.zeros((128,), dtype=np.float32)
		state["fRec149"] = np.zeros((3,), dtype=np.float32)
		state["fVec83"] = np.zeros((128,), dtype=np.float32)
		state["fRec151"] = np.zeros((3,), dtype=np.float32)
		state["fVec84"] = np.zeros((64,), dtype=np.float32)
		state["fRec153"] = np.zeros((3,), dtype=np.float32)
		state["fVec85"] = np.zeros((64,), dtype=np.float32)
		state["fRec155"] = np.zeros((3,), dtype=np.float32)
		state["fVec86"] = np.zeros((64,), dtype=np.float32)
		state["fRec157"] = np.zeros((3,), dtype=np.float32)
		state["fVec87"] = np.zeros((64,), dtype=np.float32)
		state["fRec159"] = np.zeros((3,), dtype=np.float32)
		state["fVec89"] = np.zeros((1024,), dtype=np.float32)
		state["fRec161"] = np.zeros((3,), dtype=np.float32)
		state["fVec90"] = np.zeros((1024,), dtype=np.float32)
		state["fRec163"] = np.zeros((3,), dtype=np.float32)
		state["fVec91"] = np.zeros((256,), dtype=np.float32)
		state["fRec165"] = np.zeros((3,), dtype=np.float32)
		state["fVec92"] = np.zeros((256,), dtype=np.float32)
		state["fRec167"] = np.zeros((3,), dtype=np.float32)
		state["fVec93"] = np.zeros((128,), dtype=np.float32)
		state["fRec169"] = np.zeros((3,), dtype=np.float32)
		state["fVec94"] = np.zeros((128,), dtype=np.float32)
		state["fRec171"] = np.zeros((3,), dtype=np.float32)
		state["fVec95"] = np.zeros((64,), dtype=np.float32)
		state["fRec173"] = np.zeros((3,), dtype=np.float32)
		state["fVec96"] = np.zeros((64,), dtype=np.float32)
		state["fRec175"] = np.zeros((3,), dtype=np.float32)
		state["fVec97"] = np.zeros((64,), dtype=np.float32)
		state["fRec177"] = np.zeros((3,), dtype=np.float32)
		state["fVec98"] = np.zeros((64,), dtype=np.float32)
		state["fRec179"] = np.zeros((3,), dtype=np.float32)
		state["fVec100"] = np.zeros((1024,), dtype=np.float32)
		state["fRec181"] = np.zeros((3,), dtype=np.float32)
		state["fVec101"] = np.zeros((1024,), dtype=np.float32)
		state["fRec183"] = np.zeros((3,), dtype=np.float32)
		state["fVec102"] = np.zeros((256,), dtype=np.float32)
		state["fRec185"] = np.zeros((3,), dtype=np.float32)
		state["fVec103"] = np.zeros((256,), dtype=np.float32)
		state["fRec187"] = np.zeros((3,), dtype=np.float32)
		state["fVec104"] = np.zeros((128,), dtype=np.float32)
		state["fRec189"] = np.zeros((3,), dtype=np.float32)
		state["fVec105"] = np.zeros((128,), dtype=np.float32)
		state["fRec191"] = np.zeros((3,), dtype=np.float32)
		state["fVec106"] = np.zeros((64,), dtype=np.float32)
		state["fRec193"] = np.zeros((3,), dtype=np.float32)
		state["fVec107"] = np.zeros((64,), dtype=np.float32)
		state["fRec195"] = np.zeros((3,), dtype=np.float32)
		state["fVec108"] = np.zeros((64,), dtype=np.float32)
		state["fRec197"] = np.zeros((3,), dtype=np.float32)
		state["fVec109"] = np.zeros((64,), dtype=np.float32)
		state["fRec199"] = np.zeros((3,), dtype=np.float32)
		state["fVec110"] = np.zeros((256,), dtype=np.float32)
		state["fRec201"] = np.zeros((3,), dtype=np.float32)
		state["fVec111"] = np.zeros((256,), dtype=np.float32)
		state["fRec203"] = np.zeros((3,), dtype=np.float32)
		state["fVec112"] = np.zeros((256,), dtype=np.float32)
		state["fRec205"] = np.zeros((3,), dtype=np.float32)
		state["fVec113"] = np.zeros((256,), dtype=np.float32)
		state["fRec207"] = np.zeros((3,), dtype=np.float32)
		state["fVec114"] = np.zeros((256,), dtype=np.float32)
		state["fRec209"] = np.zeros((3,), dtype=np.float32)
		state["fVec115"] = np.zeros((256,), dtype=np.float32)
		state["fRec211"] = np.zeros((3,), dtype=np.float32)
		state["fVec116"] = np.zeros((256,), dtype=np.float32)
		state["fRec213"] = np.zeros((3,), dtype=np.float32)
		state["fVec117"] = np.zeros((256,), dtype=np.float32)
		state["fRec215"] = np.zeros((3,), dtype=np.float32)
		state["fVec118"] = np.zeros((256,), dtype=np.float32)
		state["fRec217"] = np.zeros((3,), dtype=np.float32)
		state["fVec119"] = np.zeros((256,), dtype=np.float32)
		state["fRec219"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry0"] + jnp.float32(-69.0)))) 
		fSlow1 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow0))) 
		fSlow2 = params["fButton0"] 
		fSlow3 = params["fEntry1"] 
		iSlow4 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow0)))) 
		fSlow5 = (((jnp.float32(3959.208) * fSlow0) < self._fConst6).astype(jnp.int32)) 
		fSlow6 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow0))) 
		iSlow7 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow0)))) 
		fSlow8 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow0) < self._fConst6).astype(jnp.int32))) 
		fSlow9 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow0))) 
		iSlow10 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow0)))) 
		fSlow11 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow0) < self._fConst6).astype(jnp.int32))) 
		fSlow12 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow0))) 
		iSlow13 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow0)))) 
		fSlow14 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow0) < self._fConst6).astype(jnp.int32))) 
		fSlow15 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow0))) 
		iSlow16 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow0)))) 
		fSlow17 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow0) < self._fConst6).astype(jnp.int32))) 
		fSlow18 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow0))) 
		iSlow19 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow0)))) 
		fSlow20 = (((jnp.float32(3966.8186) * fSlow0) < self._fConst6).astype(jnp.int32)) 
		fSlow21 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow0))) 
		iSlow22 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow0)))) 
		fSlow23 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow0) < self._fConst6).astype(jnp.int32))) 
		fSlow24 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow0))) 
		iSlow25 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow0)))) 
		fSlow26 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow0) < self._fConst6).astype(jnp.int32))) 
		fSlow27 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow0))) 
		iSlow28 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow0)))) 
		fSlow29 = (((jnp.float32(7603.562) * fSlow0) < self._fConst6).astype(jnp.int32)) 
		fSlow30 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow0))) 
		iSlow31 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow0)))) 
		fSlow32 = (((jnp.float32(9669.452) * fSlow0) < self._fConst6).astype(jnp.int32)) 
		fSlow33 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry2"] + jnp.float32(-69.0)))) 
		fSlow34 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow33))) 
		fSlow35 = params["fButton1"] 
		iSlow36 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow33)))) 
		fSlow37 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow33) < self._fConst6).astype(jnp.int32))) 
		fSlow38 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow33))) 
		iSlow39 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow33)))) 
		fSlow40 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow33) < self._fConst6).astype(jnp.int32))) 
		fSlow41 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow33))) 
		iSlow42 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow33)))) 
		fSlow43 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow33) < self._fConst6).astype(jnp.int32))) 
		fSlow44 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow33))) 
		iSlow45 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow33)))) 
		fSlow46 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow33) < self._fConst6).astype(jnp.int32))) 
		fSlow47 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow33))) 
		iSlow48 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow33)))) 
		fSlow49 = (((jnp.float32(3959.208) * fSlow33) < self._fConst6).astype(jnp.int32)) 
		fSlow50 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow33))) 
		iSlow51 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow33)))) 
		fSlow52 = (((jnp.float32(3966.8186) * fSlow33) < self._fConst6).astype(jnp.int32)) 
		fSlow53 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow33))) 
		iSlow54 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow33)))) 
		fSlow55 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow33) < self._fConst6).astype(jnp.int32))) 
		fSlow56 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow33))) 
		iSlow57 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow33)))) 
		fSlow58 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow33) < self._fConst6).astype(jnp.int32))) 
		fSlow59 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow33))) 
		iSlow60 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow33)))) 
		fSlow61 = (((jnp.float32(7603.562) * fSlow33) < self._fConst6).astype(jnp.int32)) 
		fSlow62 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow33))) 
		iSlow63 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow33)))) 
		fSlow64 = (((jnp.float32(9669.452) * fSlow33) < self._fConst6).astype(jnp.int32)) 
		fSlow65 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry3"] + jnp.float32(-69.0)))) 
		fSlow66 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow65))) 
		fSlow67 = params["fButton2"] 
		iSlow68 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow65)))) 
		fSlow69 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow65) < self._fConst6).astype(jnp.int32))) 
		fSlow70 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow65))) 
		iSlow71 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow65)))) 
		fSlow72 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow65) < self._fConst6).astype(jnp.int32))) 
		fSlow73 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow65))) 
		iSlow74 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow65)))) 
		fSlow75 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow65) < self._fConst6).astype(jnp.int32))) 
		fSlow76 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow65))) 
		iSlow77 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow65)))) 
		fSlow78 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow65) < self._fConst6).astype(jnp.int32))) 
		fSlow79 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow65))) 
		iSlow80 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow65)))) 
		fSlow81 = (((jnp.float32(3959.208) * fSlow65) < self._fConst6).astype(jnp.int32)) 
		fSlow82 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow65))) 
		iSlow83 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow65)))) 
		fSlow84 = (((jnp.float32(3966.8186) * fSlow65) < self._fConst6).astype(jnp.int32)) 
		fSlow85 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow65))) 
		iSlow86 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow65)))) 
		fSlow87 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow65) < self._fConst6).astype(jnp.int32))) 
		fSlow88 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow65))) 
		iSlow89 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow65)))) 
		fSlow90 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow65) < self._fConst6).astype(jnp.int32))) 
		fSlow91 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow65))) 
		iSlow92 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow65)))) 
		fSlow93 = (((jnp.float32(7603.562) * fSlow65) < self._fConst6).astype(jnp.int32)) 
		fSlow94 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow65))) 
		iSlow95 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow65)))) 
		fSlow96 = (((jnp.float32(9669.452) * fSlow65) < self._fConst6).astype(jnp.int32)) 
		fSlow97 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry4"] + jnp.float32(-69.0)))) 
		fSlow98 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow97))) 
		fSlow99 = params["fButton3"] 
		iSlow100 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow97)))) 
		fSlow101 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow97) < self._fConst6).astype(jnp.int32))) 
		fSlow102 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow97))) 
		iSlow103 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow97)))) 
		fSlow104 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow97) < self._fConst6).astype(jnp.int32))) 
		fSlow105 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow97))) 
		iSlow106 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow97)))) 
		fSlow107 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow97) < self._fConst6).astype(jnp.int32))) 
		fSlow108 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow97))) 
		iSlow109 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow97)))) 
		fSlow110 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow97) < self._fConst6).astype(jnp.int32))) 
		fSlow111 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow97))) 
		iSlow112 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow97)))) 
		fSlow113 = (((jnp.float32(3959.208) * fSlow97) < self._fConst6).astype(jnp.int32)) 
		fSlow114 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow97))) 
		iSlow115 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow97)))) 
		fSlow116 = (((jnp.float32(3966.8186) * fSlow97) < self._fConst6).astype(jnp.int32)) 
		fSlow117 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow97))) 
		iSlow118 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow97)))) 
		fSlow119 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow97) < self._fConst6).astype(jnp.int32))) 
		fSlow120 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow97))) 
		iSlow121 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow97)))) 
		fSlow122 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow97) < self._fConst6).astype(jnp.int32))) 
		fSlow123 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow97))) 
		iSlow124 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow97)))) 
		fSlow125 = (((jnp.float32(7603.562) * fSlow97) < self._fConst6).astype(jnp.int32)) 
		fSlow126 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow97))) 
		iSlow127 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow97)))) 
		fSlow128 = (((jnp.float32(9669.452) * fSlow97) < self._fConst6).astype(jnp.int32)) 
		fSlow129 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry5"] + jnp.float32(-69.0)))) 
		fSlow130 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow129))) 
		fSlow131 = params["fButton4"] 
		iSlow132 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow129)))) 
		fSlow133 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow129) < self._fConst6).astype(jnp.int32))) 
		fSlow134 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow129))) 
		iSlow135 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow129)))) 
		fSlow136 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow129) < self._fConst6).astype(jnp.int32))) 
		fSlow137 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow129))) 
		iSlow138 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow129)))) 
		fSlow139 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow129) < self._fConst6).astype(jnp.int32))) 
		fSlow140 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow129))) 
		iSlow141 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow129)))) 
		fSlow142 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow129) < self._fConst6).astype(jnp.int32))) 
		fSlow143 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow129))) 
		iSlow144 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow129)))) 
		fSlow145 = (((jnp.float32(3959.208) * fSlow129) < self._fConst6).astype(jnp.int32)) 
		fSlow146 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow129))) 
		iSlow147 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow129)))) 
		fSlow148 = (((jnp.float32(3966.8186) * fSlow129) < self._fConst6).astype(jnp.int32)) 
		fSlow149 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow129))) 
		iSlow150 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow129)))) 
		fSlow151 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow129) < self._fConst6).astype(jnp.int32))) 
		fSlow152 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow129))) 
		iSlow153 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow129)))) 
		fSlow154 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow129) < self._fConst6).astype(jnp.int32))) 
		fSlow155 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow129))) 
		iSlow156 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow129)))) 
		fSlow157 = (((jnp.float32(7603.562) * fSlow129) < self._fConst6).astype(jnp.int32)) 
		fSlow158 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow129))) 
		iSlow159 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow129)))) 
		fSlow160 = (((jnp.float32(9669.452) * fSlow129) < self._fConst6).astype(jnp.int32)) 
		fSlow161 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry6"] + jnp.float32(-69.0)))) 
		fSlow162 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow161))) 
		fSlow163 = params["fButton5"] 
		iSlow164 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow161)))) 
		fSlow165 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow161) < self._fConst6).astype(jnp.int32))) 
		fSlow166 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow161))) 
		iSlow167 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow161)))) 
		fSlow168 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow161) < self._fConst6).astype(jnp.int32))) 
		fSlow169 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow161))) 
		iSlow170 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow161)))) 
		fSlow171 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow161) < self._fConst6).astype(jnp.int32))) 
		fSlow172 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow161))) 
		iSlow173 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow161)))) 
		fSlow174 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow161) < self._fConst6).astype(jnp.int32))) 
		fSlow175 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow161))) 
		iSlow176 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow161)))) 
		fSlow177 = (((jnp.float32(3959.208) * fSlow161) < self._fConst6).astype(jnp.int32)) 
		fSlow178 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow161))) 
		iSlow179 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow161)))) 
		fSlow180 = (((jnp.float32(3966.8186) * fSlow161) < self._fConst6).astype(jnp.int32)) 
		fSlow181 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow161))) 
		iSlow182 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow161)))) 
		fSlow183 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow161) < self._fConst6).astype(jnp.int32))) 
		fSlow184 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow161))) 
		iSlow185 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow161)))) 
		fSlow186 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow161) < self._fConst6).astype(jnp.int32))) 
		fSlow187 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow161))) 
		iSlow188 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow161)))) 
		fSlow189 = (((jnp.float32(7603.562) * fSlow161) < self._fConst6).astype(jnp.int32)) 
		fSlow190 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow161))) 
		iSlow191 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow161)))) 
		fSlow192 = (((jnp.float32(9669.452) * fSlow161) < self._fConst6).astype(jnp.int32)) 
		fSlow193 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry7"] + jnp.float32(-69.0)))) 
		fSlow194 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow193))) 
		fSlow195 = params["fButton6"] 
		iSlow196 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow193)))) 
		fSlow197 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow193) < self._fConst6).astype(jnp.int32))) 
		fSlow198 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow193))) 
		iSlow199 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow193)))) 
		fSlow200 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow193) < self._fConst6).astype(jnp.int32))) 
		fSlow201 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow193))) 
		iSlow202 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow193)))) 
		fSlow203 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow193) < self._fConst6).astype(jnp.int32))) 
		fSlow204 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow193))) 
		iSlow205 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow193)))) 
		fSlow206 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow193) < self._fConst6).astype(jnp.int32))) 
		fSlow207 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow193))) 
		iSlow208 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow193)))) 
		fSlow209 = (((jnp.float32(3959.208) * fSlow193) < self._fConst6).astype(jnp.int32)) 
		fSlow210 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow193))) 
		iSlow211 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow193)))) 
		fSlow212 = (((jnp.float32(3966.8186) * fSlow193) < self._fConst6).astype(jnp.int32)) 
		fSlow213 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow193))) 
		iSlow214 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow193)))) 
		fSlow215 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow193) < self._fConst6).astype(jnp.int32))) 
		fSlow216 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow193))) 
		iSlow217 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow193)))) 
		fSlow218 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow193) < self._fConst6).astype(jnp.int32))) 
		fSlow219 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow193))) 
		iSlow220 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow193)))) 
		fSlow221 = (((jnp.float32(7603.562) * fSlow193) < self._fConst6).astype(jnp.int32)) 
		fSlow222 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow193))) 
		iSlow223 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow193)))) 
		fSlow224 = (((jnp.float32(9669.452) * fSlow193) < self._fConst6).astype(jnp.int32)) 
		fSlow225 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry8"] + jnp.float32(-69.0)))) 
		fSlow226 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow225))) 
		fSlow227 = params["fButton7"] 
		iSlow228 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow225)))) 
		fSlow229 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow225) < self._fConst6).astype(jnp.int32))) 
		fSlow230 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow225))) 
		iSlow231 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow225)))) 
		fSlow232 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow225) < self._fConst6).astype(jnp.int32))) 
		fSlow233 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow225))) 
		iSlow234 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow225)))) 
		fSlow235 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow225) < self._fConst6).astype(jnp.int32))) 
		fSlow236 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow225))) 
		iSlow237 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow225)))) 
		fSlow238 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow225) < self._fConst6).astype(jnp.int32))) 
		fSlow239 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow225))) 
		iSlow240 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow225)))) 
		fSlow241 = (((jnp.float32(3959.208) * fSlow225) < self._fConst6).astype(jnp.int32)) 
		fSlow242 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow225))) 
		iSlow243 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow225)))) 
		fSlow244 = (((jnp.float32(3966.8186) * fSlow225) < self._fConst6).astype(jnp.int32)) 
		fSlow245 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow225))) 
		iSlow246 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow225)))) 
		fSlow247 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow225) < self._fConst6).astype(jnp.int32))) 
		fSlow248 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow225))) 
		iSlow249 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow225)))) 
		fSlow250 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow225) < self._fConst6).astype(jnp.int32))) 
		fSlow251 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow225))) 
		iSlow252 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow225)))) 
		fSlow253 = (((jnp.float32(7603.562) * fSlow225) < self._fConst6).astype(jnp.int32)) 
		fSlow254 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow225))) 
		iSlow255 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow225)))) 
		fSlow256 = (((jnp.float32(9669.452) * fSlow225) < self._fConst6).astype(jnp.int32)) 
		fSlow257 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry9"] + jnp.float32(-69.0)))) 
		fSlow258 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow257))) 
		fSlow259 = params["fButton8"] 
		iSlow260 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow257)))) 
		fSlow261 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow257) < self._fConst6).astype(jnp.int32))) 
		fSlow262 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow257))) 
		iSlow263 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow257)))) 
		fSlow264 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow257) < self._fConst6).astype(jnp.int32))) 
		fSlow265 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow257))) 
		iSlow266 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow257)))) 
		fSlow267 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow257) < self._fConst6).astype(jnp.int32))) 
		fSlow268 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow257))) 
		iSlow269 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow257)))) 
		fSlow270 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow257) < self._fConst6).astype(jnp.int32))) 
		fSlow271 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow257))) 
		iSlow272 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow257)))) 
		fSlow273 = (((jnp.float32(3959.208) * fSlow257) < self._fConst6).astype(jnp.int32)) 
		fSlow274 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow257))) 
		iSlow275 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow257)))) 
		fSlow276 = (((jnp.float32(3966.8186) * fSlow257) < self._fConst6).astype(jnp.int32)) 
		fSlow277 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow257))) 
		iSlow278 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow257)))) 
		fSlow279 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow257) < self._fConst6).astype(jnp.int32))) 
		fSlow280 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow257))) 
		iSlow281 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow257)))) 
		fSlow282 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow257) < self._fConst6).astype(jnp.int32))) 
		fSlow283 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow257))) 
		iSlow284 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow257)))) 
		fSlow285 = (((jnp.float32(7603.562) * fSlow257) < self._fConst6).astype(jnp.int32)) 
		fSlow286 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow257))) 
		iSlow287 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow257)))) 
		fSlow288 = (((jnp.float32(9669.452) * fSlow257) < self._fConst6).astype(jnp.int32)) 
		fSlow289 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fEntry10"] + jnp.float32(-69.0)))) 
		fSlow290 = (self._fConst4 * jnp.cos((self._fConst14 * fSlow289))) 
		fSlow291 = params["fButton9"] 
		iSlow292 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst15 / fSlow289)))) 
		fSlow293 = (jnp.float32(0.999926) * (((jnp.float32(438.2877) * fSlow289) < self._fConst6).astype(jnp.int32))) 
		fSlow294 = (self._fConst4 * jnp.cos((self._fConst12 * fSlow289))) 
		iSlow295 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst13 / fSlow289)))) 
		fSlow296 = (jnp.float32(0.999926) * (((jnp.float32(441.71234) * fSlow289) < self._fConst6).astype(jnp.int32))) 
		fSlow297 = (self._fConst4 * jnp.cos((self._fConst10 * fSlow289))) 
		iSlow298 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst11 / fSlow289)))) 
		fSlow299 = (jnp.float32(0.9999828) * (((jnp.float32(1310.8384) * fSlow289) < self._fConst6).astype(jnp.int32))) 
		fSlow300 = (self._fConst4 * jnp.cos((self._fConst8 * fSlow289))) 
		iSlow301 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst9 / fSlow289)))) 
		fSlow302 = (jnp.float32(0.9999828) * (((jnp.float32(1317.0509) * fSlow289) < self._fConst6).astype(jnp.int32))) 
		fSlow303 = (self._fConst4 * jnp.cos((self._fConst3 * fSlow289))) 
		iSlow304 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst5 / fSlow289)))) 
		fSlow305 = (((jnp.float32(3959.208) * fSlow289) < self._fConst6).astype(jnp.int32)) 
		fSlow306 = (self._fConst4 * jnp.cos((self._fConst16 * fSlow289))) 
		iSlow307 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst17 / fSlow289)))) 
		fSlow308 = (((jnp.float32(3966.8186) * fSlow289) < self._fConst6).astype(jnp.int32)) 
		fSlow309 = (self._fConst4 * jnp.cos((self._fConst18 * fSlow289))) 
		iSlow310 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst19 / fSlow289)))) 
		fSlow311 = (jnp.float32(0.9999655) * (((jnp.float32(5646.533) * fSlow289) < self._fConst6).astype(jnp.int32))) 
		fSlow312 = (self._fConst4 * jnp.cos((self._fConst20 * fSlow289))) 
		iSlow313 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst21 / fSlow289)))) 
		fSlow314 = (jnp.float32(0.9999655) * (((jnp.float32(5635.248) * fSlow289) < self._fConst6).astype(jnp.int32))) 
		fSlow315 = (self._fConst4 * jnp.cos((self._fConst22 * fSlow289))) 
		iSlow316 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst23 / fSlow289)))) 
		fSlow317 = (((jnp.float32(7603.562) * fSlow289) < self._fConst6).astype(jnp.int32)) 
		fSlow318 = (self._fConst4 * jnp.cos((self._fConst24 * fSlow289))) 
		iSlow319 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst25 / fSlow289)))) 
		fSlow320 = (((jnp.float32(9669.452) * fSlow289) < self._fConst6).astype(jnp.int32)) 
		fSlow321 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow33))) 
		iSlow322 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow33)))) 
		fSlow323 = (((jnp.float32(2509.959) * fSlow33) < self._fConst6).astype(jnp.int32)) 
		fSlow324 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow0))) 
		iSlow325 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow0)))) 
		fSlow326 = (((jnp.float32(2509.959) * fSlow0) < self._fConst6).astype(jnp.int32)) 
		fSlow327 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow65))) 
		iSlow328 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow65)))) 
		fSlow329 = (((jnp.float32(2509.959) * fSlow65) < self._fConst6).astype(jnp.int32)) 
		fSlow330 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow97))) 
		iSlow331 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow97)))) 
		fSlow332 = (((jnp.float32(2509.959) * fSlow97) < self._fConst6).astype(jnp.int32)) 
		fSlow333 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow129))) 
		iSlow334 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow129)))) 
		fSlow335 = (((jnp.float32(2509.959) * fSlow129) < self._fConst6).astype(jnp.int32)) 
		fSlow336 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow161))) 
		iSlow337 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow161)))) 
		fSlow338 = (((jnp.float32(2509.959) * fSlow161) < self._fConst6).astype(jnp.int32)) 
		fSlow339 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow193))) 
		iSlow340 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow193)))) 
		fSlow341 = (((jnp.float32(2509.959) * fSlow193) < self._fConst6).astype(jnp.int32)) 
		fSlow342 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow225))) 
		iSlow343 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow225)))) 
		fSlow344 = (((jnp.float32(2509.959) * fSlow225) < self._fConst6).astype(jnp.int32)) 
		fSlow345 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow257))) 
		iSlow346 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow257)))) 
		fSlow347 = (((jnp.float32(2509.959) * fSlow257) < self._fConst6).astype(jnp.int32)) 
		fSlow348 = (self._fConst4 * jnp.cos((self._fConst26 * fSlow289))) 
		iSlow349 = jnp.int32(jnp.minimum(jnp.float32(4096.0), jnp.maximum(jnp.float32(0.0), (self._fConst27 / fSlow289)))) 
		fSlow350 = (((jnp.float32(2509.959) * fSlow289) < self._fConst6).astype(jnp.int32)) 
		fVec0_temp = state["fVec0"] 
		fRec0_temp = state["fRec0"] 
		fRec2_temp = state["fRec2"] 
		fRec4_temp = state["fRec4"] 
		fRec6_temp = state["fRec6"] 
		fRec8_temp = state["fRec8"] 
		fRec10_temp = state["fRec10"] 
		fRec12_temp = state["fRec12"] 
		fRec14_temp = state["fRec14"] 
		fRec16_temp = state["fRec16"] 
		fRec18_temp = state["fRec18"] 
		fVec11_temp = state["fVec11"] 
		fRec20_temp = state["fRec20"] 
		fRec22_temp = state["fRec22"] 
		fRec24_temp = state["fRec24"] 
		fRec26_temp = state["fRec26"] 
		fRec28_temp = state["fRec28"] 
		fRec30_temp = state["fRec30"] 
		fRec32_temp = state["fRec32"] 
		fRec34_temp = state["fRec34"] 
		fRec36_temp = state["fRec36"] 
		fRec38_temp = state["fRec38"] 
		fVec22_temp = state["fVec22"] 
		fRec40_temp = state["fRec40"] 
		fRec42_temp = state["fRec42"] 
		fRec44_temp = state["fRec44"] 
		fRec46_temp = state["fRec46"] 
		fRec48_temp = state["fRec48"] 
		fRec50_temp = state["fRec50"] 
		fRec52_temp = state["fRec52"] 
		fRec54_temp = state["fRec54"] 
		fRec56_temp = state["fRec56"] 
		fRec58_temp = state["fRec58"] 
		fVec33_temp = state["fVec33"] 
		fRec60_temp = state["fRec60"] 
		fRec62_temp = state["fRec62"] 
		fRec64_temp = state["fRec64"] 
		fRec66_temp = state["fRec66"] 
		fRec68_temp = state["fRec68"] 
		fRec70_temp = state["fRec70"] 
		fRec72_temp = state["fRec72"] 
		fRec74_temp = state["fRec74"] 
		fRec76_temp = state["fRec76"] 
		fRec78_temp = state["fRec78"] 
		fVec44_temp = state["fVec44"] 
		fRec80_temp = state["fRec80"] 
		fRec82_temp = state["fRec82"] 
		fRec84_temp = state["fRec84"] 
		fRec86_temp = state["fRec86"] 
		fRec88_temp = state["fRec88"] 
		fRec90_temp = state["fRec90"] 
		fRec92_temp = state["fRec92"] 
		fRec94_temp = state["fRec94"] 
		fRec96_temp = state["fRec96"] 
		fRec98_temp = state["fRec98"] 
		fVec55_temp = state["fVec55"] 
		fRec100_temp = state["fRec100"] 
		fRec102_temp = state["fRec102"] 
		fRec104_temp = state["fRec104"] 
		fRec106_temp = state["fRec106"] 
		fRec108_temp = state["fRec108"] 
		fRec110_temp = state["fRec110"] 
		fRec112_temp = state["fRec112"] 
		fRec114_temp = state["fRec114"] 
		fRec116_temp = state["fRec116"] 
		fRec118_temp = state["fRec118"] 
		fVec66_temp = state["fVec66"] 
		fRec120_temp = state["fRec120"] 
		fRec122_temp = state["fRec122"] 
		fRec124_temp = state["fRec124"] 
		fRec126_temp = state["fRec126"] 
		fRec128_temp = state["fRec128"] 
		fRec130_temp = state["fRec130"] 
		fRec132_temp = state["fRec132"] 
		fRec134_temp = state["fRec134"] 
		fRec136_temp = state["fRec136"] 
		fRec138_temp = state["fRec138"] 
		fVec77_temp = state["fVec77"] 
		fRec140_temp = state["fRec140"] 
		fRec142_temp = state["fRec142"] 
		fRec144_temp = state["fRec144"] 
		fRec146_temp = state["fRec146"] 
		fRec148_temp = state["fRec148"] 
		fRec150_temp = state["fRec150"] 
		fRec152_temp = state["fRec152"] 
		fRec154_temp = state["fRec154"] 
		fRec156_temp = state["fRec156"] 
		fRec158_temp = state["fRec158"] 
		fVec88_temp = state["fVec88"] 
		fRec160_temp = state["fRec160"] 
		fRec162_temp = state["fRec162"] 
		fRec164_temp = state["fRec164"] 
		fRec166_temp = state["fRec166"] 
		fRec168_temp = state["fRec168"] 
		fRec170_temp = state["fRec170"] 
		fRec172_temp = state["fRec172"] 
		fRec174_temp = state["fRec174"] 
		fRec176_temp = state["fRec176"] 
		fRec178_temp = state["fRec178"] 
		fVec99_temp = state["fVec99"] 
		fRec180_temp = state["fRec180"] 
		fRec182_temp = state["fRec182"] 
		fRec184_temp = state["fRec184"] 
		fRec186_temp = state["fRec186"] 
		fRec188_temp = state["fRec188"] 
		fRec190_temp = state["fRec190"] 
		fRec192_temp = state["fRec192"] 
		fRec194_temp = state["fRec194"] 
		fRec196_temp = state["fRec196"] 
		fRec198_temp = state["fRec198"] 
		fRec200_temp = state["fRec200"] 
		fRec202_temp = state["fRec202"] 
		fRec204_temp = state["fRec204"] 
		fRec206_temp = state["fRec206"] 
		fRec208_temp = state["fRec208"] 
		fRec210_temp = state["fRec210"] 
		fRec212_temp = state["fRec212"] 
		fRec214_temp = state["fRec214"] 
		fRec216_temp = state["fRec216"] 
		fRec218_temp = state["fRec218"] 
		state["fVec0"] = fSlow2 
		fTemp0 = ((fSlow2 > fVec0_temp).astype(jnp.int32)) 
		fTemp1 = (jnp.float32(5.0063033) * fTemp0) 
		state["fVec1"] = state["fVec1"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec0_temp) + fTemp1)) 
		state["fRec1"] = state["fRec1"].at[0].set(((fSlow5 * state["fVec1"][((state["IOTA0"] - iSlow4) & 127).astype(jnp.int32)]) - ((fSlow1 * state["fRec1"][1]) + (self._fConst2 * state["fRec1"][2])))) 
		state["fRec0"] = (self._fConst7 * (state["fRec1"][0] - state["fRec1"][2])) 
		fTemp2 = (jnp.float32(2.0914886) * fTemp0) 
		state["fVec2"] = state["fVec2"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp2 + (fSlow3 * fRec2_temp))) 
		state["fRec3"] = state["fRec3"].at[0].set(((fSlow8 * state["fVec2"][((state["IOTA0"] - iSlow7) & 255).astype(jnp.int32)]) - ((fSlow6 * state["fRec3"][1]) + (self._fConst2 * state["fRec3"][2])))) 
		state["fRec2"] = (self._fConst7 * (state["fRec3"][0] - state["fRec3"][2])) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec4_temp) + fTemp2)) 
		state["fRec5"] = state["fRec5"].at[0].set(((fSlow11 * state["fVec3"][((state["IOTA0"] - iSlow10) & 255).astype(jnp.int32)]) - ((fSlow9 * state["fRec5"][1]) + (self._fConst2 * state["fRec5"][2])))) 
		state["fRec4"] = (self._fConst7 * (state["fRec5"][0] - state["fRec5"][2])) 
		fTemp3 = (jnp.float32(2.1900356) * fTemp0) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp3 + (fSlow3 * fRec6_temp))) 
		state["fRec7"] = state["fRec7"].at[0].set(((fSlow14 * state["fVec4"][((state["IOTA0"] - iSlow13) & 1023).astype(jnp.int32)]) - ((fSlow12 * state["fRec7"][1]) + (self._fConst2 * state["fRec7"][2])))) 
		state["fRec6"] = (self._fConst7 * (state["fRec7"][0] - state["fRec7"][2])) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec8_temp) + fTemp3)) 
		state["fRec9"] = state["fRec9"].at[0].set(((fSlow17 * state["fVec5"][((state["IOTA0"] - iSlow16) & 1023).astype(jnp.int32)]) - ((fSlow15 * state["fRec9"][1]) + (self._fConst2 * state["fRec9"][2])))) 
		state["fRec8"] = (self._fConst7 * (state["fRec9"][0] - state["fRec9"][2])) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp1 + (fSlow3 * fRec10_temp))) 
		state["fRec11"] = state["fRec11"].at[0].set(((fSlow20 * state["fVec6"][((state["IOTA0"] - iSlow19) & 127).astype(jnp.int32)]) - ((fSlow18 * state["fRec11"][1]) + (self._fConst2 * state["fRec11"][2])))) 
		state["fRec10"] = (self._fConst7 * (state["fRec11"][0] - state["fRec11"][2])) 
		fTemp4 = (jnp.float32(1.7063034) * fTemp0) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec12_temp) + fTemp4)) 
		state["fRec13"] = state["fRec13"].at[0].set(((fSlow23 * state["fVec7"][((state["IOTA0"] - iSlow22) & 63).astype(jnp.int32)]) - ((fSlow21 * state["fRec13"][1]) + (self._fConst2 * state["fRec13"][2])))) 
		state["fRec12"] = (self._fConst7 * (state["fRec13"][0] - state["fRec13"][2])) 
		state["fVec8"] = state["fVec8"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp4 + (fSlow3 * fRec14_temp))) 
		state["fRec15"] = state["fRec15"].at[0].set(((fSlow26 * state["fVec8"][((state["IOTA0"] - iSlow25) & 63).astype(jnp.int32)]) - ((fSlow24 * state["fRec15"][1]) + (self._fConst2 * state["fRec15"][2])))) 
		state["fRec14"] = (self._fConst7 * (state["fRec15"][0] - state["fRec15"][2])) 
		fTemp5 = (jnp.float32(6.7063036) * fTemp0) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec16_temp) + fTemp5)) 
		state["fRec17"] = state["fRec17"].at[0].set(((fSlow29 * state["fVec9"][((state["IOTA0"] - iSlow28) & 63).astype(jnp.int32)]) - ((fSlow27 * state["fRec17"][1]) + (self._fConst2 * state["fRec17"][2])))) 
		state["fRec16"] = (self._fConst7 * (state["fRec17"][0] - state["fRec17"][2])) 
		state["fVec10"] = state["fVec10"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp5 + (fSlow3 * fRec18_temp))) 
		state["fRec19"] = state["fRec19"].at[0].set(((fSlow32 * state["fVec10"][((state["IOTA0"] - iSlow31) & 63).astype(jnp.int32)]) - ((fSlow30 * state["fRec19"][1]) + (self._fConst2 * state["fRec19"][2])))) 
		state["fRec18"] = (self._fConst7 * (state["fRec19"][0] - state["fRec19"][2])) 
		state["fVec11"] = fSlow35 
		fTemp6 = ((fSlow35 > fVec11_temp).astype(jnp.int32)) 
		fTemp7 = (jnp.float32(2.1900356) * fTemp6) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec20_temp) + fTemp7)) 
		state["fRec21"] = state["fRec21"].at[0].set(((fSlow37 * state["fVec12"][((state["IOTA0"] - iSlow36) & 1023).astype(jnp.int32)]) - ((fSlow34 * state["fRec21"][1]) + (self._fConst2 * state["fRec21"][2])))) 
		state["fRec20"] = (self._fConst7 * (state["fRec21"][0] - state["fRec21"][2])) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp7 + (fSlow3 * fRec22_temp))) 
		state["fRec23"] = state["fRec23"].at[0].set(((fSlow40 * state["fVec13"][((state["IOTA0"] - iSlow39) & 1023).astype(jnp.int32)]) - ((fSlow38 * state["fRec23"][1]) + (self._fConst2 * state["fRec23"][2])))) 
		state["fRec22"] = (self._fConst7 * (state["fRec23"][0] - state["fRec23"][2])) 
		fTemp8 = (jnp.float32(2.0914886) * fTemp6) 
		state["fVec14"] = state["fVec14"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec24_temp) + fTemp8)) 
		state["fRec25"] = state["fRec25"].at[0].set(((fSlow43 * state["fVec14"][((state["IOTA0"] - iSlow42) & 255).astype(jnp.int32)]) - ((fSlow41 * state["fRec25"][1]) + (self._fConst2 * state["fRec25"][2])))) 
		state["fRec24"] = (self._fConst7 * (state["fRec25"][0] - state["fRec25"][2])) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp8 + (fSlow3 * fRec26_temp))) 
		state["fRec27"] = state["fRec27"].at[0].set(((fSlow46 * state["fVec15"][((state["IOTA0"] - iSlow45) & 255).astype(jnp.int32)]) - ((fSlow44 * state["fRec27"][1]) + (self._fConst2 * state["fRec27"][2])))) 
		state["fRec26"] = (self._fConst7 * (state["fRec27"][0] - state["fRec27"][2])) 
		fTemp9 = (jnp.float32(5.0063033) * fTemp6) 
		state["fVec16"] = state["fVec16"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec28_temp) + fTemp9)) 
		state["fRec29"] = state["fRec29"].at[0].set(((fSlow49 * state["fVec16"][((state["IOTA0"] - iSlow48) & 127).astype(jnp.int32)]) - ((fSlow47 * state["fRec29"][1]) + (self._fConst2 * state["fRec29"][2])))) 
		state["fRec28"] = (self._fConst7 * (state["fRec29"][0] - state["fRec29"][2])) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp9 + (fSlow3 * fRec30_temp))) 
		state["fRec31"] = state["fRec31"].at[0].set(((fSlow52 * state["fVec17"][((state["IOTA0"] - iSlow51) & 127).astype(jnp.int32)]) - ((fSlow50 * state["fRec31"][1]) + (self._fConst2 * state["fRec31"][2])))) 
		state["fRec30"] = (self._fConst7 * (state["fRec31"][0] - state["fRec31"][2])) 
		fTemp10 = (jnp.float32(1.7063034) * fTemp6) 
		state["fVec18"] = state["fVec18"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec32_temp) + fTemp10)) 
		state["fRec33"] = state["fRec33"].at[0].set(((fSlow55 * state["fVec18"][((state["IOTA0"] - iSlow54) & 63).astype(jnp.int32)]) - ((fSlow53 * state["fRec33"][1]) + (self._fConst2 * state["fRec33"][2])))) 
		state["fRec32"] = (self._fConst7 * (state["fRec33"][0] - state["fRec33"][2])) 
		state["fVec19"] = state["fVec19"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp10 + (fSlow3 * fRec34_temp))) 
		state["fRec35"] = state["fRec35"].at[0].set(((fSlow58 * state["fVec19"][((state["IOTA0"] - iSlow57) & 63).astype(jnp.int32)]) - ((fSlow56 * state["fRec35"][1]) + (self._fConst2 * state["fRec35"][2])))) 
		state["fRec34"] = (self._fConst7 * (state["fRec35"][0] - state["fRec35"][2])) 
		fTemp11 = (jnp.float32(6.7063036) * fTemp6) 
		state["fVec20"] = state["fVec20"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec36_temp) + fTemp11)) 
		state["fRec37"] = state["fRec37"].at[0].set(((fSlow61 * state["fVec20"][((state["IOTA0"] - iSlow60) & 63).astype(jnp.int32)]) - ((fSlow59 * state["fRec37"][1]) + (self._fConst2 * state["fRec37"][2])))) 
		state["fRec36"] = (self._fConst7 * (state["fRec37"][0] - state["fRec37"][2])) 
		state["fVec21"] = state["fVec21"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp11 + (fSlow3 * fRec38_temp))) 
		state["fRec39"] = state["fRec39"].at[0].set(((fSlow64 * state["fVec21"][((state["IOTA0"] - iSlow63) & 63).astype(jnp.int32)]) - ((fSlow62 * state["fRec39"][1]) + (self._fConst2 * state["fRec39"][2])))) 
		state["fRec38"] = (self._fConst7 * (state["fRec39"][0] - state["fRec39"][2])) 
		state["fVec22"] = fSlow67 
		fTemp12 = ((fSlow67 > fVec22_temp).astype(jnp.int32)) 
		fTemp13 = (jnp.float32(2.1900356) * fTemp12) 
		state["fVec23"] = state["fVec23"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec40_temp) + fTemp13)) 
		state["fRec41"] = state["fRec41"].at[0].set(((fSlow69 * state["fVec23"][((state["IOTA0"] - iSlow68) & 1023).astype(jnp.int32)]) - ((fSlow66 * state["fRec41"][1]) + (self._fConst2 * state["fRec41"][2])))) 
		state["fRec40"] = (self._fConst7 * (state["fRec41"][0] - state["fRec41"][2])) 
		state["fVec24"] = state["fVec24"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp13 + (fSlow3 * fRec42_temp))) 
		state["fRec43"] = state["fRec43"].at[0].set(((fSlow72 * state["fVec24"][((state["IOTA0"] - iSlow71) & 1023).astype(jnp.int32)]) - ((fSlow70 * state["fRec43"][1]) + (self._fConst2 * state["fRec43"][2])))) 
		state["fRec42"] = (self._fConst7 * (state["fRec43"][0] - state["fRec43"][2])) 
		fTemp14 = (jnp.float32(2.0914886) * fTemp12) 
		state["fVec25"] = state["fVec25"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec44_temp) + fTemp14)) 
		state["fRec45"] = state["fRec45"].at[0].set(((fSlow75 * state["fVec25"][((state["IOTA0"] - iSlow74) & 255).astype(jnp.int32)]) - ((fSlow73 * state["fRec45"][1]) + (self._fConst2 * state["fRec45"][2])))) 
		state["fRec44"] = (self._fConst7 * (state["fRec45"][0] - state["fRec45"][2])) 
		state["fVec26"] = state["fVec26"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp14 + (fSlow3 * fRec46_temp))) 
		state["fRec47"] = state["fRec47"].at[0].set(((fSlow78 * state["fVec26"][((state["IOTA0"] - iSlow77) & 255).astype(jnp.int32)]) - ((fSlow76 * state["fRec47"][1]) + (self._fConst2 * state["fRec47"][2])))) 
		state["fRec46"] = (self._fConst7 * (state["fRec47"][0] - state["fRec47"][2])) 
		fTemp15 = (jnp.float32(5.0063033) * fTemp12) 
		state["fVec27"] = state["fVec27"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec48_temp) + fTemp15)) 
		state["fRec49"] = state["fRec49"].at[0].set(((fSlow81 * state["fVec27"][((state["IOTA0"] - iSlow80) & 127).astype(jnp.int32)]) - ((fSlow79 * state["fRec49"][1]) + (self._fConst2 * state["fRec49"][2])))) 
		state["fRec48"] = (self._fConst7 * (state["fRec49"][0] - state["fRec49"][2])) 
		state["fVec28"] = state["fVec28"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp15 + (fSlow3 * fRec50_temp))) 
		state["fRec51"] = state["fRec51"].at[0].set(((fSlow84 * state["fVec28"][((state["IOTA0"] - iSlow83) & 127).astype(jnp.int32)]) - ((fSlow82 * state["fRec51"][1]) + (self._fConst2 * state["fRec51"][2])))) 
		state["fRec50"] = (self._fConst7 * (state["fRec51"][0] - state["fRec51"][2])) 
		fTemp16 = (jnp.float32(1.7063034) * fTemp12) 
		state["fVec29"] = state["fVec29"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec52_temp) + fTemp16)) 
		state["fRec53"] = state["fRec53"].at[0].set(((fSlow87 * state["fVec29"][((state["IOTA0"] - iSlow86) & 63).astype(jnp.int32)]) - ((fSlow85 * state["fRec53"][1]) + (self._fConst2 * state["fRec53"][2])))) 
		state["fRec52"] = (self._fConst7 * (state["fRec53"][0] - state["fRec53"][2])) 
		state["fVec30"] = state["fVec30"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp16 + (fSlow3 * fRec54_temp))) 
		state["fRec55"] = state["fRec55"].at[0].set(((fSlow90 * state["fVec30"][((state["IOTA0"] - iSlow89) & 63).astype(jnp.int32)]) - ((fSlow88 * state["fRec55"][1]) + (self._fConst2 * state["fRec55"][2])))) 
		state["fRec54"] = (self._fConst7 * (state["fRec55"][0] - state["fRec55"][2])) 
		fTemp17 = (jnp.float32(6.7063036) * fTemp12) 
		state["fVec31"] = state["fVec31"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec56_temp) + fTemp17)) 
		state["fRec57"] = state["fRec57"].at[0].set(((fSlow93 * state["fVec31"][((state["IOTA0"] - iSlow92) & 63).astype(jnp.int32)]) - ((fSlow91 * state["fRec57"][1]) + (self._fConst2 * state["fRec57"][2])))) 
		state["fRec56"] = (self._fConst7 * (state["fRec57"][0] - state["fRec57"][2])) 
		state["fVec32"] = state["fVec32"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp17 + (fSlow3 * fRec58_temp))) 
		state["fRec59"] = state["fRec59"].at[0].set(((fSlow96 * state["fVec32"][((state["IOTA0"] - iSlow95) & 63).astype(jnp.int32)]) - ((fSlow94 * state["fRec59"][1]) + (self._fConst2 * state["fRec59"][2])))) 
		state["fRec58"] = (self._fConst7 * (state["fRec59"][0] - state["fRec59"][2])) 
		state["fVec33"] = fSlow99 
		fTemp18 = ((fSlow99 > fVec33_temp).astype(jnp.int32)) 
		fTemp19 = (jnp.float32(2.1900356) * fTemp18) 
		state["fVec34"] = state["fVec34"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec60_temp) + fTemp19)) 
		state["fRec61"] = state["fRec61"].at[0].set(((fSlow101 * state["fVec34"][((state["IOTA0"] - iSlow100) & 1023).astype(jnp.int32)]) - ((fSlow98 * state["fRec61"][1]) + (self._fConst2 * state["fRec61"][2])))) 
		state["fRec60"] = (self._fConst7 * (state["fRec61"][0] - state["fRec61"][2])) 
		state["fVec35"] = state["fVec35"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp19 + (fSlow3 * fRec62_temp))) 
		state["fRec63"] = state["fRec63"].at[0].set(((fSlow104 * state["fVec35"][((state["IOTA0"] - iSlow103) & 1023).astype(jnp.int32)]) - ((fSlow102 * state["fRec63"][1]) + (self._fConst2 * state["fRec63"][2])))) 
		state["fRec62"] = (self._fConst7 * (state["fRec63"][0] - state["fRec63"][2])) 
		fTemp20 = (jnp.float32(2.0914886) * fTemp18) 
		state["fVec36"] = state["fVec36"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec64_temp) + fTemp20)) 
		state["fRec65"] = state["fRec65"].at[0].set(((fSlow107 * state["fVec36"][((state["IOTA0"] - iSlow106) & 255).astype(jnp.int32)]) - ((fSlow105 * state["fRec65"][1]) + (self._fConst2 * state["fRec65"][2])))) 
		state["fRec64"] = (self._fConst7 * (state["fRec65"][0] - state["fRec65"][2])) 
		state["fVec37"] = state["fVec37"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp20 + (fSlow3 * fRec66_temp))) 
		state["fRec67"] = state["fRec67"].at[0].set(((fSlow110 * state["fVec37"][((state["IOTA0"] - iSlow109) & 255).astype(jnp.int32)]) - ((fSlow108 * state["fRec67"][1]) + (self._fConst2 * state["fRec67"][2])))) 
		state["fRec66"] = (self._fConst7 * (state["fRec67"][0] - state["fRec67"][2])) 
		fTemp21 = (jnp.float32(5.0063033) * fTemp18) 
		state["fVec38"] = state["fVec38"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec68_temp) + fTemp21)) 
		state["fRec69"] = state["fRec69"].at[0].set(((fSlow113 * state["fVec38"][((state["IOTA0"] - iSlow112) & 127).astype(jnp.int32)]) - ((fSlow111 * state["fRec69"][1]) + (self._fConst2 * state["fRec69"][2])))) 
		state["fRec68"] = (self._fConst7 * (state["fRec69"][0] - state["fRec69"][2])) 
		state["fVec39"] = state["fVec39"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp21 + (fSlow3 * fRec70_temp))) 
		state["fRec71"] = state["fRec71"].at[0].set(((fSlow116 * state["fVec39"][((state["IOTA0"] - iSlow115) & 127).astype(jnp.int32)]) - ((fSlow114 * state["fRec71"][1]) + (self._fConst2 * state["fRec71"][2])))) 
		state["fRec70"] = (self._fConst7 * (state["fRec71"][0] - state["fRec71"][2])) 
		fTemp22 = (jnp.float32(1.7063034) * fTemp18) 
		state["fVec40"] = state["fVec40"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec72_temp) + fTemp22)) 
		state["fRec73"] = state["fRec73"].at[0].set(((fSlow119 * state["fVec40"][((state["IOTA0"] - iSlow118) & 63).astype(jnp.int32)]) - ((fSlow117 * state["fRec73"][1]) + (self._fConst2 * state["fRec73"][2])))) 
		state["fRec72"] = (self._fConst7 * (state["fRec73"][0] - state["fRec73"][2])) 
		state["fVec41"] = state["fVec41"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp22 + (fSlow3 * fRec74_temp))) 
		state["fRec75"] = state["fRec75"].at[0].set(((fSlow122 * state["fVec41"][((state["IOTA0"] - iSlow121) & 63).astype(jnp.int32)]) - ((fSlow120 * state["fRec75"][1]) + (self._fConst2 * state["fRec75"][2])))) 
		state["fRec74"] = (self._fConst7 * (state["fRec75"][0] - state["fRec75"][2])) 
		fTemp23 = (jnp.float32(6.7063036) * fTemp18) 
		state["fVec42"] = state["fVec42"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec76_temp) + fTemp23)) 
		state["fRec77"] = state["fRec77"].at[0].set(((fSlow125 * state["fVec42"][((state["IOTA0"] - iSlow124) & 63).astype(jnp.int32)]) - ((fSlow123 * state["fRec77"][1]) + (self._fConst2 * state["fRec77"][2])))) 
		state["fRec76"] = (self._fConst7 * (state["fRec77"][0] - state["fRec77"][2])) 
		state["fVec43"] = state["fVec43"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp23 + (fSlow3 * fRec78_temp))) 
		state["fRec79"] = state["fRec79"].at[0].set(((fSlow128 * state["fVec43"][((state["IOTA0"] - iSlow127) & 63).astype(jnp.int32)]) - ((fSlow126 * state["fRec79"][1]) + (self._fConst2 * state["fRec79"][2])))) 
		state["fRec78"] = (self._fConst7 * (state["fRec79"][0] - state["fRec79"][2])) 
		state["fVec44"] = fSlow131 
		fTemp24 = ((fSlow131 > fVec44_temp).astype(jnp.int32)) 
		fTemp25 = (jnp.float32(2.1900356) * fTemp24) 
		state["fVec45"] = state["fVec45"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec80_temp) + fTemp25)) 
		state["fRec81"] = state["fRec81"].at[0].set(((fSlow133 * state["fVec45"][((state["IOTA0"] - iSlow132) & 1023).astype(jnp.int32)]) - ((fSlow130 * state["fRec81"][1]) + (self._fConst2 * state["fRec81"][2])))) 
		state["fRec80"] = (self._fConst7 * (state["fRec81"][0] - state["fRec81"][2])) 
		state["fVec46"] = state["fVec46"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp25 + (fSlow3 * fRec82_temp))) 
		state["fRec83"] = state["fRec83"].at[0].set(((fSlow136 * state["fVec46"][((state["IOTA0"] - iSlow135) & 1023).astype(jnp.int32)]) - ((fSlow134 * state["fRec83"][1]) + (self._fConst2 * state["fRec83"][2])))) 
		state["fRec82"] = (self._fConst7 * (state["fRec83"][0] - state["fRec83"][2])) 
		fTemp26 = (jnp.float32(2.0914886) * fTemp24) 
		state["fVec47"] = state["fVec47"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec84_temp) + fTemp26)) 
		state["fRec85"] = state["fRec85"].at[0].set(((fSlow139 * state["fVec47"][((state["IOTA0"] - iSlow138) & 255).astype(jnp.int32)]) - ((fSlow137 * state["fRec85"][1]) + (self._fConst2 * state["fRec85"][2])))) 
		state["fRec84"] = (self._fConst7 * (state["fRec85"][0] - state["fRec85"][2])) 
		state["fVec48"] = state["fVec48"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp26 + (fSlow3 * fRec86_temp))) 
		state["fRec87"] = state["fRec87"].at[0].set(((fSlow142 * state["fVec48"][((state["IOTA0"] - iSlow141) & 255).astype(jnp.int32)]) - ((fSlow140 * state["fRec87"][1]) + (self._fConst2 * state["fRec87"][2])))) 
		state["fRec86"] = (self._fConst7 * (state["fRec87"][0] - state["fRec87"][2])) 
		fTemp27 = (jnp.float32(5.0063033) * fTemp24) 
		state["fVec49"] = state["fVec49"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec88_temp) + fTemp27)) 
		state["fRec89"] = state["fRec89"].at[0].set(((fSlow145 * state["fVec49"][((state["IOTA0"] - iSlow144) & 127).astype(jnp.int32)]) - ((fSlow143 * state["fRec89"][1]) + (self._fConst2 * state["fRec89"][2])))) 
		state["fRec88"] = (self._fConst7 * (state["fRec89"][0] - state["fRec89"][2])) 
		state["fVec50"] = state["fVec50"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp27 + (fSlow3 * fRec90_temp))) 
		state["fRec91"] = state["fRec91"].at[0].set(((fSlow148 * state["fVec50"][((state["IOTA0"] - iSlow147) & 127).astype(jnp.int32)]) - ((fSlow146 * state["fRec91"][1]) + (self._fConst2 * state["fRec91"][2])))) 
		state["fRec90"] = (self._fConst7 * (state["fRec91"][0] - state["fRec91"][2])) 
		fTemp28 = (jnp.float32(1.7063034) * fTemp24) 
		state["fVec51"] = state["fVec51"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec92_temp) + fTemp28)) 
		state["fRec93"] = state["fRec93"].at[0].set(((fSlow151 * state["fVec51"][((state["IOTA0"] - iSlow150) & 63).astype(jnp.int32)]) - ((fSlow149 * state["fRec93"][1]) + (self._fConst2 * state["fRec93"][2])))) 
		state["fRec92"] = (self._fConst7 * (state["fRec93"][0] - state["fRec93"][2])) 
		state["fVec52"] = state["fVec52"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp28 + (fSlow3 * fRec94_temp))) 
		state["fRec95"] = state["fRec95"].at[0].set(((fSlow154 * state["fVec52"][((state["IOTA0"] - iSlow153) & 63).astype(jnp.int32)]) - ((fSlow152 * state["fRec95"][1]) + (self._fConst2 * state["fRec95"][2])))) 
		state["fRec94"] = (self._fConst7 * (state["fRec95"][0] - state["fRec95"][2])) 
		fTemp29 = (jnp.float32(6.7063036) * fTemp24) 
		state["fVec53"] = state["fVec53"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec96_temp) + fTemp29)) 
		state["fRec97"] = state["fRec97"].at[0].set(((fSlow157 * state["fVec53"][((state["IOTA0"] - iSlow156) & 63).astype(jnp.int32)]) - ((fSlow155 * state["fRec97"][1]) + (self._fConst2 * state["fRec97"][2])))) 
		state["fRec96"] = (self._fConst7 * (state["fRec97"][0] - state["fRec97"][2])) 
		state["fVec54"] = state["fVec54"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp29 + (fSlow3 * fRec98_temp))) 
		state["fRec99"] = state["fRec99"].at[0].set(((fSlow160 * state["fVec54"][((state["IOTA0"] - iSlow159) & 63).astype(jnp.int32)]) - ((fSlow158 * state["fRec99"][1]) + (self._fConst2 * state["fRec99"][2])))) 
		state["fRec98"] = (self._fConst7 * (state["fRec99"][0] - state["fRec99"][2])) 
		state["fVec55"] = fSlow163 
		fTemp30 = ((fSlow163 > fVec55_temp).astype(jnp.int32)) 
		fTemp31 = (jnp.float32(2.1900356) * fTemp30) 
		state["fVec56"] = state["fVec56"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec100_temp) + fTemp31)) 
		state["fRec101"] = state["fRec101"].at[0].set(((fSlow165 * state["fVec56"][((state["IOTA0"] - iSlow164) & 1023).astype(jnp.int32)]) - ((fSlow162 * state["fRec101"][1]) + (self._fConst2 * state["fRec101"][2])))) 
		state["fRec100"] = (self._fConst7 * (state["fRec101"][0] - state["fRec101"][2])) 
		state["fVec57"] = state["fVec57"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp31 + (fSlow3 * fRec102_temp))) 
		state["fRec103"] = state["fRec103"].at[0].set(((fSlow168 * state["fVec57"][((state["IOTA0"] - iSlow167) & 1023).astype(jnp.int32)]) - ((fSlow166 * state["fRec103"][1]) + (self._fConst2 * state["fRec103"][2])))) 
		state["fRec102"] = (self._fConst7 * (state["fRec103"][0] - state["fRec103"][2])) 
		fTemp32 = (jnp.float32(2.0914886) * fTemp30) 
		state["fVec58"] = state["fVec58"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec104_temp) + fTemp32)) 
		state["fRec105"] = state["fRec105"].at[0].set(((fSlow171 * state["fVec58"][((state["IOTA0"] - iSlow170) & 255).astype(jnp.int32)]) - ((fSlow169 * state["fRec105"][1]) + (self._fConst2 * state["fRec105"][2])))) 
		state["fRec104"] = (self._fConst7 * (state["fRec105"][0] - state["fRec105"][2])) 
		state["fVec59"] = state["fVec59"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp32 + (fSlow3 * fRec106_temp))) 
		state["fRec107"] = state["fRec107"].at[0].set(((fSlow174 * state["fVec59"][((state["IOTA0"] - iSlow173) & 255).astype(jnp.int32)]) - ((fSlow172 * state["fRec107"][1]) + (self._fConst2 * state["fRec107"][2])))) 
		state["fRec106"] = (self._fConst7 * (state["fRec107"][0] - state["fRec107"][2])) 
		fTemp33 = (jnp.float32(5.0063033) * fTemp30) 
		state["fVec60"] = state["fVec60"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec108_temp) + fTemp33)) 
		state["fRec109"] = state["fRec109"].at[0].set(((fSlow177 * state["fVec60"][((state["IOTA0"] - iSlow176) & 127).astype(jnp.int32)]) - ((fSlow175 * state["fRec109"][1]) + (self._fConst2 * state["fRec109"][2])))) 
		state["fRec108"] = (self._fConst7 * (state["fRec109"][0] - state["fRec109"][2])) 
		state["fVec61"] = state["fVec61"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp33 + (fSlow3 * fRec110_temp))) 
		state["fRec111"] = state["fRec111"].at[0].set(((fSlow180 * state["fVec61"][((state["IOTA0"] - iSlow179) & 127).astype(jnp.int32)]) - ((fSlow178 * state["fRec111"][1]) + (self._fConst2 * state["fRec111"][2])))) 
		state["fRec110"] = (self._fConst7 * (state["fRec111"][0] - state["fRec111"][2])) 
		fTemp34 = (jnp.float32(1.7063034) * fTemp30) 
		state["fVec62"] = state["fVec62"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec112_temp) + fTemp34)) 
		state["fRec113"] = state["fRec113"].at[0].set(((fSlow183 * state["fVec62"][((state["IOTA0"] - iSlow182) & 63).astype(jnp.int32)]) - ((fSlow181 * state["fRec113"][1]) + (self._fConst2 * state["fRec113"][2])))) 
		state["fRec112"] = (self._fConst7 * (state["fRec113"][0] - state["fRec113"][2])) 
		state["fVec63"] = state["fVec63"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp34 + (fSlow3 * fRec114_temp))) 
		state["fRec115"] = state["fRec115"].at[0].set(((fSlow186 * state["fVec63"][((state["IOTA0"] - iSlow185) & 63).astype(jnp.int32)]) - ((fSlow184 * state["fRec115"][1]) + (self._fConst2 * state["fRec115"][2])))) 
		state["fRec114"] = (self._fConst7 * (state["fRec115"][0] - state["fRec115"][2])) 
		fTemp35 = (jnp.float32(6.7063036) * fTemp30) 
		state["fVec64"] = state["fVec64"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec116_temp) + fTemp35)) 
		state["fRec117"] = state["fRec117"].at[0].set(((fSlow189 * state["fVec64"][((state["IOTA0"] - iSlow188) & 63).astype(jnp.int32)]) - ((fSlow187 * state["fRec117"][1]) + (self._fConst2 * state["fRec117"][2])))) 
		state["fRec116"] = (self._fConst7 * (state["fRec117"][0] - state["fRec117"][2])) 
		state["fVec65"] = state["fVec65"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp35 + (fSlow3 * fRec118_temp))) 
		state["fRec119"] = state["fRec119"].at[0].set(((fSlow192 * state["fVec65"][((state["IOTA0"] - iSlow191) & 63).astype(jnp.int32)]) - ((fSlow190 * state["fRec119"][1]) + (self._fConst2 * state["fRec119"][2])))) 
		state["fRec118"] = (self._fConst7 * (state["fRec119"][0] - state["fRec119"][2])) 
		state["fVec66"] = fSlow195 
		fTemp36 = ((fSlow195 > fVec66_temp).astype(jnp.int32)) 
		fTemp37 = (jnp.float32(2.1900356) * fTemp36) 
		state["fVec67"] = state["fVec67"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec120_temp) + fTemp37)) 
		state["fRec121"] = state["fRec121"].at[0].set(((fSlow197 * state["fVec67"][((state["IOTA0"] - iSlow196) & 1023).astype(jnp.int32)]) - ((fSlow194 * state["fRec121"][1]) + (self._fConst2 * state["fRec121"][2])))) 
		state["fRec120"] = (self._fConst7 * (state["fRec121"][0] - state["fRec121"][2])) 
		state["fVec68"] = state["fVec68"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp37 + (fSlow3 * fRec122_temp))) 
		state["fRec123"] = state["fRec123"].at[0].set(((fSlow200 * state["fVec68"][((state["IOTA0"] - iSlow199) & 1023).astype(jnp.int32)]) - ((fSlow198 * state["fRec123"][1]) + (self._fConst2 * state["fRec123"][2])))) 
		state["fRec122"] = (self._fConst7 * (state["fRec123"][0] - state["fRec123"][2])) 
		fTemp38 = (jnp.float32(2.0914886) * fTemp36) 
		state["fVec69"] = state["fVec69"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec124_temp) + fTemp38)) 
		state["fRec125"] = state["fRec125"].at[0].set(((fSlow203 * state["fVec69"][((state["IOTA0"] - iSlow202) & 255).astype(jnp.int32)]) - ((fSlow201 * state["fRec125"][1]) + (self._fConst2 * state["fRec125"][2])))) 
		state["fRec124"] = (self._fConst7 * (state["fRec125"][0] - state["fRec125"][2])) 
		state["fVec70"] = state["fVec70"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp38 + (fSlow3 * fRec126_temp))) 
		state["fRec127"] = state["fRec127"].at[0].set(((fSlow206 * state["fVec70"][((state["IOTA0"] - iSlow205) & 255).astype(jnp.int32)]) - ((fSlow204 * state["fRec127"][1]) + (self._fConst2 * state["fRec127"][2])))) 
		state["fRec126"] = (self._fConst7 * (state["fRec127"][0] - state["fRec127"][2])) 
		fTemp39 = (jnp.float32(5.0063033) * fTemp36) 
		state["fVec71"] = state["fVec71"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec128_temp) + fTemp39)) 
		state["fRec129"] = state["fRec129"].at[0].set(((fSlow209 * state["fVec71"][((state["IOTA0"] - iSlow208) & 127).astype(jnp.int32)]) - ((fSlow207 * state["fRec129"][1]) + (self._fConst2 * state["fRec129"][2])))) 
		state["fRec128"] = (self._fConst7 * (state["fRec129"][0] - state["fRec129"][2])) 
		state["fVec72"] = state["fVec72"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp39 + (fSlow3 * fRec130_temp))) 
		state["fRec131"] = state["fRec131"].at[0].set(((fSlow212 * state["fVec72"][((state["IOTA0"] - iSlow211) & 127).astype(jnp.int32)]) - ((fSlow210 * state["fRec131"][1]) + (self._fConst2 * state["fRec131"][2])))) 
		state["fRec130"] = (self._fConst7 * (state["fRec131"][0] - state["fRec131"][2])) 
		fTemp40 = (jnp.float32(1.7063034) * fTemp36) 
		state["fVec73"] = state["fVec73"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec132_temp) + fTemp40)) 
		state["fRec133"] = state["fRec133"].at[0].set(((fSlow215 * state["fVec73"][((state["IOTA0"] - iSlow214) & 63).astype(jnp.int32)]) - ((fSlow213 * state["fRec133"][1]) + (self._fConst2 * state["fRec133"][2])))) 
		state["fRec132"] = (self._fConst7 * (state["fRec133"][0] - state["fRec133"][2])) 
		state["fVec74"] = state["fVec74"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp40 + (fSlow3 * fRec134_temp))) 
		state["fRec135"] = state["fRec135"].at[0].set(((fSlow218 * state["fVec74"][((state["IOTA0"] - iSlow217) & 63).astype(jnp.int32)]) - ((fSlow216 * state["fRec135"][1]) + (self._fConst2 * state["fRec135"][2])))) 
		state["fRec134"] = (self._fConst7 * (state["fRec135"][0] - state["fRec135"][2])) 
		fTemp41 = (jnp.float32(6.7063036) * fTemp36) 
		state["fVec75"] = state["fVec75"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec136_temp) + fTemp41)) 
		state["fRec137"] = state["fRec137"].at[0].set(((fSlow221 * state["fVec75"][((state["IOTA0"] - iSlow220) & 63).astype(jnp.int32)]) - ((fSlow219 * state["fRec137"][1]) + (self._fConst2 * state["fRec137"][2])))) 
		state["fRec136"] = (self._fConst7 * (state["fRec137"][0] - state["fRec137"][2])) 
		state["fVec76"] = state["fVec76"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp41 + (fSlow3 * fRec138_temp))) 
		state["fRec139"] = state["fRec139"].at[0].set(((fSlow224 * state["fVec76"][((state["IOTA0"] - iSlow223) & 63).astype(jnp.int32)]) - ((fSlow222 * state["fRec139"][1]) + (self._fConst2 * state["fRec139"][2])))) 
		state["fRec138"] = (self._fConst7 * (state["fRec139"][0] - state["fRec139"][2])) 
		state["fVec77"] = fSlow227 
		fTemp42 = ((fSlow227 > fVec77_temp).astype(jnp.int32)) 
		fTemp43 = (jnp.float32(2.1900356) * fTemp42) 
		state["fVec78"] = state["fVec78"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec140_temp) + fTemp43)) 
		state["fRec141"] = state["fRec141"].at[0].set(((fSlow229 * state["fVec78"][((state["IOTA0"] - iSlow228) & 1023).astype(jnp.int32)]) - ((fSlow226 * state["fRec141"][1]) + (self._fConst2 * state["fRec141"][2])))) 
		state["fRec140"] = (self._fConst7 * (state["fRec141"][0] - state["fRec141"][2])) 
		state["fVec79"] = state["fVec79"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp43 + (fSlow3 * fRec142_temp))) 
		state["fRec143"] = state["fRec143"].at[0].set(((fSlow232 * state["fVec79"][((state["IOTA0"] - iSlow231) & 1023).astype(jnp.int32)]) - ((fSlow230 * state["fRec143"][1]) + (self._fConst2 * state["fRec143"][2])))) 
		state["fRec142"] = (self._fConst7 * (state["fRec143"][0] - state["fRec143"][2])) 
		fTemp44 = (jnp.float32(2.0914886) * fTemp42) 
		state["fVec80"] = state["fVec80"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec144_temp) + fTemp44)) 
		state["fRec145"] = state["fRec145"].at[0].set(((fSlow235 * state["fVec80"][((state["IOTA0"] - iSlow234) & 255).astype(jnp.int32)]) - ((fSlow233 * state["fRec145"][1]) + (self._fConst2 * state["fRec145"][2])))) 
		state["fRec144"] = (self._fConst7 * (state["fRec145"][0] - state["fRec145"][2])) 
		state["fVec81"] = state["fVec81"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp44 + (fSlow3 * fRec146_temp))) 
		state["fRec147"] = state["fRec147"].at[0].set(((fSlow238 * state["fVec81"][((state["IOTA0"] - iSlow237) & 255).astype(jnp.int32)]) - ((fSlow236 * state["fRec147"][1]) + (self._fConst2 * state["fRec147"][2])))) 
		state["fRec146"] = (self._fConst7 * (state["fRec147"][0] - state["fRec147"][2])) 
		fTemp45 = (jnp.float32(5.0063033) * fTemp42) 
		state["fVec82"] = state["fVec82"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec148_temp) + fTemp45)) 
		state["fRec149"] = state["fRec149"].at[0].set(((fSlow241 * state["fVec82"][((state["IOTA0"] - iSlow240) & 127).astype(jnp.int32)]) - ((fSlow239 * state["fRec149"][1]) + (self._fConst2 * state["fRec149"][2])))) 
		state["fRec148"] = (self._fConst7 * (state["fRec149"][0] - state["fRec149"][2])) 
		state["fVec83"] = state["fVec83"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp45 + (fSlow3 * fRec150_temp))) 
		state["fRec151"] = state["fRec151"].at[0].set(((fSlow244 * state["fVec83"][((state["IOTA0"] - iSlow243) & 127).astype(jnp.int32)]) - ((fSlow242 * state["fRec151"][1]) + (self._fConst2 * state["fRec151"][2])))) 
		state["fRec150"] = (self._fConst7 * (state["fRec151"][0] - state["fRec151"][2])) 
		fTemp46 = (jnp.float32(1.7063034) * fTemp42) 
		state["fVec84"] = state["fVec84"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec152_temp) + fTemp46)) 
		state["fRec153"] = state["fRec153"].at[0].set(((fSlow247 * state["fVec84"][((state["IOTA0"] - iSlow246) & 63).astype(jnp.int32)]) - ((fSlow245 * state["fRec153"][1]) + (self._fConst2 * state["fRec153"][2])))) 
		state["fRec152"] = (self._fConst7 * (state["fRec153"][0] - state["fRec153"][2])) 
		state["fVec85"] = state["fVec85"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp46 + (fSlow3 * fRec154_temp))) 
		state["fRec155"] = state["fRec155"].at[0].set(((fSlow250 * state["fVec85"][((state["IOTA0"] - iSlow249) & 63).astype(jnp.int32)]) - ((fSlow248 * state["fRec155"][1]) + (self._fConst2 * state["fRec155"][2])))) 
		state["fRec154"] = (self._fConst7 * (state["fRec155"][0] - state["fRec155"][2])) 
		fTemp47 = (jnp.float32(6.7063036) * fTemp42) 
		state["fVec86"] = state["fVec86"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec156_temp) + fTemp47)) 
		state["fRec157"] = state["fRec157"].at[0].set(((fSlow253 * state["fVec86"][((state["IOTA0"] - iSlow252) & 63).astype(jnp.int32)]) - ((fSlow251 * state["fRec157"][1]) + (self._fConst2 * state["fRec157"][2])))) 
		state["fRec156"] = (self._fConst7 * (state["fRec157"][0] - state["fRec157"][2])) 
		state["fVec87"] = state["fVec87"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp47 + (fSlow3 * fRec158_temp))) 
		state["fRec159"] = state["fRec159"].at[0].set(((fSlow256 * state["fVec87"][((state["IOTA0"] - iSlow255) & 63).astype(jnp.int32)]) - ((fSlow254 * state["fRec159"][1]) + (self._fConst2 * state["fRec159"][2])))) 
		state["fRec158"] = (self._fConst7 * (state["fRec159"][0] - state["fRec159"][2])) 
		state["fVec88"] = fSlow259 
		fTemp48 = ((fSlow259 > fVec88_temp).astype(jnp.int32)) 
		fTemp49 = (jnp.float32(2.1900356) * fTemp48) 
		state["fVec89"] = state["fVec89"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec160_temp) + fTemp49)) 
		state["fRec161"] = state["fRec161"].at[0].set(((fSlow261 * state["fVec89"][((state["IOTA0"] - iSlow260) & 1023).astype(jnp.int32)]) - ((fSlow258 * state["fRec161"][1]) + (self._fConst2 * state["fRec161"][2])))) 
		state["fRec160"] = (self._fConst7 * (state["fRec161"][0] - state["fRec161"][2])) 
		state["fVec90"] = state["fVec90"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp49 + (fSlow3 * fRec162_temp))) 
		state["fRec163"] = state["fRec163"].at[0].set(((fSlow264 * state["fVec90"][((state["IOTA0"] - iSlow263) & 1023).astype(jnp.int32)]) - ((fSlow262 * state["fRec163"][1]) + (self._fConst2 * state["fRec163"][2])))) 
		state["fRec162"] = (self._fConst7 * (state["fRec163"][0] - state["fRec163"][2])) 
		fTemp50 = (jnp.float32(2.0914886) * fTemp48) 
		state["fVec91"] = state["fVec91"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec164_temp) + fTemp50)) 
		state["fRec165"] = state["fRec165"].at[0].set(((fSlow267 * state["fVec91"][((state["IOTA0"] - iSlow266) & 255).astype(jnp.int32)]) - ((fSlow265 * state["fRec165"][1]) + (self._fConst2 * state["fRec165"][2])))) 
		state["fRec164"] = (self._fConst7 * (state["fRec165"][0] - state["fRec165"][2])) 
		state["fVec92"] = state["fVec92"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp50 + (fSlow3 * fRec166_temp))) 
		state["fRec167"] = state["fRec167"].at[0].set(((fSlow270 * state["fVec92"][((state["IOTA0"] - iSlow269) & 255).astype(jnp.int32)]) - ((fSlow268 * state["fRec167"][1]) + (self._fConst2 * state["fRec167"][2])))) 
		state["fRec166"] = (self._fConst7 * (state["fRec167"][0] - state["fRec167"][2])) 
		fTemp51 = (jnp.float32(5.0063033) * fTemp48) 
		state["fVec93"] = state["fVec93"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec168_temp) + fTemp51)) 
		state["fRec169"] = state["fRec169"].at[0].set(((fSlow273 * state["fVec93"][((state["IOTA0"] - iSlow272) & 127).astype(jnp.int32)]) - ((fSlow271 * state["fRec169"][1]) + (self._fConst2 * state["fRec169"][2])))) 
		state["fRec168"] = (self._fConst7 * (state["fRec169"][0] - state["fRec169"][2])) 
		state["fVec94"] = state["fVec94"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp51 + (fSlow3 * fRec170_temp))) 
		state["fRec171"] = state["fRec171"].at[0].set(((fSlow276 * state["fVec94"][((state["IOTA0"] - iSlow275) & 127).astype(jnp.int32)]) - ((fSlow274 * state["fRec171"][1]) + (self._fConst2 * state["fRec171"][2])))) 
		state["fRec170"] = (self._fConst7 * (state["fRec171"][0] - state["fRec171"][2])) 
		fTemp52 = (jnp.float32(1.7063034) * fTemp48) 
		state["fVec95"] = state["fVec95"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec172_temp) + fTemp52)) 
		state["fRec173"] = state["fRec173"].at[0].set(((fSlow279 * state["fVec95"][((state["IOTA0"] - iSlow278) & 63).astype(jnp.int32)]) - ((fSlow277 * state["fRec173"][1]) + (self._fConst2 * state["fRec173"][2])))) 
		state["fRec172"] = (self._fConst7 * (state["fRec173"][0] - state["fRec173"][2])) 
		state["fVec96"] = state["fVec96"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp52 + (fSlow3 * fRec174_temp))) 
		state["fRec175"] = state["fRec175"].at[0].set(((fSlow282 * state["fVec96"][((state["IOTA0"] - iSlow281) & 63).astype(jnp.int32)]) - ((fSlow280 * state["fRec175"][1]) + (self._fConst2 * state["fRec175"][2])))) 
		state["fRec174"] = (self._fConst7 * (state["fRec175"][0] - state["fRec175"][2])) 
		fTemp53 = (jnp.float32(6.7063036) * fTemp48) 
		state["fVec97"] = state["fVec97"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec176_temp) + fTemp53)) 
		state["fRec177"] = state["fRec177"].at[0].set(((fSlow285 * state["fVec97"][((state["IOTA0"] - iSlow284) & 63).astype(jnp.int32)]) - ((fSlow283 * state["fRec177"][1]) + (self._fConst2 * state["fRec177"][2])))) 
		state["fRec176"] = (self._fConst7 * (state["fRec177"][0] - state["fRec177"][2])) 
		state["fVec98"] = state["fVec98"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp53 + (fSlow3 * fRec178_temp))) 
		state["fRec179"] = state["fRec179"].at[0].set(((fSlow288 * state["fVec98"][((state["IOTA0"] - iSlow287) & 63).astype(jnp.int32)]) - ((fSlow286 * state["fRec179"][1]) + (self._fConst2 * state["fRec179"][2])))) 
		state["fRec178"] = (self._fConst7 * (state["fRec179"][0] - state["fRec179"][2])) 
		state["fVec99"] = fSlow291 
		fTemp54 = ((fSlow291 > fVec99_temp).astype(jnp.int32)) 
		fTemp55 = (jnp.float32(2.1900356) * fTemp54) 
		state["fVec100"] = state["fVec100"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set(((fSlow3 * fRec180_temp) + fTemp55)) 
		state["fRec181"] = state["fRec181"].at[0].set(((fSlow293 * state["fVec100"][((state["IOTA0"] - iSlow292) & 1023).astype(jnp.int32)]) - ((fSlow290 * state["fRec181"][1]) + (self._fConst2 * state["fRec181"][2])))) 
		state["fRec180"] = (self._fConst7 * (state["fRec181"][0] - state["fRec181"][2])) 
		state["fVec101"] = state["fVec101"].at[(state["IOTA0"] & 1023).astype(jnp.int32)].set((fTemp55 + (fSlow3 * fRec182_temp))) 
		state["fRec183"] = state["fRec183"].at[0].set(((fSlow296 * state["fVec101"][((state["IOTA0"] - iSlow295) & 1023).astype(jnp.int32)]) - ((fSlow294 * state["fRec183"][1]) + (self._fConst2 * state["fRec183"][2])))) 
		state["fRec182"] = (self._fConst7 * (state["fRec183"][0] - state["fRec183"][2])) 
		fTemp56 = (jnp.float32(2.0914886) * fTemp54) 
		state["fVec102"] = state["fVec102"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec184_temp) + fTemp56)) 
		state["fRec185"] = state["fRec185"].at[0].set(((fSlow299 * state["fVec102"][((state["IOTA0"] - iSlow298) & 255).astype(jnp.int32)]) - ((fSlow297 * state["fRec185"][1]) + (self._fConst2 * state["fRec185"][2])))) 
		state["fRec184"] = (self._fConst7 * (state["fRec185"][0] - state["fRec185"][2])) 
		state["fVec103"] = state["fVec103"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set((fTemp56 + (fSlow3 * fRec186_temp))) 
		state["fRec187"] = state["fRec187"].at[0].set(((fSlow302 * state["fVec103"][((state["IOTA0"] - iSlow301) & 255).astype(jnp.int32)]) - ((fSlow300 * state["fRec187"][1]) + (self._fConst2 * state["fRec187"][2])))) 
		state["fRec186"] = (self._fConst7 * (state["fRec187"][0] - state["fRec187"][2])) 
		fTemp57 = (jnp.float32(5.0063033) * fTemp54) 
		state["fVec104"] = state["fVec104"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow3 * fRec188_temp) + fTemp57)) 
		state["fRec189"] = state["fRec189"].at[0].set(((fSlow305 * state["fVec104"][((state["IOTA0"] - iSlow304) & 127).astype(jnp.int32)]) - ((fSlow303 * state["fRec189"][1]) + (self._fConst2 * state["fRec189"][2])))) 
		state["fRec188"] = (self._fConst7 * (state["fRec189"][0] - state["fRec189"][2])) 
		state["fVec105"] = state["fVec105"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set((fTemp57 + (fSlow3 * fRec190_temp))) 
		state["fRec191"] = state["fRec191"].at[0].set(((fSlow308 * state["fVec105"][((state["IOTA0"] - iSlow307) & 127).astype(jnp.int32)]) - ((fSlow306 * state["fRec191"][1]) + (self._fConst2 * state["fRec191"][2])))) 
		state["fRec190"] = (self._fConst7 * (state["fRec191"][0] - state["fRec191"][2])) 
		fTemp58 = (jnp.float32(1.7063034) * fTemp54) 
		state["fVec106"] = state["fVec106"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec192_temp) + fTemp58)) 
		state["fRec193"] = state["fRec193"].at[0].set(((fSlow311 * state["fVec106"][((state["IOTA0"] - iSlow310) & 63).astype(jnp.int32)]) - ((fSlow309 * state["fRec193"][1]) + (self._fConst2 * state["fRec193"][2])))) 
		state["fRec192"] = (self._fConst7 * (state["fRec193"][0] - state["fRec193"][2])) 
		state["fVec107"] = state["fVec107"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp58 + (fSlow3 * fRec194_temp))) 
		state["fRec195"] = state["fRec195"].at[0].set(((fSlow314 * state["fVec107"][((state["IOTA0"] - iSlow313) & 63).astype(jnp.int32)]) - ((fSlow312 * state["fRec195"][1]) + (self._fConst2 * state["fRec195"][2])))) 
		state["fRec194"] = (self._fConst7 * (state["fRec195"][0] - state["fRec195"][2])) 
		fTemp59 = (jnp.float32(6.7063036) * fTemp54) 
		state["fVec108"] = state["fVec108"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow3 * fRec196_temp) + fTemp59)) 
		state["fRec197"] = state["fRec197"].at[0].set(((fSlow317 * state["fVec108"][((state["IOTA0"] - iSlow316) & 63).astype(jnp.int32)]) - ((fSlow315 * state["fRec197"][1]) + (self._fConst2 * state["fRec197"][2])))) 
		state["fRec196"] = (self._fConst7 * (state["fRec197"][0] - state["fRec197"][2])) 
		state["fVec109"] = state["fVec109"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set((fTemp59 + (fSlow3 * fRec198_temp))) 
		state["fRec199"] = state["fRec199"].at[0].set(((fSlow320 * state["fVec109"][((state["IOTA0"] - iSlow319) & 63).astype(jnp.int32)]) - ((fSlow318 * state["fRec199"][1]) + (self._fConst2 * state["fRec199"][2])))) 
		state["fRec198"] = (self._fConst7 * (state["fRec199"][0] - state["fRec199"][2])) 
		state["fVec110"] = state["fVec110"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec200_temp) + (jnp.float32(5.2995043) * fTemp6))) 
		state["fRec201"] = state["fRec201"].at[0].set(((fSlow323 * state["fVec110"][((state["IOTA0"] - iSlow322) & 255).astype(jnp.int32)]) - ((fSlow321 * state["fRec201"][1]) + (self._fConst2 * state["fRec201"][2])))) 
		state["fRec200"] = (self._fConst7 * (state["fRec201"][0] - state["fRec201"][2])) 
		state["fVec111"] = state["fVec111"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec202_temp) + (jnp.float32(5.2995043) * fTemp0))) 
		state["fRec203"] = state["fRec203"].at[0].set(((fSlow326 * state["fVec111"][((state["IOTA0"] - iSlow325) & 255).astype(jnp.int32)]) - ((fSlow324 * state["fRec203"][1]) + (self._fConst2 * state["fRec203"][2])))) 
		state["fRec202"] = (self._fConst7 * (state["fRec203"][0] - state["fRec203"][2])) 
		state["fVec112"] = state["fVec112"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec204_temp) + (jnp.float32(5.2995043) * fTemp12))) 
		state["fRec205"] = state["fRec205"].at[0].set(((fSlow329 * state["fVec112"][((state["IOTA0"] - iSlow328) & 255).astype(jnp.int32)]) - ((fSlow327 * state["fRec205"][1]) + (self._fConst2 * state["fRec205"][2])))) 
		state["fRec204"] = (self._fConst7 * (state["fRec205"][0] - state["fRec205"][2])) 
		state["fVec113"] = state["fVec113"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec206_temp) + (jnp.float32(5.2995043) * fTemp18))) 
		state["fRec207"] = state["fRec207"].at[0].set(((fSlow332 * state["fVec113"][((state["IOTA0"] - iSlow331) & 255).astype(jnp.int32)]) - ((fSlow330 * state["fRec207"][1]) + (self._fConst2 * state["fRec207"][2])))) 
		state["fRec206"] = (self._fConst7 * (state["fRec207"][0] - state["fRec207"][2])) 
		state["fVec114"] = state["fVec114"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec208_temp) + (jnp.float32(5.2995043) * fTemp24))) 
		state["fRec209"] = state["fRec209"].at[0].set(((fSlow335 * state["fVec114"][((state["IOTA0"] - iSlow334) & 255).astype(jnp.int32)]) - ((fSlow333 * state["fRec209"][1]) + (self._fConst2 * state["fRec209"][2])))) 
		state["fRec208"] = (self._fConst7 * (state["fRec209"][0] - state["fRec209"][2])) 
		state["fVec115"] = state["fVec115"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec210_temp) + (jnp.float32(5.2995043) * fTemp30))) 
		state["fRec211"] = state["fRec211"].at[0].set(((fSlow338 * state["fVec115"][((state["IOTA0"] - iSlow337) & 255).astype(jnp.int32)]) - ((fSlow336 * state["fRec211"][1]) + (self._fConst2 * state["fRec211"][2])))) 
		state["fRec210"] = (self._fConst7 * (state["fRec211"][0] - state["fRec211"][2])) 
		state["fVec116"] = state["fVec116"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec212_temp) + (jnp.float32(5.2995043) * fTemp36))) 
		state["fRec213"] = state["fRec213"].at[0].set(((fSlow341 * state["fVec116"][((state["IOTA0"] - iSlow340) & 255).astype(jnp.int32)]) - ((fSlow339 * state["fRec213"][1]) + (self._fConst2 * state["fRec213"][2])))) 
		state["fRec212"] = (self._fConst7 * (state["fRec213"][0] - state["fRec213"][2])) 
		state["fVec117"] = state["fVec117"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec214_temp) + (jnp.float32(5.2995043) * fTemp42))) 
		state["fRec215"] = state["fRec215"].at[0].set(((fSlow344 * state["fVec117"][((state["IOTA0"] - iSlow343) & 255).astype(jnp.int32)]) - ((fSlow342 * state["fRec215"][1]) + (self._fConst2 * state["fRec215"][2])))) 
		state["fRec214"] = (self._fConst7 * (state["fRec215"][0] - state["fRec215"][2])) 
		state["fVec118"] = state["fVec118"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec216_temp) + (jnp.float32(5.2995043) * fTemp48))) 
		state["fRec217"] = state["fRec217"].at[0].set(((fSlow347 * state["fVec118"][((state["IOTA0"] - iSlow346) & 255).astype(jnp.int32)]) - ((fSlow345 * state["fRec217"][1]) + (self._fConst2 * state["fRec217"][2])))) 
		state["fRec216"] = (self._fConst7 * (state["fRec217"][0] - state["fRec217"][2])) 
		state["fVec119"] = state["fVec119"].at[(state["IOTA0"] & 255).astype(jnp.int32)].set(((fSlow3 * fRec218_temp) + (jnp.float32(5.2995043) * fTemp54))) 
		state["fRec219"] = state["fRec219"].at[0].set(((fSlow350 * state["fVec119"][((state["IOTA0"] - iSlow349) & 255).astype(jnp.int32)]) - ((fSlow348 * state["fRec219"][1]) + (self._fConst2 * state["fRec219"][2])))) 
		state["fRec218"] = (self._fConst7 * (state["fRec219"][0] - state["fRec219"][2])) 
		_result0 = ((jnp.float32(2.0) * (state["fRec218"] + (state["fRec216"] + (state["fRec214"] + (state["fRec212"] + (state["fRec210"] + (state["fRec208"] + (state["fRec206"] + (state["fRec204"] + (state["fRec202"] + state["fRec200"])))))))))) + (state["fRec198"] + (state["fRec196"] + (state["fRec194"] + (state["fRec192"] + (state["fRec190"] + (state["fRec188"] + (state["fRec186"] + (state["fRec184"] + (state["fRec182"] + (state["fRec180"] + (state["fRec178"] + (state["fRec176"] + (state["fRec174"] + (state["fRec172"] + (state["fRec170"] + (state["fRec168"] + (state["fRec166"] + (state["fRec164"] + (state["fRec162"] + (state["fRec160"] + (state["fRec158"] + (state["fRec156"] + (state["fRec154"] + (state["fRec152"] + (state["fRec150"] + (state["fRec148"] + (state["fRec146"] + (state["fRec144"] + (state["fRec142"] + (state["fRec140"] + (state["fRec138"] + (state["fRec136"] + (state["fRec134"] + (state["fRec132"] + (state["fRec130"] + (state["fRec128"] + (state["fRec126"] + (state["fRec124"] + (state["fRec122"] + (state["fRec120"] + (state["fRec118"] + (state["fRec116"] + (state["fRec114"] + (state["fRec112"] + (state["fRec110"] + (state["fRec108"] + (state["fRec106"] + (state["fRec104"] + (state["fRec102"] + (state["fRec100"] + (state["fRec98"] + (state["fRec96"] + (state["fRec94"] + (state["fRec92"] + (state["fRec90"] + (state["fRec88"] + (state["fRec86"] + (state["fRec84"] + (state["fRec82"] + (state["fRec80"] + (state["fRec78"] + (state["fRec76"] + (state["fRec74"] + (state["fRec72"] + (state["fRec70"] + (state["fRec68"] + (state["fRec66"] + (state["fRec64"] + (state["fRec62"] + (state["fRec60"] + (state["fRec58"] + (state["fRec56"] + (state["fRec54"] + (state["fRec52"] + (state["fRec50"] + (state["fRec48"] + (state["fRec46"] + (state["fRec44"] + (state["fRec42"] + (state["fRec40"] + (state["fRec38"] + (state["fRec36"] + (state["fRec34"] + (state["fRec32"] + (state["fRec30"] + (state["fRec28"] + (state["fRec26"] + (state["fRec24"] + (state["fRec22"] + (state["fRec20"] + (state["fRec18"] + (state["fRec16"] + (state["fRec14"] + (state["fRec12"] + (state["fRec10"] + ((((state["fRec8"] + state["fRec6"]) + state["fRec4"]) + state["fRec2"]) + state["fRec0"]))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec11"] = jnp.roll(state["fRec11"], 1) 
		state["fRec13"] = jnp.roll(state["fRec13"], 1) 
		state["fRec15"] = jnp.roll(state["fRec15"], 1) 
		state["fRec17"] = jnp.roll(state["fRec17"], 1) 
		state["fRec19"] = jnp.roll(state["fRec19"], 1) 
		state["fRec21"] = jnp.roll(state["fRec21"], 1) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec25"] = jnp.roll(state["fRec25"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec31"] = jnp.roll(state["fRec31"], 1) 
		state["fRec33"] = jnp.roll(state["fRec33"], 1) 
		state["fRec35"] = jnp.roll(state["fRec35"], 1) 
		state["fRec37"] = jnp.roll(state["fRec37"], 1) 
		state["fRec39"] = jnp.roll(state["fRec39"], 1) 
		state["fRec41"] = jnp.roll(state["fRec41"], 1) 
		state["fRec43"] = jnp.roll(state["fRec43"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec47"] = jnp.roll(state["fRec47"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec53"] = jnp.roll(state["fRec53"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
		state["fRec57"] = jnp.roll(state["fRec57"], 1) 
		state["fRec59"] = jnp.roll(state["fRec59"], 1) 
		state["fRec61"] = jnp.roll(state["fRec61"], 1) 
		state["fRec63"] = jnp.roll(state["fRec63"], 1) 
		state["fRec65"] = jnp.roll(state["fRec65"], 1) 
		state["fRec67"] = jnp.roll(state["fRec67"], 1) 
		state["fRec69"] = jnp.roll(state["fRec69"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec75"] = jnp.roll(state["fRec75"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec81"] = jnp.roll(state["fRec81"], 1) 
		state["fRec83"] = jnp.roll(state["fRec83"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec91"] = jnp.roll(state["fRec91"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec103"] = jnp.roll(state["fRec103"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec119"] = jnp.roll(state["fRec119"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec123"] = jnp.roll(state["fRec123"], 1) 
		state["fRec125"] = jnp.roll(state["fRec125"], 1) 
		state["fRec127"] = jnp.roll(state["fRec127"], 1) 
		state["fRec129"] = jnp.roll(state["fRec129"], 1) 
		state["fRec131"] = jnp.roll(state["fRec131"], 1) 
		state["fRec133"] = jnp.roll(state["fRec133"], 1) 
		state["fRec135"] = jnp.roll(state["fRec135"], 1) 
		state["fRec137"] = jnp.roll(state["fRec137"], 1) 
		state["fRec139"] = jnp.roll(state["fRec139"], 1) 
		state["fRec141"] = jnp.roll(state["fRec141"], 1) 
		state["fRec143"] = jnp.roll(state["fRec143"], 1) 
		state["fRec145"] = jnp.roll(state["fRec145"], 1) 
		state["fRec147"] = jnp.roll(state["fRec147"], 1) 
		state["fRec149"] = jnp.roll(state["fRec149"], 1) 
		state["fRec151"] = jnp.roll(state["fRec151"], 1) 
		state["fRec153"] = jnp.roll(state["fRec153"], 1) 
		state["fRec155"] = jnp.roll(state["fRec155"], 1) 
		state["fRec157"] = jnp.roll(state["fRec157"], 1) 
		state["fRec159"] = jnp.roll(state["fRec159"], 1) 
		state["fRec161"] = jnp.roll(state["fRec161"], 1) 
		state["fRec163"] = jnp.roll(state["fRec163"], 1) 
		state["fRec165"] = jnp.roll(state["fRec165"], 1) 
		state["fRec167"] = jnp.roll(state["fRec167"], 1) 
		state["fRec169"] = jnp.roll(state["fRec169"], 1) 
		state["fRec171"] = jnp.roll(state["fRec171"], 1) 
		state["fRec173"] = jnp.roll(state["fRec173"], 1) 
		state["fRec175"] = jnp.roll(state["fRec175"], 1) 
		state["fRec177"] = jnp.roll(state["fRec177"], 1) 
		state["fRec179"] = jnp.roll(state["fRec179"], 1) 
		state["fRec181"] = jnp.roll(state["fRec181"], 1) 
		state["fRec183"] = jnp.roll(state["fRec183"], 1) 
		state["fRec185"] = jnp.roll(state["fRec185"], 1) 
		state["fRec187"] = jnp.roll(state["fRec187"], 1) 
		state["fRec189"] = jnp.roll(state["fRec189"], 1) 
		state["fRec191"] = jnp.roll(state["fRec191"], 1) 
		state["fRec193"] = jnp.roll(state["fRec193"], 1) 
		state["fRec195"] = jnp.roll(state["fRec195"], 1) 
		state["fRec197"] = jnp.roll(state["fRec197"], 1) 
		state["fRec199"] = jnp.roll(state["fRec199"], 1) 
		state["fRec201"] = jnp.roll(state["fRec201"], 1) 
		state["fRec203"] = jnp.roll(state["fRec203"], 1) 
		state["fRec205"] = jnp.roll(state["fRec205"], 1) 
		state["fRec207"] = jnp.roll(state["fRec207"], 1) 
		state["fRec209"] = jnp.roll(state["fRec209"], 1) 
		state["fRec211"] = jnp.roll(state["fRec211"], 1) 
		state["fRec213"] = jnp.roll(state["fRec213"], 1) 
		state["fRec215"] = jnp.roll(state["fRec215"], 1) 
		state["fRec217"] = jnp.roll(state["fRec217"], 1) 
		state["fRec219"] = jnp.roll(state["fRec219"], 1) 
		return state, jnp.stack([_result0]) 
		
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
