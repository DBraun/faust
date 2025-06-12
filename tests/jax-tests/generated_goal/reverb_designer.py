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
		ui_path.append("reverb_designer") 
		ui_path.append("FEEDBACK DELAY NETWORK (FDN) REVERBERATOR, ORDER 16") 
		ui_path.append("Band Crossover Frequencies") 
		self.add_hslider("fHslider0", ui_path, "Band 0 upper edge in Hz", 5e+02, 1e+02, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider1", ui_path, "Band 1 upper edge in Hz", 1e+03, 1e+02, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider2", ui_path, "Band 2 upper edge in Hz", 2e+03, 1e+02, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider3", ui_path, "Band 3 upper edge in Hz", 4e+03, 1e+02, 1e+04, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Band Decay Times (T60)") 
		self.add_vslider("fVslider0", ui_path, "0", 8.4, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider1", ui_path, "1", 6.5, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider2", ui_path, "2", 5.0, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider3", ui_path, "3", 3.8, 0.1, 1e+02, unnorm_funcs, "log") 
		self.add_vslider("fVslider4", ui_path, "4", 2.7, 0.1, 1e+02, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Room Dimensions") 
		self.add_hslider("fHslider5", ui_path, "min acoustic ray length", 46.0, 0.1, 63.0, unnorm_funcs, "log") 
		self.add_hslider("fHslider4", ui_path, "max acoustic ray length", 63.0, 0.1, 63.0, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Input Controls") 
		ui_path.append("Input Config") 
		self.add_button("fCheckbox0", ui_path, "Mute Ext Inputs", unnorm_funcs) 
		self.add_button("fCheckbox1", ui_path, "Pink Noise", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Impulse Selection") 
		self.add_button("fButton2", ui_path, "Left", unnorm_funcs) 
		self.add_button("fButton1", ui_path, "Center", unnorm_funcs) 
		self.add_button("fButton3", ui_path, "Right", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Reverb State") 
		self.add_button("fButton0", ui_path, "Quench", unnorm_funcs) 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		self.add_hslider("fHslider6", ui_path, "Output Level (dB)", -4e+01, -7e+01, 2e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = (np.float32(3.1415927) / self._fConst0) 
		self._fConst2 = (np.float32(0.002915452) * self._fConst0) 
		self._fConst3 = (np.float32(6.9077554) / self._fConst0) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec103"] = np.float32(0)
		state["fRec105"] = np.float32(0)
		state["fRec107"] = np.float32(0)
		state["fRec109"] = np.float32(0)
		state["fRec111"] = np.float32(0)
		state["fRec113"] = np.float32(0)
		state["fRec116"] = np.float32(0)
		state["fRec120"] = np.float32(0)
		state["fRec125"] = np.float32(0)
		state["fRec127"] = np.float32(0)
		state["fRec129"] = np.float32(0)
		state["fRec131"] = np.float32(0)
		state["fRec133"] = np.float32(0)
		state["fRec135"] = np.float32(0)
		state["fRec138"] = np.float32(0)
		state["fRec142"] = np.float32(0)
		state["fRec147"] = np.float32(0)
		state["fRec149"] = np.float32(0)
		state["fRec151"] = np.float32(0)
		state["fRec153"] = np.float32(0)
		state["fRec155"] = np.float32(0)
		state["fRec157"] = np.float32(0)
		state["fRec160"] = np.float32(0)
		state["fRec164"] = np.float32(0)
		state["fRec169"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec171"] = np.float32(0)
		state["fRec173"] = np.float32(0)
		state["fRec175"] = np.float32(0)
		state["fRec177"] = np.float32(0)
		state["fRec179"] = np.float32(0)
		state["fRec182"] = np.float32(0)
		state["fRec186"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec191"] = np.float32(0)
		state["fRec193"] = np.float32(0)
		state["fRec195"] = np.float32(0)
		state["fRec197"] = np.float32(0)
		state["fRec199"] = np.float32(0)
		state["fRec201"] = np.float32(0)
		state["fRec204"] = np.float32(0)
		state["fRec208"] = np.float32(0)
		state["fRec21"] = np.float32(0)
		state["fRec213"] = np.float32(0)
		state["fRec215"] = np.float32(0)
		state["fRec217"] = np.float32(0)
		state["fRec219"] = np.float32(0)
		state["fRec221"] = np.float32(0)
		state["fRec223"] = np.float32(0)
		state["fRec226"] = np.float32(0)
		state["fRec23"] = np.float32(0)
		state["fRec230"] = np.float32(0)
		state["fRec235"] = np.float32(0)
		state["fRec237"] = np.float32(0)
		state["fRec239"] = np.float32(0)
		state["fRec241"] = np.float32(0)
		state["fRec243"] = np.float32(0)
		state["fRec245"] = np.float32(0)
		state["fRec248"] = np.float32(0)
		state["fRec25"] = np.float32(0)
		state["fRec252"] = np.float32(0)
		state["fRec257"] = np.float32(0)
		state["fRec259"] = np.float32(0)
		state["fRec261"] = np.float32(0)
		state["fRec263"] = np.float32(0)
		state["fRec265"] = np.float32(0)
		state["fRec267"] = np.float32(0)
		state["fRec270"] = np.float32(0)
		state["fRec274"] = np.float32(0)
		state["fRec279"] = np.float32(0)
		state["fRec28"] = np.float32(0)
		state["fRec281"] = np.float32(0)
		state["fRec283"] = np.float32(0)
		state["fRec285"] = np.float32(0)
		state["fRec287"] = np.float32(0)
		state["fRec289"] = np.float32(0)
		state["fRec292"] = np.float32(0)
		state["fRec296"] = np.float32(0)
		state["fRec301"] = np.float32(0)
		state["fRec303"] = np.float32(0)
		state["fRec305"] = np.float32(0)
		state["fRec307"] = np.float32(0)
		state["fRec309"] = np.float32(0)
		state["fRec311"] = np.float32(0)
		state["fRec314"] = np.float32(0)
		state["fRec318"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec323"] = np.float32(0)
		state["fRec325"] = np.float32(0)
		state["fRec327"] = np.float32(0)
		state["fRec329"] = np.float32(0)
		state["fRec331"] = np.float32(0)
		state["fRec333"] = np.float32(0)
		state["fRec336"] = np.float32(0)
		state["fRec340"] = np.float32(0)
		state["fRec345"] = np.float32(0)
		state["fRec347"] = np.float32(0)
		state["fRec349"] = np.float32(0)
		state["fRec351"] = np.float32(0)
		state["fRec353"] = np.float32(0)
		state["fRec355"] = np.float32(0)
		state["fRec358"] = np.float32(0)
		state["fRec362"] = np.float32(0)
		state["fRec367"] = np.float32(0)
		state["fRec37"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec41"] = np.float32(0)
		state["fRec43"] = np.float32(0)
		state["fRec45"] = np.float32(0)
		state["fRec47"] = np.float32(0)
		state["fRec50"] = np.float32(0)
		state["fRec54"] = np.float32(0)
		state["fRec59"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec63"] = np.float32(0)
		state["fRec65"] = np.float32(0)
		state["fRec67"] = np.float32(0)
		state["fRec69"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec76"] = np.float32(0)
		state["fRec81"] = np.float32(0)
		state["fRec83"] = np.float32(0)
		state["fRec85"] = np.float32(0)
		state["fRec87"] = np.float32(0)
		state["fRec89"] = np.float32(0)
		state["fRec91"] = np.float32(0)
		state["fRec94"] = np.float32(0)
		state["fRec98"] = np.float32(0)
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
		state["iRec369"] = np.int32(0)
		# Initialize array delays
		state["fRec22"] = np.zeros((3,), dtype=np.float32)
		state["fRec20"] = np.zeros((3,), dtype=np.float32)
		state["fRec18"] = np.zeros((3,), dtype=np.float32)
		state["fRec16"] = np.zeros((3,), dtype=np.float32)
		state["fRec24"] = np.zeros((3,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fRec26"] = np.zeros((3,), dtype=np.float32)
		state["fRec31"] = np.zeros((3,), dtype=np.float32)
		state["fRec30"] = np.zeros((3,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fRec36"] = np.zeros((3,), dtype=np.float32)
		state["fRec35"] = np.zeros((3,), dtype=np.float32)
		state["fRec34"] = np.zeros((3,), dtype=np.float32)
		state["fRec33"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		state["fRec42"] = np.zeros((3,), dtype=np.float32)
		state["fRec40"] = np.zeros((3,), dtype=np.float32)
		state["fRec38"] = np.zeros((3,), dtype=np.float32)
		state["fRec46"] = np.zeros((3,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fRec48"] = np.zeros((3,), dtype=np.float32)
		state["fRec53"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec58"] = np.zeros((3,), dtype=np.float32)
		state["fRec57"] = np.zeros((3,), dtype=np.float32)
		state["fRec56"] = np.zeros((3,), dtype=np.float32)
		state["fRec55"] = np.zeros((3,), dtype=np.float32)
		state["fRec66"] = np.zeros((3,), dtype=np.float32)
		state["fRec64"] = np.zeros((3,), dtype=np.float32)
		state["fRec62"] = np.zeros((3,), dtype=np.float32)
		state["fRec60"] = np.zeros((3,), dtype=np.float32)
		state["fRec68"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec70"] = np.zeros((3,), dtype=np.float32)
		state["fRec75"] = np.zeros((3,), dtype=np.float32)
		state["fRec74"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fRec88"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec84"] = np.zeros((3,), dtype=np.float32)
		state["fRec82"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec92"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec96"] = np.zeros((3,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fRec102"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fRec110"] = np.zeros((3,), dtype=np.float32)
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec112"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec114"] = np.zeros((3,), dtype=np.float32)
		state["fRec119"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec124"] = np.zeros((3,), dtype=np.float32)
		state["fRec123"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec132"] = np.zeros((3,), dtype=np.float32)
		state["fRec130"] = np.zeros((3,), dtype=np.float32)
		state["fRec128"] = np.zeros((3,), dtype=np.float32)
		state["fRec126"] = np.zeros((3,), dtype=np.float32)
		state["fRec134"] = np.zeros((3,), dtype=np.float32)
		state["fRec137"] = np.zeros((3,), dtype=np.float32)
		state["fRec136"] = np.zeros((3,), dtype=np.float32)
		state["fRec141"] = np.zeros((3,), dtype=np.float32)
		state["fRec140"] = np.zeros((3,), dtype=np.float32)
		state["fRec139"] = np.zeros((3,), dtype=np.float32)
		state["fRec146"] = np.zeros((3,), dtype=np.float32)
		state["fRec145"] = np.zeros((3,), dtype=np.float32)
		state["fRec144"] = np.zeros((3,), dtype=np.float32)
		state["fRec143"] = np.zeros((3,), dtype=np.float32)
		state["fRec154"] = np.zeros((3,), dtype=np.float32)
		state["fRec152"] = np.zeros((3,), dtype=np.float32)
		state["fRec150"] = np.zeros((3,), dtype=np.float32)
		state["fRec148"] = np.zeros((3,), dtype=np.float32)
		state["fRec156"] = np.zeros((3,), dtype=np.float32)
		state["fRec159"] = np.zeros((3,), dtype=np.float32)
		state["fRec158"] = np.zeros((3,), dtype=np.float32)
		state["fRec163"] = np.zeros((3,), dtype=np.float32)
		state["fRec162"] = np.zeros((3,), dtype=np.float32)
		state["fRec161"] = np.zeros((3,), dtype=np.float32)
		state["fRec168"] = np.zeros((3,), dtype=np.float32)
		state["fRec167"] = np.zeros((3,), dtype=np.float32)
		state["fRec166"] = np.zeros((3,), dtype=np.float32)
		state["fRec165"] = np.zeros((3,), dtype=np.float32)
		state["fRec176"] = np.zeros((3,), dtype=np.float32)
		state["fRec174"] = np.zeros((3,), dtype=np.float32)
		state["fRec172"] = np.zeros((3,), dtype=np.float32)
		state["fRec170"] = np.zeros((3,), dtype=np.float32)
		state["fRec178"] = np.zeros((3,), dtype=np.float32)
		state["fRec181"] = np.zeros((3,), dtype=np.float32)
		state["fRec180"] = np.zeros((3,), dtype=np.float32)
		state["fRec185"] = np.zeros((3,), dtype=np.float32)
		state["fRec184"] = np.zeros((3,), dtype=np.float32)
		state["fRec183"] = np.zeros((3,), dtype=np.float32)
		state["fRec190"] = np.zeros((3,), dtype=np.float32)
		state["fRec189"] = np.zeros((3,), dtype=np.float32)
		state["fRec188"] = np.zeros((3,), dtype=np.float32)
		state["fRec187"] = np.zeros((3,), dtype=np.float32)
		state["fRec198"] = np.zeros((3,), dtype=np.float32)
		state["fRec196"] = np.zeros((3,), dtype=np.float32)
		state["fRec194"] = np.zeros((3,), dtype=np.float32)
		state["fRec192"] = np.zeros((3,), dtype=np.float32)
		state["fRec200"] = np.zeros((3,), dtype=np.float32)
		state["fRec203"] = np.zeros((3,), dtype=np.float32)
		state["fRec202"] = np.zeros((3,), dtype=np.float32)
		state["fRec207"] = np.zeros((3,), dtype=np.float32)
		state["fRec206"] = np.zeros((3,), dtype=np.float32)
		state["fRec205"] = np.zeros((3,), dtype=np.float32)
		state["fRec212"] = np.zeros((3,), dtype=np.float32)
		state["fRec211"] = np.zeros((3,), dtype=np.float32)
		state["fRec210"] = np.zeros((3,), dtype=np.float32)
		state["fRec209"] = np.zeros((3,), dtype=np.float32)
		state["fRec220"] = np.zeros((3,), dtype=np.float32)
		state["fRec218"] = np.zeros((3,), dtype=np.float32)
		state["fRec216"] = np.zeros((3,), dtype=np.float32)
		state["fRec214"] = np.zeros((3,), dtype=np.float32)
		state["fRec222"] = np.zeros((3,), dtype=np.float32)
		state["fRec225"] = np.zeros((3,), dtype=np.float32)
		state["fRec224"] = np.zeros((3,), dtype=np.float32)
		state["fRec229"] = np.zeros((3,), dtype=np.float32)
		state["fRec228"] = np.zeros((3,), dtype=np.float32)
		state["fRec227"] = np.zeros((3,), dtype=np.float32)
		state["fRec234"] = np.zeros((3,), dtype=np.float32)
		state["fRec233"] = np.zeros((3,), dtype=np.float32)
		state["fRec232"] = np.zeros((3,), dtype=np.float32)
		state["fRec231"] = np.zeros((3,), dtype=np.float32)
		state["fRec242"] = np.zeros((3,), dtype=np.float32)
		state["fRec240"] = np.zeros((3,), dtype=np.float32)
		state["fRec238"] = np.zeros((3,), dtype=np.float32)
		state["fRec236"] = np.zeros((3,), dtype=np.float32)
		state["fRec244"] = np.zeros((3,), dtype=np.float32)
		state["fRec247"] = np.zeros((3,), dtype=np.float32)
		state["fRec246"] = np.zeros((3,), dtype=np.float32)
		state["fRec251"] = np.zeros((3,), dtype=np.float32)
		state["fRec250"] = np.zeros((3,), dtype=np.float32)
		state["fRec249"] = np.zeros((3,), dtype=np.float32)
		state["fRec256"] = np.zeros((3,), dtype=np.float32)
		state["fRec255"] = np.zeros((3,), dtype=np.float32)
		state["fRec254"] = np.zeros((3,), dtype=np.float32)
		state["fRec253"] = np.zeros((3,), dtype=np.float32)
		state["fRec264"] = np.zeros((3,), dtype=np.float32)
		state["fRec262"] = np.zeros((3,), dtype=np.float32)
		state["fRec260"] = np.zeros((3,), dtype=np.float32)
		state["fRec258"] = np.zeros((3,), dtype=np.float32)
		state["fRec266"] = np.zeros((3,), dtype=np.float32)
		state["fRec269"] = np.zeros((3,), dtype=np.float32)
		state["fRec268"] = np.zeros((3,), dtype=np.float32)
		state["fRec273"] = np.zeros((3,), dtype=np.float32)
		state["fRec272"] = np.zeros((3,), dtype=np.float32)
		state["fRec271"] = np.zeros((3,), dtype=np.float32)
		state["fRec278"] = np.zeros((3,), dtype=np.float32)
		state["fRec277"] = np.zeros((3,), dtype=np.float32)
		state["fRec276"] = np.zeros((3,), dtype=np.float32)
		state["fRec275"] = np.zeros((3,), dtype=np.float32)
		state["fRec286"] = np.zeros((3,), dtype=np.float32)
		state["fRec284"] = np.zeros((3,), dtype=np.float32)
		state["fRec282"] = np.zeros((3,), dtype=np.float32)
		state["fRec280"] = np.zeros((3,), dtype=np.float32)
		state["fRec288"] = np.zeros((3,), dtype=np.float32)
		state["fRec291"] = np.zeros((3,), dtype=np.float32)
		state["fRec290"] = np.zeros((3,), dtype=np.float32)
		state["fRec295"] = np.zeros((3,), dtype=np.float32)
		state["fRec294"] = np.zeros((3,), dtype=np.float32)
		state["fRec293"] = np.zeros((3,), dtype=np.float32)
		state["fRec300"] = np.zeros((3,), dtype=np.float32)
		state["fRec299"] = np.zeros((3,), dtype=np.float32)
		state["fRec298"] = np.zeros((3,), dtype=np.float32)
		state["fRec297"] = np.zeros((3,), dtype=np.float32)
		state["fRec308"] = np.zeros((3,), dtype=np.float32)
		state["fRec306"] = np.zeros((3,), dtype=np.float32)
		state["fRec304"] = np.zeros((3,), dtype=np.float32)
		state["fRec302"] = np.zeros((3,), dtype=np.float32)
		state["fRec310"] = np.zeros((3,), dtype=np.float32)
		state["fRec313"] = np.zeros((3,), dtype=np.float32)
		state["fRec312"] = np.zeros((3,), dtype=np.float32)
		state["fRec317"] = np.zeros((3,), dtype=np.float32)
		state["fRec316"] = np.zeros((3,), dtype=np.float32)
		state["fRec315"] = np.zeros((3,), dtype=np.float32)
		state["fRec322"] = np.zeros((3,), dtype=np.float32)
		state["fRec321"] = np.zeros((3,), dtype=np.float32)
		state["fRec320"] = np.zeros((3,), dtype=np.float32)
		state["fRec319"] = np.zeros((3,), dtype=np.float32)
		state["fRec330"] = np.zeros((3,), dtype=np.float32)
		state["fRec328"] = np.zeros((3,), dtype=np.float32)
		state["fRec326"] = np.zeros((3,), dtype=np.float32)
		state["fRec324"] = np.zeros((3,), dtype=np.float32)
		state["fRec332"] = np.zeros((3,), dtype=np.float32)
		state["fRec335"] = np.zeros((3,), dtype=np.float32)
		state["fRec334"] = np.zeros((3,), dtype=np.float32)
		state["fRec339"] = np.zeros((3,), dtype=np.float32)
		state["fRec338"] = np.zeros((3,), dtype=np.float32)
		state["fRec337"] = np.zeros((3,), dtype=np.float32)
		state["fRec344"] = np.zeros((3,), dtype=np.float32)
		state["fRec343"] = np.zeros((3,), dtype=np.float32)
		state["fRec342"] = np.zeros((3,), dtype=np.float32)
		state["fRec341"] = np.zeros((3,), dtype=np.float32)
		state["fRec352"] = np.zeros((3,), dtype=np.float32)
		state["fRec350"] = np.zeros((3,), dtype=np.float32)
		state["fRec348"] = np.zeros((3,), dtype=np.float32)
		state["fRec346"] = np.zeros((3,), dtype=np.float32)
		state["fRec354"] = np.zeros((3,), dtype=np.float32)
		state["fRec357"] = np.zeros((3,), dtype=np.float32)
		state["fRec356"] = np.zeros((3,), dtype=np.float32)
		state["fRec361"] = np.zeros((3,), dtype=np.float32)
		state["fRec360"] = np.zeros((3,), dtype=np.float32)
		state["fRec359"] = np.zeros((3,), dtype=np.float32)
		state["fRec366"] = np.zeros((3,), dtype=np.float32)
		state["fRec365"] = np.zeros((3,), dtype=np.float32)
		state["fRec364"] = np.zeros((3,), dtype=np.float32)
		state["fRec363"] = np.zeros((3,), dtype=np.float32)
		state["fRec368"] = np.zeros((4,), dtype=np.float32)
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

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = (jnp.float32(1.0) - params["fCheckbox0"]) 
		fSlow1 = jnp.tan((self._fConst1 * params["fHslider0"])) 
		fSlow2 = jnp.power(fSlow1, jnp.float32(2.0)) 
		fSlow3 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow2))) 
		fSlow4 = (jnp.float32(1.0) / fSlow1) 
		fSlow5 = (((fSlow4 + jnp.float32(-1.0)) / fSlow1) + jnp.float32(1.0)) 
		fSlow6 = (jnp.float32(1.0) / (((fSlow4 + jnp.float32(1.0)) / fSlow1) + jnp.float32(1.0))) 
		fSlow7 = jnp.tan((self._fConst1 * params["fHslider1"])) 
		fSlow8 = jnp.power(fSlow7, jnp.float32(2.0)) 
		fSlow9 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow8))) 
		fSlow10 = (jnp.float32(1.0) / fSlow7) 
		fSlow11 = (((fSlow10 + jnp.float32(-1.0)) / fSlow7) + jnp.float32(1.0)) 
		fSlow12 = (((fSlow10 + jnp.float32(1.0)) / fSlow7) + jnp.float32(1.0)) 
		fSlow13 = (jnp.float32(1.0) / fSlow12) 
		fSlow14 = jnp.tan((self._fConst1 * params["fHslider2"])) 
		fSlow15 = jnp.power(fSlow14, jnp.float32(2.0)) 
		fSlow16 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow15))) 
		fSlow17 = (jnp.float32(1.0) / fSlow14) 
		fSlow18 = (((fSlow17 + jnp.float32(-1.0)) / fSlow14) + jnp.float32(1.0)) 
		fSlow19 = (((fSlow17 + jnp.float32(1.0)) / fSlow14) + jnp.float32(1.0)) 
		fSlow20 = (jnp.float32(1.0) / fSlow19) 
		fSlow21 = jnp.tan((self._fConst1 * params["fHslider3"])) 
		fSlow22 = jnp.power(fSlow21, jnp.float32(2.0)) 
		fSlow23 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow22))) 
		fSlow24 = (jnp.float32(1.0) / fSlow21) 
		fSlow25 = (((fSlow24 + jnp.float32(-1.0)) / fSlow21) + jnp.float32(1.0)) 
		fSlow26 = (((fSlow24 + jnp.float32(1.0)) / fSlow21) + jnp.float32(1.0)) 
		fSlow27 = (jnp.float32(1.0) / fSlow26) 
		fSlow28 = (jnp.float32(1.0) - fSlow24) 
		fSlow29 = (jnp.float32(1.0) / (fSlow24 + jnp.float32(1.0))) 
		fSlow30 = (jnp.float32(1.0) - fSlow17) 
		fSlow31 = (fSlow17 + jnp.float32(1.0)) 
		fSlow32 = (jnp.float32(1.0) / fSlow31) 
		fSlow33 = (jnp.float32(1.0) - fSlow10) 
		fSlow34 = (fSlow10 + jnp.float32(1.0)) 
		fSlow35 = (jnp.float32(1.0) / fSlow34) 
		fSlow36 = (jnp.float32(1.0) - fSlow4) 
		fSlow37 = (fSlow4 + jnp.float32(1.0)) 
		fSlow38 = (jnp.float32(1.0) / fSlow37) 
		fSlow39 = params["fVslider0"] 
		fSlow40 = params["fHslider4"] 
		fSlow41 = jnp.power(jnp.float32(53.0), jnp.floor(((jnp.float32(0.25187066) * jnp.log((self._fConst2 * fSlow40))) + jnp.float32(0.5)))) 
		fSlow42 = jnp.exp(-((self._fConst3 * (fSlow41 / fSlow39)))) 
		fSlow43 = params["fVslider1"] 
		fSlow44 = (jnp.exp(-((self._fConst3 * (fSlow41 / fSlow43)))) / fSlow2) 
		fSlow45 = (jnp.float32(1.0) - (fSlow36 / fSlow1)) 
		fSlow46 = (jnp.float32(1.0) / ((fSlow37 / fSlow1) + jnp.float32(1.0))) 
		fSlow47 = (jnp.float32(1.0) / (fSlow8 * fSlow12)) 
		fSlow48 = params["fVslider2"] 
		fSlow49 = jnp.exp(-((self._fConst3 * (fSlow41 / fSlow48)))) 
		fSlow50 = (jnp.float32(1.0) - (fSlow33 / fSlow7)) 
		fSlow51 = (jnp.float32(1.0) / ((fSlow34 / fSlow7) + jnp.float32(1.0))) 
		fSlow52 = (jnp.float32(1.0) / (fSlow15 * fSlow19)) 
		fSlow53 = params["fVslider3"] 
		fSlow54 = jnp.exp(-((self._fConst3 * (fSlow41 / fSlow53)))) 
		fSlow55 = (jnp.float32(1.0) - (fSlow30 / fSlow14)) 
		fSlow56 = (jnp.float32(1.0) / ((fSlow31 / fSlow14) + jnp.float32(1.0))) 
		fSlow57 = (jnp.float32(1.0) / (fSlow22 * fSlow26)) 
		fSlow58 = params["fVslider4"] 
		fSlow59 = jnp.exp(-((self._fConst3 * (fSlow41 / fSlow58)))) 
		fSlow60 = params["fHslider5"] 
		fSlow61 = (fSlow40 / fSlow60) 
		fSlow62 = jnp.power(jnp.float32(19.0), jnp.floor(((jnp.float32(0.33962327) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.46666667)))))) + jnp.float32(0.5)))) 
		fSlow63 = jnp.exp(-((self._fConst3 * (fSlow62 / fSlow39)))) 
		fSlow64 = (jnp.exp(-((self._fConst3 * (fSlow62 / fSlow43)))) / fSlow2) 
		fSlow65 = jnp.exp(-((self._fConst3 * (fSlow62 / fSlow48)))) 
		fSlow66 = jnp.exp(-((self._fConst3 * (fSlow62 / fSlow53)))) 
		fSlow67 = jnp.exp(-((self._fConst3 * (fSlow62 / fSlow58)))) 
		fSlow68 = jnp.power(jnp.float32(37.0), jnp.floor(((jnp.float32(0.2769379) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.73333335)))))) + jnp.float32(0.5)))) 
		fSlow69 = jnp.exp(-((self._fConst3 * (fSlow68 / fSlow39)))) 
		fSlow70 = (jnp.exp(-((self._fConst3 * (fSlow68 / fSlow43)))) / fSlow2) 
		fSlow71 = jnp.exp(-((self._fConst3 * (fSlow68 / fSlow48)))) 
		fSlow72 = jnp.exp(-((self._fConst3 * (fSlow68 / fSlow53)))) 
		fSlow73 = jnp.exp(-((self._fConst3 * (fSlow68 / fSlow58)))) 
		fSlow74 = jnp.power(jnp.float32(7.0), jnp.floor(((jnp.float32(0.5138983) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.2)))))) + jnp.float32(0.5)))) 
		fSlow75 = jnp.exp(-((self._fConst3 * (fSlow74 / fSlow39)))) 
		fSlow76 = (jnp.exp(-((self._fConst3 * (fSlow74 / fSlow43)))) / fSlow2) 
		fSlow77 = jnp.exp(-((self._fConst3 * (fSlow74 / fSlow48)))) 
		fSlow78 = jnp.exp(-((self._fConst3 * (fSlow74 / fSlow53)))) 
		fSlow79 = jnp.exp(-((self._fConst3 * (fSlow74 / fSlow58)))) 
		fSlow80 = jnp.power(jnp.float32(43.0), jnp.floor(((jnp.float32(0.2658726) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.8666667)))))) + jnp.float32(0.5)))) 
		fSlow81 = jnp.exp(-((self._fConst3 * (fSlow80 / fSlow39)))) 
		fSlow82 = (jnp.exp(-((self._fConst3 * (fSlow80 / fSlow43)))) / fSlow2) 
		fSlow83 = jnp.exp(-((self._fConst3 * (fSlow80 / fSlow48)))) 
		fSlow84 = jnp.exp(-((self._fConst3 * (fSlow80 / fSlow53)))) 
		fSlow85 = jnp.exp(-((self._fConst3 * (fSlow80 / fSlow58)))) 
		fSlow86 = jnp.power(jnp.float32(13.0), jnp.floor(((jnp.float32(0.38987124) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.33333334)))))) + jnp.float32(0.5)))) 
		fSlow87 = jnp.exp(-((self._fConst3 * (fSlow86 / fSlow39)))) 
		fSlow88 = (jnp.exp(-((self._fConst3 * (fSlow86 / fSlow43)))) / fSlow2) 
		fSlow89 = jnp.exp(-((self._fConst3 * (fSlow86 / fSlow48)))) 
		fSlow90 = jnp.exp(-((self._fConst3 * (fSlow86 / fSlow53)))) 
		fSlow91 = jnp.exp(-((self._fConst3 * (fSlow86 / fSlow58)))) 
		fSlow92 = jnp.power(jnp.float32(29.0), jnp.floor(((jnp.float32(0.2969742) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.6)))))) + jnp.float32(0.5)))) 
		fSlow93 = jnp.exp(-((self._fConst3 * (fSlow92 / fSlow39)))) 
		fSlow94 = (jnp.exp(-((self._fConst3 * (fSlow92 / fSlow43)))) / fSlow2) 
		fSlow95 = jnp.exp(-((self._fConst3 * (fSlow92 / fSlow48)))) 
		fSlow96 = jnp.exp(-((self._fConst3 * (fSlow92 / fSlow53)))) 
		fSlow97 = jnp.exp(-((self._fConst3 * (fSlow92 / fSlow58)))) 
		fSlow98 = jnp.power(jnp.float32(3.0), jnp.floor(((jnp.float32(0.9102392) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.06666667)))))) + jnp.float32(0.5)))) 
		fSlow99 = jnp.exp(-((self._fConst3 * (fSlow98 / fSlow39)))) 
		fSlow100 = (jnp.exp(-((self._fConst3 * (fSlow98 / fSlow43)))) / fSlow2) 
		fSlow101 = jnp.exp(-((self._fConst3 * (fSlow98 / fSlow48)))) 
		fSlow102 = jnp.exp(-((self._fConst3 * (fSlow98 / fSlow53)))) 
		fSlow103 = jnp.exp(-((self._fConst3 * (fSlow98 / fSlow58)))) 
		fSlow104 = jnp.power(jnp.float32(47.0), jnp.floor(((jnp.float32(0.2597303) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.93333334)))))) + jnp.float32(0.5)))) 
		fSlow105 = jnp.exp(-((self._fConst3 * (fSlow104 / fSlow39)))) 
		fSlow106 = (jnp.exp(-((self._fConst3 * (fSlow104 / fSlow43)))) / fSlow2) 
		fSlow107 = jnp.exp(-((self._fConst3 * (fSlow104 / fSlow48)))) 
		fSlow108 = jnp.exp(-((self._fConst3 * (fSlow104 / fSlow53)))) 
		fSlow109 = jnp.exp(-((self._fConst3 * (fSlow104 / fSlow58)))) 
		fSlow110 = jnp.power(jnp.float32(17.0), jnp.floor(((jnp.float32(0.35295612) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.4)))))) + jnp.float32(0.5)))) 
		fSlow111 = jnp.exp(-((self._fConst3 * (fSlow110 / fSlow39)))) 
		fSlow112 = (jnp.exp(-((self._fConst3 * (fSlow110 / fSlow43)))) / fSlow2) 
		fSlow113 = jnp.exp(-((self._fConst3 * (fSlow110 / fSlow48)))) 
		fSlow114 = jnp.exp(-((self._fConst3 * (fSlow110 / fSlow53)))) 
		fSlow115 = jnp.exp(-((self._fConst3 * (fSlow110 / fSlow58)))) 
		fSlow116 = jnp.power(jnp.float32(31.0), jnp.floor(((jnp.float32(0.2912067) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.6666667)))))) + jnp.float32(0.5)))) 
		fSlow117 = jnp.exp(-((self._fConst3 * (fSlow116 / fSlow39)))) 
		fSlow118 = (jnp.exp(-((self._fConst3 * (fSlow116 / fSlow43)))) / fSlow2) 
		fSlow119 = jnp.exp(-((self._fConst3 * (fSlow116 / fSlow48)))) 
		fSlow120 = jnp.exp(-((self._fConst3 * (fSlow116 / fSlow53)))) 
		fSlow121 = jnp.exp(-((self._fConst3 * (fSlow116 / fSlow58)))) 
		fSlow122 = jnp.power(jnp.float32(5.0), jnp.floor(((jnp.float32(0.6213349) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.13333334)))))) + jnp.float32(0.5)))) 
		fSlow123 = jnp.exp(-((self._fConst3 * (fSlow122 / fSlow39)))) 
		fSlow124 = (jnp.exp(-((self._fConst3 * (fSlow122 / fSlow43)))) / fSlow2) 
		fSlow125 = jnp.exp(-((self._fConst3 * (fSlow122 / fSlow48)))) 
		fSlow126 = jnp.exp(-((self._fConst3 * (fSlow122 / fSlow53)))) 
		fSlow127 = jnp.exp(-((self._fConst3 * (fSlow122 / fSlow58)))) 
		fSlow128 = jnp.power(jnp.float32(41.0), jnp.floor(((jnp.float32(0.26928252) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.8)))))) + jnp.float32(0.5)))) 
		fSlow129 = jnp.exp(-((self._fConst3 * (fSlow128 / fSlow39)))) 
		fSlow130 = (jnp.exp(-((self._fConst3 * (fSlow128 / fSlow43)))) / fSlow2) 
		fSlow131 = jnp.exp(-((self._fConst3 * (fSlow128 / fSlow48)))) 
		fSlow132 = jnp.exp(-((self._fConst3 * (fSlow128 / fSlow53)))) 
		fSlow133 = jnp.exp(-((self._fConst3 * (fSlow128 / fSlow58)))) 
		fSlow134 = jnp.power(jnp.float32(11.0), jnp.floor(((jnp.float32(0.4170324) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.26666668)))))) + jnp.float32(0.5)))) 
		fSlow135 = jnp.exp(-((self._fConst3 * (fSlow134 / fSlow39)))) 
		fSlow136 = (jnp.exp(-((self._fConst3 * (fSlow134 / fSlow43)))) / fSlow2) 
		fSlow137 = jnp.exp(-((self._fConst3 * (fSlow134 / fSlow48)))) 
		fSlow138 = jnp.exp(-((self._fConst3 * (fSlow134 / fSlow53)))) 
		fSlow139 = jnp.exp(-((self._fConst3 * (fSlow134 / fSlow58)))) 
		fSlow140 = jnp.power(jnp.float32(23.0), jnp.floor(((jnp.float32(0.318929) * jnp.log((self._fConst2 * (fSlow60 * jnp.power(fSlow61, jnp.float32(0.53333336)))))) + jnp.float32(0.5)))) 
		fSlow141 = jnp.exp(-((self._fConst3 * (fSlow140 / fSlow39)))) 
		fSlow142 = (jnp.exp(-((self._fConst3 * (fSlow140 / fSlow43)))) / fSlow2) 
		fSlow143 = jnp.exp(-((self._fConst3 * (fSlow140 / fSlow48)))) 
		fSlow144 = jnp.exp(-((self._fConst3 * (fSlow140 / fSlow53)))) 
		fSlow145 = jnp.exp(-((self._fConst3 * (fSlow140 / fSlow58)))) 
		fSlow146 = jnp.power(jnp.float32(2.0), jnp.floor(((jnp.float32(1.442695) * jnp.log((self._fConst2 * fSlow60))) + jnp.float32(0.5)))) 
		fSlow147 = jnp.exp(-((self._fConst3 * (fSlow146 / fSlow39)))) 
		fSlow148 = (jnp.exp(-((self._fConst3 * (fSlow146 / fSlow43)))) / fSlow2) 
		fSlow149 = jnp.exp(-((self._fConst3 * (fSlow146 / fSlow48)))) 
		fSlow150 = jnp.exp(-((self._fConst3 * (fSlow146 / fSlow53)))) 
		fSlow151 = jnp.exp(-((self._fConst3 * (fSlow146 / fSlow58)))) 
		fSlow152 = (jnp.float32(0.25) * (jnp.float32(1.0) - (jnp.float32(0.5) * params["fButton0"]))) 
		fSlow153 = params["fButton1"] 
		fSlow154 = params["fButton2"] 
		fSlow155 = (jnp.float32(0.1) * params["fCheckbox1"]) 
		iSlow156 = (jnp.int32((fSlow146 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		fSlow157 = params["fButton3"] 
		iSlow158 = (jnp.int32((fSlow98 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow159 = (jnp.int32((fSlow122 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow160 = (jnp.int32((fSlow74 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow161 = (jnp.int32((fSlow134 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow162 = (jnp.int32((fSlow86 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow163 = (jnp.int32((fSlow110 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow164 = (jnp.int32((fSlow62 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow165 = (jnp.int32((fSlow140 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow166 = (jnp.int32((fSlow92 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow167 = (jnp.int32((fSlow116 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow168 = (jnp.int32((fSlow68 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow169 = (jnp.int32((fSlow128 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow170 = (jnp.int32((fSlow80 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow171 = (jnp.int32((fSlow104 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		iSlow172 = (jnp.int32((fSlow41 + jnp.float32(-1.0))) & jnp.int32(8191)).astype(jnp.int32) 
		fSlow173 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider6"])) 
		fRec23_temp = state["fRec23"] 
		fVec0_temp = state["fVec0"] 
		fRec21_temp = state["fRec21"] 
		fVec1_temp = state["fVec1"] 
		fRec19_temp = state["fRec19"] 
		fVec2_temp = state["fVec2"] 
		fRec17_temp = state["fRec17"] 
		fRec25_temp = state["fRec25"] 
		fRec28_temp = state["fRec28"] 
		fRec32_temp = state["fRec32"] 
		fRec37_temp = state["fRec37"] 
		fRec45_temp = state["fRec45"] 
		fVec3_temp = state["fVec3"] 
		fRec43_temp = state["fRec43"] 
		fVec4_temp = state["fVec4"] 
		fRec41_temp = state["fRec41"] 
		fVec5_temp = state["fVec5"] 
		fRec39_temp = state["fRec39"] 
		fRec47_temp = state["fRec47"] 
		fRec50_temp = state["fRec50"] 
		fRec54_temp = state["fRec54"] 
		fRec59_temp = state["fRec59"] 
		fRec67_temp = state["fRec67"] 
		fVec6_temp = state["fVec6"] 
		fRec65_temp = state["fRec65"] 
		fVec7_temp = state["fVec7"] 
		fRec63_temp = state["fRec63"] 
		fVec8_temp = state["fVec8"] 
		fRec61_temp = state["fRec61"] 
		fRec69_temp = state["fRec69"] 
		fRec72_temp = state["fRec72"] 
		fRec76_temp = state["fRec76"] 
		fRec81_temp = state["fRec81"] 
		fRec89_temp = state["fRec89"] 
		fVec9_temp = state["fVec9"] 
		fRec87_temp = state["fRec87"] 
		fVec10_temp = state["fVec10"] 
		fRec85_temp = state["fRec85"] 
		fVec11_temp = state["fVec11"] 
		fRec83_temp = state["fRec83"] 
		fRec91_temp = state["fRec91"] 
		fRec94_temp = state["fRec94"] 
		fRec98_temp = state["fRec98"] 
		fRec103_temp = state["fRec103"] 
		fRec111_temp = state["fRec111"] 
		fVec12_temp = state["fVec12"] 
		fRec109_temp = state["fRec109"] 
		fVec13_temp = state["fVec13"] 
		fRec107_temp = state["fRec107"] 
		fVec14_temp = state["fVec14"] 
		fRec105_temp = state["fRec105"] 
		fRec113_temp = state["fRec113"] 
		fRec116_temp = state["fRec116"] 
		fRec120_temp = state["fRec120"] 
		fRec125_temp = state["fRec125"] 
		fRec133_temp = state["fRec133"] 
		fVec15_temp = state["fVec15"] 
		fRec131_temp = state["fRec131"] 
		fVec16_temp = state["fVec16"] 
		fRec129_temp = state["fRec129"] 
		fVec17_temp = state["fVec17"] 
		fRec127_temp = state["fRec127"] 
		fRec135_temp = state["fRec135"] 
		fRec138_temp = state["fRec138"] 
		fRec142_temp = state["fRec142"] 
		fRec147_temp = state["fRec147"] 
		fRec155_temp = state["fRec155"] 
		fVec18_temp = state["fVec18"] 
		fRec153_temp = state["fRec153"] 
		fVec19_temp = state["fVec19"] 
		fRec151_temp = state["fRec151"] 
		fVec20_temp = state["fVec20"] 
		fRec149_temp = state["fRec149"] 
		fRec157_temp = state["fRec157"] 
		fRec160_temp = state["fRec160"] 
		fRec164_temp = state["fRec164"] 
		fRec169_temp = state["fRec169"] 
		fRec177_temp = state["fRec177"] 
		fVec21_temp = state["fVec21"] 
		fRec175_temp = state["fRec175"] 
		fVec22_temp = state["fVec22"] 
		fRec173_temp = state["fRec173"] 
		fVec23_temp = state["fVec23"] 
		fRec171_temp = state["fRec171"] 
		fRec179_temp = state["fRec179"] 
		fRec182_temp = state["fRec182"] 
		fRec186_temp = state["fRec186"] 
		fRec191_temp = state["fRec191"] 
		fRec199_temp = state["fRec199"] 
		fVec24_temp = state["fVec24"] 
		fRec197_temp = state["fRec197"] 
		fVec25_temp = state["fVec25"] 
		fRec195_temp = state["fRec195"] 
		fVec26_temp = state["fVec26"] 
		fRec193_temp = state["fRec193"] 
		fRec201_temp = state["fRec201"] 
		fRec204_temp = state["fRec204"] 
		fRec208_temp = state["fRec208"] 
		fRec213_temp = state["fRec213"] 
		fRec221_temp = state["fRec221"] 
		fVec27_temp = state["fVec27"] 
		fRec219_temp = state["fRec219"] 
		fVec28_temp = state["fVec28"] 
		fRec217_temp = state["fRec217"] 
		fVec29_temp = state["fVec29"] 
		fRec215_temp = state["fRec215"] 
		fRec223_temp = state["fRec223"] 
		fRec226_temp = state["fRec226"] 
		fRec230_temp = state["fRec230"] 
		fRec235_temp = state["fRec235"] 
		fRec243_temp = state["fRec243"] 
		fVec30_temp = state["fVec30"] 
		fRec241_temp = state["fRec241"] 
		fVec31_temp = state["fVec31"] 
		fRec239_temp = state["fRec239"] 
		fVec32_temp = state["fVec32"] 
		fRec237_temp = state["fRec237"] 
		fRec245_temp = state["fRec245"] 
		fRec248_temp = state["fRec248"] 
		fRec252_temp = state["fRec252"] 
		fRec257_temp = state["fRec257"] 
		fRec265_temp = state["fRec265"] 
		fVec33_temp = state["fVec33"] 
		fRec263_temp = state["fRec263"] 
		fVec34_temp = state["fVec34"] 
		fRec261_temp = state["fRec261"] 
		fVec35_temp = state["fVec35"] 
		fRec259_temp = state["fRec259"] 
		fRec267_temp = state["fRec267"] 
		fRec270_temp = state["fRec270"] 
		fRec274_temp = state["fRec274"] 
		fRec279_temp = state["fRec279"] 
		fRec287_temp = state["fRec287"] 
		fVec36_temp = state["fVec36"] 
		fRec285_temp = state["fRec285"] 
		fVec37_temp = state["fVec37"] 
		fRec283_temp = state["fRec283"] 
		fVec38_temp = state["fVec38"] 
		fRec281_temp = state["fRec281"] 
		fRec289_temp = state["fRec289"] 
		fRec292_temp = state["fRec292"] 
		fRec296_temp = state["fRec296"] 
		fRec301_temp = state["fRec301"] 
		fRec309_temp = state["fRec309"] 
		fVec39_temp = state["fVec39"] 
		fRec307_temp = state["fRec307"] 
		fVec40_temp = state["fVec40"] 
		fRec305_temp = state["fRec305"] 
		fVec41_temp = state["fVec41"] 
		fRec303_temp = state["fRec303"] 
		fRec311_temp = state["fRec311"] 
		fRec314_temp = state["fRec314"] 
		fRec318_temp = state["fRec318"] 
		fRec323_temp = state["fRec323"] 
		fRec331_temp = state["fRec331"] 
		fVec42_temp = state["fVec42"] 
		fRec329_temp = state["fRec329"] 
		fVec43_temp = state["fVec43"] 
		fRec327_temp = state["fRec327"] 
		fVec44_temp = state["fVec44"] 
		fRec325_temp = state["fRec325"] 
		fRec333_temp = state["fRec333"] 
		fRec336_temp = state["fRec336"] 
		fRec340_temp = state["fRec340"] 
		fRec345_temp = state["fRec345"] 
		fRec353_temp = state["fRec353"] 
		fVec45_temp = state["fVec45"] 
		fRec351_temp = state["fRec351"] 
		fVec46_temp = state["fVec46"] 
		fRec349_temp = state["fRec349"] 
		fVec47_temp = state["fVec47"] 
		fRec347_temp = state["fRec347"] 
		fRec355_temp = state["fRec355"] 
		fRec358_temp = state["fRec358"] 
		fRec362_temp = state["fRec362"] 
		fRec367_temp = state["fRec367"] 
		fVec48_temp = state["fVec48"] 
		fVec49_temp = state["fVec49"] 
		iRec369_temp = state["iRec369"] 
		fVec51_temp = state["fVec51"] 
		fTemp0 = (fSlow0 * inputs[0]) 
		state["fRec23"] = -((fSlow29 * ((fSlow28 * fRec23_temp) - (state["fRec15"][1] + state["fRec15"][2])))) 
		state["fRec22"] = state["fRec22"].at[0].set((state["fRec23"] - (fSlow27 * ((fSlow25 * state["fRec22"][2]) + (fSlow23 * state["fRec22"][1]))))) 
		fTemp1 = (fSlow27 * (state["fRec22"][2] + (state["fRec22"][0] + (jnp.float32(2.0) * state["fRec22"][1])))) 
		state["fVec0"] = jnp.float32(fTemp1) 
		state["fRec21"] = -((fSlow32 * ((fSlow30 * fRec21_temp) - (fTemp1 + fVec0_temp)))) 
		state["fRec20"] = state["fRec20"].at[0].set((state["fRec21"] - (fSlow20 * ((fSlow18 * state["fRec20"][2]) + (fSlow16 * state["fRec20"][1]))))) 
		fTemp2 = (fSlow20 * (state["fRec20"][2] + (state["fRec20"][0] + (jnp.float32(2.0) * state["fRec20"][1])))) 
		state["fVec1"] = jnp.float32(fTemp2) 
		state["fRec19"] = -((fSlow35 * ((fSlow33 * fRec19_temp) - (fTemp2 + fVec1_temp)))) 
		state["fRec18"] = state["fRec18"].at[0].set((state["fRec19"] - (fSlow13 * ((fSlow11 * state["fRec18"][2]) + (fSlow9 * state["fRec18"][1]))))) 
		fTemp3 = (fSlow13 * (state["fRec18"][2] + (state["fRec18"][0] + (jnp.float32(2.0) * state["fRec18"][1])))) 
		state["fVec2"] = jnp.float32(fTemp3) 
		state["fRec17"] = -((fSlow38 * ((fSlow36 * fRec17_temp) - (fTemp3 + fVec2_temp)))) 
		state["fRec16"] = state["fRec16"].at[0].set((state["fRec17"] - (fSlow6 * ((fSlow5 * state["fRec16"][2]) + (fSlow3 * state["fRec16"][1]))))) 
		state["fRec25"] = -((fSlow38 * ((fSlow36 * fRec25_temp) - (fSlow4 * (fTemp3 - fVec2_temp))))) 
		state["fRec24"] = state["fRec24"].at[0].set((state["fRec25"] - (fSlow6 * ((fSlow5 * state["fRec24"][2]) + (fSlow3 * state["fRec24"][1]))))) 
		fTemp4 = (fSlow3 * state["fRec26"][1]) 
		state["fRec28"] = -((fSlow35 * ((fSlow33 * fRec28_temp) - (fSlow10 * (fTemp2 - fVec1_temp))))) 
		state["fRec27"] = state["fRec27"].at[0].set((state["fRec28"] - (fSlow13 * ((fSlow11 * state["fRec27"][2]) + (fSlow9 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((fSlow47 * (state["fRec27"][2] + (state["fRec27"][0] - (jnp.float32(2.0) * state["fRec27"][1])))) - (fSlow46 * ((fSlow45 * state["fRec26"][2]) + fTemp4)))) 
		fTemp5 = (fSlow3 * state["fRec29"][1]) 
		fTemp6 = (fSlow9 * state["fRec30"][1]) 
		state["fRec32"] = -((fSlow32 * ((fSlow30 * fRec32_temp) - (fSlow17 * (fTemp1 - fVec0_temp))))) 
		state["fRec31"] = state["fRec31"].at[0].set((state["fRec32"] - (fSlow20 * ((fSlow18 * state["fRec31"][2]) + (fSlow16 * state["fRec31"][1]))))) 
		state["fRec30"] = state["fRec30"].at[0].set(((fSlow52 * (state["fRec31"][2] + (state["fRec31"][0] - (jnp.float32(2.0) * state["fRec31"][1])))) - (fSlow51 * ((fSlow50 * state["fRec30"][2]) + fTemp6)))) 
		state["fRec29"] = state["fRec29"].at[0].set(((state["fRec30"][2] + (fSlow51 * (fTemp6 + (fSlow50 * state["fRec30"][0])))) - (fSlow46 * ((fSlow45 * state["fRec29"][2]) + fTemp5)))) 
		fTemp7 = (fSlow3 * state["fRec33"][1]) 
		fTemp8 = (fSlow9 * state["fRec34"][1]) 
		fTemp9 = (fSlow16 * state["fRec35"][1]) 
		state["fRec37"] = -((fSlow29 * ((fSlow28 * fRec37_temp) - (fSlow24 * (state["fRec15"][1] - state["fRec15"][2]))))) 
		state["fRec36"] = state["fRec36"].at[0].set((state["fRec37"] - (fSlow27 * ((fSlow25 * state["fRec36"][2]) + (fSlow23 * state["fRec36"][1]))))) 
		state["fRec35"] = state["fRec35"].at[0].set(((fSlow57 * (state["fRec36"][2] + (state["fRec36"][0] - (jnp.float32(2.0) * state["fRec36"][1])))) - (fSlow56 * ((fSlow55 * state["fRec35"][2]) + fTemp9)))) 
		state["fRec34"] = state["fRec34"].at[0].set(((state["fRec35"][2] + (fSlow56 * (fTemp9 + (fSlow55 * state["fRec35"][0])))) - (fSlow51 * ((fSlow50 * state["fRec34"][2]) + fTemp8)))) 
		state["fRec33"] = state["fRec33"].at[0].set(((state["fRec34"][2] + (fSlow51 * (fTemp8 + (fSlow50 * state["fRec34"][0])))) - (fSlow46 * ((fSlow45 * state["fRec33"][2]) + fTemp7)))) 
		fTemp10 = ((((fSlow59 * (state["fRec33"][2] + (fSlow46 * (fTemp7 + (fSlow45 * state["fRec33"][0]))))) + (fSlow54 * (state["fRec29"][2] + (fSlow46 * (fTemp5 + (fSlow45 * state["fRec29"][0])))))) + (fSlow49 * (state["fRec26"][2] + (fSlow46 * (fTemp4 + (fSlow45 * state["fRec26"][0])))))) + (fSlow6 * ((fSlow44 * (state["fRec24"][2] + (state["fRec24"][0] - (jnp.float32(2.0) * state["fRec24"][1])))) + (fSlow42 * (state["fRec16"][2] + (state["fRec16"][0] + (jnp.float32(2.0) * state["fRec16"][1]))))))) 
		state["fRec45"] = -((fSlow29 * ((fSlow28 * fRec45_temp) - (state["fRec7"][1] + state["fRec7"][2])))) 
		state["fRec44"] = state["fRec44"].at[0].set((state["fRec45"] - (fSlow27 * ((fSlow25 * state["fRec44"][2]) + (fSlow23 * state["fRec44"][1]))))) 
		fTemp11 = (fSlow27 * (state["fRec44"][2] + (state["fRec44"][0] + (jnp.float32(2.0) * state["fRec44"][1])))) 
		state["fVec3"] = jnp.float32(fTemp11) 
		state["fRec43"] = -((fSlow32 * ((fSlow30 * fRec43_temp) - (fTemp11 + fVec3_temp)))) 
		state["fRec42"] = state["fRec42"].at[0].set((state["fRec43"] - (fSlow20 * ((fSlow18 * state["fRec42"][2]) + (fSlow16 * state["fRec42"][1]))))) 
		fTemp12 = (fSlow20 * (state["fRec42"][2] + (state["fRec42"][0] + (jnp.float32(2.0) * state["fRec42"][1])))) 
		state["fVec4"] = jnp.float32(fTemp12) 
		state["fRec41"] = -((fSlow35 * ((fSlow33 * fRec41_temp) - (fTemp12 + fVec4_temp)))) 
		state["fRec40"] = state["fRec40"].at[0].set((state["fRec41"] - (fSlow13 * ((fSlow11 * state["fRec40"][2]) + (fSlow9 * state["fRec40"][1]))))) 
		fTemp13 = (fSlow13 * (state["fRec40"][2] + (state["fRec40"][0] + (jnp.float32(2.0) * state["fRec40"][1])))) 
		state["fVec5"] = jnp.float32(fTemp13) 
		state["fRec39"] = -((fSlow38 * ((fSlow36 * fRec39_temp) - (fTemp13 + fVec5_temp)))) 
		state["fRec38"] = state["fRec38"].at[0].set((state["fRec39"] - (fSlow6 * ((fSlow5 * state["fRec38"][2]) + (fSlow3 * state["fRec38"][1]))))) 
		state["fRec47"] = -((fSlow38 * ((fSlow36 * fRec47_temp) - (fSlow4 * (fTemp13 - fVec5_temp))))) 
		state["fRec46"] = state["fRec46"].at[0].set((state["fRec47"] - (fSlow6 * ((fSlow5 * state["fRec46"][2]) + (fSlow3 * state["fRec46"][1]))))) 
		fTemp14 = (fSlow3 * state["fRec48"][1]) 
		state["fRec50"] = -((fSlow35 * ((fSlow33 * fRec50_temp) - (fSlow10 * (fTemp12 - fVec4_temp))))) 
		state["fRec49"] = state["fRec49"].at[0].set((state["fRec50"] - (fSlow13 * ((fSlow11 * state["fRec49"][2]) + (fSlow9 * state["fRec49"][1]))))) 
		state["fRec48"] = state["fRec48"].at[0].set(((fSlow47 * (state["fRec49"][2] + (state["fRec49"][0] - (jnp.float32(2.0) * state["fRec49"][1])))) - (fSlow46 * ((fSlow45 * state["fRec48"][2]) + fTemp14)))) 
		fTemp15 = (fSlow3 * state["fRec51"][1]) 
		fTemp16 = (fSlow9 * state["fRec52"][1]) 
		state["fRec54"] = -((fSlow32 * ((fSlow30 * fRec54_temp) - (fSlow17 * (fTemp11 - fVec3_temp))))) 
		state["fRec53"] = state["fRec53"].at[0].set((state["fRec54"] - (fSlow20 * ((fSlow18 * state["fRec53"][2]) + (fSlow16 * state["fRec53"][1]))))) 
		state["fRec52"] = state["fRec52"].at[0].set(((fSlow52 * (state["fRec53"][2] + (state["fRec53"][0] - (jnp.float32(2.0) * state["fRec53"][1])))) - (fSlow51 * ((fSlow50 * state["fRec52"][2]) + fTemp16)))) 
		state["fRec51"] = state["fRec51"].at[0].set(((state["fRec52"][2] + (fSlow51 * (fTemp16 + (fSlow50 * state["fRec52"][0])))) - (fSlow46 * ((fSlow45 * state["fRec51"][2]) + fTemp15)))) 
		fTemp17 = (fSlow3 * state["fRec55"][1]) 
		fTemp18 = (fSlow9 * state["fRec56"][1]) 
		fTemp19 = (fSlow16 * state["fRec57"][1]) 
		state["fRec59"] = -((fSlow29 * ((fSlow28 * fRec59_temp) - (fSlow24 * (state["fRec7"][1] - state["fRec7"][2]))))) 
		state["fRec58"] = state["fRec58"].at[0].set((state["fRec59"] - (fSlow27 * ((fSlow25 * state["fRec58"][2]) + (fSlow23 * state["fRec58"][1]))))) 
		state["fRec57"] = state["fRec57"].at[0].set(((fSlow57 * (state["fRec58"][2] + (state["fRec58"][0] - (jnp.float32(2.0) * state["fRec58"][1])))) - (fSlow56 * ((fSlow55 * state["fRec57"][2]) + fTemp19)))) 
		state["fRec56"] = state["fRec56"].at[0].set(((state["fRec57"][2] + (fSlow56 * (fTemp19 + (fSlow55 * state["fRec57"][0])))) - (fSlow51 * ((fSlow50 * state["fRec56"][2]) + fTemp18)))) 
		state["fRec55"] = state["fRec55"].at[0].set(((state["fRec56"][2] + (fSlow51 * (fTemp18 + (fSlow50 * state["fRec56"][0])))) - (fSlow46 * ((fSlow45 * state["fRec55"][2]) + fTemp17)))) 
		fTemp20 = ((((fSlow67 * (state["fRec55"][2] + (fSlow46 * (fTemp17 + (fSlow45 * state["fRec55"][0]))))) + (fSlow66 * (state["fRec51"][2] + (fSlow46 * (fTemp15 + (fSlow45 * state["fRec51"][0])))))) + (fSlow65 * (state["fRec48"][2] + (fSlow46 * (fTemp14 + (fSlow45 * state["fRec48"][0])))))) + (fSlow6 * ((fSlow64 * (state["fRec46"][2] + (state["fRec46"][0] - (jnp.float32(2.0) * state["fRec46"][1])))) + (fSlow63 * (state["fRec38"][2] + (state["fRec38"][0] + (jnp.float32(2.0) * state["fRec38"][1]))))))) 
		fTemp21 = (fTemp20 + fTemp10) 
		state["fRec67"] = -((fSlow29 * ((fSlow28 * fRec67_temp) - (state["fRec11"][1] + state["fRec11"][2])))) 
		state["fRec66"] = state["fRec66"].at[0].set((state["fRec67"] - (fSlow27 * ((fSlow25 * state["fRec66"][2]) + (fSlow23 * state["fRec66"][1]))))) 
		fTemp22 = (fSlow27 * (state["fRec66"][2] + (state["fRec66"][0] + (jnp.float32(2.0) * state["fRec66"][1])))) 
		state["fVec6"] = jnp.float32(fTemp22) 
		state["fRec65"] = -((fSlow32 * ((fSlow30 * fRec65_temp) - (fTemp22 + fVec6_temp)))) 
		state["fRec64"] = state["fRec64"].at[0].set((state["fRec65"] - (fSlow20 * ((fSlow18 * state["fRec64"][2]) + (fSlow16 * state["fRec64"][1]))))) 
		fTemp23 = (fSlow20 * (state["fRec64"][2] + (state["fRec64"][0] + (jnp.float32(2.0) * state["fRec64"][1])))) 
		state["fVec7"] = jnp.float32(fTemp23) 
		state["fRec63"] = -((fSlow35 * ((fSlow33 * fRec63_temp) - (fTemp23 + fVec7_temp)))) 
		state["fRec62"] = state["fRec62"].at[0].set((state["fRec63"] - (fSlow13 * ((fSlow11 * state["fRec62"][2]) + (fSlow9 * state["fRec62"][1]))))) 
		fTemp24 = (fSlow13 * (state["fRec62"][2] + (state["fRec62"][0] + (jnp.float32(2.0) * state["fRec62"][1])))) 
		state["fVec8"] = jnp.float32(fTemp24) 
		state["fRec61"] = -((fSlow38 * ((fSlow36 * fRec61_temp) - (fTemp24 + fVec8_temp)))) 
		state["fRec60"] = state["fRec60"].at[0].set((state["fRec61"] - (fSlow6 * ((fSlow5 * state["fRec60"][2]) + (fSlow3 * state["fRec60"][1]))))) 
		state["fRec69"] = -((fSlow38 * ((fSlow36 * fRec69_temp) - (fSlow4 * (fTemp24 - fVec8_temp))))) 
		state["fRec68"] = state["fRec68"].at[0].set((state["fRec69"] - (fSlow6 * ((fSlow5 * state["fRec68"][2]) + (fSlow3 * state["fRec68"][1]))))) 
		fTemp25 = (fSlow3 * state["fRec70"][1]) 
		state["fRec72"] = -((fSlow35 * ((fSlow33 * fRec72_temp) - (fSlow10 * (fTemp23 - fVec7_temp))))) 
		state["fRec71"] = state["fRec71"].at[0].set((state["fRec72"] - (fSlow13 * ((fSlow11 * state["fRec71"][2]) + (fSlow9 * state["fRec71"][1]))))) 
		state["fRec70"] = state["fRec70"].at[0].set(((fSlow47 * (state["fRec71"][2] + (state["fRec71"][0] - (jnp.float32(2.0) * state["fRec71"][1])))) - (fSlow46 * ((fSlow45 * state["fRec70"][2]) + fTemp25)))) 
		fTemp26 = (fSlow3 * state["fRec73"][1]) 
		fTemp27 = (fSlow9 * state["fRec74"][1]) 
		state["fRec76"] = -((fSlow32 * ((fSlow30 * fRec76_temp) - (fSlow17 * (fTemp22 - fVec6_temp))))) 
		state["fRec75"] = state["fRec75"].at[0].set((state["fRec76"] - (fSlow20 * ((fSlow18 * state["fRec75"][2]) + (fSlow16 * state["fRec75"][1]))))) 
		state["fRec74"] = state["fRec74"].at[0].set(((fSlow52 * (state["fRec75"][2] + (state["fRec75"][0] - (jnp.float32(2.0) * state["fRec75"][1])))) - (fSlow51 * ((fSlow50 * state["fRec74"][2]) + fTemp27)))) 
		state["fRec73"] = state["fRec73"].at[0].set(((state["fRec74"][2] + (fSlow51 * (fTemp27 + (fSlow50 * state["fRec74"][0])))) - (fSlow46 * ((fSlow45 * state["fRec73"][2]) + fTemp26)))) 
		fTemp28 = (fSlow3 * state["fRec77"][1]) 
		fTemp29 = (fSlow9 * state["fRec78"][1]) 
		fTemp30 = (fSlow16 * state["fRec79"][1]) 
		state["fRec81"] = -((fSlow29 * ((fSlow28 * fRec81_temp) - (fSlow24 * (state["fRec11"][1] - state["fRec11"][2]))))) 
		state["fRec80"] = state["fRec80"].at[0].set((state["fRec81"] - (fSlow27 * ((fSlow25 * state["fRec80"][2]) + (fSlow23 * state["fRec80"][1]))))) 
		state["fRec79"] = state["fRec79"].at[0].set(((fSlow57 * (state["fRec80"][2] + (state["fRec80"][0] - (jnp.float32(2.0) * state["fRec80"][1])))) - (fSlow56 * ((fSlow55 * state["fRec79"][2]) + fTemp30)))) 
		state["fRec78"] = state["fRec78"].at[0].set(((state["fRec79"][2] + (fSlow56 * (fTemp30 + (fSlow55 * state["fRec79"][0])))) - (fSlow51 * ((fSlow50 * state["fRec78"][2]) + fTemp29)))) 
		state["fRec77"] = state["fRec77"].at[0].set(((state["fRec78"][2] + (fSlow51 * (fTemp29 + (fSlow50 * state["fRec78"][0])))) - (fSlow46 * ((fSlow45 * state["fRec77"][2]) + fTemp28)))) 
		fTemp31 = ((((fSlow73 * (state["fRec77"][2] + (fSlow46 * (fTemp28 + (fSlow45 * state["fRec77"][0]))))) + (fSlow72 * (state["fRec73"][2] + (fSlow46 * (fTemp26 + (fSlow45 * state["fRec73"][0])))))) + (fSlow71 * (state["fRec70"][2] + (fSlow46 * (fTemp25 + (fSlow45 * state["fRec70"][0])))))) + (fSlow6 * ((fSlow70 * (state["fRec68"][2] + (state["fRec68"][0] - (jnp.float32(2.0) * state["fRec68"][1])))) + (fSlow69 * (state["fRec60"][2] + (state["fRec60"][0] + (jnp.float32(2.0) * state["fRec60"][1]))))))) 
		state["fRec89"] = -((fSlow29 * ((fSlow28 * fRec89_temp) - (state["fRec3"][1] + state["fRec3"][2])))) 
		state["fRec88"] = state["fRec88"].at[0].set((state["fRec89"] - (fSlow27 * ((fSlow25 * state["fRec88"][2]) + (fSlow23 * state["fRec88"][1]))))) 
		fTemp32 = (fSlow27 * (state["fRec88"][2] + (state["fRec88"][0] + (jnp.float32(2.0) * state["fRec88"][1])))) 
		state["fVec9"] = jnp.float32(fTemp32) 
		state["fRec87"] = -((fSlow32 * ((fSlow30 * fRec87_temp) - (fTemp32 + fVec9_temp)))) 
		state["fRec86"] = state["fRec86"].at[0].set((state["fRec87"] - (fSlow20 * ((fSlow18 * state["fRec86"][2]) + (fSlow16 * state["fRec86"][1]))))) 
		fTemp33 = (fSlow20 * (state["fRec86"][2] + (state["fRec86"][0] + (jnp.float32(2.0) * state["fRec86"][1])))) 
		state["fVec10"] = jnp.float32(fTemp33) 
		state["fRec85"] = -((fSlow35 * ((fSlow33 * fRec85_temp) - (fTemp33 + fVec10_temp)))) 
		state["fRec84"] = state["fRec84"].at[0].set((state["fRec85"] - (fSlow13 * ((fSlow11 * state["fRec84"][2]) + (fSlow9 * state["fRec84"][1]))))) 
		fTemp34 = (fSlow13 * (state["fRec84"][2] + (state["fRec84"][0] + (jnp.float32(2.0) * state["fRec84"][1])))) 
		state["fVec11"] = jnp.float32(fTemp34) 
		state["fRec83"] = -((fSlow38 * ((fSlow36 * fRec83_temp) - (fTemp34 + fVec11_temp)))) 
		state["fRec82"] = state["fRec82"].at[0].set((state["fRec83"] - (fSlow6 * ((fSlow5 * state["fRec82"][2]) + (fSlow3 * state["fRec82"][1]))))) 
		state["fRec91"] = -((fSlow38 * ((fSlow36 * fRec91_temp) - (fSlow4 * (fTemp34 - fVec11_temp))))) 
		state["fRec90"] = state["fRec90"].at[0].set((state["fRec91"] - (fSlow6 * ((fSlow5 * state["fRec90"][2]) + (fSlow3 * state["fRec90"][1]))))) 
		fTemp35 = (fSlow3 * state["fRec92"][1]) 
		state["fRec94"] = -((fSlow35 * ((fSlow33 * fRec94_temp) - (fSlow10 * (fTemp33 - fVec10_temp))))) 
		state["fRec93"] = state["fRec93"].at[0].set((state["fRec94"] - (fSlow13 * ((fSlow11 * state["fRec93"][2]) + (fSlow9 * state["fRec93"][1]))))) 
		state["fRec92"] = state["fRec92"].at[0].set(((fSlow47 * (state["fRec93"][2] + (state["fRec93"][0] - (jnp.float32(2.0) * state["fRec93"][1])))) - (fSlow46 * ((fSlow45 * state["fRec92"][2]) + fTemp35)))) 
		fTemp36 = (fSlow3 * state["fRec95"][1]) 
		fTemp37 = (fSlow9 * state["fRec96"][1]) 
		state["fRec98"] = -((fSlow32 * ((fSlow30 * fRec98_temp) - (fSlow17 * (fTemp32 - fVec9_temp))))) 
		state["fRec97"] = state["fRec97"].at[0].set((state["fRec98"] - (fSlow20 * ((fSlow18 * state["fRec97"][2]) + (fSlow16 * state["fRec97"][1]))))) 
		state["fRec96"] = state["fRec96"].at[0].set(((fSlow52 * (state["fRec97"][2] + (state["fRec97"][0] - (jnp.float32(2.0) * state["fRec97"][1])))) - (fSlow51 * ((fSlow50 * state["fRec96"][2]) + fTemp37)))) 
		state["fRec95"] = state["fRec95"].at[0].set(((state["fRec96"][2] + (fSlow51 * (fTemp37 + (fSlow50 * state["fRec96"][0])))) - (fSlow46 * ((fSlow45 * state["fRec95"][2]) + fTemp36)))) 
		fTemp38 = (fSlow3 * state["fRec99"][1]) 
		fTemp39 = (fSlow9 * state["fRec100"][1]) 
		fTemp40 = (fSlow16 * state["fRec101"][1]) 
		state["fRec103"] = -((fSlow29 * ((fSlow28 * fRec103_temp) - (fSlow24 * (state["fRec3"][1] - state["fRec3"][2]))))) 
		state["fRec102"] = state["fRec102"].at[0].set((state["fRec103"] - (fSlow27 * ((fSlow25 * state["fRec102"][2]) + (fSlow23 * state["fRec102"][1]))))) 
		state["fRec101"] = state["fRec101"].at[0].set(((fSlow57 * (state["fRec102"][2] + (state["fRec102"][0] - (jnp.float32(2.0) * state["fRec102"][1])))) - (fSlow56 * ((fSlow55 * state["fRec101"][2]) + fTemp40)))) 
		state["fRec100"] = state["fRec100"].at[0].set(((state["fRec101"][2] + (fSlow56 * (fTemp40 + (fSlow55 * state["fRec101"][0])))) - (fSlow51 * ((fSlow50 * state["fRec100"][2]) + fTemp39)))) 
		state["fRec99"] = state["fRec99"].at[0].set(((state["fRec100"][2] + (fSlow51 * (fTemp39 + (fSlow50 * state["fRec100"][0])))) - (fSlow46 * ((fSlow45 * state["fRec99"][2]) + fTemp38)))) 
		fTemp41 = ((((fSlow79 * (state["fRec99"][2] + (fSlow46 * (fTemp38 + (fSlow45 * state["fRec99"][0]))))) + (fSlow78 * (state["fRec95"][2] + (fSlow46 * (fTemp36 + (fSlow45 * state["fRec95"][0])))))) + (fSlow77 * (state["fRec92"][2] + (fSlow46 * (fTemp35 + (fSlow45 * state["fRec92"][0])))))) + (fSlow6 * ((fSlow76 * (state["fRec90"][2] + (state["fRec90"][0] - (jnp.float32(2.0) * state["fRec90"][1])))) + (fSlow75 * (state["fRec82"][2] + (state["fRec82"][0] + (jnp.float32(2.0) * state["fRec82"][1]))))))) 
		fTemp42 = (fTemp41 + fTemp31) 
		fTemp43 = (fTemp42 + fTemp21) 
		state["fRec111"] = -((fSlow29 * ((fSlow28 * fRec111_temp) - (state["fRec13"][1] + state["fRec13"][2])))) 
		state["fRec110"] = state["fRec110"].at[0].set((state["fRec111"] - (fSlow27 * ((fSlow25 * state["fRec110"][2]) + (fSlow23 * state["fRec110"][1]))))) 
		fTemp44 = (fSlow27 * (state["fRec110"][2] + (state["fRec110"][0] + (jnp.float32(2.0) * state["fRec110"][1])))) 
		state["fVec12"] = jnp.float32(fTemp44) 
		state["fRec109"] = -((fSlow32 * ((fSlow30 * fRec109_temp) - (fTemp44 + fVec12_temp)))) 
		state["fRec108"] = state["fRec108"].at[0].set((state["fRec109"] - (fSlow20 * ((fSlow18 * state["fRec108"][2]) + (fSlow16 * state["fRec108"][1]))))) 
		fTemp45 = (fSlow20 * (state["fRec108"][2] + (state["fRec108"][0] + (jnp.float32(2.0) * state["fRec108"][1])))) 
		state["fVec13"] = jnp.float32(fTemp45) 
		state["fRec107"] = -((fSlow35 * ((fSlow33 * fRec107_temp) - (fTemp45 + fVec13_temp)))) 
		state["fRec106"] = state["fRec106"].at[0].set((state["fRec107"] - (fSlow13 * ((fSlow11 * state["fRec106"][2]) + (fSlow9 * state["fRec106"][1]))))) 
		fTemp46 = (fSlow13 * (state["fRec106"][2] + (state["fRec106"][0] + (jnp.float32(2.0) * state["fRec106"][1])))) 
		state["fVec14"] = jnp.float32(fTemp46) 
		state["fRec105"] = -((fSlow38 * ((fSlow36 * fRec105_temp) - (fTemp46 + fVec14_temp)))) 
		state["fRec104"] = state["fRec104"].at[0].set((state["fRec105"] - (fSlow6 * ((fSlow5 * state["fRec104"][2]) + (fSlow3 * state["fRec104"][1]))))) 
		state["fRec113"] = -((fSlow38 * ((fSlow36 * fRec113_temp) - (fSlow4 * (fTemp46 - fVec14_temp))))) 
		state["fRec112"] = state["fRec112"].at[0].set((state["fRec113"] - (fSlow6 * ((fSlow5 * state["fRec112"][2]) + (fSlow3 * state["fRec112"][1]))))) 
		fTemp47 = (fSlow3 * state["fRec114"][1]) 
		state["fRec116"] = -((fSlow35 * ((fSlow33 * fRec116_temp) - (fSlow10 * (fTemp45 - fVec13_temp))))) 
		state["fRec115"] = state["fRec115"].at[0].set((state["fRec116"] - (fSlow13 * ((fSlow11 * state["fRec115"][2]) + (fSlow9 * state["fRec115"][1]))))) 
		state["fRec114"] = state["fRec114"].at[0].set(((fSlow47 * (state["fRec115"][2] + (state["fRec115"][0] - (jnp.float32(2.0) * state["fRec115"][1])))) - (fSlow46 * ((fSlow45 * state["fRec114"][2]) + fTemp47)))) 
		fTemp48 = (fSlow3 * state["fRec117"][1]) 
		fTemp49 = (fSlow9 * state["fRec118"][1]) 
		state["fRec120"] = -((fSlow32 * ((fSlow30 * fRec120_temp) - (fSlow17 * (fTemp44 - fVec12_temp))))) 
		state["fRec119"] = state["fRec119"].at[0].set((state["fRec120"] - (fSlow20 * ((fSlow18 * state["fRec119"][2]) + (fSlow16 * state["fRec119"][1]))))) 
		state["fRec118"] = state["fRec118"].at[0].set(((fSlow52 * (state["fRec119"][2] + (state["fRec119"][0] - (jnp.float32(2.0) * state["fRec119"][1])))) - (fSlow51 * ((fSlow50 * state["fRec118"][2]) + fTemp49)))) 
		state["fRec117"] = state["fRec117"].at[0].set(((state["fRec118"][2] + (fSlow51 * (fTemp49 + (fSlow50 * state["fRec118"][0])))) - (fSlow46 * ((fSlow45 * state["fRec117"][2]) + fTemp48)))) 
		fTemp50 = (fSlow3 * state["fRec121"][1]) 
		fTemp51 = (fSlow9 * state["fRec122"][1]) 
		fTemp52 = (fSlow16 * state["fRec123"][1]) 
		state["fRec125"] = -((fSlow29 * ((fSlow28 * fRec125_temp) - (fSlow24 * (state["fRec13"][1] - state["fRec13"][2]))))) 
		state["fRec124"] = state["fRec124"].at[0].set((state["fRec125"] - (fSlow27 * ((fSlow25 * state["fRec124"][2]) + (fSlow23 * state["fRec124"][1]))))) 
		state["fRec123"] = state["fRec123"].at[0].set(((fSlow57 * (state["fRec124"][2] + (state["fRec124"][0] - (jnp.float32(2.0) * state["fRec124"][1])))) - (fSlow56 * ((fSlow55 * state["fRec123"][2]) + fTemp52)))) 
		state["fRec122"] = state["fRec122"].at[0].set(((state["fRec123"][2] + (fSlow56 * (fTemp52 + (fSlow55 * state["fRec123"][0])))) - (fSlow51 * ((fSlow50 * state["fRec122"][2]) + fTemp51)))) 
		state["fRec121"] = state["fRec121"].at[0].set(((state["fRec122"][2] + (fSlow51 * (fTemp51 + (fSlow50 * state["fRec122"][0])))) - (fSlow46 * ((fSlow45 * state["fRec121"][2]) + fTemp50)))) 
		fTemp53 = ((((fSlow85 * (state["fRec121"][2] + (fSlow46 * (fTemp50 + (fSlow45 * state["fRec121"][0]))))) + (fSlow84 * (state["fRec117"][2] + (fSlow46 * (fTemp48 + (fSlow45 * state["fRec117"][0])))))) + (fSlow83 * (state["fRec114"][2] + (fSlow46 * (fTemp47 + (fSlow45 * state["fRec114"][0])))))) + (fSlow6 * ((fSlow82 * (state["fRec112"][2] + (state["fRec112"][0] - (jnp.float32(2.0) * state["fRec112"][1])))) + (fSlow81 * (state["fRec104"][2] + (state["fRec104"][0] + (jnp.float32(2.0) * state["fRec104"][1]))))))) 
		state["fRec133"] = -((fSlow29 * ((fSlow28 * fRec133_temp) - (state["fRec5"][1] + state["fRec5"][2])))) 
		state["fRec132"] = state["fRec132"].at[0].set((state["fRec133"] - (fSlow27 * ((fSlow25 * state["fRec132"][2]) + (fSlow23 * state["fRec132"][1]))))) 
		fTemp54 = (fSlow27 * (state["fRec132"][2] + (state["fRec132"][0] + (jnp.float32(2.0) * state["fRec132"][1])))) 
		state["fVec15"] = jnp.float32(fTemp54) 
		state["fRec131"] = -((fSlow32 * ((fSlow30 * fRec131_temp) - (fTemp54 + fVec15_temp)))) 
		state["fRec130"] = state["fRec130"].at[0].set((state["fRec131"] - (fSlow20 * ((fSlow18 * state["fRec130"][2]) + (fSlow16 * state["fRec130"][1]))))) 
		fTemp55 = (fSlow20 * (state["fRec130"][2] + (state["fRec130"][0] + (jnp.float32(2.0) * state["fRec130"][1])))) 
		state["fVec16"] = jnp.float32(fTemp55) 
		state["fRec129"] = -((fSlow35 * ((fSlow33 * fRec129_temp) - (fTemp55 + fVec16_temp)))) 
		state["fRec128"] = state["fRec128"].at[0].set((state["fRec129"] - (fSlow13 * ((fSlow11 * state["fRec128"][2]) + (fSlow9 * state["fRec128"][1]))))) 
		fTemp56 = (fSlow13 * (state["fRec128"][2] + (state["fRec128"][0] + (jnp.float32(2.0) * state["fRec128"][1])))) 
		state["fVec17"] = jnp.float32(fTemp56) 
		state["fRec127"] = -((fSlow38 * ((fSlow36 * fRec127_temp) - (fTemp56 + fVec17_temp)))) 
		state["fRec126"] = state["fRec126"].at[0].set((state["fRec127"] - (fSlow6 * ((fSlow5 * state["fRec126"][2]) + (fSlow3 * state["fRec126"][1]))))) 
		state["fRec135"] = -((fSlow38 * ((fSlow36 * fRec135_temp) - (fSlow4 * (fTemp56 - fVec17_temp))))) 
		state["fRec134"] = state["fRec134"].at[0].set((state["fRec135"] - (fSlow6 * ((fSlow5 * state["fRec134"][2]) + (fSlow3 * state["fRec134"][1]))))) 
		fTemp57 = (fSlow3 * state["fRec136"][1]) 
		state["fRec138"] = -((fSlow35 * ((fSlow33 * fRec138_temp) - (fSlow10 * (fTemp55 - fVec16_temp))))) 
		state["fRec137"] = state["fRec137"].at[0].set((state["fRec138"] - (fSlow13 * ((fSlow11 * state["fRec137"][2]) + (fSlow9 * state["fRec137"][1]))))) 
		state["fRec136"] = state["fRec136"].at[0].set(((fSlow47 * (state["fRec137"][2] + (state["fRec137"][0] - (jnp.float32(2.0) * state["fRec137"][1])))) - (fSlow46 * ((fSlow45 * state["fRec136"][2]) + fTemp57)))) 
		fTemp58 = (fSlow3 * state["fRec139"][1]) 
		fTemp59 = (fSlow9 * state["fRec140"][1]) 
		state["fRec142"] = -((fSlow32 * ((fSlow30 * fRec142_temp) - (fSlow17 * (fTemp54 - fVec15_temp))))) 
		state["fRec141"] = state["fRec141"].at[0].set((state["fRec142"] - (fSlow20 * ((fSlow18 * state["fRec141"][2]) + (fSlow16 * state["fRec141"][1]))))) 
		state["fRec140"] = state["fRec140"].at[0].set(((fSlow52 * (state["fRec141"][2] + (state["fRec141"][0] - (jnp.float32(2.0) * state["fRec141"][1])))) - (fSlow51 * ((fSlow50 * state["fRec140"][2]) + fTemp59)))) 
		state["fRec139"] = state["fRec139"].at[0].set(((state["fRec140"][2] + (fSlow51 * (fTemp59 + (fSlow50 * state["fRec140"][0])))) - (fSlow46 * ((fSlow45 * state["fRec139"][2]) + fTemp58)))) 
		fTemp60 = (fSlow3 * state["fRec143"][1]) 
		fTemp61 = (fSlow9 * state["fRec144"][1]) 
		fTemp62 = (fSlow16 * state["fRec145"][1]) 
		state["fRec147"] = -((fSlow29 * ((fSlow28 * fRec147_temp) - (fSlow24 * (state["fRec5"][1] - state["fRec5"][2]))))) 
		state["fRec146"] = state["fRec146"].at[0].set((state["fRec147"] - (fSlow27 * ((fSlow25 * state["fRec146"][2]) + (fSlow23 * state["fRec146"][1]))))) 
		state["fRec145"] = state["fRec145"].at[0].set(((fSlow57 * (state["fRec146"][2] + (state["fRec146"][0] - (jnp.float32(2.0) * state["fRec146"][1])))) - (fSlow56 * ((fSlow55 * state["fRec145"][2]) + fTemp62)))) 
		state["fRec144"] = state["fRec144"].at[0].set(((state["fRec145"][2] + (fSlow56 * (fTemp62 + (fSlow55 * state["fRec145"][0])))) - (fSlow51 * ((fSlow50 * state["fRec144"][2]) + fTemp61)))) 
		state["fRec143"] = state["fRec143"].at[0].set(((state["fRec144"][2] + (fSlow51 * (fTemp61 + (fSlow50 * state["fRec144"][0])))) - (fSlow46 * ((fSlow45 * state["fRec143"][2]) + fTemp60)))) 
		fTemp63 = ((((fSlow91 * (state["fRec143"][2] + (fSlow46 * (fTemp60 + (fSlow45 * state["fRec143"][0]))))) + (fSlow90 * (state["fRec139"][2] + (fSlow46 * (fTemp58 + (fSlow45 * state["fRec139"][0])))))) + (fSlow89 * (state["fRec136"][2] + (fSlow46 * (fTemp57 + (fSlow45 * state["fRec136"][0])))))) + (fSlow6 * ((fSlow88 * (state["fRec134"][2] + (state["fRec134"][0] - (jnp.float32(2.0) * state["fRec134"][1])))) + (fSlow87 * (state["fRec126"][2] + (state["fRec126"][0] + (jnp.float32(2.0) * state["fRec126"][1]))))))) 
		fTemp64 = (fTemp63 + fTemp53) 
		state["fRec155"] = -((fSlow29 * ((fSlow28 * fRec155_temp) - (state["fRec9"][1] + state["fRec9"][2])))) 
		state["fRec154"] = state["fRec154"].at[0].set((state["fRec155"] - (fSlow27 * ((fSlow25 * state["fRec154"][2]) + (fSlow23 * state["fRec154"][1]))))) 
		fTemp65 = (fSlow27 * (state["fRec154"][2] + (state["fRec154"][0] + (jnp.float32(2.0) * state["fRec154"][1])))) 
		state["fVec18"] = jnp.float32(fTemp65) 
		state["fRec153"] = -((fSlow32 * ((fSlow30 * fRec153_temp) - (fTemp65 + fVec18_temp)))) 
		state["fRec152"] = state["fRec152"].at[0].set((state["fRec153"] - (fSlow20 * ((fSlow18 * state["fRec152"][2]) + (fSlow16 * state["fRec152"][1]))))) 
		fTemp66 = (fSlow20 * (state["fRec152"][2] + (state["fRec152"][0] + (jnp.float32(2.0) * state["fRec152"][1])))) 
		state["fVec19"] = jnp.float32(fTemp66) 
		state["fRec151"] = -((fSlow35 * ((fSlow33 * fRec151_temp) - (fTemp66 + fVec19_temp)))) 
		state["fRec150"] = state["fRec150"].at[0].set((state["fRec151"] - (fSlow13 * ((fSlow11 * state["fRec150"][2]) + (fSlow9 * state["fRec150"][1]))))) 
		fTemp67 = (fSlow13 * (state["fRec150"][2] + (state["fRec150"][0] + (jnp.float32(2.0) * state["fRec150"][1])))) 
		state["fVec20"] = jnp.float32(fTemp67) 
		state["fRec149"] = -((fSlow38 * ((fSlow36 * fRec149_temp) - (fTemp67 + fVec20_temp)))) 
		state["fRec148"] = state["fRec148"].at[0].set((state["fRec149"] - (fSlow6 * ((fSlow5 * state["fRec148"][2]) + (fSlow3 * state["fRec148"][1]))))) 
		state["fRec157"] = -((fSlow38 * ((fSlow36 * fRec157_temp) - (fSlow4 * (fTemp67 - fVec20_temp))))) 
		state["fRec156"] = state["fRec156"].at[0].set((state["fRec157"] - (fSlow6 * ((fSlow5 * state["fRec156"][2]) + (fSlow3 * state["fRec156"][1]))))) 
		fTemp68 = (fSlow3 * state["fRec158"][1]) 
		state["fRec160"] = -((fSlow35 * ((fSlow33 * fRec160_temp) - (fSlow10 * (fTemp66 - fVec19_temp))))) 
		state["fRec159"] = state["fRec159"].at[0].set((state["fRec160"] - (fSlow13 * ((fSlow11 * state["fRec159"][2]) + (fSlow9 * state["fRec159"][1]))))) 
		state["fRec158"] = state["fRec158"].at[0].set(((fSlow47 * (state["fRec159"][2] + (state["fRec159"][0] - (jnp.float32(2.0) * state["fRec159"][1])))) - (fSlow46 * ((fSlow45 * state["fRec158"][2]) + fTemp68)))) 
		fTemp69 = (fSlow3 * state["fRec161"][1]) 
		fTemp70 = (fSlow9 * state["fRec162"][1]) 
		state["fRec164"] = -((fSlow32 * ((fSlow30 * fRec164_temp) - (fSlow17 * (fTemp65 - fVec18_temp))))) 
		state["fRec163"] = state["fRec163"].at[0].set((state["fRec164"] - (fSlow20 * ((fSlow18 * state["fRec163"][2]) + (fSlow16 * state["fRec163"][1]))))) 
		state["fRec162"] = state["fRec162"].at[0].set(((fSlow52 * (state["fRec163"][2] + (state["fRec163"][0] - (jnp.float32(2.0) * state["fRec163"][1])))) - (fSlow51 * ((fSlow50 * state["fRec162"][2]) + fTemp70)))) 
		state["fRec161"] = state["fRec161"].at[0].set(((state["fRec162"][2] + (fSlow51 * (fTemp70 + (fSlow50 * state["fRec162"][0])))) - (fSlow46 * ((fSlow45 * state["fRec161"][2]) + fTemp69)))) 
		fTemp71 = (fSlow3 * state["fRec165"][1]) 
		fTemp72 = (fSlow9 * state["fRec166"][1]) 
		fTemp73 = (fSlow16 * state["fRec167"][1]) 
		state["fRec169"] = -((fSlow29 * ((fSlow28 * fRec169_temp) - (fSlow24 * (state["fRec9"][1] - state["fRec9"][2]))))) 
		state["fRec168"] = state["fRec168"].at[0].set((state["fRec169"] - (fSlow27 * ((fSlow25 * state["fRec168"][2]) + (fSlow23 * state["fRec168"][1]))))) 
		state["fRec167"] = state["fRec167"].at[0].set(((fSlow57 * (state["fRec168"][2] + (state["fRec168"][0] - (jnp.float32(2.0) * state["fRec168"][1])))) - (fSlow56 * ((fSlow55 * state["fRec167"][2]) + fTemp73)))) 
		state["fRec166"] = state["fRec166"].at[0].set(((state["fRec167"][2] + (fSlow56 * (fTemp73 + (fSlow55 * state["fRec167"][0])))) - (fSlow51 * ((fSlow50 * state["fRec166"][2]) + fTemp72)))) 
		state["fRec165"] = state["fRec165"].at[0].set(((state["fRec166"][2] + (fSlow51 * (fTemp72 + (fSlow50 * state["fRec166"][0])))) - (fSlow46 * ((fSlow45 * state["fRec165"][2]) + fTemp71)))) 
		fTemp74 = ((((fSlow97 * (state["fRec165"][2] + (fSlow46 * (fTemp71 + (fSlow45 * state["fRec165"][0]))))) + (fSlow96 * (state["fRec161"][2] + (fSlow46 * (fTemp69 + (fSlow45 * state["fRec161"][0])))))) + (fSlow95 * (state["fRec158"][2] + (fSlow46 * (fTemp68 + (fSlow45 * state["fRec158"][0])))))) + (fSlow6 * ((fSlow94 * (state["fRec156"][2] + (state["fRec156"][0] - (jnp.float32(2.0) * state["fRec156"][1])))) + (fSlow93 * (state["fRec148"][2] + (state["fRec148"][0] + (jnp.float32(2.0) * state["fRec148"][1]))))))) 
		state["fRec177"] = -((fSlow29 * ((fSlow28 * fRec177_temp) - (state["fRec1"][1] + state["fRec1"][2])))) 
		state["fRec176"] = state["fRec176"].at[0].set((state["fRec177"] - (fSlow27 * ((fSlow25 * state["fRec176"][2]) + (fSlow23 * state["fRec176"][1]))))) 
		fTemp75 = (fSlow27 * (state["fRec176"][2] + (state["fRec176"][0] + (jnp.float32(2.0) * state["fRec176"][1])))) 
		state["fVec21"] = jnp.float32(fTemp75) 
		state["fRec175"] = -((fSlow32 * ((fSlow30 * fRec175_temp) - (fTemp75 + fVec21_temp)))) 
		state["fRec174"] = state["fRec174"].at[0].set((state["fRec175"] - (fSlow20 * ((fSlow18 * state["fRec174"][2]) + (fSlow16 * state["fRec174"][1]))))) 
		fTemp76 = (fSlow20 * (state["fRec174"][2] + (state["fRec174"][0] + (jnp.float32(2.0) * state["fRec174"][1])))) 
		state["fVec22"] = jnp.float32(fTemp76) 
		state["fRec173"] = -((fSlow35 * ((fSlow33 * fRec173_temp) - (fTemp76 + fVec22_temp)))) 
		state["fRec172"] = state["fRec172"].at[0].set((state["fRec173"] - (fSlow13 * ((fSlow11 * state["fRec172"][2]) + (fSlow9 * state["fRec172"][1]))))) 
		fTemp77 = (fSlow13 * (state["fRec172"][2] + (state["fRec172"][0] + (jnp.float32(2.0) * state["fRec172"][1])))) 
		state["fVec23"] = jnp.float32(fTemp77) 
		state["fRec171"] = -((fSlow38 * ((fSlow36 * fRec171_temp) - (fTemp77 + fVec23_temp)))) 
		state["fRec170"] = state["fRec170"].at[0].set((state["fRec171"] - (fSlow6 * ((fSlow5 * state["fRec170"][2]) + (fSlow3 * state["fRec170"][1]))))) 
		state["fRec179"] = -((fSlow38 * ((fSlow36 * fRec179_temp) - (fSlow4 * (fTemp77 - fVec23_temp))))) 
		state["fRec178"] = state["fRec178"].at[0].set((state["fRec179"] - (fSlow6 * ((fSlow5 * state["fRec178"][2]) + (fSlow3 * state["fRec178"][1]))))) 
		fTemp78 = (fSlow3 * state["fRec180"][1]) 
		state["fRec182"] = -((fSlow35 * ((fSlow33 * fRec182_temp) - (fSlow10 * (fTemp76 - fVec22_temp))))) 
		state["fRec181"] = state["fRec181"].at[0].set((state["fRec182"] - (fSlow13 * ((fSlow11 * state["fRec181"][2]) + (fSlow9 * state["fRec181"][1]))))) 
		state["fRec180"] = state["fRec180"].at[0].set(((fSlow47 * (state["fRec181"][2] + (state["fRec181"][0] - (jnp.float32(2.0) * state["fRec181"][1])))) - (fSlow46 * ((fSlow45 * state["fRec180"][2]) + fTemp78)))) 
		fTemp79 = (fSlow3 * state["fRec183"][1]) 
		fTemp80 = (fSlow9 * state["fRec184"][1]) 
		state["fRec186"] = -((fSlow32 * ((fSlow30 * fRec186_temp) - (fSlow17 * (fTemp75 - fVec21_temp))))) 
		state["fRec185"] = state["fRec185"].at[0].set((state["fRec186"] - (fSlow20 * ((fSlow18 * state["fRec185"][2]) + (fSlow16 * state["fRec185"][1]))))) 
		state["fRec184"] = state["fRec184"].at[0].set(((fSlow52 * (state["fRec185"][2] + (state["fRec185"][0] - (jnp.float32(2.0) * state["fRec185"][1])))) - (fSlow51 * ((fSlow50 * state["fRec184"][2]) + fTemp80)))) 
		state["fRec183"] = state["fRec183"].at[0].set(((state["fRec184"][2] + (fSlow51 * (fTemp80 + (fSlow50 * state["fRec184"][0])))) - (fSlow46 * ((fSlow45 * state["fRec183"][2]) + fTemp79)))) 
		fTemp81 = (fSlow3 * state["fRec187"][1]) 
		fTemp82 = (fSlow9 * state["fRec188"][1]) 
		fTemp83 = (fSlow16 * state["fRec189"][1]) 
		state["fRec191"] = -((fSlow29 * ((fSlow28 * fRec191_temp) - (fSlow24 * (state["fRec1"][1] - state["fRec1"][2]))))) 
		state["fRec190"] = state["fRec190"].at[0].set((state["fRec191"] - (fSlow27 * ((fSlow25 * state["fRec190"][2]) + (fSlow23 * state["fRec190"][1]))))) 
		state["fRec189"] = state["fRec189"].at[0].set(((fSlow57 * (state["fRec190"][2] + (state["fRec190"][0] - (jnp.float32(2.0) * state["fRec190"][1])))) - (fSlow56 * ((fSlow55 * state["fRec189"][2]) + fTemp83)))) 
		state["fRec188"] = state["fRec188"].at[0].set(((state["fRec189"][2] + (fSlow56 * (fTemp83 + (fSlow55 * state["fRec189"][0])))) - (fSlow51 * ((fSlow50 * state["fRec188"][2]) + fTemp82)))) 
		state["fRec187"] = state["fRec187"].at[0].set(((state["fRec188"][2] + (fSlow51 * (fTemp82 + (fSlow50 * state["fRec188"][0])))) - (fSlow46 * ((fSlow45 * state["fRec187"][2]) + fTemp81)))) 
		fTemp84 = ((((fSlow103 * (state["fRec187"][2] + (fSlow46 * (fTemp81 + (fSlow45 * state["fRec187"][0]))))) + (fSlow102 * (state["fRec183"][2] + (fSlow46 * (fTemp79 + (fSlow45 * state["fRec183"][0])))))) + (fSlow101 * (state["fRec180"][2] + (fSlow46 * (fTemp78 + (fSlow45 * state["fRec180"][0])))))) + (fSlow6 * ((fSlow100 * (state["fRec178"][2] + (state["fRec178"][0] - (jnp.float32(2.0) * state["fRec178"][1])))) + (fSlow99 * (state["fRec170"][2] + (state["fRec170"][0] + (jnp.float32(2.0) * state["fRec170"][1]))))))) 
		fTemp85 = (fTemp84 + fTemp74) 
		fTemp86 = (fTemp85 + fTemp64) 
		fTemp87 = (fTemp86 + fTemp43) 
		state["fRec199"] = -((fSlow29 * ((fSlow28 * fRec199_temp) - (state["fRec14"][1] + state["fRec14"][2])))) 
		state["fRec198"] = state["fRec198"].at[0].set((state["fRec199"] - (fSlow27 * ((fSlow25 * state["fRec198"][2]) + (fSlow23 * state["fRec198"][1]))))) 
		fTemp88 = (fSlow27 * (state["fRec198"][2] + (state["fRec198"][0] + (jnp.float32(2.0) * state["fRec198"][1])))) 
		state["fVec24"] = jnp.float32(fTemp88) 
		state["fRec197"] = -((fSlow32 * ((fSlow30 * fRec197_temp) - (fTemp88 + fVec24_temp)))) 
		state["fRec196"] = state["fRec196"].at[0].set((state["fRec197"] - (fSlow20 * ((fSlow18 * state["fRec196"][2]) + (fSlow16 * state["fRec196"][1]))))) 
		fTemp89 = (fSlow20 * (state["fRec196"][2] + (state["fRec196"][0] + (jnp.float32(2.0) * state["fRec196"][1])))) 
		state["fVec25"] = jnp.float32(fTemp89) 
		state["fRec195"] = -((fSlow35 * ((fSlow33 * fRec195_temp) - (fTemp89 + fVec25_temp)))) 
		state["fRec194"] = state["fRec194"].at[0].set((state["fRec195"] - (fSlow13 * ((fSlow11 * state["fRec194"][2]) + (fSlow9 * state["fRec194"][1]))))) 
		fTemp90 = (fSlow13 * (state["fRec194"][2] + (state["fRec194"][0] + (jnp.float32(2.0) * state["fRec194"][1])))) 
		state["fVec26"] = jnp.float32(fTemp90) 
		state["fRec193"] = -((fSlow38 * ((fSlow36 * fRec193_temp) - (fTemp90 + fVec26_temp)))) 
		state["fRec192"] = state["fRec192"].at[0].set((state["fRec193"] - (fSlow6 * ((fSlow5 * state["fRec192"][2]) + (fSlow3 * state["fRec192"][1]))))) 
		state["fRec201"] = -((fSlow38 * ((fSlow36 * fRec201_temp) - (fSlow4 * (fTemp90 - fVec26_temp))))) 
		state["fRec200"] = state["fRec200"].at[0].set((state["fRec201"] - (fSlow6 * ((fSlow5 * state["fRec200"][2]) + (fSlow3 * state["fRec200"][1]))))) 
		fTemp91 = (fSlow3 * state["fRec202"][1]) 
		state["fRec204"] = -((fSlow35 * ((fSlow33 * fRec204_temp) - (fSlow10 * (fTemp89 - fVec25_temp))))) 
		state["fRec203"] = state["fRec203"].at[0].set((state["fRec204"] - (fSlow13 * ((fSlow11 * state["fRec203"][2]) + (fSlow9 * state["fRec203"][1]))))) 
		state["fRec202"] = state["fRec202"].at[0].set(((fSlow47 * (state["fRec203"][2] + (state["fRec203"][0] - (jnp.float32(2.0) * state["fRec203"][1])))) - (fSlow46 * ((fSlow45 * state["fRec202"][2]) + fTemp91)))) 
		fTemp92 = (fSlow3 * state["fRec205"][1]) 
		fTemp93 = (fSlow9 * state["fRec206"][1]) 
		state["fRec208"] = -((fSlow32 * ((fSlow30 * fRec208_temp) - (fSlow17 * (fTemp88 - fVec24_temp))))) 
		state["fRec207"] = state["fRec207"].at[0].set((state["fRec208"] - (fSlow20 * ((fSlow18 * state["fRec207"][2]) + (fSlow16 * state["fRec207"][1]))))) 
		state["fRec206"] = state["fRec206"].at[0].set(((fSlow52 * (state["fRec207"][2] + (state["fRec207"][0] - (jnp.float32(2.0) * state["fRec207"][1])))) - (fSlow51 * ((fSlow50 * state["fRec206"][2]) + fTemp93)))) 
		state["fRec205"] = state["fRec205"].at[0].set(((state["fRec206"][2] + (fSlow51 * (fTemp93 + (fSlow50 * state["fRec206"][0])))) - (fSlow46 * ((fSlow45 * state["fRec205"][2]) + fTemp92)))) 
		fTemp94 = (fSlow3 * state["fRec209"][1]) 
		fTemp95 = (fSlow9 * state["fRec210"][1]) 
		fTemp96 = (fSlow16 * state["fRec211"][1]) 
		state["fRec213"] = -((fSlow29 * ((fSlow28 * fRec213_temp) - (fSlow24 * (state["fRec14"][1] - state["fRec14"][2]))))) 
		state["fRec212"] = state["fRec212"].at[0].set((state["fRec213"] - (fSlow27 * ((fSlow25 * state["fRec212"][2]) + (fSlow23 * state["fRec212"][1]))))) 
		state["fRec211"] = state["fRec211"].at[0].set(((fSlow57 * (state["fRec212"][2] + (state["fRec212"][0] - (jnp.float32(2.0) * state["fRec212"][1])))) - (fSlow56 * ((fSlow55 * state["fRec211"][2]) + fTemp96)))) 
		state["fRec210"] = state["fRec210"].at[0].set(((state["fRec211"][2] + (fSlow56 * (fTemp96 + (fSlow55 * state["fRec211"][0])))) - (fSlow51 * ((fSlow50 * state["fRec210"][2]) + fTemp95)))) 
		state["fRec209"] = state["fRec209"].at[0].set(((state["fRec210"][2] + (fSlow51 * (fTemp95 + (fSlow50 * state["fRec210"][0])))) - (fSlow46 * ((fSlow45 * state["fRec209"][2]) + fTemp94)))) 
		fTemp97 = ((((fSlow109 * (state["fRec209"][2] + (fSlow46 * (fTemp94 + (fSlow45 * state["fRec209"][0]))))) + (fSlow108 * (state["fRec205"][2] + (fSlow46 * (fTemp92 + (fSlow45 * state["fRec205"][0])))))) + (fSlow107 * (state["fRec202"][2] + (fSlow46 * (fTemp91 + (fSlow45 * state["fRec202"][0])))))) + (fSlow6 * ((fSlow106 * (state["fRec200"][2] + (state["fRec200"][0] - (jnp.float32(2.0) * state["fRec200"][1])))) + (fSlow105 * (state["fRec192"][2] + (state["fRec192"][0] + (jnp.float32(2.0) * state["fRec192"][1]))))))) 
		state["fRec221"] = -((fSlow29 * ((fSlow28 * fRec221_temp) - (state["fRec6"][1] + state["fRec6"][2])))) 
		state["fRec220"] = state["fRec220"].at[0].set((state["fRec221"] - (fSlow27 * ((fSlow25 * state["fRec220"][2]) + (fSlow23 * state["fRec220"][1]))))) 
		fTemp98 = (fSlow27 * (state["fRec220"][2] + (state["fRec220"][0] + (jnp.float32(2.0) * state["fRec220"][1])))) 
		state["fVec27"] = jnp.float32(fTemp98) 
		state["fRec219"] = -((fSlow32 * ((fSlow30 * fRec219_temp) - (fTemp98 + fVec27_temp)))) 
		state["fRec218"] = state["fRec218"].at[0].set((state["fRec219"] - (fSlow20 * ((fSlow18 * state["fRec218"][2]) + (fSlow16 * state["fRec218"][1]))))) 
		fTemp99 = (fSlow20 * (state["fRec218"][2] + (state["fRec218"][0] + (jnp.float32(2.0) * state["fRec218"][1])))) 
		state["fVec28"] = jnp.float32(fTemp99) 
		state["fRec217"] = -((fSlow35 * ((fSlow33 * fRec217_temp) - (fTemp99 + fVec28_temp)))) 
		state["fRec216"] = state["fRec216"].at[0].set((state["fRec217"] - (fSlow13 * ((fSlow11 * state["fRec216"][2]) + (fSlow9 * state["fRec216"][1]))))) 
		fTemp100 = (fSlow13 * (state["fRec216"][2] + (state["fRec216"][0] + (jnp.float32(2.0) * state["fRec216"][1])))) 
		state["fVec29"] = jnp.float32(fTemp100) 
		state["fRec215"] = -((fSlow38 * ((fSlow36 * fRec215_temp) - (fTemp100 + fVec29_temp)))) 
		state["fRec214"] = state["fRec214"].at[0].set((state["fRec215"] - (fSlow6 * ((fSlow5 * state["fRec214"][2]) + (fSlow3 * state["fRec214"][1]))))) 
		state["fRec223"] = -((fSlow38 * ((fSlow36 * fRec223_temp) - (fSlow4 * (fTemp100 - fVec29_temp))))) 
		state["fRec222"] = state["fRec222"].at[0].set((state["fRec223"] - (fSlow6 * ((fSlow5 * state["fRec222"][2]) + (fSlow3 * state["fRec222"][1]))))) 
		fTemp101 = (fSlow3 * state["fRec224"][1]) 
		state["fRec226"] = -((fSlow35 * ((fSlow33 * fRec226_temp) - (fSlow10 * (fTemp99 - fVec28_temp))))) 
		state["fRec225"] = state["fRec225"].at[0].set((state["fRec226"] - (fSlow13 * ((fSlow11 * state["fRec225"][2]) + (fSlow9 * state["fRec225"][1]))))) 
		state["fRec224"] = state["fRec224"].at[0].set(((fSlow47 * (state["fRec225"][2] + (state["fRec225"][0] - (jnp.float32(2.0) * state["fRec225"][1])))) - (fSlow46 * ((fSlow45 * state["fRec224"][2]) + fTemp101)))) 
		fTemp102 = (fSlow3 * state["fRec227"][1]) 
		fTemp103 = (fSlow9 * state["fRec228"][1]) 
		state["fRec230"] = -((fSlow32 * ((fSlow30 * fRec230_temp) - (fSlow17 * (fTemp98 - fVec27_temp))))) 
		state["fRec229"] = state["fRec229"].at[0].set((state["fRec230"] - (fSlow20 * ((fSlow18 * state["fRec229"][2]) + (fSlow16 * state["fRec229"][1]))))) 
		state["fRec228"] = state["fRec228"].at[0].set(((fSlow52 * (state["fRec229"][2] + (state["fRec229"][0] - (jnp.float32(2.0) * state["fRec229"][1])))) - (fSlow51 * ((fSlow50 * state["fRec228"][2]) + fTemp103)))) 
		state["fRec227"] = state["fRec227"].at[0].set(((state["fRec228"][2] + (fSlow51 * (fTemp103 + (fSlow50 * state["fRec228"][0])))) - (fSlow46 * ((fSlow45 * state["fRec227"][2]) + fTemp102)))) 
		fTemp104 = (fSlow3 * state["fRec231"][1]) 
		fTemp105 = (fSlow9 * state["fRec232"][1]) 
		fTemp106 = (fSlow16 * state["fRec233"][1]) 
		state["fRec235"] = -((fSlow29 * ((fSlow28 * fRec235_temp) - (fSlow24 * (state["fRec6"][1] - state["fRec6"][2]))))) 
		state["fRec234"] = state["fRec234"].at[0].set((state["fRec235"] - (fSlow27 * ((fSlow25 * state["fRec234"][2]) + (fSlow23 * state["fRec234"][1]))))) 
		state["fRec233"] = state["fRec233"].at[0].set(((fSlow57 * (state["fRec234"][2] + (state["fRec234"][0] - (jnp.float32(2.0) * state["fRec234"][1])))) - (fSlow56 * ((fSlow55 * state["fRec233"][2]) + fTemp106)))) 
		state["fRec232"] = state["fRec232"].at[0].set(((state["fRec233"][2] + (fSlow56 * (fTemp106 + (fSlow55 * state["fRec233"][0])))) - (fSlow51 * ((fSlow50 * state["fRec232"][2]) + fTemp105)))) 
		state["fRec231"] = state["fRec231"].at[0].set(((state["fRec232"][2] + (fSlow51 * (fTemp105 + (fSlow50 * state["fRec232"][0])))) - (fSlow46 * ((fSlow45 * state["fRec231"][2]) + fTemp104)))) 
		fTemp107 = ((((fSlow115 * (state["fRec231"][2] + (fSlow46 * (fTemp104 + (fSlow45 * state["fRec231"][0]))))) + (fSlow114 * (state["fRec227"][2] + (fSlow46 * (fTemp102 + (fSlow45 * state["fRec227"][0])))))) + (fSlow113 * (state["fRec224"][2] + (fSlow46 * (fTemp101 + (fSlow45 * state["fRec224"][0])))))) + (fSlow6 * ((fSlow112 * (state["fRec222"][2] + (state["fRec222"][0] - (jnp.float32(2.0) * state["fRec222"][1])))) + (fSlow111 * (state["fRec214"][2] + (state["fRec214"][0] + (jnp.float32(2.0) * state["fRec214"][1]))))))) 
		fTemp108 = (fTemp107 + fTemp97) 
		state["fRec243"] = -((fSlow29 * ((fSlow28 * fRec243_temp) - (state["fRec10"][1] + state["fRec10"][2])))) 
		state["fRec242"] = state["fRec242"].at[0].set((state["fRec243"] - (fSlow27 * ((fSlow25 * state["fRec242"][2]) + (fSlow23 * state["fRec242"][1]))))) 
		fTemp109 = (fSlow27 * (state["fRec242"][2] + (state["fRec242"][0] + (jnp.float32(2.0) * state["fRec242"][1])))) 
		state["fVec30"] = jnp.float32(fTemp109) 
		state["fRec241"] = -((fSlow32 * ((fSlow30 * fRec241_temp) - (fTemp109 + fVec30_temp)))) 
		state["fRec240"] = state["fRec240"].at[0].set((state["fRec241"] - (fSlow20 * ((fSlow18 * state["fRec240"][2]) + (fSlow16 * state["fRec240"][1]))))) 
		fTemp110 = (fSlow20 * (state["fRec240"][2] + (state["fRec240"][0] + (jnp.float32(2.0) * state["fRec240"][1])))) 
		state["fVec31"] = jnp.float32(fTemp110) 
		state["fRec239"] = -((fSlow35 * ((fSlow33 * fRec239_temp) - (fTemp110 + fVec31_temp)))) 
		state["fRec238"] = state["fRec238"].at[0].set((state["fRec239"] - (fSlow13 * ((fSlow11 * state["fRec238"][2]) + (fSlow9 * state["fRec238"][1]))))) 
		fTemp111 = (fSlow13 * (state["fRec238"][2] + (state["fRec238"][0] + (jnp.float32(2.0) * state["fRec238"][1])))) 
		state["fVec32"] = jnp.float32(fTemp111) 
		state["fRec237"] = -((fSlow38 * ((fSlow36 * fRec237_temp) - (fTemp111 + fVec32_temp)))) 
		state["fRec236"] = state["fRec236"].at[0].set((state["fRec237"] - (fSlow6 * ((fSlow5 * state["fRec236"][2]) + (fSlow3 * state["fRec236"][1]))))) 
		state["fRec245"] = -((fSlow38 * ((fSlow36 * fRec245_temp) - (fSlow4 * (fTemp111 - fVec32_temp))))) 
		state["fRec244"] = state["fRec244"].at[0].set((state["fRec245"] - (fSlow6 * ((fSlow5 * state["fRec244"][2]) + (fSlow3 * state["fRec244"][1]))))) 
		fTemp112 = (fSlow3 * state["fRec246"][1]) 
		state["fRec248"] = -((fSlow35 * ((fSlow33 * fRec248_temp) - (fSlow10 * (fTemp110 - fVec31_temp))))) 
		state["fRec247"] = state["fRec247"].at[0].set((state["fRec248"] - (fSlow13 * ((fSlow11 * state["fRec247"][2]) + (fSlow9 * state["fRec247"][1]))))) 
		state["fRec246"] = state["fRec246"].at[0].set(((fSlow47 * (state["fRec247"][2] + (state["fRec247"][0] - (jnp.float32(2.0) * state["fRec247"][1])))) - (fSlow46 * ((fSlow45 * state["fRec246"][2]) + fTemp112)))) 
		fTemp113 = (fSlow3 * state["fRec249"][1]) 
		fTemp114 = (fSlow9 * state["fRec250"][1]) 
		state["fRec252"] = -((fSlow32 * ((fSlow30 * fRec252_temp) - (fSlow17 * (fTemp109 - fVec30_temp))))) 
		state["fRec251"] = state["fRec251"].at[0].set((state["fRec252"] - (fSlow20 * ((fSlow18 * state["fRec251"][2]) + (fSlow16 * state["fRec251"][1]))))) 
		state["fRec250"] = state["fRec250"].at[0].set(((fSlow52 * (state["fRec251"][2] + (state["fRec251"][0] - (jnp.float32(2.0) * state["fRec251"][1])))) - (fSlow51 * ((fSlow50 * state["fRec250"][2]) + fTemp114)))) 
		state["fRec249"] = state["fRec249"].at[0].set(((state["fRec250"][2] + (fSlow51 * (fTemp114 + (fSlow50 * state["fRec250"][0])))) - (fSlow46 * ((fSlow45 * state["fRec249"][2]) + fTemp113)))) 
		fTemp115 = (fSlow3 * state["fRec253"][1]) 
		fTemp116 = (fSlow9 * state["fRec254"][1]) 
		fTemp117 = (fSlow16 * state["fRec255"][1]) 
		state["fRec257"] = -((fSlow29 * ((fSlow28 * fRec257_temp) - (fSlow24 * (state["fRec10"][1] - state["fRec10"][2]))))) 
		state["fRec256"] = state["fRec256"].at[0].set((state["fRec257"] - (fSlow27 * ((fSlow25 * state["fRec256"][2]) + (fSlow23 * state["fRec256"][1]))))) 
		state["fRec255"] = state["fRec255"].at[0].set(((fSlow57 * (state["fRec256"][2] + (state["fRec256"][0] - (jnp.float32(2.0) * state["fRec256"][1])))) - (fSlow56 * ((fSlow55 * state["fRec255"][2]) + fTemp117)))) 
		state["fRec254"] = state["fRec254"].at[0].set(((state["fRec255"][2] + (fSlow56 * (fTemp117 + (fSlow55 * state["fRec255"][0])))) - (fSlow51 * ((fSlow50 * state["fRec254"][2]) + fTemp116)))) 
		state["fRec253"] = state["fRec253"].at[0].set(((state["fRec254"][2] + (fSlow51 * (fTemp116 + (fSlow50 * state["fRec254"][0])))) - (fSlow46 * ((fSlow45 * state["fRec253"][2]) + fTemp115)))) 
		fTemp118 = ((((fSlow121 * (state["fRec253"][2] + (fSlow46 * (fTemp115 + (fSlow45 * state["fRec253"][0]))))) + (fSlow120 * (state["fRec249"][2] + (fSlow46 * (fTemp113 + (fSlow45 * state["fRec249"][0])))))) + (fSlow119 * (state["fRec246"][2] + (fSlow46 * (fTemp112 + (fSlow45 * state["fRec246"][0])))))) + (fSlow6 * ((fSlow118 * (state["fRec244"][2] + (state["fRec244"][0] - (jnp.float32(2.0) * state["fRec244"][1])))) + (fSlow117 * (state["fRec236"][2] + (state["fRec236"][0] + (jnp.float32(2.0) * state["fRec236"][1]))))))) 
		state["fRec265"] = -((fSlow29 * ((fSlow28 * fRec265_temp) - (state["fRec2"][1] + state["fRec2"][2])))) 
		state["fRec264"] = state["fRec264"].at[0].set((state["fRec265"] - (fSlow27 * ((fSlow25 * state["fRec264"][2]) + (fSlow23 * state["fRec264"][1]))))) 
		fTemp119 = (fSlow27 * (state["fRec264"][2] + (state["fRec264"][0] + (jnp.float32(2.0) * state["fRec264"][1])))) 
		state["fVec33"] = jnp.float32(fTemp119) 
		state["fRec263"] = -((fSlow32 * ((fSlow30 * fRec263_temp) - (fTemp119 + fVec33_temp)))) 
		state["fRec262"] = state["fRec262"].at[0].set((state["fRec263"] - (fSlow20 * ((fSlow18 * state["fRec262"][2]) + (fSlow16 * state["fRec262"][1]))))) 
		fTemp120 = (fSlow20 * (state["fRec262"][2] + (state["fRec262"][0] + (jnp.float32(2.0) * state["fRec262"][1])))) 
		state["fVec34"] = jnp.float32(fTemp120) 
		state["fRec261"] = -((fSlow35 * ((fSlow33 * fRec261_temp) - (fTemp120 + fVec34_temp)))) 
		state["fRec260"] = state["fRec260"].at[0].set((state["fRec261"] - (fSlow13 * ((fSlow11 * state["fRec260"][2]) + (fSlow9 * state["fRec260"][1]))))) 
		fTemp121 = (fSlow13 * (state["fRec260"][2] + (state["fRec260"][0] + (jnp.float32(2.0) * state["fRec260"][1])))) 
		state["fVec35"] = jnp.float32(fTemp121) 
		state["fRec259"] = -((fSlow38 * ((fSlow36 * fRec259_temp) - (fTemp121 + fVec35_temp)))) 
		state["fRec258"] = state["fRec258"].at[0].set((state["fRec259"] - (fSlow6 * ((fSlow5 * state["fRec258"][2]) + (fSlow3 * state["fRec258"][1]))))) 
		state["fRec267"] = -((fSlow38 * ((fSlow36 * fRec267_temp) - (fSlow4 * (fTemp121 - fVec35_temp))))) 
		state["fRec266"] = state["fRec266"].at[0].set((state["fRec267"] - (fSlow6 * ((fSlow5 * state["fRec266"][2]) + (fSlow3 * state["fRec266"][1]))))) 
		fTemp122 = (fSlow3 * state["fRec268"][1]) 
		state["fRec270"] = -((fSlow35 * ((fSlow33 * fRec270_temp) - (fSlow10 * (fTemp120 - fVec34_temp))))) 
		state["fRec269"] = state["fRec269"].at[0].set((state["fRec270"] - (fSlow13 * ((fSlow11 * state["fRec269"][2]) + (fSlow9 * state["fRec269"][1]))))) 
		state["fRec268"] = state["fRec268"].at[0].set(((fSlow47 * (state["fRec269"][2] + (state["fRec269"][0] - (jnp.float32(2.0) * state["fRec269"][1])))) - (fSlow46 * ((fSlow45 * state["fRec268"][2]) + fTemp122)))) 
		fTemp123 = (fSlow3 * state["fRec271"][1]) 
		fTemp124 = (fSlow9 * state["fRec272"][1]) 
		state["fRec274"] = -((fSlow32 * ((fSlow30 * fRec274_temp) - (fSlow17 * (fTemp119 - fVec33_temp))))) 
		state["fRec273"] = state["fRec273"].at[0].set((state["fRec274"] - (fSlow20 * ((fSlow18 * state["fRec273"][2]) + (fSlow16 * state["fRec273"][1]))))) 
		state["fRec272"] = state["fRec272"].at[0].set(((fSlow52 * (state["fRec273"][2] + (state["fRec273"][0] - (jnp.float32(2.0) * state["fRec273"][1])))) - (fSlow51 * ((fSlow50 * state["fRec272"][2]) + fTemp124)))) 
		state["fRec271"] = state["fRec271"].at[0].set(((state["fRec272"][2] + (fSlow51 * (fTemp124 + (fSlow50 * state["fRec272"][0])))) - (fSlow46 * ((fSlow45 * state["fRec271"][2]) + fTemp123)))) 
		fTemp125 = (fSlow3 * state["fRec275"][1]) 
		fTemp126 = (fSlow9 * state["fRec276"][1]) 
		fTemp127 = (fSlow16 * state["fRec277"][1]) 
		state["fRec279"] = -((fSlow29 * ((fSlow28 * fRec279_temp) - (fSlow24 * (state["fRec2"][1] - state["fRec2"][2]))))) 
		state["fRec278"] = state["fRec278"].at[0].set((state["fRec279"] - (fSlow27 * ((fSlow25 * state["fRec278"][2]) + (fSlow23 * state["fRec278"][1]))))) 
		state["fRec277"] = state["fRec277"].at[0].set(((fSlow57 * (state["fRec278"][2] + (state["fRec278"][0] - (jnp.float32(2.0) * state["fRec278"][1])))) - (fSlow56 * ((fSlow55 * state["fRec277"][2]) + fTemp127)))) 
		state["fRec276"] = state["fRec276"].at[0].set(((state["fRec277"][2] + (fSlow56 * (fTemp127 + (fSlow55 * state["fRec277"][0])))) - (fSlow51 * ((fSlow50 * state["fRec276"][2]) + fTemp126)))) 
		state["fRec275"] = state["fRec275"].at[0].set(((state["fRec276"][2] + (fSlow51 * (fTemp126 + (fSlow50 * state["fRec276"][0])))) - (fSlow46 * ((fSlow45 * state["fRec275"][2]) + fTemp125)))) 
		fTemp128 = ((((fSlow127 * (state["fRec275"][2] + (fSlow46 * (fTemp125 + (fSlow45 * state["fRec275"][0]))))) + (fSlow126 * (state["fRec271"][2] + (fSlow46 * (fTemp123 + (fSlow45 * state["fRec271"][0])))))) + (fSlow125 * (state["fRec268"][2] + (fSlow46 * (fTemp122 + (fSlow45 * state["fRec268"][0])))))) + (fSlow6 * ((fSlow124 * (state["fRec266"][2] + (state["fRec266"][0] - (jnp.float32(2.0) * state["fRec266"][1])))) + (fSlow123 * (state["fRec258"][2] + (state["fRec258"][0] + (jnp.float32(2.0) * state["fRec258"][1]))))))) 
		fTemp129 = (fTemp128 + fTemp118) 
		fTemp130 = (fTemp129 + fTemp108) 
		state["fRec287"] = -((fSlow29 * ((fSlow28 * fRec287_temp) - (state["fRec12"][1] + state["fRec12"][2])))) 
		state["fRec286"] = state["fRec286"].at[0].set((state["fRec287"] - (fSlow27 * ((fSlow25 * state["fRec286"][2]) + (fSlow23 * state["fRec286"][1]))))) 
		fTemp131 = (fSlow27 * (state["fRec286"][2] + (state["fRec286"][0] + (jnp.float32(2.0) * state["fRec286"][1])))) 
		state["fVec36"] = jnp.float32(fTemp131) 
		state["fRec285"] = -((fSlow32 * ((fSlow30 * fRec285_temp) - (fTemp131 + fVec36_temp)))) 
		state["fRec284"] = state["fRec284"].at[0].set((state["fRec285"] - (fSlow20 * ((fSlow18 * state["fRec284"][2]) + (fSlow16 * state["fRec284"][1]))))) 
		fTemp132 = (fSlow20 * (state["fRec284"][2] + (state["fRec284"][0] + (jnp.float32(2.0) * state["fRec284"][1])))) 
		state["fVec37"] = jnp.float32(fTemp132) 
		state["fRec283"] = -((fSlow35 * ((fSlow33 * fRec283_temp) - (fTemp132 + fVec37_temp)))) 
		state["fRec282"] = state["fRec282"].at[0].set((state["fRec283"] - (fSlow13 * ((fSlow11 * state["fRec282"][2]) + (fSlow9 * state["fRec282"][1]))))) 
		fTemp133 = (fSlow13 * (state["fRec282"][2] + (state["fRec282"][0] + (jnp.float32(2.0) * state["fRec282"][1])))) 
		state["fVec38"] = jnp.float32(fTemp133) 
		state["fRec281"] = -((fSlow38 * ((fSlow36 * fRec281_temp) - (fTemp133 + fVec38_temp)))) 
		state["fRec280"] = state["fRec280"].at[0].set((state["fRec281"] - (fSlow6 * ((fSlow5 * state["fRec280"][2]) + (fSlow3 * state["fRec280"][1]))))) 
		state["fRec289"] = -((fSlow38 * ((fSlow36 * fRec289_temp) - (fSlow4 * (fTemp133 - fVec38_temp))))) 
		state["fRec288"] = state["fRec288"].at[0].set((state["fRec289"] - (fSlow6 * ((fSlow5 * state["fRec288"][2]) + (fSlow3 * state["fRec288"][1]))))) 
		fTemp134 = (fSlow3 * state["fRec290"][1]) 
		state["fRec292"] = -((fSlow35 * ((fSlow33 * fRec292_temp) - (fSlow10 * (fTemp132 - fVec37_temp))))) 
		state["fRec291"] = state["fRec291"].at[0].set((state["fRec292"] - (fSlow13 * ((fSlow11 * state["fRec291"][2]) + (fSlow9 * state["fRec291"][1]))))) 
		state["fRec290"] = state["fRec290"].at[0].set(((fSlow47 * (state["fRec291"][2] + (state["fRec291"][0] - (jnp.float32(2.0) * state["fRec291"][1])))) - (fSlow46 * ((fSlow45 * state["fRec290"][2]) + fTemp134)))) 
		fTemp135 = (fSlow3 * state["fRec293"][1]) 
		fTemp136 = (fSlow9 * state["fRec294"][1]) 
		state["fRec296"] = -((fSlow32 * ((fSlow30 * fRec296_temp) - (fSlow17 * (fTemp131 - fVec36_temp))))) 
		state["fRec295"] = state["fRec295"].at[0].set((state["fRec296"] - (fSlow20 * ((fSlow18 * state["fRec295"][2]) + (fSlow16 * state["fRec295"][1]))))) 
		state["fRec294"] = state["fRec294"].at[0].set(((fSlow52 * (state["fRec295"][2] + (state["fRec295"][0] - (jnp.float32(2.0) * state["fRec295"][1])))) - (fSlow51 * ((fSlow50 * state["fRec294"][2]) + fTemp136)))) 
		state["fRec293"] = state["fRec293"].at[0].set(((state["fRec294"][2] + (fSlow51 * (fTemp136 + (fSlow50 * state["fRec294"][0])))) - (fSlow46 * ((fSlow45 * state["fRec293"][2]) + fTemp135)))) 
		fTemp137 = (fSlow3 * state["fRec297"][1]) 
		fTemp138 = (fSlow9 * state["fRec298"][1]) 
		fTemp139 = (fSlow16 * state["fRec299"][1]) 
		state["fRec301"] = -((fSlow29 * ((fSlow28 * fRec301_temp) - (fSlow24 * (state["fRec12"][1] - state["fRec12"][2]))))) 
		state["fRec300"] = state["fRec300"].at[0].set((state["fRec301"] - (fSlow27 * ((fSlow25 * state["fRec300"][2]) + (fSlow23 * state["fRec300"][1]))))) 
		state["fRec299"] = state["fRec299"].at[0].set(((fSlow57 * (state["fRec300"][2] + (state["fRec300"][0] - (jnp.float32(2.0) * state["fRec300"][1])))) - (fSlow56 * ((fSlow55 * state["fRec299"][2]) + fTemp139)))) 
		state["fRec298"] = state["fRec298"].at[0].set(((state["fRec299"][2] + (fSlow56 * (fTemp139 + (fSlow55 * state["fRec299"][0])))) - (fSlow51 * ((fSlow50 * state["fRec298"][2]) + fTemp138)))) 
		state["fRec297"] = state["fRec297"].at[0].set(((state["fRec298"][2] + (fSlow51 * (fTemp138 + (fSlow50 * state["fRec298"][0])))) - (fSlow46 * ((fSlow45 * state["fRec297"][2]) + fTemp137)))) 
		fTemp140 = ((((fSlow133 * (state["fRec297"][2] + (fSlow46 * (fTemp137 + (fSlow45 * state["fRec297"][0]))))) + (fSlow132 * (state["fRec293"][2] + (fSlow46 * (fTemp135 + (fSlow45 * state["fRec293"][0])))))) + (fSlow131 * (state["fRec290"][2] + (fSlow46 * (fTemp134 + (fSlow45 * state["fRec290"][0])))))) + (fSlow6 * ((fSlow130 * (state["fRec288"][2] + (state["fRec288"][0] - (jnp.float32(2.0) * state["fRec288"][1])))) + (fSlow129 * (state["fRec280"][2] + (state["fRec280"][0] + (jnp.float32(2.0) * state["fRec280"][1]))))))) 
		state["fRec309"] = -((fSlow29 * ((fSlow28 * fRec309_temp) - (state["fRec4"][1] + state["fRec4"][2])))) 
		state["fRec308"] = state["fRec308"].at[0].set((state["fRec309"] - (fSlow27 * ((fSlow25 * state["fRec308"][2]) + (fSlow23 * state["fRec308"][1]))))) 
		fTemp141 = (fSlow27 * (state["fRec308"][2] + (state["fRec308"][0] + (jnp.float32(2.0) * state["fRec308"][1])))) 
		state["fVec39"] = jnp.float32(fTemp141) 
		state["fRec307"] = -((fSlow32 * ((fSlow30 * fRec307_temp) - (fTemp141 + fVec39_temp)))) 
		state["fRec306"] = state["fRec306"].at[0].set((state["fRec307"] - (fSlow20 * ((fSlow18 * state["fRec306"][2]) + (fSlow16 * state["fRec306"][1]))))) 
		fTemp142 = (fSlow20 * (state["fRec306"][2] + (state["fRec306"][0] + (jnp.float32(2.0) * state["fRec306"][1])))) 
		state["fVec40"] = jnp.float32(fTemp142) 
		state["fRec305"] = -((fSlow35 * ((fSlow33 * fRec305_temp) - (fTemp142 + fVec40_temp)))) 
		state["fRec304"] = state["fRec304"].at[0].set((state["fRec305"] - (fSlow13 * ((fSlow11 * state["fRec304"][2]) + (fSlow9 * state["fRec304"][1]))))) 
		fTemp143 = (fSlow13 * (state["fRec304"][2] + (state["fRec304"][0] + (jnp.float32(2.0) * state["fRec304"][1])))) 
		state["fVec41"] = jnp.float32(fTemp143) 
		state["fRec303"] = -((fSlow38 * ((fSlow36 * fRec303_temp) - (fTemp143 + fVec41_temp)))) 
		state["fRec302"] = state["fRec302"].at[0].set((state["fRec303"] - (fSlow6 * ((fSlow5 * state["fRec302"][2]) + (fSlow3 * state["fRec302"][1]))))) 
		state["fRec311"] = -((fSlow38 * ((fSlow36 * fRec311_temp) - (fSlow4 * (fTemp143 - fVec41_temp))))) 
		state["fRec310"] = state["fRec310"].at[0].set((state["fRec311"] - (fSlow6 * ((fSlow5 * state["fRec310"][2]) + (fSlow3 * state["fRec310"][1]))))) 
		fTemp144 = (fSlow3 * state["fRec312"][1]) 
		state["fRec314"] = -((fSlow35 * ((fSlow33 * fRec314_temp) - (fSlow10 * (fTemp142 - fVec40_temp))))) 
		state["fRec313"] = state["fRec313"].at[0].set((state["fRec314"] - (fSlow13 * ((fSlow11 * state["fRec313"][2]) + (fSlow9 * state["fRec313"][1]))))) 
		state["fRec312"] = state["fRec312"].at[0].set(((fSlow47 * (state["fRec313"][2] + (state["fRec313"][0] - (jnp.float32(2.0) * state["fRec313"][1])))) - (fSlow46 * ((fSlow45 * state["fRec312"][2]) + fTemp144)))) 
		fTemp145 = (fSlow3 * state["fRec315"][1]) 
		fTemp146 = (fSlow9 * state["fRec316"][1]) 
		state["fRec318"] = -((fSlow32 * ((fSlow30 * fRec318_temp) - (fSlow17 * (fTemp141 - fVec39_temp))))) 
		state["fRec317"] = state["fRec317"].at[0].set((state["fRec318"] - (fSlow20 * ((fSlow18 * state["fRec317"][2]) + (fSlow16 * state["fRec317"][1]))))) 
		state["fRec316"] = state["fRec316"].at[0].set(((fSlow52 * (state["fRec317"][2] + (state["fRec317"][0] - (jnp.float32(2.0) * state["fRec317"][1])))) - (fSlow51 * ((fSlow50 * state["fRec316"][2]) + fTemp146)))) 
		state["fRec315"] = state["fRec315"].at[0].set(((state["fRec316"][2] + (fSlow51 * (fTemp146 + (fSlow50 * state["fRec316"][0])))) - (fSlow46 * ((fSlow45 * state["fRec315"][2]) + fTemp145)))) 
		fTemp147 = (fSlow3 * state["fRec319"][1]) 
		fTemp148 = (fSlow9 * state["fRec320"][1]) 
		fTemp149 = (fSlow16 * state["fRec321"][1]) 
		state["fRec323"] = -((fSlow29 * ((fSlow28 * fRec323_temp) - (fSlow24 * (state["fRec4"][1] - state["fRec4"][2]))))) 
		state["fRec322"] = state["fRec322"].at[0].set((state["fRec323"] - (fSlow27 * ((fSlow25 * state["fRec322"][2]) + (fSlow23 * state["fRec322"][1]))))) 
		state["fRec321"] = state["fRec321"].at[0].set(((fSlow57 * (state["fRec322"][2] + (state["fRec322"][0] - (jnp.float32(2.0) * state["fRec322"][1])))) - (fSlow56 * ((fSlow55 * state["fRec321"][2]) + fTemp149)))) 
		state["fRec320"] = state["fRec320"].at[0].set(((state["fRec321"][2] + (fSlow56 * (fTemp149 + (fSlow55 * state["fRec321"][0])))) - (fSlow51 * ((fSlow50 * state["fRec320"][2]) + fTemp148)))) 
		state["fRec319"] = state["fRec319"].at[0].set(((state["fRec320"][2] + (fSlow51 * (fTemp148 + (fSlow50 * state["fRec320"][0])))) - (fSlow46 * ((fSlow45 * state["fRec319"][2]) + fTemp147)))) 
		fTemp150 = ((((fSlow139 * (state["fRec319"][2] + (fSlow46 * (fTemp147 + (fSlow45 * state["fRec319"][0]))))) + (fSlow138 * (state["fRec315"][2] + (fSlow46 * (fTemp145 + (fSlow45 * state["fRec315"][0])))))) + (fSlow137 * (state["fRec312"][2] + (fSlow46 * (fTemp144 + (fSlow45 * state["fRec312"][0])))))) + (fSlow6 * ((fSlow136 * (state["fRec310"][2] + (state["fRec310"][0] - (jnp.float32(2.0) * state["fRec310"][1])))) + (fSlow135 * (state["fRec302"][2] + (state["fRec302"][0] + (jnp.float32(2.0) * state["fRec302"][1]))))))) 
		fTemp151 = (fTemp150 + fTemp140) 
		state["fRec331"] = -((fSlow29 * ((fSlow28 * fRec331_temp) - (state["fRec8"][1] + state["fRec8"][2])))) 
		state["fRec330"] = state["fRec330"].at[0].set((state["fRec331"] - (fSlow27 * ((fSlow25 * state["fRec330"][2]) + (fSlow23 * state["fRec330"][1]))))) 
		fTemp152 = (fSlow27 * (state["fRec330"][2] + (state["fRec330"][0] + (jnp.float32(2.0) * state["fRec330"][1])))) 
		state["fVec42"] = jnp.float32(fTemp152) 
		state["fRec329"] = -((fSlow32 * ((fSlow30 * fRec329_temp) - (fTemp152 + fVec42_temp)))) 
		state["fRec328"] = state["fRec328"].at[0].set((state["fRec329"] - (fSlow20 * ((fSlow18 * state["fRec328"][2]) + (fSlow16 * state["fRec328"][1]))))) 
		fTemp153 = (fSlow20 * (state["fRec328"][2] + (state["fRec328"][0] + (jnp.float32(2.0) * state["fRec328"][1])))) 
		state["fVec43"] = jnp.float32(fTemp153) 
		state["fRec327"] = -((fSlow35 * ((fSlow33 * fRec327_temp) - (fTemp153 + fVec43_temp)))) 
		state["fRec326"] = state["fRec326"].at[0].set((state["fRec327"] - (fSlow13 * ((fSlow11 * state["fRec326"][2]) + (fSlow9 * state["fRec326"][1]))))) 
		fTemp154 = (fSlow13 * (state["fRec326"][2] + (state["fRec326"][0] + (jnp.float32(2.0) * state["fRec326"][1])))) 
		state["fVec44"] = jnp.float32(fTemp154) 
		state["fRec325"] = -((fSlow38 * ((fSlow36 * fRec325_temp) - (fTemp154 + fVec44_temp)))) 
		state["fRec324"] = state["fRec324"].at[0].set((state["fRec325"] - (fSlow6 * ((fSlow5 * state["fRec324"][2]) + (fSlow3 * state["fRec324"][1]))))) 
		state["fRec333"] = -((fSlow38 * ((fSlow36 * fRec333_temp) - (fSlow4 * (fTemp154 - fVec44_temp))))) 
		state["fRec332"] = state["fRec332"].at[0].set((state["fRec333"] - (fSlow6 * ((fSlow5 * state["fRec332"][2]) + (fSlow3 * state["fRec332"][1]))))) 
		fTemp155 = (fSlow3 * state["fRec334"][1]) 
		state["fRec336"] = -((fSlow35 * ((fSlow33 * fRec336_temp) - (fSlow10 * (fTemp153 - fVec43_temp))))) 
		state["fRec335"] = state["fRec335"].at[0].set((state["fRec336"] - (fSlow13 * ((fSlow11 * state["fRec335"][2]) + (fSlow9 * state["fRec335"][1]))))) 
		state["fRec334"] = state["fRec334"].at[0].set(((fSlow47 * (state["fRec335"][2] + (state["fRec335"][0] - (jnp.float32(2.0) * state["fRec335"][1])))) - (fSlow46 * ((fSlow45 * state["fRec334"][2]) + fTemp155)))) 
		fTemp156 = (fSlow3 * state["fRec337"][1]) 
		fTemp157 = (fSlow9 * state["fRec338"][1]) 
		state["fRec340"] = -((fSlow32 * ((fSlow30 * fRec340_temp) - (fSlow17 * (fTemp152 - fVec42_temp))))) 
		state["fRec339"] = state["fRec339"].at[0].set((state["fRec340"] - (fSlow20 * ((fSlow18 * state["fRec339"][2]) + (fSlow16 * state["fRec339"][1]))))) 
		state["fRec338"] = state["fRec338"].at[0].set(((fSlow52 * (state["fRec339"][2] + (state["fRec339"][0] - (jnp.float32(2.0) * state["fRec339"][1])))) - (fSlow51 * ((fSlow50 * state["fRec338"][2]) + fTemp157)))) 
		state["fRec337"] = state["fRec337"].at[0].set(((state["fRec338"][2] + (fSlow51 * (fTemp157 + (fSlow50 * state["fRec338"][0])))) - (fSlow46 * ((fSlow45 * state["fRec337"][2]) + fTemp156)))) 
		fTemp158 = (fSlow3 * state["fRec341"][1]) 
		fTemp159 = (fSlow9 * state["fRec342"][1]) 
		fTemp160 = (fSlow16 * state["fRec343"][1]) 
		state["fRec345"] = -((fSlow29 * ((fSlow28 * fRec345_temp) - (fSlow24 * (state["fRec8"][1] - state["fRec8"][2]))))) 
		state["fRec344"] = state["fRec344"].at[0].set((state["fRec345"] - (fSlow27 * ((fSlow25 * state["fRec344"][2]) + (fSlow23 * state["fRec344"][1]))))) 
		state["fRec343"] = state["fRec343"].at[0].set(((fSlow57 * (state["fRec344"][2] + (state["fRec344"][0] - (jnp.float32(2.0) * state["fRec344"][1])))) - (fSlow56 * ((fSlow55 * state["fRec343"][2]) + fTemp160)))) 
		state["fRec342"] = state["fRec342"].at[0].set(((state["fRec343"][2] + (fSlow56 * (fTemp160 + (fSlow55 * state["fRec343"][0])))) - (fSlow51 * ((fSlow50 * state["fRec342"][2]) + fTemp159)))) 
		state["fRec341"] = state["fRec341"].at[0].set(((state["fRec342"][2] + (fSlow51 * (fTemp159 + (fSlow50 * state["fRec342"][0])))) - (fSlow46 * ((fSlow45 * state["fRec341"][2]) + fTemp158)))) 
		fTemp161 = ((((fSlow145 * (state["fRec341"][2] + (fSlow46 * (fTemp158 + (fSlow45 * state["fRec341"][0]))))) + (fSlow144 * (state["fRec337"][2] + (fSlow46 * (fTemp156 + (fSlow45 * state["fRec337"][0])))))) + (fSlow143 * (state["fRec334"][2] + (fSlow46 * (fTemp155 + (fSlow45 * state["fRec334"][0])))))) + (fSlow6 * ((fSlow142 * (state["fRec332"][2] + (state["fRec332"][0] - (jnp.float32(2.0) * state["fRec332"][1])))) + (fSlow141 * (state["fRec324"][2] + (state["fRec324"][0] + (jnp.float32(2.0) * state["fRec324"][1]))))))) 
		state["fRec353"] = -((fSlow29 * ((fSlow28 * fRec353_temp) - (state["fRec0"][1] + state["fRec0"][2])))) 
		state["fRec352"] = state["fRec352"].at[0].set((state["fRec353"] - (fSlow27 * ((fSlow25 * state["fRec352"][2]) + (fSlow23 * state["fRec352"][1]))))) 
		fTemp162 = (fSlow27 * (state["fRec352"][2] + (state["fRec352"][0] + (jnp.float32(2.0) * state["fRec352"][1])))) 
		state["fVec45"] = jnp.float32(fTemp162) 
		state["fRec351"] = -((fSlow32 * ((fSlow30 * fRec351_temp) - (fTemp162 + fVec45_temp)))) 
		state["fRec350"] = state["fRec350"].at[0].set((state["fRec351"] - (fSlow20 * ((fSlow18 * state["fRec350"][2]) + (fSlow16 * state["fRec350"][1]))))) 
		fTemp163 = (fSlow20 * (state["fRec350"][2] + (state["fRec350"][0] + (jnp.float32(2.0) * state["fRec350"][1])))) 
		state["fVec46"] = jnp.float32(fTemp163) 
		state["fRec349"] = -((fSlow35 * ((fSlow33 * fRec349_temp) - (fTemp163 + fVec46_temp)))) 
		state["fRec348"] = state["fRec348"].at[0].set((state["fRec349"] - (fSlow13 * ((fSlow11 * state["fRec348"][2]) + (fSlow9 * state["fRec348"][1]))))) 
		fTemp164 = (fSlow13 * (state["fRec348"][2] + (state["fRec348"][0] + (jnp.float32(2.0) * state["fRec348"][1])))) 
		state["fVec47"] = jnp.float32(fTemp164) 
		state["fRec347"] = -((fSlow38 * ((fSlow36 * fRec347_temp) - (fTemp164 + fVec47_temp)))) 
		state["fRec346"] = state["fRec346"].at[0].set((state["fRec347"] - (fSlow6 * ((fSlow5 * state["fRec346"][2]) + (fSlow3 * state["fRec346"][1]))))) 
		state["fRec355"] = -((fSlow38 * ((fSlow36 * fRec355_temp) - (fSlow4 * (fTemp164 - fVec47_temp))))) 
		state["fRec354"] = state["fRec354"].at[0].set((state["fRec355"] - (fSlow6 * ((fSlow5 * state["fRec354"][2]) + (fSlow3 * state["fRec354"][1]))))) 
		fTemp165 = (fSlow3 * state["fRec356"][1]) 
		state["fRec358"] = -((fSlow35 * ((fSlow33 * fRec358_temp) - (fSlow10 * (fTemp163 - fVec46_temp))))) 
		state["fRec357"] = state["fRec357"].at[0].set((state["fRec358"] - (fSlow13 * ((fSlow11 * state["fRec357"][2]) + (fSlow9 * state["fRec357"][1]))))) 
		state["fRec356"] = state["fRec356"].at[0].set(((fSlow47 * (state["fRec357"][2] + (state["fRec357"][0] - (jnp.float32(2.0) * state["fRec357"][1])))) - (fSlow46 * ((fSlow45 * state["fRec356"][2]) + fTemp165)))) 
		fTemp166 = (fSlow3 * state["fRec359"][1]) 
		fTemp167 = (fSlow9 * state["fRec360"][1]) 
		state["fRec362"] = -((fSlow32 * ((fSlow30 * fRec362_temp) - (fSlow17 * (fTemp162 - fVec45_temp))))) 
		state["fRec361"] = state["fRec361"].at[0].set((state["fRec362"] - (fSlow20 * ((fSlow18 * state["fRec361"][2]) + (fSlow16 * state["fRec361"][1]))))) 
		state["fRec360"] = state["fRec360"].at[0].set(((fSlow52 * (state["fRec361"][2] + (state["fRec361"][0] - (jnp.float32(2.0) * state["fRec361"][1])))) - (fSlow51 * ((fSlow50 * state["fRec360"][2]) + fTemp167)))) 
		state["fRec359"] = state["fRec359"].at[0].set(((state["fRec360"][2] + (fSlow51 * (fTemp167 + (fSlow50 * state["fRec360"][0])))) - (fSlow46 * ((fSlow45 * state["fRec359"][2]) + fTemp166)))) 
		fTemp168 = (fSlow3 * state["fRec363"][1]) 
		fTemp169 = (fSlow9 * state["fRec364"][1]) 
		fTemp170 = (fSlow16 * state["fRec365"][1]) 
		state["fRec367"] = -((fSlow29 * ((fSlow28 * fRec367_temp) - (fSlow24 * (state["fRec0"][1] - state["fRec0"][2]))))) 
		state["fRec366"] = state["fRec366"].at[0].set((state["fRec367"] - (fSlow27 * ((fSlow25 * state["fRec366"][2]) + (fSlow23 * state["fRec366"][1]))))) 
		state["fRec365"] = state["fRec365"].at[0].set(((fSlow57 * (state["fRec366"][2] + (state["fRec366"][0] - (jnp.float32(2.0) * state["fRec366"][1])))) - (fSlow56 * ((fSlow55 * state["fRec365"][2]) + fTemp170)))) 
		state["fRec364"] = state["fRec364"].at[0].set(((state["fRec365"][2] + (fSlow56 * (fTemp170 + (fSlow55 * state["fRec365"][0])))) - (fSlow51 * ((fSlow50 * state["fRec364"][2]) + fTemp169)))) 
		state["fRec363"] = state["fRec363"].at[0].set(((state["fRec364"][2] + (fSlow51 * (fTemp169 + (fSlow50 * state["fRec364"][0])))) - (fSlow46 * ((fSlow45 * state["fRec363"][2]) + fTemp168)))) 
		fTemp171 = ((((fSlow151 * (state["fRec363"][2] + (fSlow46 * (fTemp168 + (fSlow45 * state["fRec363"][0]))))) + (fSlow150 * (state["fRec359"][2] + (fSlow46 * (fTemp166 + (fSlow45 * state["fRec359"][0])))))) + (fSlow149 * (state["fRec356"][2] + (fSlow46 * (fTemp165 + (fSlow45 * state["fRec356"][0])))))) + (fSlow6 * ((fSlow148 * (state["fRec354"][2] + (state["fRec354"][0] - (jnp.float32(2.0) * state["fRec354"][1])))) + (fSlow147 * (state["fRec346"][2] + (state["fRec346"][0] + (jnp.float32(2.0) * state["fRec346"][1]))))))) 
		fTemp172 = (fTemp171 + fTemp161) 
		fTemp173 = (fTemp172 + fTemp151) 
		fTemp174 = (fTemp173 + fTemp130) 
		state["fVec48"] = fSlow153 
		iTemp175 = ((fSlow153 - fVec48_temp) > jnp.float32(0.0)).astype(jnp.int32) 
		state["fVec49"] = fSlow154 
		fTemp176 = ((((fSlow154 - fVec49_temp) > jnp.float32(0.0)).astype(jnp.int32) + iTemp175)) 
		state["iRec369"] = ((jnp.int32(1103515245) * iRec369_temp) + jnp.int32(12345)) 
		state["fRec368"] = state["fRec368"].at[0].set((((jnp.float32(0.5221894) * state["fRec368"][3]) + ((jnp.float32(4.656613e-10) * (state["iRec369"])) + (jnp.float32(2.494956) * state["fRec368"][1]))) - (jnp.float32(2.0172658) * state["fRec368"][2]))) 
		fTemp177 = (fSlow155 * (((jnp.float32(0.049922034) * state["fRec368"][0]) + (jnp.float32(0.0506127) * state["fRec368"][2])) - ((jnp.float32(0.095993534) * state["fRec368"][1]) + (jnp.float32(0.004408786) * state["fRec368"][3])))) 
		state["fVec50"] = state["fVec50"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp177 + (fTemp176 + ((fSlow152 * (fTemp174 + fTemp87)) + fTemp0)))) 
		state["fRec0"] = state["fRec0"].at[0].set(state["fVec50"][((state["IOTA0"] - iSlow156) & 8191).astype(jnp.int32)]) 
		fTemp178 = (fSlow0 * inputs[1]) 
		state["fVec51"] = fSlow157 
		fTemp179 = ((iTemp175 + ((fSlow157 - fVec51_temp) > jnp.float32(0.0)).astype(jnp.int32))) 
		state["fVec52"] = state["fVec52"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp179 + (fTemp178 + (fTemp177 + (fSlow152 * (fTemp174 - fTemp87)))))) 
		state["fRec1"] = state["fRec1"].at[0].set(state["fVec52"][((state["IOTA0"] - iSlow158) & 8191).astype(jnp.int32)]) 
		fTemp180 = (fTemp86 - fTemp43) 
		fTemp181 = (fTemp173 - fTemp130) 
		fTemp182 = ((fTemp0 + fTemp176) + fTemp177) 
		state["fVec53"] = state["fVec53"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp181 + fTemp180)))) 
		state["fRec2"] = state["fRec2"].at[0].set(state["fVec53"][((state["IOTA0"] - iSlow159) & 8191).astype(jnp.int32)]) 
		fTemp183 = (fTemp179 + (fTemp177 + fTemp178)) 
		state["fVec54"] = state["fVec54"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp181 - fTemp180)))) 
		state["fRec3"] = state["fRec3"].at[0].set(state["fVec54"][((state["IOTA0"] - iSlow160) & 8191).astype(jnp.int32)]) 
		fTemp184 = (fTemp42 - fTemp21) 
		fTemp185 = (fTemp85 - fTemp64) 
		fTemp186 = (fTemp185 + fTemp184) 
		fTemp187 = (fTemp129 - fTemp108) 
		fTemp188 = (fTemp172 - fTemp151) 
		fTemp189 = (fTemp188 + fTemp187) 
		state["fVec55"] = state["fVec55"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp189 + fTemp186)))) 
		state["fRec4"] = state["fRec4"].at[0].set(state["fVec55"][((state["IOTA0"] - iSlow161) & 8191).astype(jnp.int32)]) 
		state["fVec56"] = state["fVec56"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp189 - fTemp186)))) 
		state["fRec5"] = state["fRec5"].at[0].set(state["fVec56"][((state["IOTA0"] - iSlow162) & 8191).astype(jnp.int32)]) 
		fTemp190 = (fTemp185 - fTemp184) 
		fTemp191 = (fTemp188 - fTemp187) 
		state["fVec57"] = state["fVec57"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp191 + fTemp190)))) 
		state["fRec6"] = state["fRec6"].at[0].set(state["fVec57"][((state["IOTA0"] - iSlow163) & 8191).astype(jnp.int32)]) 
		state["fVec58"] = state["fVec58"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp191 - fTemp190)))) 
		state["fRec7"] = state["fRec7"].at[0].set(state["fVec58"][((state["IOTA0"] - iSlow164) & 8191).astype(jnp.int32)]) 
		fTemp192 = (fTemp20 - fTemp10) 
		fTemp193 = (fTemp41 - fTemp31) 
		fTemp194 = (fTemp193 + fTemp192) 
		fTemp195 = (fTemp63 - fTemp53) 
		fTemp196 = (fTemp84 - fTemp74) 
		fTemp197 = (fTemp196 + fTemp195) 
		fTemp198 = (fTemp197 + fTemp194) 
		fTemp199 = (fTemp107 - fTemp97) 
		fTemp200 = (fTemp128 - fTemp118) 
		fTemp201 = (fTemp200 + fTemp199) 
		fTemp202 = (fTemp150 - fTemp140) 
		fTemp203 = (fTemp171 - fTemp161) 
		fTemp204 = (fTemp203 + fTemp202) 
		fTemp205 = (fTemp204 + fTemp201) 
		state["fVec59"] = state["fVec59"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp205 + fTemp198)))) 
		state["fRec8"] = state["fRec8"].at[0].set(state["fVec59"][((state["IOTA0"] - iSlow165) & 8191).astype(jnp.int32)]) 
		state["fVec60"] = state["fVec60"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp205 - fTemp198)))) 
		state["fRec9"] = state["fRec9"].at[0].set(state["fVec60"][((state["IOTA0"] - iSlow166) & 8191).astype(jnp.int32)]) 
		fTemp206 = (fTemp197 - fTemp194) 
		fTemp207 = (fTemp204 - fTemp201) 
		state["fVec61"] = state["fVec61"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp207 + fTemp206)))) 
		state["fRec10"] = state["fRec10"].at[0].set(state["fVec61"][((state["IOTA0"] - iSlow167) & 8191).astype(jnp.int32)]) 
		state["fVec62"] = state["fVec62"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp207 - fTemp206)))) 
		state["fRec11"] = state["fRec11"].at[0].set(state["fVec62"][((state["IOTA0"] - iSlow168) & 8191).astype(jnp.int32)]) 
		fTemp208 = (fTemp193 - fTemp192) 
		fTemp209 = (fTemp196 - fTemp195) 
		fTemp210 = (fTemp209 + fTemp208) 
		fTemp211 = (fTemp200 - fTemp199) 
		fTemp212 = (fTemp203 - fTemp202) 
		fTemp213 = (fTemp212 + fTemp211) 
		state["fVec63"] = state["fVec63"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp213 + fTemp210)))) 
		state["fRec12"] = state["fRec12"].at[0].set(state["fVec63"][((state["IOTA0"] - iSlow169) & 8191).astype(jnp.int32)]) 
		state["fVec64"] = state["fVec64"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp213 - fTemp210)))) 
		state["fRec13"] = state["fRec13"].at[0].set(state["fVec64"][((state["IOTA0"] - iSlow170) & 8191).astype(jnp.int32)]) 
		fTemp214 = (fTemp209 - fTemp208) 
		fTemp215 = (fTemp212 - fTemp211) 
		state["fVec65"] = state["fVec65"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp182 + (fSlow152 * (fTemp215 + fTemp214)))) 
		state["fRec14"] = state["fRec14"].at[0].set(state["fVec65"][((state["IOTA0"] - iSlow171) & 8191).astype(jnp.int32)]) 
		state["fVec66"] = state["fVec66"].at[(state["IOTA0"] & 8191).astype(jnp.int32)].set((fTemp183 + (fSlow152 * (fTemp215 - fTemp214)))) 
		state["fRec15"] = state["fRec15"].at[0].set(state["fVec66"][((state["IOTA0"] - iSlow172) & 8191).astype(jnp.int32)]) 
		_result0 = (fSlow173 * (((((((state["fRec0"][0] + state["fRec2"][0]) + state["fRec4"][0]) + state["fRec6"][0]) + state["fRec8"][0]) + state["fRec10"][0]) + state["fRec12"][0]) + state["fRec14"][0])) 
		_result1 = (fSlow173 * (((((((state["fRec1"][0] + state["fRec3"][0]) + state["fRec5"][0]) + state["fRec7"][0]) + state["fRec9"][0]) + state["fRec11"][0]) + state["fRec13"][0]) + state["fRec15"][0])) 
		state["fRec22"] = jnp.roll(state["fRec22"], 1) 
		state["fRec20"] = jnp.roll(state["fRec20"], 1) 
		state["fRec18"] = jnp.roll(state["fRec18"], 1) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec24"] = jnp.roll(state["fRec24"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec26"] = jnp.roll(state["fRec26"], 1) 
		state["fRec31"] = jnp.roll(state["fRec31"], 1) 
		state["fRec30"] = jnp.roll(state["fRec30"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec36"] = jnp.roll(state["fRec36"], 1) 
		state["fRec35"] = jnp.roll(state["fRec35"], 1) 
		state["fRec34"] = jnp.roll(state["fRec34"], 1) 
		state["fRec33"] = jnp.roll(state["fRec33"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
		state["fRec42"] = jnp.roll(state["fRec42"], 1) 
		state["fRec40"] = jnp.roll(state["fRec40"], 1) 
		state["fRec38"] = jnp.roll(state["fRec38"], 1) 
		state["fRec46"] = jnp.roll(state["fRec46"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec48"] = jnp.roll(state["fRec48"], 1) 
		state["fRec53"] = jnp.roll(state["fRec53"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec58"] = jnp.roll(state["fRec58"], 1) 
		state["fRec57"] = jnp.roll(state["fRec57"], 1) 
		state["fRec56"] = jnp.roll(state["fRec56"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
		state["fRec66"] = jnp.roll(state["fRec66"], 1) 
		state["fRec64"] = jnp.roll(state["fRec64"], 1) 
		state["fRec62"] = jnp.roll(state["fRec62"], 1) 
		state["fRec60"] = jnp.roll(state["fRec60"], 1) 
		state["fRec68"] = jnp.roll(state["fRec68"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec70"] = jnp.roll(state["fRec70"], 1) 
		state["fRec75"] = jnp.roll(state["fRec75"], 1) 
		state["fRec74"] = jnp.roll(state["fRec74"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec88"] = jnp.roll(state["fRec88"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec84"] = jnp.roll(state["fRec84"], 1) 
		state["fRec82"] = jnp.roll(state["fRec82"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec92"] = jnp.roll(state["fRec92"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec96"] = jnp.roll(state["fRec96"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec102"] = jnp.roll(state["fRec102"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec110"] = jnp.roll(state["fRec110"], 1) 
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec112"] = jnp.roll(state["fRec112"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec114"] = jnp.roll(state["fRec114"], 1) 
		state["fRec119"] = jnp.roll(state["fRec119"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec124"] = jnp.roll(state["fRec124"], 1) 
		state["fRec123"] = jnp.roll(state["fRec123"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec132"] = jnp.roll(state["fRec132"], 1) 
		state["fRec130"] = jnp.roll(state["fRec130"], 1) 
		state["fRec128"] = jnp.roll(state["fRec128"], 1) 
		state["fRec126"] = jnp.roll(state["fRec126"], 1) 
		state["fRec134"] = jnp.roll(state["fRec134"], 1) 
		state["fRec137"] = jnp.roll(state["fRec137"], 1) 
		state["fRec136"] = jnp.roll(state["fRec136"], 1) 
		state["fRec141"] = jnp.roll(state["fRec141"], 1) 
		state["fRec140"] = jnp.roll(state["fRec140"], 1) 
		state["fRec139"] = jnp.roll(state["fRec139"], 1) 
		state["fRec146"] = jnp.roll(state["fRec146"], 1) 
		state["fRec145"] = jnp.roll(state["fRec145"], 1) 
		state["fRec144"] = jnp.roll(state["fRec144"], 1) 
		state["fRec143"] = jnp.roll(state["fRec143"], 1) 
		state["fRec154"] = jnp.roll(state["fRec154"], 1) 
		state["fRec152"] = jnp.roll(state["fRec152"], 1) 
		state["fRec150"] = jnp.roll(state["fRec150"], 1) 
		state["fRec148"] = jnp.roll(state["fRec148"], 1) 
		state["fRec156"] = jnp.roll(state["fRec156"], 1) 
		state["fRec159"] = jnp.roll(state["fRec159"], 1) 
		state["fRec158"] = jnp.roll(state["fRec158"], 1) 
		state["fRec163"] = jnp.roll(state["fRec163"], 1) 
		state["fRec162"] = jnp.roll(state["fRec162"], 1) 
		state["fRec161"] = jnp.roll(state["fRec161"], 1) 
		state["fRec168"] = jnp.roll(state["fRec168"], 1) 
		state["fRec167"] = jnp.roll(state["fRec167"], 1) 
		state["fRec166"] = jnp.roll(state["fRec166"], 1) 
		state["fRec165"] = jnp.roll(state["fRec165"], 1) 
		state["fRec176"] = jnp.roll(state["fRec176"], 1) 
		state["fRec174"] = jnp.roll(state["fRec174"], 1) 
		state["fRec172"] = jnp.roll(state["fRec172"], 1) 
		state["fRec170"] = jnp.roll(state["fRec170"], 1) 
		state["fRec178"] = jnp.roll(state["fRec178"], 1) 
		state["fRec181"] = jnp.roll(state["fRec181"], 1) 
		state["fRec180"] = jnp.roll(state["fRec180"], 1) 
		state["fRec185"] = jnp.roll(state["fRec185"], 1) 
		state["fRec184"] = jnp.roll(state["fRec184"], 1) 
		state["fRec183"] = jnp.roll(state["fRec183"], 1) 
		state["fRec190"] = jnp.roll(state["fRec190"], 1) 
		state["fRec189"] = jnp.roll(state["fRec189"], 1) 
		state["fRec188"] = jnp.roll(state["fRec188"], 1) 
		state["fRec187"] = jnp.roll(state["fRec187"], 1) 
		state["fRec198"] = jnp.roll(state["fRec198"], 1) 
		state["fRec196"] = jnp.roll(state["fRec196"], 1) 
		state["fRec194"] = jnp.roll(state["fRec194"], 1) 
		state["fRec192"] = jnp.roll(state["fRec192"], 1) 
		state["fRec200"] = jnp.roll(state["fRec200"], 1) 
		state["fRec203"] = jnp.roll(state["fRec203"], 1) 
		state["fRec202"] = jnp.roll(state["fRec202"], 1) 
		state["fRec207"] = jnp.roll(state["fRec207"], 1) 
		state["fRec206"] = jnp.roll(state["fRec206"], 1) 
		state["fRec205"] = jnp.roll(state["fRec205"], 1) 
		state["fRec212"] = jnp.roll(state["fRec212"], 1) 
		state["fRec211"] = jnp.roll(state["fRec211"], 1) 
		state["fRec210"] = jnp.roll(state["fRec210"], 1) 
		state["fRec209"] = jnp.roll(state["fRec209"], 1) 
		state["fRec220"] = jnp.roll(state["fRec220"], 1) 
		state["fRec218"] = jnp.roll(state["fRec218"], 1) 
		state["fRec216"] = jnp.roll(state["fRec216"], 1) 
		state["fRec214"] = jnp.roll(state["fRec214"], 1) 
		state["fRec222"] = jnp.roll(state["fRec222"], 1) 
		state["fRec225"] = jnp.roll(state["fRec225"], 1) 
		state["fRec224"] = jnp.roll(state["fRec224"], 1) 
		state["fRec229"] = jnp.roll(state["fRec229"], 1) 
		state["fRec228"] = jnp.roll(state["fRec228"], 1) 
		state["fRec227"] = jnp.roll(state["fRec227"], 1) 
		state["fRec234"] = jnp.roll(state["fRec234"], 1) 
		state["fRec233"] = jnp.roll(state["fRec233"], 1) 
		state["fRec232"] = jnp.roll(state["fRec232"], 1) 
		state["fRec231"] = jnp.roll(state["fRec231"], 1) 
		state["fRec242"] = jnp.roll(state["fRec242"], 1) 
		state["fRec240"] = jnp.roll(state["fRec240"], 1) 
		state["fRec238"] = jnp.roll(state["fRec238"], 1) 
		state["fRec236"] = jnp.roll(state["fRec236"], 1) 
		state["fRec244"] = jnp.roll(state["fRec244"], 1) 
		state["fRec247"] = jnp.roll(state["fRec247"], 1) 
		state["fRec246"] = jnp.roll(state["fRec246"], 1) 
		state["fRec251"] = jnp.roll(state["fRec251"], 1) 
		state["fRec250"] = jnp.roll(state["fRec250"], 1) 
		state["fRec249"] = jnp.roll(state["fRec249"], 1) 
		state["fRec256"] = jnp.roll(state["fRec256"], 1) 
		state["fRec255"] = jnp.roll(state["fRec255"], 1) 
		state["fRec254"] = jnp.roll(state["fRec254"], 1) 
		state["fRec253"] = jnp.roll(state["fRec253"], 1) 
		state["fRec264"] = jnp.roll(state["fRec264"], 1) 
		state["fRec262"] = jnp.roll(state["fRec262"], 1) 
		state["fRec260"] = jnp.roll(state["fRec260"], 1) 
		state["fRec258"] = jnp.roll(state["fRec258"], 1) 
		state["fRec266"] = jnp.roll(state["fRec266"], 1) 
		state["fRec269"] = jnp.roll(state["fRec269"], 1) 
		state["fRec268"] = jnp.roll(state["fRec268"], 1) 
		state["fRec273"] = jnp.roll(state["fRec273"], 1) 
		state["fRec272"] = jnp.roll(state["fRec272"], 1) 
		state["fRec271"] = jnp.roll(state["fRec271"], 1) 
		state["fRec278"] = jnp.roll(state["fRec278"], 1) 
		state["fRec277"] = jnp.roll(state["fRec277"], 1) 
		state["fRec276"] = jnp.roll(state["fRec276"], 1) 
		state["fRec275"] = jnp.roll(state["fRec275"], 1) 
		state["fRec286"] = jnp.roll(state["fRec286"], 1) 
		state["fRec284"] = jnp.roll(state["fRec284"], 1) 
		state["fRec282"] = jnp.roll(state["fRec282"], 1) 
		state["fRec280"] = jnp.roll(state["fRec280"], 1) 
		state["fRec288"] = jnp.roll(state["fRec288"], 1) 
		state["fRec291"] = jnp.roll(state["fRec291"], 1) 
		state["fRec290"] = jnp.roll(state["fRec290"], 1) 
		state["fRec295"] = jnp.roll(state["fRec295"], 1) 
		state["fRec294"] = jnp.roll(state["fRec294"], 1) 
		state["fRec293"] = jnp.roll(state["fRec293"], 1) 
		state["fRec300"] = jnp.roll(state["fRec300"], 1) 
		state["fRec299"] = jnp.roll(state["fRec299"], 1) 
		state["fRec298"] = jnp.roll(state["fRec298"], 1) 
		state["fRec297"] = jnp.roll(state["fRec297"], 1) 
		state["fRec308"] = jnp.roll(state["fRec308"], 1) 
		state["fRec306"] = jnp.roll(state["fRec306"], 1) 
		state["fRec304"] = jnp.roll(state["fRec304"], 1) 
		state["fRec302"] = jnp.roll(state["fRec302"], 1) 
		state["fRec310"] = jnp.roll(state["fRec310"], 1) 
		state["fRec313"] = jnp.roll(state["fRec313"], 1) 
		state["fRec312"] = jnp.roll(state["fRec312"], 1) 
		state["fRec317"] = jnp.roll(state["fRec317"], 1) 
		state["fRec316"] = jnp.roll(state["fRec316"], 1) 
		state["fRec315"] = jnp.roll(state["fRec315"], 1) 
		state["fRec322"] = jnp.roll(state["fRec322"], 1) 
		state["fRec321"] = jnp.roll(state["fRec321"], 1) 
		state["fRec320"] = jnp.roll(state["fRec320"], 1) 
		state["fRec319"] = jnp.roll(state["fRec319"], 1) 
		state["fRec330"] = jnp.roll(state["fRec330"], 1) 
		state["fRec328"] = jnp.roll(state["fRec328"], 1) 
		state["fRec326"] = jnp.roll(state["fRec326"], 1) 
		state["fRec324"] = jnp.roll(state["fRec324"], 1) 
		state["fRec332"] = jnp.roll(state["fRec332"], 1) 
		state["fRec335"] = jnp.roll(state["fRec335"], 1) 
		state["fRec334"] = jnp.roll(state["fRec334"], 1) 
		state["fRec339"] = jnp.roll(state["fRec339"], 1) 
		state["fRec338"] = jnp.roll(state["fRec338"], 1) 
		state["fRec337"] = jnp.roll(state["fRec337"], 1) 
		state["fRec344"] = jnp.roll(state["fRec344"], 1) 
		state["fRec343"] = jnp.roll(state["fRec343"], 1) 
		state["fRec342"] = jnp.roll(state["fRec342"], 1) 
		state["fRec341"] = jnp.roll(state["fRec341"], 1) 
		state["fRec352"] = jnp.roll(state["fRec352"], 1) 
		state["fRec350"] = jnp.roll(state["fRec350"], 1) 
		state["fRec348"] = jnp.roll(state["fRec348"], 1) 
		state["fRec346"] = jnp.roll(state["fRec346"], 1) 
		state["fRec354"] = jnp.roll(state["fRec354"], 1) 
		state["fRec357"] = jnp.roll(state["fRec357"], 1) 
		state["fRec356"] = jnp.roll(state["fRec356"], 1) 
		state["fRec361"] = jnp.roll(state["fRec361"], 1) 
		state["fRec360"] = jnp.roll(state["fRec360"], 1) 
		state["fRec359"] = jnp.roll(state["fRec359"], 1) 
		state["fRec366"] = jnp.roll(state["fRec366"], 1) 
		state["fRec365"] = jnp.roll(state["fRec365"], 1) 
		state["fRec364"] = jnp.roll(state["fRec364"], 1) 
		state["fRec363"] = jnp.roll(state["fRec363"], 1) 
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
