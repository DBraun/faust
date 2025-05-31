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

import dataclasses
from typing import Dict, List, Tuple
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
Compilation options: -a ../../architecture/jax/minimal.py -lang jax -ct 1 -es 1 -mcd 16 -mdd 1024 -mdy 33 -single -ftz 0 
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
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec10"] = np.float32(0)
		state["fRec102"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec25"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec4"] = np.float32(0)
		state["fRec46"] = np.float32(0)
		state["fRec5"] = np.float32(0)
		state["fRec53"] = np.float32(0)
		state["fRec6"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec67"] = np.float32(0)
		state["fRec7"] = np.float32(0)
		state["fRec74"] = np.float32(0)
		state["fRec8"] = np.float32(0)
		state["fRec81"] = np.float32(0)
		state["fRec88"] = np.float32(0)
		state["fRec9"] = np.float32(0)
		state["fRec95"] = np.float32(0)
		state["fVec1"] = np.float32(0)
		state["iVec0"] = np.int32(0)
		# Initialize array delays
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec17"] = np.zeros((3,), dtype=np.float32)
		state["fRec16"] = np.zeros((3,), dtype=np.float32)
		state["fRec15"] = np.zeros((3,), dtype=np.float32)
		state["fRec14"] = np.zeros((3,), dtype=np.float32)
		state["fRec13"] = np.zeros((3,), dtype=np.float32)
		state["fRec12"] = np.zeros((3,), dtype=np.float32)
		state["fRec24"] = np.zeros((3,), dtype=np.float32)
		state["fRec23"] = np.zeros((3,), dtype=np.float32)
		state["fRec22"] = np.zeros((3,), dtype=np.float32)
		state["fRec21"] = np.zeros((3,), dtype=np.float32)
		state["fRec20"] = np.zeros((3,), dtype=np.float32)
		state["fRec19"] = np.zeros((3,), dtype=np.float32)
		state["fRec31"] = np.zeros((3,), dtype=np.float32)
		state["fRec30"] = np.zeros((3,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fRec28"] = np.zeros((3,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fRec26"] = np.zeros((3,), dtype=np.float32)
		state["fRec38"] = np.zeros((3,), dtype=np.float32)
		state["fRec37"] = np.zeros((3,), dtype=np.float32)
		state["fRec36"] = np.zeros((3,), dtype=np.float32)
		state["fRec35"] = np.zeros((3,), dtype=np.float32)
		state["fRec34"] = np.zeros((3,), dtype=np.float32)
		state["fRec33"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		state["fRec43"] = np.zeros((3,), dtype=np.float32)
		state["fRec42"] = np.zeros((3,), dtype=np.float32)
		state["fRec41"] = np.zeros((3,), dtype=np.float32)
		state["fRec40"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec50"] = np.zeros((3,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fRec48"] = np.zeros((3,), dtype=np.float32)
		state["fRec47"] = np.zeros((3,), dtype=np.float32)
		state["fRec59"] = np.zeros((3,), dtype=np.float32)
		state["fRec58"] = np.zeros((3,), dtype=np.float32)
		state["fRec57"] = np.zeros((3,), dtype=np.float32)
		state["fRec56"] = np.zeros((3,), dtype=np.float32)
		state["fRec55"] = np.zeros((3,), dtype=np.float32)
		state["fRec54"] = np.zeros((3,), dtype=np.float32)
		state["fRec66"] = np.zeros((3,), dtype=np.float32)
		state["fRec65"] = np.zeros((3,), dtype=np.float32)
		state["fRec64"] = np.zeros((3,), dtype=np.float32)
		state["fRec63"] = np.zeros((3,), dtype=np.float32)
		state["fRec62"] = np.zeros((3,), dtype=np.float32)
		state["fRec61"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec72"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec70"] = np.zeros((3,), dtype=np.float32)
		state["fRec69"] = np.zeros((3,), dtype=np.float32)
		state["fRec68"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fRec76"] = np.zeros((3,), dtype=np.float32)
		state["fRec75"] = np.zeros((3,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec84"] = np.zeros((3,), dtype=np.float32)
		state["fRec83"] = np.zeros((3,), dtype=np.float32)
		state["fRec82"] = np.zeros((3,), dtype=np.float32)
		state["fRec94"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec92"] = np.zeros((3,), dtype=np.float32)
		state["fRec91"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fRec98"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec96"] = np.zeros((3,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec103"] = np.zeros((3,), dtype=np.float32)
		# Initialize read-write tables
		# Initialize waveform arrays for read-write tables
		return state

	def setup(self):
		# Initialize static tables
		# Initialize waveform data
		# Convert static tables and waveform data to JAX arrays
		# Initialize UI parameters
		unnorm_funcs = {}
		ui_path = []
		ui_path.append("cubic_distortion") 
		ui_path.append("0x00") 
		ui_path.append("SINE WAVE OSCILLATOR oscrs") 
		self.add_hslider("fHslider3", ui_path, "Amplitude", -2e+01, -1.2e+02, 1e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider5", ui_path, "Frequency", 49.0, 1.0, 88.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider4", ui_path, "Portamento", 0.1, 0.001, 1e+01, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("CUBIC NONLINEARITY cubicnl") 
		self.add_button("fCheckbox0", ui_path, "Bypass", unnorm_funcs) 
		self.add_hslider("fHslider6", ui_path, "Drive", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "Offset", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("CONSTANT-Q SPECTRUM ANALYZER (6E), 15 bands spanning LP, 9 octaves below 16000 Hz, HP") 
		self.add_vbargraph("fVbargraph14", ui_path, "vbargraph0", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph13", ui_path, "vbargraph1", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph12", ui_path, "vbargraph2", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph11", ui_path, "vbargraph3", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph10", ui_path, "vbargraph4", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph9", ui_path, "vbargraph5", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph8", ui_path, "vbargraph6", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph7", ui_path, "vbargraph7", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph6", ui_path, "vbargraph8", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph5", ui_path, "vbargraph9", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph4", ui_path, "vbargraph10", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph3", ui_path, "vbargraph11", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph2", ui_path, "vbargraph12", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph1", ui_path, "vbargraph13", -5e+01, 1e+01) 
		self.add_vbargraph("fVbargraph0", ui_path, "vbargraph14", -5e+01, 1e+01) 
		ui_path.pop()
		ui_path.append("SPECTRUM ANALYZER CONTROLS") 
		self.add_hslider("fHslider1", ui_path, "Level Averaging Time", 1e+02, 1.0, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider0", ui_path, "Level dB Offset", 5e+01, 0.0, 1e+02, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		
		self._fConst1 = (np.float32(1e+03) / self._fConst0) 
		
		self._fConst2 = np.tan((np.float32(50265.484) / self._fConst0)) 
		
		self._fConst3 = (np.float32(1.0) / self._fConst2) 
		
		self._fConst4 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.15748216)) / self._fConst2) + np.float32(0.9351402))) 
		
		self._fConst5 = np.power(self._fConst2, np.float32(2.0)) 
		
		self._fConst6 = (np.float32(50.06381) / self._fConst5) 
		
		self._fConst7 = (self._fConst6 + np.float32(0.9351402)) 
		
		self._fConst8 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.74313045)) / self._fConst2) + np.float32(1.4500711))) 
		
		self._fConst9 = (np.float32(11.0520525) / self._fConst5) 
		
		self._fConst10 = (self._fConst9 + np.float32(1.4500711)) 
		
		self._fConst11 = (np.float32(1.0) / (((self._fConst3 + np.float32(3.1897273)) / self._fConst2) + np.float32(4.0767817))) 
		
		self._fConst12 = (np.float32(0.0017661728) / self._fConst5) 
		
		self._fConst13 = (self._fConst12 + np.float32(0.0004076782)) 
		
		self._fConst14 = (np.float32(6.2831855) / self._fConst0) 
		
		self._fConst15 = (np.float32(1.0) / self._fConst0) 
		
		self._fConst16 = (((self._fConst3 + np.float32(-3.1897273)) / self._fConst2) + np.float32(4.0767817)) 
		
		self._fConst17 = (np.float32(1.0) / self._fConst5) 
		
		self._fConst18 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst17)) 
		
		self._fConst19 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst12)) 
		
		self._fConst20 = (((self._fConst3 + np.float32(-0.74313045)) / self._fConst2) + np.float32(1.4500711)) 
		
		self._fConst21 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst17)) 
		
		self._fConst22 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst9)) 
		
		self._fConst23 = (((self._fConst3 + np.float32(-0.15748216)) / self._fConst2) + np.float32(0.9351402)) 
		
		self._fConst24 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst17)) 
		
		self._fConst25 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst6)) 
		
		self._fConst26 = np.tan((np.float32(31665.27) / self._fConst0)) 
		
		self._fConst27 = (np.float32(1.0) / self._fConst26) 
		
		self._fConst28 = (np.float32(1.0) / (((self._fConst27 + np.float32(0.15748216)) / self._fConst26) + np.float32(0.9351402))) 
		
		self._fConst29 = np.power(self._fConst26, np.float32(2.0)) 
		
		self._fConst30 = (np.float32(50.06381) / self._fConst29) 
		
		self._fConst31 = (self._fConst30 + np.float32(0.9351402)) 
		
		self._fConst32 = (np.float32(1.0) / (((self._fConst27 + np.float32(0.74313045)) / self._fConst26) + np.float32(1.4500711))) 
		
		self._fConst33 = (np.float32(11.0520525) / self._fConst29) 
		
		self._fConst34 = (self._fConst33 + np.float32(1.4500711)) 
		
		self._fConst35 = (np.float32(1.0) / (((self._fConst27 + np.float32(3.1897273)) / self._fConst26) + np.float32(4.0767817))) 
		
		self._fConst36 = (np.float32(0.0017661728) / self._fConst29) 
		
		self._fConst37 = (self._fConst36 + np.float32(0.0004076782)) 
		
		self._fConst38 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.16840488)) / self._fConst2) + np.float32(1.0693583))) 
		
		self._fConst39 = (self._fConst17 + np.float32(53.53615)) 
		
		self._fConst40 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.51247865)) / self._fConst2) + np.float32(0.6896214))) 
		
		self._fConst41 = (self._fConst17 + np.float32(7.6217313)) 
		
		self._fConst42 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.78241307)) / self._fConst2) + np.float32(0.2452915))) 
		
		self._fConst43 = (np.float32(0.0001) / self._fConst5) 
		
		self._fConst44 = (self._fConst43 + np.float32(0.0004332272)) 
		
		self._fConst45 = (((self._fConst3 + np.float32(-0.78241307)) / self._fConst2) + np.float32(0.2452915)) 
		
		self._fConst46 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst17)) 
		
		self._fConst47 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst43)) 
		
		self._fConst48 = (((self._fConst3 + np.float32(-0.51247865)) / self._fConst2) + np.float32(0.6896214)) 
		
		self._fConst49 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst17)) 
		
		self._fConst50 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst17)) 
		
		self._fConst51 = (((self._fConst3 + np.float32(-0.16840488)) / self._fConst2) + np.float32(1.0693583)) 
		
		self._fConst52 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst17)) 
		
		self._fConst53 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst17)) 
		
		self._fConst54 = (((self._fConst27 + np.float32(-3.1897273)) / self._fConst26) + np.float32(4.0767817)) 
		
		self._fConst55 = (np.float32(1.0) / self._fConst29) 
		
		self._fConst56 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst55)) 
		
		self._fConst57 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst36)) 
		
		self._fConst58 = (((self._fConst27 + np.float32(-0.74313045)) / self._fConst26) + np.float32(1.4500711)) 
		
		self._fConst59 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst55)) 
		
		self._fConst60 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst33)) 
		
		self._fConst61 = (((self._fConst27 + np.float32(-0.15748216)) / self._fConst26) + np.float32(0.9351402)) 
		
		self._fConst62 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst55)) 
		
		self._fConst63 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst30)) 
		
		self._fConst64 = np.tan((np.float32(19947.87) / self._fConst0)) 
		
		self._fConst65 = (np.float32(1.0) / self._fConst64) 
		
		self._fConst66 = (np.float32(1.0) / (((self._fConst65 + np.float32(0.15748216)) / self._fConst64) + np.float32(0.9351402))) 
		
		self._fConst67 = np.power(self._fConst64, np.float32(2.0)) 
		
		self._fConst68 = (np.float32(50.06381) / self._fConst67) 
		
		self._fConst69 = (self._fConst68 + np.float32(0.9351402)) 
		
		self._fConst70 = (np.float32(1.0) / (((self._fConst65 + np.float32(0.74313045)) / self._fConst64) + np.float32(1.4500711))) 
		
		self._fConst71 = (np.float32(11.0520525) / self._fConst67) 
		
		self._fConst72 = (self._fConst71 + np.float32(1.4500711)) 
		
		self._fConst73 = (np.float32(1.0) / (((self._fConst65 + np.float32(3.1897273)) / self._fConst64) + np.float32(4.0767817))) 
		
		self._fConst74 = (np.float32(0.0017661728) / self._fConst67) 
		
		self._fConst75 = (self._fConst74 + np.float32(0.0004076782)) 
		
		self._fConst76 = (np.float32(1.0) / (((self._fConst27 + np.float32(0.16840488)) / self._fConst26) + np.float32(1.0693583))) 
		
		self._fConst77 = (self._fConst55 + np.float32(53.53615)) 
		
		self._fConst78 = (np.float32(1.0) / (((self._fConst27 + np.float32(0.51247865)) / self._fConst26) + np.float32(0.6896214))) 
		
		self._fConst79 = (self._fConst55 + np.float32(7.6217313)) 
		
		self._fConst80 = (np.float32(1.0) / (((self._fConst27 + np.float32(0.78241307)) / self._fConst26) + np.float32(0.2452915))) 
		
		self._fConst81 = (np.float32(0.0001) / self._fConst29) 
		
		self._fConst82 = (self._fConst81 + np.float32(0.0004332272)) 
		
		self._fConst83 = (((self._fConst27 + np.float32(-0.78241307)) / self._fConst26) + np.float32(0.2452915)) 
		
		self._fConst84 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst55)) 
		
		self._fConst85 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst81)) 
		
		self._fConst86 = (((self._fConst27 + np.float32(-0.51247865)) / self._fConst26) + np.float32(0.6896214)) 
		
		self._fConst87 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst55)) 
		
		self._fConst88 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst55)) 
		
		self._fConst89 = (((self._fConst27 + np.float32(-0.16840488)) / self._fConst26) + np.float32(1.0693583)) 
		
		self._fConst90 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst55)) 
		
		self._fConst91 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst55)) 
		
		self._fConst92 = (((self._fConst65 + np.float32(-3.1897273)) / self._fConst64) + np.float32(4.0767817)) 
		
		self._fConst93 = (np.float32(1.0) / self._fConst67) 
		
		self._fConst94 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst93)) 
		
		self._fConst95 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst74)) 
		
		self._fConst96 = (((self._fConst65 + np.float32(-0.74313045)) / self._fConst64) + np.float32(1.4500711)) 
		
		self._fConst97 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst93)) 
		
		self._fConst98 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst71)) 
		
		self._fConst99 = (((self._fConst65 + np.float32(-0.15748216)) / self._fConst64) + np.float32(0.9351402)) 
		
		self._fConst100 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst93)) 
		
		self._fConst101 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst68)) 
		
		self._fConst102 = np.tan((np.float32(12566.371) / self._fConst0)) 
		
		self._fConst103 = (np.float32(1.0) / self._fConst102) 
		
		self._fConst104 = (np.float32(1.0) / (((self._fConst103 + np.float32(0.15748216)) / self._fConst102) + np.float32(0.9351402))) 
		
		self._fConst105 = np.power(self._fConst102, np.float32(2.0)) 
		
		self._fConst106 = (np.float32(50.06381) / self._fConst105) 
		
		self._fConst107 = (self._fConst106 + np.float32(0.9351402)) 
		
		self._fConst108 = (np.float32(1.0) / (((self._fConst103 + np.float32(0.74313045)) / self._fConst102) + np.float32(1.4500711))) 
		
		self._fConst109 = (np.float32(11.0520525) / self._fConst105) 
		
		self._fConst110 = (self._fConst109 + np.float32(1.4500711)) 
		
		self._fConst111 = (np.float32(1.0) / (((self._fConst103 + np.float32(3.1897273)) / self._fConst102) + np.float32(4.0767817))) 
		
		self._fConst112 = (np.float32(0.0017661728) / self._fConst105) 
		
		self._fConst113 = (self._fConst112 + np.float32(0.0004076782)) 
		
		self._fConst114 = (np.float32(1.0) / (((self._fConst65 + np.float32(0.16840488)) / self._fConst64) + np.float32(1.0693583))) 
		
		self._fConst115 = (self._fConst93 + np.float32(53.53615)) 
		
		self._fConst116 = (np.float32(1.0) / (((self._fConst65 + np.float32(0.51247865)) / self._fConst64) + np.float32(0.6896214))) 
		
		self._fConst117 = (self._fConst93 + np.float32(7.6217313)) 
		
		self._fConst118 = (np.float32(1.0) / (((self._fConst65 + np.float32(0.78241307)) / self._fConst64) + np.float32(0.2452915))) 
		
		self._fConst119 = (np.float32(0.0001) / self._fConst67) 
		
		self._fConst120 = (self._fConst119 + np.float32(0.0004332272)) 
		
		self._fConst121 = (((self._fConst65 + np.float32(-0.78241307)) / self._fConst64) + np.float32(0.2452915)) 
		
		self._fConst122 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst93)) 
		
		self._fConst123 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst119)) 
		
		self._fConst124 = (((self._fConst65 + np.float32(-0.51247865)) / self._fConst64) + np.float32(0.6896214)) 
		
		self._fConst125 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst93)) 
		
		self._fConst126 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst93)) 
		
		self._fConst127 = (((self._fConst65 + np.float32(-0.16840488)) / self._fConst64) + np.float32(1.0693583)) 
		
		self._fConst128 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst93)) 
		
		self._fConst129 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst93)) 
		
		self._fConst130 = (((self._fConst103 + np.float32(-3.1897273)) / self._fConst102) + np.float32(4.0767817)) 
		
		self._fConst131 = (np.float32(1.0) / self._fConst105) 
		
		self._fConst132 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst131)) 
		
		self._fConst133 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst112)) 
		
		self._fConst134 = (((self._fConst103 + np.float32(-0.74313045)) / self._fConst102) + np.float32(1.4500711)) 
		
		self._fConst135 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst131)) 
		
		self._fConst136 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst109)) 
		
		self._fConst137 = (((self._fConst103 + np.float32(-0.15748216)) / self._fConst102) + np.float32(0.9351402)) 
		
		self._fConst138 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst131)) 
		
		self._fConst139 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst106)) 
		
		self._fConst140 = np.tan((np.float32(7916.3174) / self._fConst0)) 
		
		self._fConst141 = (np.float32(1.0) / self._fConst140) 
		
		self._fConst142 = (np.float32(1.0) / (((self._fConst141 + np.float32(0.15748216)) / self._fConst140) + np.float32(0.9351402))) 
		
		self._fConst143 = np.power(self._fConst140, np.float32(2.0)) 
		
		self._fConst144 = (np.float32(50.06381) / self._fConst143) 
		
		self._fConst145 = (self._fConst144 + np.float32(0.9351402)) 
		
		self._fConst146 = (np.float32(1.0) / (((self._fConst141 + np.float32(0.74313045)) / self._fConst140) + np.float32(1.4500711))) 
		
		self._fConst147 = (np.float32(11.0520525) / self._fConst143) 
		
		self._fConst148 = (self._fConst147 + np.float32(1.4500711)) 
		
		self._fConst149 = (np.float32(1.0) / (((self._fConst141 + np.float32(3.1897273)) / self._fConst140) + np.float32(4.0767817))) 
		
		self._fConst150 = (np.float32(0.0017661728) / self._fConst143) 
		
		self._fConst151 = (self._fConst150 + np.float32(0.0004076782)) 
		
		self._fConst152 = (np.float32(1.0) / (((self._fConst103 + np.float32(0.16840488)) / self._fConst102) + np.float32(1.0693583))) 
		
		self._fConst153 = (self._fConst131 + np.float32(53.53615)) 
		
		self._fConst154 = (np.float32(1.0) / (((self._fConst103 + np.float32(0.51247865)) / self._fConst102) + np.float32(0.6896214))) 
		
		self._fConst155 = (self._fConst131 + np.float32(7.6217313)) 
		
		self._fConst156 = (np.float32(1.0) / (((self._fConst103 + np.float32(0.78241307)) / self._fConst102) + np.float32(0.2452915))) 
		
		self._fConst157 = (np.float32(0.0001) / self._fConst105) 
		
		self._fConst158 = (self._fConst157 + np.float32(0.0004332272)) 
		
		self._fConst159 = (((self._fConst103 + np.float32(-0.78241307)) / self._fConst102) + np.float32(0.2452915)) 
		
		self._fConst160 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst131)) 
		
		self._fConst161 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst157)) 
		
		self._fConst162 = (((self._fConst103 + np.float32(-0.51247865)) / self._fConst102) + np.float32(0.6896214)) 
		
		self._fConst163 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst131)) 
		
		self._fConst164 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst131)) 
		
		self._fConst165 = (((self._fConst103 + np.float32(-0.16840488)) / self._fConst102) + np.float32(1.0693583)) 
		
		self._fConst166 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst131)) 
		
		self._fConst167 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst131)) 
		
		self._fConst168 = (((self._fConst141 + np.float32(-3.1897273)) / self._fConst140) + np.float32(4.0767817)) 
		
		self._fConst169 = (np.float32(1.0) / self._fConst143) 
		
		self._fConst170 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst169)) 
		
		self._fConst171 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst150)) 
		
		self._fConst172 = (((self._fConst141 + np.float32(-0.74313045)) / self._fConst140) + np.float32(1.4500711)) 
		
		self._fConst173 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst169)) 
		
		self._fConst174 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst147)) 
		
		self._fConst175 = (((self._fConst141 + np.float32(-0.15748216)) / self._fConst140) + np.float32(0.9351402)) 
		
		self._fConst176 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst169)) 
		
		self._fConst177 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst144)) 
		
		self._fConst178 = np.tan((np.float32(4986.9673) / self._fConst0)) 
		
		self._fConst179 = (np.float32(1.0) / self._fConst178) 
		
		self._fConst180 = (np.float32(1.0) / (((self._fConst179 + np.float32(0.15748216)) / self._fConst178) + np.float32(0.9351402))) 
		
		self._fConst181 = np.power(self._fConst178, np.float32(2.0)) 
		
		self._fConst182 = (np.float32(50.06381) / self._fConst181) 
		
		self._fConst183 = (self._fConst182 + np.float32(0.9351402)) 
		
		self._fConst184 = (np.float32(1.0) / (((self._fConst179 + np.float32(0.74313045)) / self._fConst178) + np.float32(1.4500711))) 
		
		self._fConst185 = (np.float32(11.0520525) / self._fConst181) 
		
		self._fConst186 = (self._fConst185 + np.float32(1.4500711)) 
		
		self._fConst187 = (np.float32(1.0) / (((self._fConst179 + np.float32(3.1897273)) / self._fConst178) + np.float32(4.0767817))) 
		
		self._fConst188 = (np.float32(0.0017661728) / self._fConst181) 
		
		self._fConst189 = (self._fConst188 + np.float32(0.0004076782)) 
		
		self._fConst190 = (np.float32(1.0) / (((self._fConst141 + np.float32(0.16840488)) / self._fConst140) + np.float32(1.0693583))) 
		
		self._fConst191 = (self._fConst169 + np.float32(53.53615)) 
		
		self._fConst192 = (np.float32(1.0) / (((self._fConst141 + np.float32(0.51247865)) / self._fConst140) + np.float32(0.6896214))) 
		
		self._fConst193 = (self._fConst169 + np.float32(7.6217313)) 
		
		self._fConst194 = (np.float32(1.0) / (((self._fConst141 + np.float32(0.78241307)) / self._fConst140) + np.float32(0.2452915))) 
		
		self._fConst195 = (np.float32(0.0001) / self._fConst143) 
		
		self._fConst196 = (self._fConst195 + np.float32(0.0004332272)) 
		
		self._fConst197 = (((self._fConst141 + np.float32(-0.78241307)) / self._fConst140) + np.float32(0.2452915)) 
		
		self._fConst198 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst169)) 
		
		self._fConst199 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst195)) 
		
		self._fConst200 = (((self._fConst141 + np.float32(-0.51247865)) / self._fConst140) + np.float32(0.6896214)) 
		
		self._fConst201 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst169)) 
		
		self._fConst202 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst169)) 
		
		self._fConst203 = (((self._fConst141 + np.float32(-0.16840488)) / self._fConst140) + np.float32(1.0693583)) 
		
		self._fConst204 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst169)) 
		
		self._fConst205 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst169)) 
		
		self._fConst206 = (((self._fConst179 + np.float32(-3.1897273)) / self._fConst178) + np.float32(4.0767817)) 
		
		self._fConst207 = (np.float32(1.0) / self._fConst181) 
		
		self._fConst208 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst207)) 
		
		self._fConst209 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst188)) 
		
		self._fConst210 = (((self._fConst179 + np.float32(-0.74313045)) / self._fConst178) + np.float32(1.4500711)) 
		
		self._fConst211 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst207)) 
		
		self._fConst212 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst185)) 
		
		self._fConst213 = (((self._fConst179 + np.float32(-0.15748216)) / self._fConst178) + np.float32(0.9351402)) 
		
		self._fConst214 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst207)) 
		
		self._fConst215 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst182)) 
		
		self._fConst216 = np.tan((np.float32(3141.5928) / self._fConst0)) 
		
		self._fConst217 = (np.float32(1.0) / self._fConst216) 
		
		self._fConst218 = (np.float32(1.0) / (((self._fConst217 + np.float32(0.15748216)) / self._fConst216) + np.float32(0.9351402))) 
		
		self._fConst219 = np.power(self._fConst216, np.float32(2.0)) 
		
		self._fConst220 = (np.float32(50.06381) / self._fConst219) 
		
		self._fConst221 = (self._fConst220 + np.float32(0.9351402)) 
		
		self._fConst222 = (np.float32(1.0) / (((self._fConst217 + np.float32(0.74313045)) / self._fConst216) + np.float32(1.4500711))) 
		
		self._fConst223 = (np.float32(11.0520525) / self._fConst219) 
		
		self._fConst224 = (self._fConst223 + np.float32(1.4500711)) 
		
		self._fConst225 = (np.float32(1.0) / (((self._fConst217 + np.float32(3.1897273)) / self._fConst216) + np.float32(4.0767817))) 
		
		self._fConst226 = (np.float32(0.0017661728) / self._fConst219) 
		
		self._fConst227 = (self._fConst226 + np.float32(0.0004076782)) 
		
		self._fConst228 = (np.float32(1.0) / (((self._fConst179 + np.float32(0.16840488)) / self._fConst178) + np.float32(1.0693583))) 
		
		self._fConst229 = (self._fConst207 + np.float32(53.53615)) 
		
		self._fConst230 = (np.float32(1.0) / (((self._fConst179 + np.float32(0.51247865)) / self._fConst178) + np.float32(0.6896214))) 
		
		self._fConst231 = (self._fConst207 + np.float32(7.6217313)) 
		
		self._fConst232 = (np.float32(1.0) / (((self._fConst179 + np.float32(0.78241307)) / self._fConst178) + np.float32(0.2452915))) 
		
		self._fConst233 = (np.float32(0.0001) / self._fConst181) 
		
		self._fConst234 = (self._fConst233 + np.float32(0.0004332272)) 
		
		self._fConst235 = (((self._fConst179 + np.float32(-0.78241307)) / self._fConst178) + np.float32(0.2452915)) 
		
		self._fConst236 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst207)) 
		
		self._fConst237 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst233)) 
		
		self._fConst238 = (((self._fConst179 + np.float32(-0.51247865)) / self._fConst178) + np.float32(0.6896214)) 
		
		self._fConst239 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst207)) 
		
		self._fConst240 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst207)) 
		
		self._fConst241 = (((self._fConst179 + np.float32(-0.16840488)) / self._fConst178) + np.float32(1.0693583)) 
		
		self._fConst242 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst207)) 
		
		self._fConst243 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst207)) 
		
		self._fConst244 = (((self._fConst217 + np.float32(-3.1897273)) / self._fConst216) + np.float32(4.0767817)) 
		
		self._fConst245 = (np.float32(1.0) / self._fConst219) 
		
		self._fConst246 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst245)) 
		
		self._fConst247 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst226)) 
		
		self._fConst248 = (((self._fConst217 + np.float32(-0.74313045)) / self._fConst216) + np.float32(1.4500711)) 
		
		self._fConst249 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst245)) 
		
		self._fConst250 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst223)) 
		
		self._fConst251 = (((self._fConst217 + np.float32(-0.15748216)) / self._fConst216) + np.float32(0.9351402)) 
		
		self._fConst252 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst245)) 
		
		self._fConst253 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst220)) 
		
		self._fConst254 = np.tan((np.float32(1979.0793) / self._fConst0)) 
		
		self._fConst255 = (np.float32(1.0) / self._fConst254) 
		
		self._fConst256 = (np.float32(1.0) / (((self._fConst255 + np.float32(0.15748216)) / self._fConst254) + np.float32(0.9351402))) 
		
		self._fConst257 = np.power(self._fConst254, np.float32(2.0)) 
		
		self._fConst258 = (np.float32(50.06381) / self._fConst257) 
		
		self._fConst259 = (self._fConst258 + np.float32(0.9351402)) 
		
		self._fConst260 = (np.float32(1.0) / (((self._fConst255 + np.float32(0.74313045)) / self._fConst254) + np.float32(1.4500711))) 
		
		self._fConst261 = (np.float32(11.0520525) / self._fConst257) 
		
		self._fConst262 = (self._fConst261 + np.float32(1.4500711)) 
		
		self._fConst263 = (np.float32(1.0) / (((self._fConst255 + np.float32(3.1897273)) / self._fConst254) + np.float32(4.0767817))) 
		
		self._fConst264 = (np.float32(0.0017661728) / self._fConst257) 
		
		self._fConst265 = (self._fConst264 + np.float32(0.0004076782)) 
		
		self._fConst266 = (np.float32(1.0) / (((self._fConst217 + np.float32(0.16840488)) / self._fConst216) + np.float32(1.0693583))) 
		
		self._fConst267 = (self._fConst245 + np.float32(53.53615)) 
		
		self._fConst268 = (np.float32(1.0) / (((self._fConst217 + np.float32(0.51247865)) / self._fConst216) + np.float32(0.6896214))) 
		
		self._fConst269 = (self._fConst245 + np.float32(7.6217313)) 
		
		self._fConst270 = (np.float32(1.0) / (((self._fConst217 + np.float32(0.78241307)) / self._fConst216) + np.float32(0.2452915))) 
		
		self._fConst271 = (np.float32(0.0001) / self._fConst219) 
		
		self._fConst272 = (self._fConst271 + np.float32(0.0004332272)) 
		
		self._fConst273 = (((self._fConst217 + np.float32(-0.78241307)) / self._fConst216) + np.float32(0.2452915)) 
		
		self._fConst274 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst245)) 
		
		self._fConst275 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst271)) 
		
		self._fConst276 = (((self._fConst217 + np.float32(-0.51247865)) / self._fConst216) + np.float32(0.6896214)) 
		
		self._fConst277 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst245)) 
		
		self._fConst278 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst245)) 
		
		self._fConst279 = (((self._fConst217 + np.float32(-0.16840488)) / self._fConst216) + np.float32(1.0693583)) 
		
		self._fConst280 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst245)) 
		
		self._fConst281 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst245)) 
		
		self._fConst282 = (((self._fConst255 + np.float32(-3.1897273)) / self._fConst254) + np.float32(4.0767817)) 
		
		self._fConst283 = (np.float32(1.0) / self._fConst257) 
		
		self._fConst284 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst283)) 
		
		self._fConst285 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst264)) 
		
		self._fConst286 = (((self._fConst255 + np.float32(-0.74313045)) / self._fConst254) + np.float32(1.4500711)) 
		
		self._fConst287 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst283)) 
		
		self._fConst288 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst261)) 
		
		self._fConst289 = (((self._fConst255 + np.float32(-0.15748216)) / self._fConst254) + np.float32(0.9351402)) 
		
		self._fConst290 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst283)) 
		
		self._fConst291 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst258)) 
		
		self._fConst292 = np.tan((np.float32(1246.7418) / self._fConst0)) 
		
		self._fConst293 = (np.float32(1.0) / self._fConst292) 
		
		self._fConst294 = (np.float32(1.0) / (((self._fConst293 + np.float32(0.15748216)) / self._fConst292) + np.float32(0.9351402))) 
		
		self._fConst295 = np.power(self._fConst292, np.float32(2.0)) 
		
		self._fConst296 = (np.float32(50.06381) / self._fConst295) 
		
		self._fConst297 = (self._fConst296 + np.float32(0.9351402)) 
		
		self._fConst298 = (np.float32(1.0) / (((self._fConst293 + np.float32(0.74313045)) / self._fConst292) + np.float32(1.4500711))) 
		
		self._fConst299 = (np.float32(11.0520525) / self._fConst295) 
		
		self._fConst300 = (self._fConst299 + np.float32(1.4500711)) 
		
		self._fConst301 = (np.float32(1.0) / (((self._fConst293 + np.float32(3.1897273)) / self._fConst292) + np.float32(4.0767817))) 
		
		self._fConst302 = (np.float32(0.0017661728) / self._fConst295) 
		
		self._fConst303 = (self._fConst302 + np.float32(0.0004076782)) 
		
		self._fConst304 = (np.float32(1.0) / (((self._fConst255 + np.float32(0.16840488)) / self._fConst254) + np.float32(1.0693583))) 
		
		self._fConst305 = (self._fConst283 + np.float32(53.53615)) 
		
		self._fConst306 = (np.float32(1.0) / (((self._fConst255 + np.float32(0.51247865)) / self._fConst254) + np.float32(0.6896214))) 
		
		self._fConst307 = (self._fConst283 + np.float32(7.6217313)) 
		
		self._fConst308 = (np.float32(1.0) / (((self._fConst255 + np.float32(0.78241307)) / self._fConst254) + np.float32(0.2452915))) 
		
		self._fConst309 = (np.float32(0.0001) / self._fConst257) 
		
		self._fConst310 = (self._fConst309 + np.float32(0.0004332272)) 
		
		self._fConst311 = (((self._fConst255 + np.float32(-0.78241307)) / self._fConst254) + np.float32(0.2452915)) 
		
		self._fConst312 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst283)) 
		
		self._fConst313 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst309)) 
		
		self._fConst314 = (((self._fConst255 + np.float32(-0.51247865)) / self._fConst254) + np.float32(0.6896214)) 
		
		self._fConst315 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst283)) 
		
		self._fConst316 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst283)) 
		
		self._fConst317 = (((self._fConst255 + np.float32(-0.16840488)) / self._fConst254) + np.float32(1.0693583)) 
		
		self._fConst318 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst283)) 
		
		self._fConst319 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst283)) 
		
		self._fConst320 = (((self._fConst293 + np.float32(-3.1897273)) / self._fConst292) + np.float32(4.0767817)) 
		
		self._fConst321 = (np.float32(1.0) / self._fConst295) 
		
		self._fConst322 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst321)) 
		
		self._fConst323 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst302)) 
		
		self._fConst324 = (((self._fConst293 + np.float32(-0.74313045)) / self._fConst292) + np.float32(1.4500711)) 
		
		self._fConst325 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst321)) 
		
		self._fConst326 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst299)) 
		
		self._fConst327 = (((self._fConst293 + np.float32(-0.15748216)) / self._fConst292) + np.float32(0.9351402)) 
		
		self._fConst328 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst321)) 
		
		self._fConst329 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst296)) 
		
		self._fConst330 = np.tan((np.float32(785.3982) / self._fConst0)) 
		
		self._fConst331 = (np.float32(1.0) / self._fConst330) 
		
		self._fConst332 = (np.float32(1.0) / (((self._fConst331 + np.float32(0.15748216)) / self._fConst330) + np.float32(0.9351402))) 
		
		self._fConst333 = np.power(self._fConst330, np.float32(2.0)) 
		
		self._fConst334 = (np.float32(50.06381) / self._fConst333) 
		
		self._fConst335 = (self._fConst334 + np.float32(0.9351402)) 
		
		self._fConst336 = (np.float32(1.0) / (((self._fConst331 + np.float32(0.74313045)) / self._fConst330) + np.float32(1.4500711))) 
		
		self._fConst337 = (np.float32(11.0520525) / self._fConst333) 
		
		self._fConst338 = (self._fConst337 + np.float32(1.4500711)) 
		
		self._fConst339 = (np.float32(1.0) / (((self._fConst331 + np.float32(3.1897273)) / self._fConst330) + np.float32(4.0767817))) 
		
		self._fConst340 = (np.float32(0.0017661728) / self._fConst333) 
		
		self._fConst341 = (self._fConst340 + np.float32(0.0004076782)) 
		
		self._fConst342 = (np.float32(1.0) / (((self._fConst293 + np.float32(0.16840488)) / self._fConst292) + np.float32(1.0693583))) 
		
		self._fConst343 = (self._fConst321 + np.float32(53.53615)) 
		
		self._fConst344 = (np.float32(1.0) / (((self._fConst293 + np.float32(0.51247865)) / self._fConst292) + np.float32(0.6896214))) 
		
		self._fConst345 = (self._fConst321 + np.float32(7.6217313)) 
		
		self._fConst346 = (np.float32(1.0) / (((self._fConst293 + np.float32(0.78241307)) / self._fConst292) + np.float32(0.2452915))) 
		
		self._fConst347 = (np.float32(0.0001) / self._fConst295) 
		
		self._fConst348 = (self._fConst347 + np.float32(0.0004332272)) 
		
		self._fConst349 = (((self._fConst293 + np.float32(-0.78241307)) / self._fConst292) + np.float32(0.2452915)) 
		
		self._fConst350 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst321)) 
		
		self._fConst351 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst347)) 
		
		self._fConst352 = (((self._fConst293 + np.float32(-0.51247865)) / self._fConst292) + np.float32(0.6896214)) 
		
		self._fConst353 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst321)) 
		
		self._fConst354 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst321)) 
		
		self._fConst355 = (((self._fConst293 + np.float32(-0.16840488)) / self._fConst292) + np.float32(1.0693583)) 
		
		self._fConst356 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst321)) 
		
		self._fConst357 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst321)) 
		
		self._fConst358 = (((self._fConst331 + np.float32(-3.1897273)) / self._fConst330) + np.float32(4.0767817)) 
		
		self._fConst359 = (np.float32(1.0) / self._fConst333) 
		
		self._fConst360 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst359)) 
		
		self._fConst361 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst340)) 
		
		self._fConst362 = (((self._fConst331 + np.float32(-0.74313045)) / self._fConst330) + np.float32(1.4500711)) 
		
		self._fConst363 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst359)) 
		
		self._fConst364 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst337)) 
		
		self._fConst365 = (((self._fConst331 + np.float32(-0.15748216)) / self._fConst330) + np.float32(0.9351402)) 
		
		self._fConst366 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst359)) 
		
		self._fConst367 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst334)) 
		
		self._fConst368 = np.tan((np.float32(494.76984) / self._fConst0)) 
		
		self._fConst369 = (np.float32(1.0) / self._fConst368) 
		
		self._fConst370 = (np.float32(1.0) / (((self._fConst369 + np.float32(0.15748216)) / self._fConst368) + np.float32(0.9351402))) 
		
		self._fConst371 = np.power(self._fConst368, np.float32(2.0)) 
		
		self._fConst372 = (np.float32(50.06381) / self._fConst371) 
		
		self._fConst373 = (self._fConst372 + np.float32(0.9351402)) 
		
		self._fConst374 = (np.float32(1.0) / (((self._fConst369 + np.float32(0.74313045)) / self._fConst368) + np.float32(1.4500711))) 
		
		self._fConst375 = (np.float32(11.0520525) / self._fConst371) 
		
		self._fConst376 = (self._fConst375 + np.float32(1.4500711)) 
		
		self._fConst377 = (np.float32(1.0) / (((self._fConst369 + np.float32(3.1897273)) / self._fConst368) + np.float32(4.0767817))) 
		
		self._fConst378 = (np.float32(0.0017661728) / self._fConst371) 
		
		self._fConst379 = (self._fConst378 + np.float32(0.0004076782)) 
		
		self._fConst380 = (np.float32(1.0) / (((self._fConst331 + np.float32(0.16840488)) / self._fConst330) + np.float32(1.0693583))) 
		
		self._fConst381 = (self._fConst359 + np.float32(53.53615)) 
		
		self._fConst382 = (np.float32(1.0) / (((self._fConst331 + np.float32(0.51247865)) / self._fConst330) + np.float32(0.6896214))) 
		
		self._fConst383 = (self._fConst359 + np.float32(7.6217313)) 
		
		self._fConst384 = (np.float32(1.0) / (((self._fConst331 + np.float32(0.78241307)) / self._fConst330) + np.float32(0.2452915))) 
		
		self._fConst385 = (np.float32(0.0001) / self._fConst333) 
		
		self._fConst386 = (self._fConst385 + np.float32(0.0004332272)) 
		
		self._fConst387 = (((self._fConst331 + np.float32(-0.78241307)) / self._fConst330) + np.float32(0.2452915)) 
		
		self._fConst388 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst359)) 
		
		self._fConst389 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst385)) 
		
		self._fConst390 = (((self._fConst331 + np.float32(-0.51247865)) / self._fConst330) + np.float32(0.6896214)) 
		
		self._fConst391 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst359)) 
		
		self._fConst392 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst359)) 
		
		self._fConst393 = (((self._fConst331 + np.float32(-0.16840488)) / self._fConst330) + np.float32(1.0693583)) 
		
		self._fConst394 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst359)) 
		
		self._fConst395 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst359)) 
		
		self._fConst396 = (((self._fConst369 + np.float32(-3.1897273)) / self._fConst368) + np.float32(4.0767817)) 
		
		self._fConst397 = (np.float32(1.0) / self._fConst371) 
		
		self._fConst398 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst397)) 
		
		self._fConst399 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst378)) 
		
		self._fConst400 = (((self._fConst369 + np.float32(-0.74313045)) / self._fConst368) + np.float32(1.4500711)) 
		
		self._fConst401 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst397)) 
		
		self._fConst402 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst375)) 
		
		self._fConst403 = (((self._fConst369 + np.float32(-0.15748216)) / self._fConst368) + np.float32(0.9351402)) 
		
		self._fConst404 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst397)) 
		
		self._fConst405 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst372)) 
		
		self._fConst406 = np.tan((np.float32(311.68546) / self._fConst0)) 
		
		self._fConst407 = (np.float32(1.0) / self._fConst406) 
		
		self._fConst408 = (np.float32(1.0) / (((self._fConst407 + np.float32(0.15748216)) / self._fConst406) + np.float32(0.9351402))) 
		
		self._fConst409 = np.power(self._fConst406, np.float32(2.0)) 
		
		self._fConst410 = (np.float32(50.06381) / self._fConst409) 
		
		self._fConst411 = (self._fConst410 + np.float32(0.9351402)) 
		
		self._fConst412 = (np.float32(1.0) / (((self._fConst407 + np.float32(0.74313045)) / self._fConst406) + np.float32(1.4500711))) 
		
		self._fConst413 = (np.float32(11.0520525) / self._fConst409) 
		
		self._fConst414 = (self._fConst413 + np.float32(1.4500711)) 
		
		self._fConst415 = (np.float32(1.0) / (((self._fConst407 + np.float32(3.1897273)) / self._fConst406) + np.float32(4.0767817))) 
		
		self._fConst416 = (np.float32(0.0017661728) / self._fConst409) 
		
		self._fConst417 = (self._fConst416 + np.float32(0.0004076782)) 
		
		self._fConst418 = (np.float32(1.0) / (((self._fConst369 + np.float32(0.16840488)) / self._fConst368) + np.float32(1.0693583))) 
		
		self._fConst419 = (self._fConst397 + np.float32(53.53615)) 
		
		self._fConst420 = (np.float32(1.0) / (((self._fConst369 + np.float32(0.51247865)) / self._fConst368) + np.float32(0.6896214))) 
		
		self._fConst421 = (self._fConst397 + np.float32(7.6217313)) 
		
		self._fConst422 = (np.float32(1.0) / (((self._fConst369 + np.float32(0.78241307)) / self._fConst368) + np.float32(0.2452915))) 
		
		self._fConst423 = (np.float32(0.0001) / self._fConst371) 
		
		self._fConst424 = (self._fConst423 + np.float32(0.0004332272)) 
		
		self._fConst425 = (((self._fConst369 + np.float32(-0.78241307)) / self._fConst368) + np.float32(0.2452915)) 
		
		self._fConst426 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst397)) 
		
		self._fConst427 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst423)) 
		
		self._fConst428 = (((self._fConst369 + np.float32(-0.51247865)) / self._fConst368) + np.float32(0.6896214)) 
		
		self._fConst429 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst397)) 
		
		self._fConst430 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst397)) 
		
		self._fConst431 = (((self._fConst369 + np.float32(-0.16840488)) / self._fConst368) + np.float32(1.0693583)) 
		
		self._fConst432 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst397)) 
		
		self._fConst433 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst397)) 
		
		self._fConst434 = (((self._fConst407 + np.float32(-3.1897273)) / self._fConst406) + np.float32(4.0767817)) 
		
		self._fConst435 = (np.float32(1.0) / self._fConst409) 
		
		self._fConst436 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst435)) 
		
		self._fConst437 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst416)) 
		
		self._fConst438 = (((self._fConst407 + np.float32(-0.74313045)) / self._fConst406) + np.float32(1.4500711)) 
		
		self._fConst439 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst435)) 
		
		self._fConst440 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst413)) 
		
		self._fConst441 = (((self._fConst407 + np.float32(-0.15748216)) / self._fConst406) + np.float32(0.9351402)) 
		
		self._fConst442 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst435)) 
		
		self._fConst443 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst410)) 
		
		self._fConst444 = np.tan((np.float32(196.34955) / self._fConst0)) 
		
		self._fConst445 = (np.float32(1.0) / self._fConst444) 
		
		self._fConst446 = (np.float32(1.0) / (((self._fConst445 + np.float32(0.15748216)) / self._fConst444) + np.float32(0.9351402))) 
		
		self._fConst447 = np.power(self._fConst444, np.float32(2.0)) 
		
		self._fConst448 = (np.float32(50.06381) / self._fConst447) 
		
		self._fConst449 = (self._fConst448 + np.float32(0.9351402)) 
		
		self._fConst450 = (np.float32(1.0) / (((self._fConst445 + np.float32(0.74313045)) / self._fConst444) + np.float32(1.4500711))) 
		
		self._fConst451 = (np.float32(11.0520525) / self._fConst447) 
		
		self._fConst452 = (self._fConst451 + np.float32(1.4500711)) 
		
		self._fConst453 = (np.float32(1.0) / (((self._fConst445 + np.float32(3.1897273)) / self._fConst444) + np.float32(4.0767817))) 
		
		self._fConst454 = (np.float32(0.0017661728) / self._fConst447) 
		
		self._fConst455 = (self._fConst454 + np.float32(0.0004076782)) 
		
		self._fConst456 = (np.float32(1.0) / (((self._fConst407 + np.float32(0.16840488)) / self._fConst406) + np.float32(1.0693583))) 
		
		self._fConst457 = (self._fConst435 + np.float32(53.53615)) 
		
		self._fConst458 = (np.float32(1.0) / (((self._fConst407 + np.float32(0.51247865)) / self._fConst406) + np.float32(0.6896214))) 
		
		self._fConst459 = (self._fConst435 + np.float32(7.6217313)) 
		
		self._fConst460 = (np.float32(1.0) / (((self._fConst407 + np.float32(0.78241307)) / self._fConst406) + np.float32(0.2452915))) 
		
		self._fConst461 = (np.float32(0.0001) / self._fConst409) 
		
		self._fConst462 = (self._fConst461 + np.float32(0.0004332272)) 
		
		self._fConst463 = (((self._fConst407 + np.float32(-0.78241307)) / self._fConst406) + np.float32(0.2452915)) 
		
		self._fConst464 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst435)) 
		
		self._fConst465 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst461)) 
		
		self._fConst466 = (((self._fConst407 + np.float32(-0.51247865)) / self._fConst406) + np.float32(0.6896214)) 
		
		self._fConst467 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst435)) 
		
		self._fConst468 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst435)) 
		
		self._fConst469 = (((self._fConst407 + np.float32(-0.16840488)) / self._fConst406) + np.float32(1.0693583)) 
		
		self._fConst470 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst435)) 
		
		self._fConst471 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst435)) 
		
		self._fConst472 = (((self._fConst445 + np.float32(-3.1897273)) / self._fConst444) + np.float32(4.0767817)) 
		
		self._fConst473 = (np.float32(1.0) / self._fConst447) 
		
		self._fConst474 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst473)) 
		
		self._fConst475 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst454)) 
		
		self._fConst476 = (((self._fConst445 + np.float32(-0.74313045)) / self._fConst444) + np.float32(1.4500711)) 
		
		self._fConst477 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst473)) 
		
		self._fConst478 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst451)) 
		
		self._fConst479 = (((self._fConst445 + np.float32(-0.15748216)) / self._fConst444) + np.float32(0.9351402)) 
		
		self._fConst480 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst473)) 
		
		self._fConst481 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst448)) 
		
		self._fConst482 = np.tan((np.float32(123.69246) / self._fConst0)) 
		
		self._fConst483 = (np.float32(1.0) / self._fConst482) 
		
		self._fConst484 = (np.float32(1.0) / (((self._fConst483 + np.float32(0.15748216)) / self._fConst482) + np.float32(0.9351402))) 
		
		self._fConst485 = np.power(self._fConst482, np.float32(2.0)) 
		
		self._fConst486 = (np.float32(50.06381) / self._fConst485) 
		
		self._fConst487 = (self._fConst486 + np.float32(0.9351402)) 
		
		self._fConst488 = (np.float32(1.0) / (((self._fConst483 + np.float32(0.74313045)) / self._fConst482) + np.float32(1.4500711))) 
		
		self._fConst489 = (np.float32(11.0520525) / self._fConst485) 
		
		self._fConst490 = (self._fConst489 + np.float32(1.4500711)) 
		
		self._fConst491 = (np.float32(1.0) / (((self._fConst483 + np.float32(3.1897273)) / self._fConst482) + np.float32(4.0767817))) 
		
		self._fConst492 = (np.float32(0.0017661728) / self._fConst485) 
		
		self._fConst493 = (self._fConst492 + np.float32(0.0004076782)) 
		
		self._fConst494 = (np.float32(1.0) / (((self._fConst445 + np.float32(0.16840488)) / self._fConst444) + np.float32(1.0693583))) 
		
		self._fConst495 = (self._fConst473 + np.float32(53.53615)) 
		
		self._fConst496 = (np.float32(1.0) / (((self._fConst445 + np.float32(0.51247865)) / self._fConst444) + np.float32(0.6896214))) 
		
		self._fConst497 = (self._fConst473 + np.float32(7.6217313)) 
		
		self._fConst498 = (np.float32(1.0) / (((self._fConst445 + np.float32(0.78241307)) / self._fConst444) + np.float32(0.2452915))) 
		
		self._fConst499 = (np.float32(0.0001) / self._fConst447) 
		
		self._fConst500 = (self._fConst499 + np.float32(0.0004332272)) 
		
		self._fConst501 = (((self._fConst445 + np.float32(-0.78241307)) / self._fConst444) + np.float32(0.2452915)) 
		
		self._fConst502 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst473)) 
		
		self._fConst503 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst499)) 
		
		self._fConst504 = (((self._fConst445 + np.float32(-0.51247865)) / self._fConst444) + np.float32(0.6896214)) 
		
		self._fConst505 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst473)) 
		
		self._fConst506 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst473)) 
		
		self._fConst507 = (((self._fConst445 + np.float32(-0.16840488)) / self._fConst444) + np.float32(1.0693583)) 
		
		self._fConst508 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst473)) 
		
		self._fConst509 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst473)) 
		
		self._fConst510 = (((self._fConst483 + np.float32(-3.1897273)) / self._fConst482) + np.float32(4.0767817)) 
		
		self._fConst511 = (np.float32(1.0) / self._fConst485) 
		
		self._fConst512 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst511)) 
		
		self._fConst513 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst492)) 
		
		self._fConst514 = (((self._fConst483 + np.float32(-0.74313045)) / self._fConst482) + np.float32(1.4500711)) 
		
		self._fConst515 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst511)) 
		
		self._fConst516 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst489)) 
		
		self._fConst517 = (((self._fConst483 + np.float32(-0.15748216)) / self._fConst482) + np.float32(0.9351402)) 
		
		self._fConst518 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst511)) 
		
		self._fConst519 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst486)) 
		
		self._fConst520 = (np.float32(1.0) / (((self._fConst483 + np.float32(0.16840488)) / self._fConst482) + np.float32(1.0693583))) 
		
		self._fConst521 = (self._fConst511 + np.float32(53.53615)) 
		
		self._fConst522 = (np.float32(1.0) / (((self._fConst483 + np.float32(0.51247865)) / self._fConst482) + np.float32(0.6896214))) 
		
		self._fConst523 = (self._fConst511 + np.float32(7.6217313)) 
		
		self._fConst524 = (np.float32(1.0) / (((self._fConst483 + np.float32(0.78241307)) / self._fConst482) + np.float32(0.2452915))) 
		
		self._fConst525 = (np.float32(0.0001) / self._fConst485) 
		
		self._fConst526 = (self._fConst525 + np.float32(0.0004332272)) 
		
		self._fConst527 = (((self._fConst483 + np.float32(-0.78241307)) / self._fConst482) + np.float32(0.2452915)) 
		
		self._fConst528 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst511)) 
		
		self._fConst529 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst525)) 
		
		self._fConst530 = (((self._fConst483 + np.float32(-0.51247865)) / self._fConst482) + np.float32(0.6896214)) 
		
		self._fConst531 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst511)) 
		
		self._fConst532 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst511)) 
		
		self._fConst533 = (((self._fConst483 + np.float32(-0.16840488)) / self._fConst482) + np.float32(1.0693583)) 
		
		self._fConst534 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst511)) 
		
		self._fConst535 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst511)) 
		
	def tick(self, params: dict, state: dict, inputs: jnp.array) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = params["fHslider0"] 
		fSlow1 = params["fHslider1"] 
		fSlow2 = jnp.where((((jnp.float32(0.001) * fSlow1) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst1 / fSlow1))), jnp.float32(0.0)) 
		iSlow3 = jnp.int32(params["fCheckbox0"]) 
		fSlow4 = (jnp.float32(0.001) * params["fHslider2"]) 
		fSlow5 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider3"]))) 
		fSlow6 = params["fHslider4"] 
		fSlow7 = jnp.where(((fSlow6 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst15 / fSlow6))), jnp.float32(0.0)) 
		fSlow8 = ((jnp.float32(4.4e+02) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fHslider5"] + jnp.float32(-49.0))))) * (jnp.float32(1.0) - fSlow7)) 
		fSlow9 = (jnp.float32(0.001) * params["fHslider6"]) 
		fSlow10 = (jnp.float32(1.0) - fSlow2) 
		fRec5_temp = state["fRec5"] 
		fRec6_temp = state["fRec6"] 
		fRec9_temp = state["fRec9"] 
		fRec7_temp = state["fRec7"] 
		iVec0_temp = state["iVec0"] 
		fRec8_temp = state["fRec8"] 
		fRec10_temp = state["fRec10"] 
		fVec1_temp = state["fVec1"] 
		fRec4_temp = state["fRec4"] 
		fRec0_temp = state["fRec0"] 
		fRec11_temp = state["fRec11"] 
		fRec18_temp = state["fRec18"] 
		fRec25_temp = state["fRec25"] 
		fRec32_temp = state["fRec32"] 
		fRec39_temp = state["fRec39"] 
		fRec46_temp = state["fRec46"] 
		fRec53_temp = state["fRec53"] 
		fRec60_temp = state["fRec60"] 
		fRec67_temp = state["fRec67"] 
		fRec74_temp = state["fRec74"] 
		fRec81_temp = state["fRec81"] 
		fRec88_temp = state["fRec88"] 
		fRec95_temp = state["fRec95"] 
		fRec102_temp = state["fRec102"] 
		state["fRec5"] = (fSlow4 + (jnp.float32(0.999) * fRec5_temp)) 
		state["fRec6"] = (fSlow5 + (jnp.float32(0.999) * fRec6_temp)) 
		state["fRec9"] = ((fRec9_temp * fSlow7) + fSlow8) 
		fTemp0 = (self._fConst14 * state["fRec9"]) 
		fTemp1 = jnp.sin(fTemp0) 
		fTemp2 = jnp.cos(fTemp0) 
		state["fRec7"] = ((fRec8_temp * fTemp1) + (fRec7_temp * fTemp2)) 
		state["iVec0"] = jnp.int32(1) 
		state["fRec8"] = ((((jnp.int32(1) - iVec0_temp)) + (fRec8_temp * fTemp2)) - (fTemp1 * fRec7_temp)) 
		fTemp3 = (state["fRec6"] * state["fRec7"]) 
		state["fRec10"] = (fSlow9 + (jnp.float32(0.999) * fRec10_temp)) 
		fTemp4 = jnp.maximum(jnp.float32(-1.0), jnp.minimum(jnp.float32(1.0), (state["fRec5"] + (jnp.where((iSlow3 != 0), jnp.float32(0.0), fTemp3) * jnp.power(jnp.float32(1e+01), (jnp.float32(2.0) * state["fRec10"])))))) 
		fTemp5 = (fTemp4 * (jnp.float32(1.0) - (jnp.float32(0.33333334) * jnp.power(fTemp4, jnp.float32(2.0))))) 
		state["fVec1"] = jnp.float32(fTemp5) 
		state["fRec4"] = (((jnp.float32(0.995) * fRec4_temp) + fTemp5) - fVec1_temp) 
		fTemp6 = jnp.where((iSlow3 != 0), fTemp3, state["fRec4"]) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp6 - (self._fConst11 * ((self._fConst16 * state["fRec3"][2]) + (self._fConst18 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst11 * (((self._fConst13 * state["fRec3"][0]) + (self._fConst19 * state["fRec3"][1])) + (self._fConst13 * state["fRec3"][2]))) - (self._fConst8 * ((self._fConst20 * state["fRec2"][2]) + (self._fConst21 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst8 * (((self._fConst10 * state["fRec2"][0]) + (self._fConst22 * state["fRec2"][1])) + (self._fConst10 * state["fRec2"][2]))) - (self._fConst4 * ((self._fConst23 * state["fRec1"][2]) + (self._fConst24 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fRec0_temp * fSlow2) + (jnp.abs((self._fConst4 * (((self._fConst7 * state["fRec1"][0]) + (self._fConst25 * state["fRec1"][1])) + (self._fConst7 * state["fRec1"][2])))) * fSlow10)) 
		fVbargraph0 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec17"] = state["fRec17"].at[0].set((fTemp6 - (self._fConst42 * ((self._fConst45 * state["fRec17"][2]) + (self._fConst46 * state["fRec17"][1]))))) 
		state["fRec16"] = state["fRec16"].at[0].set(((self._fConst42 * (((self._fConst44 * state["fRec17"][0]) + (self._fConst47 * state["fRec17"][1])) + (self._fConst44 * state["fRec17"][2]))) - (self._fConst40 * ((self._fConst48 * state["fRec16"][2]) + (self._fConst49 * state["fRec16"][1]))))) 
		state["fRec15"] = state["fRec15"].at[0].set(((self._fConst40 * (((self._fConst41 * state["fRec16"][0]) + (self._fConst50 * state["fRec16"][1])) + (self._fConst41 * state["fRec16"][2]))) - (self._fConst38 * ((self._fConst51 * state["fRec15"][2]) + (self._fConst52 * state["fRec15"][1]))))) 
		fTemp7 = (self._fConst38 * (((self._fConst39 * state["fRec15"][0]) + (self._fConst53 * state["fRec15"][1])) + (self._fConst39 * state["fRec15"][2]))) 
		state["fRec14"] = state["fRec14"].at[0].set((fTemp7 - (self._fConst35 * ((self._fConst54 * state["fRec14"][2]) + (self._fConst56 * state["fRec14"][1]))))) 
		state["fRec13"] = state["fRec13"].at[0].set(((self._fConst35 * (((self._fConst37 * state["fRec14"][0]) + (self._fConst57 * state["fRec14"][1])) + (self._fConst37 * state["fRec14"][2]))) - (self._fConst32 * ((self._fConst58 * state["fRec13"][2]) + (self._fConst59 * state["fRec13"][1]))))) 
		state["fRec12"] = state["fRec12"].at[0].set(((self._fConst32 * (((self._fConst34 * state["fRec13"][0]) + (self._fConst60 * state["fRec13"][1])) + (self._fConst34 * state["fRec13"][2]))) - (self._fConst28 * ((self._fConst61 * state["fRec12"][2]) + (self._fConst62 * state["fRec12"][1]))))) 
		state["fRec11"] = ((fSlow2 * fRec11_temp) + (fSlow10 * jnp.abs((self._fConst28 * (((self._fConst31 * state["fRec12"][0]) + (self._fConst63 * state["fRec12"][1])) + (self._fConst31 * state["fRec12"][2])))))) 
		fVbargraph1 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec11"])))
		self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec24"] = state["fRec24"].at[0].set((fTemp7 - (self._fConst80 * ((self._fConst83 * state["fRec24"][2]) + (self._fConst84 * state["fRec24"][1]))))) 
		state["fRec23"] = state["fRec23"].at[0].set(((self._fConst80 * (((self._fConst82 * state["fRec24"][0]) + (self._fConst85 * state["fRec24"][1])) + (self._fConst82 * state["fRec24"][2]))) - (self._fConst78 * ((self._fConst86 * state["fRec23"][2]) + (self._fConst87 * state["fRec23"][1]))))) 
		state["fRec22"] = state["fRec22"].at[0].set(((self._fConst78 * (((self._fConst79 * state["fRec23"][0]) + (self._fConst88 * state["fRec23"][1])) + (self._fConst79 * state["fRec23"][2]))) - (self._fConst76 * ((self._fConst89 * state["fRec22"][2]) + (self._fConst90 * state["fRec22"][1]))))) 
		fTemp8 = (self._fConst76 * (((self._fConst77 * state["fRec22"][0]) + (self._fConst91 * state["fRec22"][1])) + (self._fConst77 * state["fRec22"][2]))) 
		state["fRec21"] = state["fRec21"].at[0].set((fTemp8 - (self._fConst73 * ((self._fConst92 * state["fRec21"][2]) + (self._fConst94 * state["fRec21"][1]))))) 
		state["fRec20"] = state["fRec20"].at[0].set(((self._fConst73 * (((self._fConst75 * state["fRec21"][0]) + (self._fConst95 * state["fRec21"][1])) + (self._fConst75 * state["fRec21"][2]))) - (self._fConst70 * ((self._fConst96 * state["fRec20"][2]) + (self._fConst97 * state["fRec20"][1]))))) 
		state["fRec19"] = state["fRec19"].at[0].set(((self._fConst70 * (((self._fConst72 * state["fRec20"][0]) + (self._fConst98 * state["fRec20"][1])) + (self._fConst72 * state["fRec20"][2]))) - (self._fConst66 * ((self._fConst99 * state["fRec19"][2]) + (self._fConst100 * state["fRec19"][1]))))) 
		state["fRec18"] = ((fSlow2 * fRec18_temp) + (fSlow10 * jnp.abs((self._fConst66 * (((self._fConst69 * state["fRec19"][0]) + (self._fConst101 * state["fRec19"][1])) + (self._fConst69 * state["fRec19"][2])))))) 
		fVbargraph2 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec18"])))
		self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec31"] = state["fRec31"].at[0].set((fTemp8 - (self._fConst118 * ((self._fConst121 * state["fRec31"][2]) + (self._fConst122 * state["fRec31"][1]))))) 
		state["fRec30"] = state["fRec30"].at[0].set(((self._fConst118 * (((self._fConst120 * state["fRec31"][0]) + (self._fConst123 * state["fRec31"][1])) + (self._fConst120 * state["fRec31"][2]))) - (self._fConst116 * ((self._fConst124 * state["fRec30"][2]) + (self._fConst125 * state["fRec30"][1]))))) 
		state["fRec29"] = state["fRec29"].at[0].set(((self._fConst116 * (((self._fConst117 * state["fRec30"][0]) + (self._fConst126 * state["fRec30"][1])) + (self._fConst117 * state["fRec30"][2]))) - (self._fConst114 * ((self._fConst127 * state["fRec29"][2]) + (self._fConst128 * state["fRec29"][1]))))) 
		fTemp9 = (self._fConst114 * (((self._fConst115 * state["fRec29"][0]) + (self._fConst129 * state["fRec29"][1])) + (self._fConst115 * state["fRec29"][2]))) 
		state["fRec28"] = state["fRec28"].at[0].set((fTemp9 - (self._fConst111 * ((self._fConst130 * state["fRec28"][2]) + (self._fConst132 * state["fRec28"][1]))))) 
		state["fRec27"] = state["fRec27"].at[0].set(((self._fConst111 * (((self._fConst113 * state["fRec28"][0]) + (self._fConst133 * state["fRec28"][1])) + (self._fConst113 * state["fRec28"][2]))) - (self._fConst108 * ((self._fConst134 * state["fRec27"][2]) + (self._fConst135 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((self._fConst108 * (((self._fConst110 * state["fRec27"][0]) + (self._fConst136 * state["fRec27"][1])) + (self._fConst110 * state["fRec27"][2]))) - (self._fConst104 * ((self._fConst137 * state["fRec26"][2]) + (self._fConst138 * state["fRec26"][1]))))) 
		state["fRec25"] = ((fSlow2 * fRec25_temp) + (fSlow10 * jnp.abs((self._fConst104 * (((self._fConst107 * state["fRec26"][0]) + (self._fConst139 * state["fRec26"][1])) + (self._fConst107 * state["fRec26"][2])))))) 
		fVbargraph3 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec25"])))
		self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec38"] = state["fRec38"].at[0].set((fTemp9 - (self._fConst156 * ((self._fConst159 * state["fRec38"][2]) + (self._fConst160 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst156 * (((self._fConst158 * state["fRec38"][0]) + (self._fConst161 * state["fRec38"][1])) + (self._fConst158 * state["fRec38"][2]))) - (self._fConst154 * ((self._fConst162 * state["fRec37"][2]) + (self._fConst163 * state["fRec37"][1]))))) 
		state["fRec36"] = state["fRec36"].at[0].set(((self._fConst154 * (((self._fConst155 * state["fRec37"][0]) + (self._fConst164 * state["fRec37"][1])) + (self._fConst155 * state["fRec37"][2]))) - (self._fConst152 * ((self._fConst165 * state["fRec36"][2]) + (self._fConst166 * state["fRec36"][1]))))) 
		fTemp10 = (self._fConst152 * (((self._fConst153 * state["fRec36"][0]) + (self._fConst167 * state["fRec36"][1])) + (self._fConst153 * state["fRec36"][2]))) 
		state["fRec35"] = state["fRec35"].at[0].set((fTemp10 - (self._fConst149 * ((self._fConst168 * state["fRec35"][2]) + (self._fConst170 * state["fRec35"][1]))))) 
		state["fRec34"] = state["fRec34"].at[0].set(((self._fConst149 * (((self._fConst151 * state["fRec35"][0]) + (self._fConst171 * state["fRec35"][1])) + (self._fConst151 * state["fRec35"][2]))) - (self._fConst146 * ((self._fConst172 * state["fRec34"][2]) + (self._fConst173 * state["fRec34"][1]))))) 
		state["fRec33"] = state["fRec33"].at[0].set(((self._fConst146 * (((self._fConst148 * state["fRec34"][0]) + (self._fConst174 * state["fRec34"][1])) + (self._fConst148 * state["fRec34"][2]))) - (self._fConst142 * ((self._fConst175 * state["fRec33"][2]) + (self._fConst176 * state["fRec33"][1]))))) 
		state["fRec32"] = ((fSlow2 * fRec32_temp) + (fSlow10 * jnp.abs((self._fConst142 * (((self._fConst145 * state["fRec33"][0]) + (self._fConst177 * state["fRec33"][1])) + (self._fConst145 * state["fRec33"][2])))))) 
		fVbargraph4 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec32"])))
		self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec45"] = state["fRec45"].at[0].set((fTemp10 - (self._fConst194 * ((self._fConst197 * state["fRec45"][2]) + (self._fConst198 * state["fRec45"][1]))))) 
		state["fRec44"] = state["fRec44"].at[0].set(((self._fConst194 * (((self._fConst196 * state["fRec45"][0]) + (self._fConst199 * state["fRec45"][1])) + (self._fConst196 * state["fRec45"][2]))) - (self._fConst192 * ((self._fConst200 * state["fRec44"][2]) + (self._fConst201 * state["fRec44"][1]))))) 
		state["fRec43"] = state["fRec43"].at[0].set(((self._fConst192 * (((self._fConst193 * state["fRec44"][0]) + (self._fConst202 * state["fRec44"][1])) + (self._fConst193 * state["fRec44"][2]))) - (self._fConst190 * ((self._fConst203 * state["fRec43"][2]) + (self._fConst204 * state["fRec43"][1]))))) 
		fTemp11 = (self._fConst190 * (((self._fConst191 * state["fRec43"][0]) + (self._fConst205 * state["fRec43"][1])) + (self._fConst191 * state["fRec43"][2]))) 
		state["fRec42"] = state["fRec42"].at[0].set((fTemp11 - (self._fConst187 * ((self._fConst206 * state["fRec42"][2]) + (self._fConst208 * state["fRec42"][1]))))) 
		state["fRec41"] = state["fRec41"].at[0].set(((self._fConst187 * (((self._fConst189 * state["fRec42"][0]) + (self._fConst209 * state["fRec42"][1])) + (self._fConst189 * state["fRec42"][2]))) - (self._fConst184 * ((self._fConst210 * state["fRec41"][2]) + (self._fConst211 * state["fRec41"][1]))))) 
		state["fRec40"] = state["fRec40"].at[0].set(((self._fConst184 * (((self._fConst186 * state["fRec41"][0]) + (self._fConst212 * state["fRec41"][1])) + (self._fConst186 * state["fRec41"][2]))) - (self._fConst180 * ((self._fConst213 * state["fRec40"][2]) + (self._fConst214 * state["fRec40"][1]))))) 
		state["fRec39"] = ((fSlow2 * fRec39_temp) + (fSlow10 * jnp.abs((self._fConst180 * (((self._fConst183 * state["fRec40"][0]) + (self._fConst215 * state["fRec40"][1])) + (self._fConst183 * state["fRec40"][2])))))) 
		fVbargraph5 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec39"])))
		self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec52"] = state["fRec52"].at[0].set((fTemp11 - (self._fConst232 * ((self._fConst235 * state["fRec52"][2]) + (self._fConst236 * state["fRec52"][1]))))) 
		state["fRec51"] = state["fRec51"].at[0].set(((self._fConst232 * (((self._fConst234 * state["fRec52"][0]) + (self._fConst237 * state["fRec52"][1])) + (self._fConst234 * state["fRec52"][2]))) - (self._fConst230 * ((self._fConst238 * state["fRec51"][2]) + (self._fConst239 * state["fRec51"][1]))))) 
		state["fRec50"] = state["fRec50"].at[0].set(((self._fConst230 * (((self._fConst231 * state["fRec51"][0]) + (self._fConst240 * state["fRec51"][1])) + (self._fConst231 * state["fRec51"][2]))) - (self._fConst228 * ((self._fConst241 * state["fRec50"][2]) + (self._fConst242 * state["fRec50"][1]))))) 
		fTemp12 = (self._fConst228 * (((self._fConst229 * state["fRec50"][0]) + (self._fConst243 * state["fRec50"][1])) + (self._fConst229 * state["fRec50"][2]))) 
		state["fRec49"] = state["fRec49"].at[0].set((fTemp12 - (self._fConst225 * ((self._fConst244 * state["fRec49"][2]) + (self._fConst246 * state["fRec49"][1]))))) 
		state["fRec48"] = state["fRec48"].at[0].set(((self._fConst225 * (((self._fConst227 * state["fRec49"][0]) + (self._fConst247 * state["fRec49"][1])) + (self._fConst227 * state["fRec49"][2]))) - (self._fConst222 * ((self._fConst248 * state["fRec48"][2]) + (self._fConst249 * state["fRec48"][1]))))) 
		state["fRec47"] = state["fRec47"].at[0].set(((self._fConst222 * (((self._fConst224 * state["fRec48"][0]) + (self._fConst250 * state["fRec48"][1])) + (self._fConst224 * state["fRec48"][2]))) - (self._fConst218 * ((self._fConst251 * state["fRec47"][2]) + (self._fConst252 * state["fRec47"][1]))))) 
		state["fRec46"] = ((fSlow2 * fRec46_temp) + (fSlow10 * jnp.abs((self._fConst218 * (((self._fConst221 * state["fRec47"][0]) + (self._fConst253 * state["fRec47"][1])) + (self._fConst221 * state["fRec47"][2])))))) 
		fVbargraph6 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec46"])))
		self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec59"] = state["fRec59"].at[0].set((fTemp12 - (self._fConst270 * ((self._fConst273 * state["fRec59"][2]) + (self._fConst274 * state["fRec59"][1]))))) 
		state["fRec58"] = state["fRec58"].at[0].set(((self._fConst270 * (((self._fConst272 * state["fRec59"][0]) + (self._fConst275 * state["fRec59"][1])) + (self._fConst272 * state["fRec59"][2]))) - (self._fConst268 * ((self._fConst276 * state["fRec58"][2]) + (self._fConst277 * state["fRec58"][1]))))) 
		state["fRec57"] = state["fRec57"].at[0].set(((self._fConst268 * (((self._fConst269 * state["fRec58"][0]) + (self._fConst278 * state["fRec58"][1])) + (self._fConst269 * state["fRec58"][2]))) - (self._fConst266 * ((self._fConst279 * state["fRec57"][2]) + (self._fConst280 * state["fRec57"][1]))))) 
		fTemp13 = (self._fConst266 * (((self._fConst267 * state["fRec57"][0]) + (self._fConst281 * state["fRec57"][1])) + (self._fConst267 * state["fRec57"][2]))) 
		state["fRec56"] = state["fRec56"].at[0].set((fTemp13 - (self._fConst263 * ((self._fConst282 * state["fRec56"][2]) + (self._fConst284 * state["fRec56"][1]))))) 
		state["fRec55"] = state["fRec55"].at[0].set(((self._fConst263 * (((self._fConst265 * state["fRec56"][0]) + (self._fConst285 * state["fRec56"][1])) + (self._fConst265 * state["fRec56"][2]))) - (self._fConst260 * ((self._fConst286 * state["fRec55"][2]) + (self._fConst287 * state["fRec55"][1]))))) 
		state["fRec54"] = state["fRec54"].at[0].set(((self._fConst260 * (((self._fConst262 * state["fRec55"][0]) + (self._fConst288 * state["fRec55"][1])) + (self._fConst262 * state["fRec55"][2]))) - (self._fConst256 * ((self._fConst289 * state["fRec54"][2]) + (self._fConst290 * state["fRec54"][1]))))) 
		state["fRec53"] = ((fSlow2 * fRec53_temp) + (fSlow10 * jnp.abs((self._fConst256 * (((self._fConst259 * state["fRec54"][0]) + (self._fConst291 * state["fRec54"][1])) + (self._fConst259 * state["fRec54"][2])))))) 
		fVbargraph7 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec53"])))
		self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec66"] = state["fRec66"].at[0].set((fTemp13 - (self._fConst308 * ((self._fConst311 * state["fRec66"][2]) + (self._fConst312 * state["fRec66"][1]))))) 
		state["fRec65"] = state["fRec65"].at[0].set(((self._fConst308 * (((self._fConst310 * state["fRec66"][0]) + (self._fConst313 * state["fRec66"][1])) + (self._fConst310 * state["fRec66"][2]))) - (self._fConst306 * ((self._fConst314 * state["fRec65"][2]) + (self._fConst315 * state["fRec65"][1]))))) 
		state["fRec64"] = state["fRec64"].at[0].set(((self._fConst306 * (((self._fConst307 * state["fRec65"][0]) + (self._fConst316 * state["fRec65"][1])) + (self._fConst307 * state["fRec65"][2]))) - (self._fConst304 * ((self._fConst317 * state["fRec64"][2]) + (self._fConst318 * state["fRec64"][1]))))) 
		fTemp14 = (self._fConst304 * (((self._fConst305 * state["fRec64"][0]) + (self._fConst319 * state["fRec64"][1])) + (self._fConst305 * state["fRec64"][2]))) 
		state["fRec63"] = state["fRec63"].at[0].set((fTemp14 - (self._fConst301 * ((self._fConst320 * state["fRec63"][2]) + (self._fConst322 * state["fRec63"][1]))))) 
		state["fRec62"] = state["fRec62"].at[0].set(((self._fConst301 * (((self._fConst303 * state["fRec63"][0]) + (self._fConst323 * state["fRec63"][1])) + (self._fConst303 * state["fRec63"][2]))) - (self._fConst298 * ((self._fConst324 * state["fRec62"][2]) + (self._fConst325 * state["fRec62"][1]))))) 
		state["fRec61"] = state["fRec61"].at[0].set(((self._fConst298 * (((self._fConst300 * state["fRec62"][0]) + (self._fConst326 * state["fRec62"][1])) + (self._fConst300 * state["fRec62"][2]))) - (self._fConst294 * ((self._fConst327 * state["fRec61"][2]) + (self._fConst328 * state["fRec61"][1]))))) 
		state["fRec60"] = ((fSlow2 * fRec60_temp) + (fSlow10 * jnp.abs((self._fConst294 * (((self._fConst297 * state["fRec61"][0]) + (self._fConst329 * state["fRec61"][1])) + (self._fConst297 * state["fRec61"][2])))))) 
		fVbargraph8 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec60"])))
		self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec73"] = state["fRec73"].at[0].set((fTemp14 - (self._fConst346 * ((self._fConst349 * state["fRec73"][2]) + (self._fConst350 * state["fRec73"][1]))))) 
		state["fRec72"] = state["fRec72"].at[0].set(((self._fConst346 * (((self._fConst348 * state["fRec73"][0]) + (self._fConst351 * state["fRec73"][1])) + (self._fConst348 * state["fRec73"][2]))) - (self._fConst344 * ((self._fConst352 * state["fRec72"][2]) + (self._fConst353 * state["fRec72"][1]))))) 
		state["fRec71"] = state["fRec71"].at[0].set(((self._fConst344 * (((self._fConst345 * state["fRec72"][0]) + (self._fConst354 * state["fRec72"][1])) + (self._fConst345 * state["fRec72"][2]))) - (self._fConst342 * ((self._fConst355 * state["fRec71"][2]) + (self._fConst356 * state["fRec71"][1]))))) 
		fTemp15 = (self._fConst342 * (((self._fConst343 * state["fRec71"][0]) + (self._fConst357 * state["fRec71"][1])) + (self._fConst343 * state["fRec71"][2]))) 
		state["fRec70"] = state["fRec70"].at[0].set((fTemp15 - (self._fConst339 * ((self._fConst358 * state["fRec70"][2]) + (self._fConst360 * state["fRec70"][1]))))) 
		state["fRec69"] = state["fRec69"].at[0].set(((self._fConst339 * (((self._fConst341 * state["fRec70"][0]) + (self._fConst361 * state["fRec70"][1])) + (self._fConst341 * state["fRec70"][2]))) - (self._fConst336 * ((self._fConst362 * state["fRec69"][2]) + (self._fConst363 * state["fRec69"][1]))))) 
		state["fRec68"] = state["fRec68"].at[0].set(((self._fConst336 * (((self._fConst338 * state["fRec69"][0]) + (self._fConst364 * state["fRec69"][1])) + (self._fConst338 * state["fRec69"][2]))) - (self._fConst332 * ((self._fConst365 * state["fRec68"][2]) + (self._fConst366 * state["fRec68"][1]))))) 
		state["fRec67"] = ((fSlow2 * fRec67_temp) + (fSlow10 * jnp.abs((self._fConst332 * (((self._fConst335 * state["fRec68"][0]) + (self._fConst367 * state["fRec68"][1])) + (self._fConst335 * state["fRec68"][2])))))) 
		fVbargraph9 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec67"])))
		self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec80"] = state["fRec80"].at[0].set((fTemp15 - (self._fConst384 * ((self._fConst387 * state["fRec80"][2]) + (self._fConst388 * state["fRec80"][1]))))) 
		state["fRec79"] = state["fRec79"].at[0].set(((self._fConst384 * (((self._fConst386 * state["fRec80"][0]) + (self._fConst389 * state["fRec80"][1])) + (self._fConst386 * state["fRec80"][2]))) - (self._fConst382 * ((self._fConst390 * state["fRec79"][2]) + (self._fConst391 * state["fRec79"][1]))))) 
		state["fRec78"] = state["fRec78"].at[0].set(((self._fConst382 * (((self._fConst383 * state["fRec79"][0]) + (self._fConst392 * state["fRec79"][1])) + (self._fConst383 * state["fRec79"][2]))) - (self._fConst380 * ((self._fConst393 * state["fRec78"][2]) + (self._fConst394 * state["fRec78"][1]))))) 
		fTemp16 = (self._fConst380 * (((self._fConst381 * state["fRec78"][0]) + (self._fConst395 * state["fRec78"][1])) + (self._fConst381 * state["fRec78"][2]))) 
		state["fRec77"] = state["fRec77"].at[0].set((fTemp16 - (self._fConst377 * ((self._fConst396 * state["fRec77"][2]) + (self._fConst398 * state["fRec77"][1]))))) 
		state["fRec76"] = state["fRec76"].at[0].set(((self._fConst377 * (((self._fConst379 * state["fRec77"][0]) + (self._fConst399 * state["fRec77"][1])) + (self._fConst379 * state["fRec77"][2]))) - (self._fConst374 * ((self._fConst400 * state["fRec76"][2]) + (self._fConst401 * state["fRec76"][1]))))) 
		state["fRec75"] = state["fRec75"].at[0].set(((self._fConst374 * (((self._fConst376 * state["fRec76"][0]) + (self._fConst402 * state["fRec76"][1])) + (self._fConst376 * state["fRec76"][2]))) - (self._fConst370 * ((self._fConst403 * state["fRec75"][2]) + (self._fConst404 * state["fRec75"][1]))))) 
		state["fRec74"] = ((fSlow2 * fRec74_temp) + (fSlow10 * jnp.abs((self._fConst370 * (((self._fConst373 * state["fRec75"][0]) + (self._fConst405 * state["fRec75"][1])) + (self._fConst373 * state["fRec75"][2])))))) 
		fVbargraph10 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec74"])))
		self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec87"] = state["fRec87"].at[0].set((fTemp16 - (self._fConst422 * ((self._fConst425 * state["fRec87"][2]) + (self._fConst426 * state["fRec87"][1]))))) 
		state["fRec86"] = state["fRec86"].at[0].set(((self._fConst422 * (((self._fConst424 * state["fRec87"][0]) + (self._fConst427 * state["fRec87"][1])) + (self._fConst424 * state["fRec87"][2]))) - (self._fConst420 * ((self._fConst428 * state["fRec86"][2]) + (self._fConst429 * state["fRec86"][1]))))) 
		state["fRec85"] = state["fRec85"].at[0].set(((self._fConst420 * (((self._fConst421 * state["fRec86"][0]) + (self._fConst430 * state["fRec86"][1])) + (self._fConst421 * state["fRec86"][2]))) - (self._fConst418 * ((self._fConst431 * state["fRec85"][2]) + (self._fConst432 * state["fRec85"][1]))))) 
		fTemp17 = (self._fConst418 * (((self._fConst419 * state["fRec85"][0]) + (self._fConst433 * state["fRec85"][1])) + (self._fConst419 * state["fRec85"][2]))) 
		state["fRec84"] = state["fRec84"].at[0].set((fTemp17 - (self._fConst415 * ((self._fConst434 * state["fRec84"][2]) + (self._fConst436 * state["fRec84"][1]))))) 
		state["fRec83"] = state["fRec83"].at[0].set(((self._fConst415 * (((self._fConst417 * state["fRec84"][0]) + (self._fConst437 * state["fRec84"][1])) + (self._fConst417 * state["fRec84"][2]))) - (self._fConst412 * ((self._fConst438 * state["fRec83"][2]) + (self._fConst439 * state["fRec83"][1]))))) 
		state["fRec82"] = state["fRec82"].at[0].set(((self._fConst412 * (((self._fConst414 * state["fRec83"][0]) + (self._fConst440 * state["fRec83"][1])) + (self._fConst414 * state["fRec83"][2]))) - (self._fConst408 * ((self._fConst441 * state["fRec82"][2]) + (self._fConst442 * state["fRec82"][1]))))) 
		state["fRec81"] = ((fSlow2 * fRec81_temp) + (fSlow10 * jnp.abs((self._fConst408 * (((self._fConst411 * state["fRec82"][0]) + (self._fConst443 * state["fRec82"][1])) + (self._fConst411 * state["fRec82"][2])))))) 
		fVbargraph11 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec81"])))
		self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec94"] = state["fRec94"].at[0].set((fTemp17 - (self._fConst460 * ((self._fConst463 * state["fRec94"][2]) + (self._fConst464 * state["fRec94"][1]))))) 
		state["fRec93"] = state["fRec93"].at[0].set(((self._fConst460 * (((self._fConst462 * state["fRec94"][0]) + (self._fConst465 * state["fRec94"][1])) + (self._fConst462 * state["fRec94"][2]))) - (self._fConst458 * ((self._fConst466 * state["fRec93"][2]) + (self._fConst467 * state["fRec93"][1]))))) 
		state["fRec92"] = state["fRec92"].at[0].set(((self._fConst458 * (((self._fConst459 * state["fRec93"][0]) + (self._fConst468 * state["fRec93"][1])) + (self._fConst459 * state["fRec93"][2]))) - (self._fConst456 * ((self._fConst469 * state["fRec92"][2]) + (self._fConst470 * state["fRec92"][1]))))) 
		fTemp18 = (self._fConst456 * (((self._fConst457 * state["fRec92"][0]) + (self._fConst471 * state["fRec92"][1])) + (self._fConst457 * state["fRec92"][2]))) 
		state["fRec91"] = state["fRec91"].at[0].set((fTemp18 - (self._fConst453 * ((self._fConst472 * state["fRec91"][2]) + (self._fConst474 * state["fRec91"][1]))))) 
		state["fRec90"] = state["fRec90"].at[0].set(((self._fConst453 * (((self._fConst455 * state["fRec91"][0]) + (self._fConst475 * state["fRec91"][1])) + (self._fConst455 * state["fRec91"][2]))) - (self._fConst450 * ((self._fConst476 * state["fRec90"][2]) + (self._fConst477 * state["fRec90"][1]))))) 
		state["fRec89"] = state["fRec89"].at[0].set(((self._fConst450 * (((self._fConst452 * state["fRec90"][0]) + (self._fConst478 * state["fRec90"][1])) + (self._fConst452 * state["fRec90"][2]))) - (self._fConst446 * ((self._fConst479 * state["fRec89"][2]) + (self._fConst480 * state["fRec89"][1]))))) 
		state["fRec88"] = ((fSlow2 * fRec88_temp) + (fSlow10 * jnp.abs((self._fConst446 * (((self._fConst449 * state["fRec89"][0]) + (self._fConst481 * state["fRec89"][1])) + (self._fConst449 * state["fRec89"][2])))))) 
		fVbargraph12 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec88"])))
		self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec101"] = state["fRec101"].at[0].set((fTemp18 - (self._fConst498 * ((self._fConst501 * state["fRec101"][2]) + (self._fConst502 * state["fRec101"][1]))))) 
		state["fRec100"] = state["fRec100"].at[0].set(((self._fConst498 * (((self._fConst500 * state["fRec101"][0]) + (self._fConst503 * state["fRec101"][1])) + (self._fConst500 * state["fRec101"][2]))) - (self._fConst496 * ((self._fConst504 * state["fRec100"][2]) + (self._fConst505 * state["fRec100"][1]))))) 
		state["fRec99"] = state["fRec99"].at[0].set(((self._fConst496 * (((self._fConst497 * state["fRec100"][0]) + (self._fConst506 * state["fRec100"][1])) + (self._fConst497 * state["fRec100"][2]))) - (self._fConst494 * ((self._fConst507 * state["fRec99"][2]) + (self._fConst508 * state["fRec99"][1]))))) 
		fTemp19 = (self._fConst494 * (((self._fConst495 * state["fRec99"][0]) + (self._fConst509 * state["fRec99"][1])) + (self._fConst495 * state["fRec99"][2]))) 
		state["fRec98"] = state["fRec98"].at[0].set((fTemp19 - (self._fConst491 * ((self._fConst510 * state["fRec98"][2]) + (self._fConst512 * state["fRec98"][1]))))) 
		state["fRec97"] = state["fRec97"].at[0].set(((self._fConst491 * (((self._fConst493 * state["fRec98"][0]) + (self._fConst513 * state["fRec98"][1])) + (self._fConst493 * state["fRec98"][2]))) - (self._fConst488 * ((self._fConst514 * state["fRec97"][2]) + (self._fConst515 * state["fRec97"][1]))))) 
		state["fRec96"] = state["fRec96"].at[0].set(((self._fConst488 * (((self._fConst490 * state["fRec97"][0]) + (self._fConst516 * state["fRec97"][1])) + (self._fConst490 * state["fRec97"][2]))) - (self._fConst484 * ((self._fConst517 * state["fRec96"][2]) + (self._fConst518 * state["fRec96"][1]))))) 
		state["fRec95"] = ((fSlow2 * fRec95_temp) + (fSlow10 * jnp.abs((self._fConst484 * (((self._fConst487 * state["fRec96"][0]) + (self._fConst519 * state["fRec96"][1])) + (self._fConst487 * state["fRec96"][2])))))) 
		fVbargraph13 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec95"])))
		self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec105"] = state["fRec105"].at[0].set((fTemp19 - (self._fConst524 * ((self._fConst527 * state["fRec105"][2]) + (self._fConst528 * state["fRec105"][1]))))) 
		state["fRec104"] = state["fRec104"].at[0].set(((self._fConst524 * (((self._fConst526 * state["fRec105"][0]) + (self._fConst529 * state["fRec105"][1])) + (self._fConst526 * state["fRec105"][2]))) - (self._fConst522 * ((self._fConst530 * state["fRec104"][2]) + (self._fConst531 * state["fRec104"][1]))))) 
		state["fRec103"] = state["fRec103"].at[0].set(((self._fConst522 * (((self._fConst523 * state["fRec104"][0]) + (self._fConst532 * state["fRec104"][1])) + (self._fConst523 * state["fRec104"][2]))) - (self._fConst520 * ((self._fConst533 * state["fRec103"][2]) + (self._fConst534 * state["fRec103"][1]))))) 
		state["fRec102"] = ((fSlow2 * fRec102_temp) + (fSlow10 * jnp.abs((self._fConst520 * (((self._fConst521 * state["fRec103"][0]) + (self._fConst535 * state["fRec103"][1])) + (self._fConst521 * state["fRec103"][2])))))) 
		fVbargraph14 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec102"])))
		self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		fTemp20 = fTemp6 
		_result0 = fTemp20 
		_result1 = fTemp20 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec17"] = jnp.roll(state["fRec17"], 1) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec15"] = jnp.roll(state["fRec15"], 1) 
		state["fRec14"] = jnp.roll(state["fRec14"], 1) 
		state["fRec13"] = jnp.roll(state["fRec13"], 1) 
		state["fRec12"] = jnp.roll(state["fRec12"], 1) 
		state["fRec24"] = jnp.roll(state["fRec24"], 1) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec22"] = jnp.roll(state["fRec22"], 1) 
		state["fRec21"] = jnp.roll(state["fRec21"], 1) 
		state["fRec20"] = jnp.roll(state["fRec20"], 1) 
		state["fRec19"] = jnp.roll(state["fRec19"], 1) 
		state["fRec31"] = jnp.roll(state["fRec31"], 1) 
		state["fRec30"] = jnp.roll(state["fRec30"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec28"] = jnp.roll(state["fRec28"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec26"] = jnp.roll(state["fRec26"], 1) 
		state["fRec38"] = jnp.roll(state["fRec38"], 1) 
		state["fRec37"] = jnp.roll(state["fRec37"], 1) 
		state["fRec36"] = jnp.roll(state["fRec36"], 1) 
		state["fRec35"] = jnp.roll(state["fRec35"], 1) 
		state["fRec34"] = jnp.roll(state["fRec34"], 1) 
		state["fRec33"] = jnp.roll(state["fRec33"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
		state["fRec43"] = jnp.roll(state["fRec43"], 1) 
		state["fRec42"] = jnp.roll(state["fRec42"], 1) 
		state["fRec41"] = jnp.roll(state["fRec41"], 1) 
		state["fRec40"] = jnp.roll(state["fRec40"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec50"] = jnp.roll(state["fRec50"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec48"] = jnp.roll(state["fRec48"], 1) 
		state["fRec47"] = jnp.roll(state["fRec47"], 1) 
		state["fRec59"] = jnp.roll(state["fRec59"], 1) 
		state["fRec58"] = jnp.roll(state["fRec58"], 1) 
		state["fRec57"] = jnp.roll(state["fRec57"], 1) 
		state["fRec56"] = jnp.roll(state["fRec56"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
		state["fRec54"] = jnp.roll(state["fRec54"], 1) 
		state["fRec66"] = jnp.roll(state["fRec66"], 1) 
		state["fRec65"] = jnp.roll(state["fRec65"], 1) 
		state["fRec64"] = jnp.roll(state["fRec64"], 1) 
		state["fRec63"] = jnp.roll(state["fRec63"], 1) 
		state["fRec62"] = jnp.roll(state["fRec62"], 1) 
		state["fRec61"] = jnp.roll(state["fRec61"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec72"] = jnp.roll(state["fRec72"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec70"] = jnp.roll(state["fRec70"], 1) 
		state["fRec69"] = jnp.roll(state["fRec69"], 1) 
		state["fRec68"] = jnp.roll(state["fRec68"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec76"] = jnp.roll(state["fRec76"], 1) 
		state["fRec75"] = jnp.roll(state["fRec75"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec84"] = jnp.roll(state["fRec84"], 1) 
		state["fRec83"] = jnp.roll(state["fRec83"], 1) 
		state["fRec82"] = jnp.roll(state["fRec82"], 1) 
		state["fRec94"] = jnp.roll(state["fRec94"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec92"] = jnp.roll(state["fRec92"], 1) 
		state["fRec91"] = jnp.roll(state["fRec91"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec98"] = jnp.roll(state["fRec98"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec96"] = jnp.roll(state["fRec96"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec103"] = jnp.roll(state["fRec103"], 1) 
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
	
	def add_soundfile(self, zone: str, ui_path: list[str], label: str, url: str):
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
				if module.has_rng("gumbel"):
					gumbel_noise = random.gumbel(module.make_rng("gumbel"), logits.shape, dtype=FAUSTFLOAT)
					logits_with_noise = logits + gumbel_noise
				else:
					logits_with_noise = logits
				probs = nn.softmax(logits_with_noise / tau)
				return jnp.dot(probs, step_values)
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
	
	def add_hbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float):
		# Bargraphs are output-only, no parameters needed
		pass
	
	def add_vbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float):
		# Bargraphs are output-only, no parameters needed
		pass

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
