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
		return 1
	
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
		ui_path.append("virtual_analog_oscillators") 
		ui_path.append("0x00") 
		ui_path.append("VIRTUAL ANALOG OSCILLATORS") 
		ui_path.append("Signal Levels") 
		self.add_vslider("fVslider8", ui_path, "Sawtooth", 1.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.append("Pulse Train") 
		self.add_button("fCheckbox2", ui_path, "Order 3", unnorm_funcs) 
		self.add_vslider("fVslider5", ui_path, "0x00", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider3", ui_path, "Duty Cycle", 0.5, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		self.add_vslider("fVslider7", ui_path, "Square", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider6", ui_path, "Triangle", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider0", ui_path, "Pink Noise", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider9", ui_path, "Ext Input", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Signal Parameters") 
		ui_path.append("0x00") 
		self.add_hslider("fHslider3", ui_path, "Mix Amplitude", -2e+01, -1.2e+02, 1e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "Frequency", 49.0, 1.0, 88.0, unnorm_funcs, "linear") 
		ui_path.pop()
		self.add_vslider("fVslider4", ui_path, "Detuning 1", -0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider2", ui_path, "Detuning 2", 0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider1", ui_path, "Portamento", 0.1, 0.001, 1e+01, unnorm_funcs, "log") 
		self.add_nentry("fEntry0", ui_path, "Saw Order", 2.0, 1.0, 4.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("MOOG VCF (Voltage Controlled Filter)") 
		ui_path.append("0x00") 
		self.add_button("fCheckbox0", ui_path, "Bypass", unnorm_funcs) 
		self.add_button("fCheckbox1", ui_path, "Use Biquads", unnorm_funcs) 
		self.add_button("fCheckbox3", ui_path, "Normalized Ladders", unnorm_funcs) 
		ui_path.pop()
		self.add_hslider("fHslider0", ui_path, "Corner Frequency", 25.0, 1.0, 88.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider1", ui_path, "Corner Resonance", 0.9, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider4", ui_path, "VCF Output Level", 5.0, -6e+01, 2e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("CONSTANT-Q SPECTRUM ANALYZER (6E), 15 bands spanning LP, 9 octaves below 16000 Hz, HP") 
		self.add_vbargraph("fVbargraph0", ui_path, "vbargraph0", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph1", ui_path, "vbargraph1", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph2", ui_path, "vbargraph2", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph3", ui_path, "vbargraph3", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph4", ui_path, "vbargraph4", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph5", ui_path, "vbargraph5", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph6", ui_path, "vbargraph6", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph7", ui_path, "vbargraph7", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph8", ui_path, "vbargraph8", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph9", ui_path, "vbargraph9", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph10", ui_path, "vbargraph10", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph11", ui_path, "vbargraph11", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph12", ui_path, "vbargraph12", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph13", ui_path, "vbargraph13", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph14", ui_path, "vbargraph14", -5e+01, 1e+01, unnorm_funcs) 
		ui_path.pop()
		ui_path.append("SPECTRUM ANALYZER CONTROLS") 
		self.add_hslider("fHslider5", ui_path, "Level Averaging Time", 1e+02, 1.0, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider6", ui_path, "Level dB Offset", 5e+01, 0.0, 1e+02, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		
		self._fConst1 = np.tan((np.float32(123.69246) / self._fConst0)) 
		
		self._fConst2 = np.power(self._fConst1, np.float32(2.0)) 
		
		self._fConst3 = (np.float32(1.0) / self._fConst2) 
		
		self._fConst4 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst3)) 
		
		self._fConst5 = (np.float32(1.0) / self._fConst1) 
		
		self._fConst6 = (((self._fConst5 + np.float32(-0.16840488)) / self._fConst1) + np.float32(1.0693583)) 
		
		self._fConst7 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.16840488)) / self._fConst1) + np.float32(1.0693583))) 
		
		self._fConst8 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst3)) 
		
		self._fConst9 = (((self._fConst5 + np.float32(-0.51247865)) / self._fConst1) + np.float32(0.6896214)) 
		
		self._fConst10 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.51247865)) / self._fConst1) + np.float32(0.6896214))) 
		
		self._fConst11 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst3)) 
		
		self._fConst12 = (((self._fConst5 + np.float32(-0.78241307)) / self._fConst1) + np.float32(0.2452915)) 
		
		self._fConst13 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.78241307)) / self._fConst1) + np.float32(0.2452915))) 
		
		self._fConst14 = np.tan((np.float32(196.34955) / self._fConst0)) 
		
		self._fConst15 = np.power(self._fConst14, np.float32(2.0)) 
		
		self._fConst16 = (np.float32(1.0) / self._fConst15) 
		
		self._fConst17 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst16)) 
		
		self._fConst18 = (np.float32(1.0) / self._fConst14) 
		
		self._fConst19 = (((self._fConst18 + np.float32(-0.16840488)) / self._fConst14) + np.float32(1.0693583)) 
		
		self._fConst20 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.16840488)) / self._fConst14) + np.float32(1.0693583))) 
		
		self._fConst21 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst16)) 
		
		self._fConst22 = (((self._fConst18 + np.float32(-0.51247865)) / self._fConst14) + np.float32(0.6896214)) 
		
		self._fConst23 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.51247865)) / self._fConst14) + np.float32(0.6896214))) 
		
		self._fConst24 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst16)) 
		
		self._fConst25 = (((self._fConst18 + np.float32(-0.78241307)) / self._fConst14) + np.float32(0.2452915)) 
		
		self._fConst26 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.78241307)) / self._fConst14) + np.float32(0.2452915))) 
		
		self._fConst27 = np.tan((np.float32(311.68546) / self._fConst0)) 
		
		self._fConst28 = np.power(self._fConst27, np.float32(2.0)) 
		
		self._fConst29 = (np.float32(1.0) / self._fConst28) 
		
		self._fConst30 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst29)) 
		
		self._fConst31 = (np.float32(1.0) / self._fConst27) 
		
		self._fConst32 = (((self._fConst31 + np.float32(-0.16840488)) / self._fConst27) + np.float32(1.0693583)) 
		
		self._fConst33 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.16840488)) / self._fConst27) + np.float32(1.0693583))) 
		
		self._fConst34 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst29)) 
		
		self._fConst35 = (((self._fConst31 + np.float32(-0.51247865)) / self._fConst27) + np.float32(0.6896214)) 
		
		self._fConst36 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.51247865)) / self._fConst27) + np.float32(0.6896214))) 
		
		self._fConst37 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst29)) 
		
		self._fConst38 = (((self._fConst31 + np.float32(-0.78241307)) / self._fConst27) + np.float32(0.2452915)) 
		
		self._fConst39 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.78241307)) / self._fConst27) + np.float32(0.2452915))) 
		
		self._fConst40 = np.tan((np.float32(494.76984) / self._fConst0)) 
		
		self._fConst41 = np.power(self._fConst40, np.float32(2.0)) 
		
		self._fConst42 = (np.float32(1.0) / self._fConst41) 
		
		self._fConst43 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst42)) 
		
		self._fConst44 = (np.float32(1.0) / self._fConst40) 
		
		self._fConst45 = (((self._fConst44 + np.float32(-0.16840488)) / self._fConst40) + np.float32(1.0693583)) 
		
		self._fConst46 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.16840488)) / self._fConst40) + np.float32(1.0693583))) 
		
		self._fConst47 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst42)) 
		
		self._fConst48 = (((self._fConst44 + np.float32(-0.51247865)) / self._fConst40) + np.float32(0.6896214)) 
		
		self._fConst49 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.51247865)) / self._fConst40) + np.float32(0.6896214))) 
		
		self._fConst50 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst42)) 
		
		self._fConst51 = (((self._fConst44 + np.float32(-0.78241307)) / self._fConst40) + np.float32(0.2452915)) 
		
		self._fConst52 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.78241307)) / self._fConst40) + np.float32(0.2452915))) 
		
		self._fConst53 = np.tan((np.float32(785.3982) / self._fConst0)) 
		
		self._fConst54 = np.power(self._fConst53, np.float32(2.0)) 
		
		self._fConst55 = (np.float32(1.0) / self._fConst54) 
		
		self._fConst56 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst55)) 
		
		self._fConst57 = (np.float32(1.0) / self._fConst53) 
		
		self._fConst58 = (((self._fConst57 + np.float32(-0.16840488)) / self._fConst53) + np.float32(1.0693583)) 
		
		self._fConst59 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.16840488)) / self._fConst53) + np.float32(1.0693583))) 
		
		self._fConst60 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst55)) 
		
		self._fConst61 = (((self._fConst57 + np.float32(-0.51247865)) / self._fConst53) + np.float32(0.6896214)) 
		
		self._fConst62 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.51247865)) / self._fConst53) + np.float32(0.6896214))) 
		
		self._fConst63 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst55)) 
		
		self._fConst64 = (((self._fConst57 + np.float32(-0.78241307)) / self._fConst53) + np.float32(0.2452915)) 
		
		self._fConst65 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.78241307)) / self._fConst53) + np.float32(0.2452915))) 
		
		self._fConst66 = np.tan((np.float32(1246.7418) / self._fConst0)) 
		
		self._fConst67 = np.power(self._fConst66, np.float32(2.0)) 
		
		self._fConst68 = (np.float32(1.0) / self._fConst67) 
		
		self._fConst69 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst68)) 
		
		self._fConst70 = (np.float32(1.0) / self._fConst66) 
		
		self._fConst71 = (((self._fConst70 + np.float32(-0.16840488)) / self._fConst66) + np.float32(1.0693583)) 
		
		self._fConst72 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.16840488)) / self._fConst66) + np.float32(1.0693583))) 
		
		self._fConst73 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst68)) 
		
		self._fConst74 = (((self._fConst70 + np.float32(-0.51247865)) / self._fConst66) + np.float32(0.6896214)) 
		
		self._fConst75 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.51247865)) / self._fConst66) + np.float32(0.6896214))) 
		
		self._fConst76 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst68)) 
		
		self._fConst77 = (((self._fConst70 + np.float32(-0.78241307)) / self._fConst66) + np.float32(0.2452915)) 
		
		self._fConst78 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.78241307)) / self._fConst66) + np.float32(0.2452915))) 
		
		self._fConst79 = np.tan((np.float32(1979.0793) / self._fConst0)) 
		
		self._fConst80 = np.power(self._fConst79, np.float32(2.0)) 
		
		self._fConst81 = (np.float32(1.0) / self._fConst80) 
		
		self._fConst82 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst81)) 
		
		self._fConst83 = (np.float32(1.0) / self._fConst79) 
		
		self._fConst84 = (((self._fConst83 + np.float32(-0.16840488)) / self._fConst79) + np.float32(1.0693583)) 
		
		self._fConst85 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.16840488)) / self._fConst79) + np.float32(1.0693583))) 
		
		self._fConst86 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst81)) 
		
		self._fConst87 = (((self._fConst83 + np.float32(-0.51247865)) / self._fConst79) + np.float32(0.6896214)) 
		
		self._fConst88 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.51247865)) / self._fConst79) + np.float32(0.6896214))) 
		
		self._fConst89 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst81)) 
		
		self._fConst90 = (((self._fConst83 + np.float32(-0.78241307)) / self._fConst79) + np.float32(0.2452915)) 
		
		self._fConst91 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.78241307)) / self._fConst79) + np.float32(0.2452915))) 
		
		self._fConst92 = np.tan((np.float32(3141.5928) / self._fConst0)) 
		
		self._fConst93 = np.power(self._fConst92, np.float32(2.0)) 
		
		self._fConst94 = (np.float32(1.0) / self._fConst93) 
		
		self._fConst95 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst94)) 
		
		self._fConst96 = (np.float32(1.0) / self._fConst92) 
		
		self._fConst97 = (((self._fConst96 + np.float32(-0.16840488)) / self._fConst92) + np.float32(1.0693583)) 
		
		self._fConst98 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.16840488)) / self._fConst92) + np.float32(1.0693583))) 
		
		self._fConst99 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst94)) 
		
		self._fConst100 = (((self._fConst96 + np.float32(-0.51247865)) / self._fConst92) + np.float32(0.6896214)) 
		
		self._fConst101 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.51247865)) / self._fConst92) + np.float32(0.6896214))) 
		
		self._fConst102 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst94)) 
		
		self._fConst103 = (((self._fConst96 + np.float32(-0.78241307)) / self._fConst92) + np.float32(0.2452915)) 
		
		self._fConst104 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.78241307)) / self._fConst92) + np.float32(0.2452915))) 
		
		self._fConst105 = np.tan((np.float32(4986.9673) / self._fConst0)) 
		
		self._fConst106 = np.power(self._fConst105, np.float32(2.0)) 
		
		self._fConst107 = (np.float32(1.0) / self._fConst106) 
		
		self._fConst108 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst107)) 
		
		self._fConst109 = (np.float32(1.0) / self._fConst105) 
		
		self._fConst110 = (((self._fConst109 + np.float32(-0.16840488)) / self._fConst105) + np.float32(1.0693583)) 
		
		self._fConst111 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.16840488)) / self._fConst105) + np.float32(1.0693583))) 
		
		self._fConst112 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst107)) 
		
		self._fConst113 = (((self._fConst109 + np.float32(-0.51247865)) / self._fConst105) + np.float32(0.6896214)) 
		
		self._fConst114 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.51247865)) / self._fConst105) + np.float32(0.6896214))) 
		
		self._fConst115 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst107)) 
		
		self._fConst116 = (((self._fConst109 + np.float32(-0.78241307)) / self._fConst105) + np.float32(0.2452915)) 
		
		self._fConst117 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.78241307)) / self._fConst105) + np.float32(0.2452915))) 
		
		self._fConst118 = np.tan((np.float32(7916.3174) / self._fConst0)) 
		
		self._fConst119 = np.power(self._fConst118, np.float32(2.0)) 
		
		self._fConst120 = (np.float32(1.0) / self._fConst119) 
		
		self._fConst121 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst120)) 
		
		self._fConst122 = (np.float32(1.0) / self._fConst118) 
		
		self._fConst123 = (((self._fConst122 + np.float32(-0.16840488)) / self._fConst118) + np.float32(1.0693583)) 
		
		self._fConst124 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.16840488)) / self._fConst118) + np.float32(1.0693583))) 
		
		self._fConst125 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst120)) 
		
		self._fConst126 = (((self._fConst122 + np.float32(-0.51247865)) / self._fConst118) + np.float32(0.6896214)) 
		
		self._fConst127 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.51247865)) / self._fConst118) + np.float32(0.6896214))) 
		
		self._fConst128 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst120)) 
		
		self._fConst129 = (((self._fConst122 + np.float32(-0.78241307)) / self._fConst118) + np.float32(0.2452915)) 
		
		self._fConst130 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.78241307)) / self._fConst118) + np.float32(0.2452915))) 
		
		self._fConst131 = np.tan((np.float32(12566.371) / self._fConst0)) 
		
		self._fConst132 = np.power(self._fConst131, np.float32(2.0)) 
		
		self._fConst133 = (np.float32(1.0) / self._fConst132) 
		
		self._fConst134 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst133)) 
		
		self._fConst135 = (np.float32(1.0) / self._fConst131) 
		
		self._fConst136 = (((self._fConst135 + np.float32(-0.16840488)) / self._fConst131) + np.float32(1.0693583)) 
		
		self._fConst137 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.16840488)) / self._fConst131) + np.float32(1.0693583))) 
		
		self._fConst138 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst133)) 
		
		self._fConst139 = (((self._fConst135 + np.float32(-0.51247865)) / self._fConst131) + np.float32(0.6896214)) 
		
		self._fConst140 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.51247865)) / self._fConst131) + np.float32(0.6896214))) 
		
		self._fConst141 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst133)) 
		
		self._fConst142 = (((self._fConst135 + np.float32(-0.78241307)) / self._fConst131) + np.float32(0.2452915)) 
		
		self._fConst143 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.78241307)) / self._fConst131) + np.float32(0.2452915))) 
		
		self._fConst144 = np.tan((np.float32(19947.87) / self._fConst0)) 
		
		self._fConst145 = np.power(self._fConst144, np.float32(2.0)) 
		
		self._fConst146 = (np.float32(1.0) / self._fConst145) 
		
		self._fConst147 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst146)) 
		
		self._fConst148 = (np.float32(1.0) / self._fConst144) 
		
		self._fConst149 = (((self._fConst148 + np.float32(-0.16840488)) / self._fConst144) + np.float32(1.0693583)) 
		
		self._fConst150 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.16840488)) / self._fConst144) + np.float32(1.0693583))) 
		
		self._fConst151 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst146)) 
		
		self._fConst152 = (((self._fConst148 + np.float32(-0.51247865)) / self._fConst144) + np.float32(0.6896214)) 
		
		self._fConst153 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.51247865)) / self._fConst144) + np.float32(0.6896214))) 
		
		self._fConst154 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst146)) 
		
		self._fConst155 = (((self._fConst148 + np.float32(-0.78241307)) / self._fConst144) + np.float32(0.2452915)) 
		
		self._fConst156 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.78241307)) / self._fConst144) + np.float32(0.2452915))) 
		
		self._fConst157 = np.tan((np.float32(31665.27) / self._fConst0)) 
		
		self._fConst158 = np.power(self._fConst157, np.float32(2.0)) 
		
		self._fConst159 = (np.float32(1.0) / self._fConst158) 
		
		self._fConst160 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst159)) 
		
		self._fConst161 = (np.float32(1.0) / self._fConst157) 
		
		self._fConst162 = (((self._fConst161 + np.float32(-0.16840488)) / self._fConst157) + np.float32(1.0693583)) 
		
		self._fConst163 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.16840488)) / self._fConst157) + np.float32(1.0693583))) 
		
		self._fConst164 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst159)) 
		
		self._fConst165 = (((self._fConst161 + np.float32(-0.51247865)) / self._fConst157) + np.float32(0.6896214)) 
		
		self._fConst166 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.51247865)) / self._fConst157) + np.float32(0.6896214))) 
		
		self._fConst167 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst159)) 
		
		self._fConst168 = (((self._fConst161 + np.float32(-0.78241307)) / self._fConst157) + np.float32(0.2452915)) 
		
		self._fConst169 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.78241307)) / self._fConst157) + np.float32(0.2452915))) 
		
		self._fConst170 = np.tan((np.float32(50265.484) / self._fConst0)) 
		
		self._fConst171 = np.power(self._fConst170, np.float32(2.0)) 
		
		self._fConst172 = (np.float32(1.0) / self._fConst171) 
		
		self._fConst173 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst172)) 
		
		self._fConst174 = (np.float32(1.0) / self._fConst170) 
		
		self._fConst175 = (((self._fConst174 + np.float32(-0.16840488)) / self._fConst170) + np.float32(1.0693583)) 
		
		self._fConst176 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.16840488)) / self._fConst170) + np.float32(1.0693583))) 
		
		self._fConst177 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst172)) 
		
		self._fConst178 = (((self._fConst174 + np.float32(-0.51247865)) / self._fConst170) + np.float32(0.6896214)) 
		
		self._fConst179 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.51247865)) / self._fConst170) + np.float32(0.6896214))) 
		
		self._fConst180 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst172)) 
		
		self._fConst181 = (((self._fConst174 + np.float32(-0.78241307)) / self._fConst170) + np.float32(0.2452915)) 
		
		self._fConst182 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.78241307)) / self._fConst170) + np.float32(0.2452915))) 
		
		self._fConst183 = (np.float32(6.2831855) / self._fConst0) 
		
		self._fConst184 = (np.float32(1.0) / self._fConst0) 
		
		self._fConst185 = (np.float32(0.013888889) * self._fConst0) 
		
		self._fConst186 = (np.float32(0.5) * self._fConst0) 
		
		self._fConst187 = (np.float32(0.25) * self._fConst0) 
		
		self._fConst188 = (np.float32(1.3333334) / self._fConst0) 
		
		self._fConst189 = (np.float32(0.083333336) * self._fConst0) 
		
		self._fConst190 = (np.float32(0.041666668) * np.power(self._fConst0, np.float32(2.0))) 
		
		self._fConst191 = (np.float32(0.0052083335) * np.power(self._fConst0, np.float32(3.0))) 
		
		self._fConst192 = (np.float32(3.1415927) / self._fConst0) 
		
		self._fConst193 = (np.float32(0.0001) / self._fConst171) 
		
		self._fConst194 = (self._fConst193 + np.float32(0.0004332272)) 
		
		self._fConst195 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst193)) 
		
		self._fConst196 = (self._fConst172 + np.float32(7.6217313)) 
		
		self._fConst197 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst172)) 
		
		self._fConst198 = (self._fConst172 + np.float32(53.53615)) 
		
		self._fConst199 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst172)) 
		
		self._fConst200 = (np.float32(0.0001) / self._fConst158) 
		
		self._fConst201 = (self._fConst200 + np.float32(0.0004332272)) 
		
		self._fConst202 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst200)) 
		
		self._fConst203 = (self._fConst159 + np.float32(7.6217313)) 
		
		self._fConst204 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst159)) 
		
		self._fConst205 = (self._fConst159 + np.float32(53.53615)) 
		
		self._fConst206 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst159)) 
		
		self._fConst207 = (np.float32(0.0001) / self._fConst145) 
		
		self._fConst208 = (self._fConst207 + np.float32(0.0004332272)) 
		
		self._fConst209 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst207)) 
		
		self._fConst210 = (self._fConst146 + np.float32(7.6217313)) 
		
		self._fConst211 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst146)) 
		
		self._fConst212 = (self._fConst146 + np.float32(53.53615)) 
		
		self._fConst213 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst146)) 
		
		self._fConst214 = (np.float32(0.0001) / self._fConst132) 
		
		self._fConst215 = (self._fConst214 + np.float32(0.0004332272)) 
		
		self._fConst216 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst214)) 
		
		self._fConst217 = (self._fConst133 + np.float32(7.6217313)) 
		
		self._fConst218 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst133)) 
		
		self._fConst219 = (self._fConst133 + np.float32(53.53615)) 
		
		self._fConst220 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst133)) 
		
		self._fConst221 = (np.float32(0.0001) / self._fConst119) 
		
		self._fConst222 = (self._fConst221 + np.float32(0.0004332272)) 
		
		self._fConst223 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst221)) 
		
		self._fConst224 = (self._fConst120 + np.float32(7.6217313)) 
		
		self._fConst225 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst120)) 
		
		self._fConst226 = (self._fConst120 + np.float32(53.53615)) 
		
		self._fConst227 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst120)) 
		
		self._fConst228 = (np.float32(0.0001) / self._fConst106) 
		
		self._fConst229 = (self._fConst228 + np.float32(0.0004332272)) 
		
		self._fConst230 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst228)) 
		
		self._fConst231 = (self._fConst107 + np.float32(7.6217313)) 
		
		self._fConst232 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst107)) 
		
		self._fConst233 = (self._fConst107 + np.float32(53.53615)) 
		
		self._fConst234 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst107)) 
		
		self._fConst235 = (np.float32(0.0001) / self._fConst93) 
		
		self._fConst236 = (self._fConst235 + np.float32(0.0004332272)) 
		
		self._fConst237 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst235)) 
		
		self._fConst238 = (self._fConst94 + np.float32(7.6217313)) 
		
		self._fConst239 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst94)) 
		
		self._fConst240 = (self._fConst94 + np.float32(53.53615)) 
		
		self._fConst241 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst94)) 
		
		self._fConst242 = (np.float32(0.0001) / self._fConst80) 
		
		self._fConst243 = (self._fConst242 + np.float32(0.0004332272)) 
		
		self._fConst244 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst242)) 
		
		self._fConst245 = (self._fConst81 + np.float32(7.6217313)) 
		
		self._fConst246 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst81)) 
		
		self._fConst247 = (self._fConst81 + np.float32(53.53615)) 
		
		self._fConst248 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst81)) 
		
		self._fConst249 = (np.float32(0.0001) / self._fConst67) 
		
		self._fConst250 = (self._fConst249 + np.float32(0.0004332272)) 
		
		self._fConst251 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst249)) 
		
		self._fConst252 = (self._fConst68 + np.float32(7.6217313)) 
		
		self._fConst253 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst68)) 
		
		self._fConst254 = (self._fConst68 + np.float32(53.53615)) 
		
		self._fConst255 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst68)) 
		
		self._fConst256 = (np.float32(0.0001) / self._fConst54) 
		
		self._fConst257 = (self._fConst256 + np.float32(0.0004332272)) 
		
		self._fConst258 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst256)) 
		
		self._fConst259 = (self._fConst55 + np.float32(7.6217313)) 
		
		self._fConst260 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst55)) 
		
		self._fConst261 = (self._fConst55 + np.float32(53.53615)) 
		
		self._fConst262 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst55)) 
		
		self._fConst263 = (np.float32(0.0001) / self._fConst41) 
		
		self._fConst264 = (self._fConst263 + np.float32(0.0004332272)) 
		
		self._fConst265 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst263)) 
		
		self._fConst266 = (self._fConst42 + np.float32(7.6217313)) 
		
		self._fConst267 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst42)) 
		
		self._fConst268 = (self._fConst42 + np.float32(53.53615)) 
		
		self._fConst269 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst42)) 
		
		self._fConst270 = (np.float32(0.0001) / self._fConst28) 
		
		self._fConst271 = (self._fConst270 + np.float32(0.0004332272)) 
		
		self._fConst272 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst270)) 
		
		self._fConst273 = (self._fConst29 + np.float32(7.6217313)) 
		
		self._fConst274 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst29)) 
		
		self._fConst275 = (self._fConst29 + np.float32(53.53615)) 
		
		self._fConst276 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst29)) 
		
		self._fConst277 = (np.float32(0.0001) / self._fConst15) 
		
		self._fConst278 = (self._fConst277 + np.float32(0.0004332272)) 
		
		self._fConst279 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst277)) 
		
		self._fConst280 = (self._fConst16 + np.float32(7.6217313)) 
		
		self._fConst281 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst16)) 
		
		self._fConst282 = (self._fConst16 + np.float32(53.53615)) 
		
		self._fConst283 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst16)) 
		
		self._fConst284 = (np.float32(0.0001) / self._fConst2) 
		
		self._fConst285 = (self._fConst284 + np.float32(0.0004332272)) 
		
		self._fConst286 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst284)) 
		
		self._fConst287 = (self._fConst3 + np.float32(7.6217313)) 
		
		self._fConst288 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst3)) 
		
		self._fConst289 = (self._fConst3 + np.float32(53.53615)) 
		
		self._fConst290 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst3)) 
		
		self._fConst291 = (np.float32(1e+03) / self._fConst0) 
		
		self._fConst292 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst3)) 
		
		self._fConst293 = (((self._fConst5 + np.float32(-0.15748216)) / self._fConst1) + np.float32(0.9351402)) 
		
		self._fConst294 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.15748216)) / self._fConst1) + np.float32(0.9351402))) 
		
		self._fConst295 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst3)) 
		
		self._fConst296 = (((self._fConst5 + np.float32(-0.74313045)) / self._fConst1) + np.float32(1.4500711)) 
		
		self._fConst297 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.74313045)) / self._fConst1) + np.float32(1.4500711))) 
		
		self._fConst298 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst3)) 
		
		self._fConst299 = (((self._fConst5 + np.float32(-3.1897273)) / self._fConst1) + np.float32(4.0767817)) 
		
		self._fConst300 = (np.float32(1.0) / (((self._fConst5 + np.float32(3.1897273)) / self._fConst1) + np.float32(4.0767817))) 
		
		self._fConst301 = (np.float32(0.0017661728) / self._fConst2) 
		
		self._fConst302 = (self._fConst301 + np.float32(0.0004076782)) 
		
		self._fConst303 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst301)) 
		
		self._fConst304 = (np.float32(11.0520525) / self._fConst2) 
		
		self._fConst305 = (self._fConst304 + np.float32(1.4500711)) 
		
		self._fConst306 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst304)) 
		
		self._fConst307 = (np.float32(50.06381) / self._fConst2) 
		
		self._fConst308 = (self._fConst307 + np.float32(0.9351402)) 
		
		self._fConst309 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst307)) 
		
		self._fConst310 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst16)) 
		
		self._fConst311 = (((self._fConst18 + np.float32(-0.15748216)) / self._fConst14) + np.float32(0.9351402)) 
		
		self._fConst312 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.15748216)) / self._fConst14) + np.float32(0.9351402))) 
		
		self._fConst313 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst16)) 
		
		self._fConst314 = (((self._fConst18 + np.float32(-0.74313045)) / self._fConst14) + np.float32(1.4500711)) 
		
		self._fConst315 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.74313045)) / self._fConst14) + np.float32(1.4500711))) 
		
		self._fConst316 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst16)) 
		
		self._fConst317 = (((self._fConst18 + np.float32(-3.1897273)) / self._fConst14) + np.float32(4.0767817)) 
		
		self._fConst318 = (np.float32(1.0) / (((self._fConst18 + np.float32(3.1897273)) / self._fConst14) + np.float32(4.0767817))) 
		
		self._fConst319 = (np.float32(0.0017661728) / self._fConst15) 
		
		self._fConst320 = (self._fConst319 + np.float32(0.0004076782)) 
		
		self._fConst321 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst319)) 
		
		self._fConst322 = (np.float32(11.0520525) / self._fConst15) 
		
		self._fConst323 = (self._fConst322 + np.float32(1.4500711)) 
		
		self._fConst324 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst322)) 
		
		self._fConst325 = (np.float32(50.06381) / self._fConst15) 
		
		self._fConst326 = (self._fConst325 + np.float32(0.9351402)) 
		
		self._fConst327 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst325)) 
		
		self._fConst328 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst29)) 
		
		self._fConst329 = (((self._fConst31 + np.float32(-0.15748216)) / self._fConst27) + np.float32(0.9351402)) 
		
		self._fConst330 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.15748216)) / self._fConst27) + np.float32(0.9351402))) 
		
		self._fConst331 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst29)) 
		
		self._fConst332 = (((self._fConst31 + np.float32(-0.74313045)) / self._fConst27) + np.float32(1.4500711)) 
		
		self._fConst333 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.74313045)) / self._fConst27) + np.float32(1.4500711))) 
		
		self._fConst334 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst29)) 
		
		self._fConst335 = (((self._fConst31 + np.float32(-3.1897273)) / self._fConst27) + np.float32(4.0767817)) 
		
		self._fConst336 = (np.float32(1.0) / (((self._fConst31 + np.float32(3.1897273)) / self._fConst27) + np.float32(4.0767817))) 
		
		self._fConst337 = (np.float32(0.0017661728) / self._fConst28) 
		
		self._fConst338 = (self._fConst337 + np.float32(0.0004076782)) 
		
		self._fConst339 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst337)) 
		
		self._fConst340 = (np.float32(11.0520525) / self._fConst28) 
		
		self._fConst341 = (self._fConst340 + np.float32(1.4500711)) 
		
		self._fConst342 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst340)) 
		
		self._fConst343 = (np.float32(50.06381) / self._fConst28) 
		
		self._fConst344 = (self._fConst343 + np.float32(0.9351402)) 
		
		self._fConst345 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst343)) 
		
		self._fConst346 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst42)) 
		
		self._fConst347 = (((self._fConst44 + np.float32(-0.15748216)) / self._fConst40) + np.float32(0.9351402)) 
		
		self._fConst348 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.15748216)) / self._fConst40) + np.float32(0.9351402))) 
		
		self._fConst349 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst42)) 
		
		self._fConst350 = (((self._fConst44 + np.float32(-0.74313045)) / self._fConst40) + np.float32(1.4500711)) 
		
		self._fConst351 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.74313045)) / self._fConst40) + np.float32(1.4500711))) 
		
		self._fConst352 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst42)) 
		
		self._fConst353 = (((self._fConst44 + np.float32(-3.1897273)) / self._fConst40) + np.float32(4.0767817)) 
		
		self._fConst354 = (np.float32(1.0) / (((self._fConst44 + np.float32(3.1897273)) / self._fConst40) + np.float32(4.0767817))) 
		
		self._fConst355 = (np.float32(0.0017661728) / self._fConst41) 
		
		self._fConst356 = (self._fConst355 + np.float32(0.0004076782)) 
		
		self._fConst357 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst355)) 
		
		self._fConst358 = (np.float32(11.0520525) / self._fConst41) 
		
		self._fConst359 = (self._fConst358 + np.float32(1.4500711)) 
		
		self._fConst360 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst358)) 
		
		self._fConst361 = (np.float32(50.06381) / self._fConst41) 
		
		self._fConst362 = (self._fConst361 + np.float32(0.9351402)) 
		
		self._fConst363 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst361)) 
		
		self._fConst364 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst55)) 
		
		self._fConst365 = (((self._fConst57 + np.float32(-0.15748216)) / self._fConst53) + np.float32(0.9351402)) 
		
		self._fConst366 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.15748216)) / self._fConst53) + np.float32(0.9351402))) 
		
		self._fConst367 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst55)) 
		
		self._fConst368 = (((self._fConst57 + np.float32(-0.74313045)) / self._fConst53) + np.float32(1.4500711)) 
		
		self._fConst369 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.74313045)) / self._fConst53) + np.float32(1.4500711))) 
		
		self._fConst370 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst55)) 
		
		self._fConst371 = (((self._fConst57 + np.float32(-3.1897273)) / self._fConst53) + np.float32(4.0767817)) 
		
		self._fConst372 = (np.float32(1.0) / (((self._fConst57 + np.float32(3.1897273)) / self._fConst53) + np.float32(4.0767817))) 
		
		self._fConst373 = (np.float32(0.0017661728) / self._fConst54) 
		
		self._fConst374 = (self._fConst373 + np.float32(0.0004076782)) 
		
		self._fConst375 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst373)) 
		
		self._fConst376 = (np.float32(11.0520525) / self._fConst54) 
		
		self._fConst377 = (self._fConst376 + np.float32(1.4500711)) 
		
		self._fConst378 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst376)) 
		
		self._fConst379 = (np.float32(50.06381) / self._fConst54) 
		
		self._fConst380 = (self._fConst379 + np.float32(0.9351402)) 
		
		self._fConst381 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst379)) 
		
		self._fConst382 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst68)) 
		
		self._fConst383 = (((self._fConst70 + np.float32(-0.15748216)) / self._fConst66) + np.float32(0.9351402)) 
		
		self._fConst384 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.15748216)) / self._fConst66) + np.float32(0.9351402))) 
		
		self._fConst385 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst68)) 
		
		self._fConst386 = (((self._fConst70 + np.float32(-0.74313045)) / self._fConst66) + np.float32(1.4500711)) 
		
		self._fConst387 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.74313045)) / self._fConst66) + np.float32(1.4500711))) 
		
		self._fConst388 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst68)) 
		
		self._fConst389 = (((self._fConst70 + np.float32(-3.1897273)) / self._fConst66) + np.float32(4.0767817)) 
		
		self._fConst390 = (np.float32(1.0) / (((self._fConst70 + np.float32(3.1897273)) / self._fConst66) + np.float32(4.0767817))) 
		
		self._fConst391 = (np.float32(0.0017661728) / self._fConst67) 
		
		self._fConst392 = (self._fConst391 + np.float32(0.0004076782)) 
		
		self._fConst393 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst391)) 
		
		self._fConst394 = (np.float32(11.0520525) / self._fConst67) 
		
		self._fConst395 = (self._fConst394 + np.float32(1.4500711)) 
		
		self._fConst396 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst394)) 
		
		self._fConst397 = (np.float32(50.06381) / self._fConst67) 
		
		self._fConst398 = (self._fConst397 + np.float32(0.9351402)) 
		
		self._fConst399 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst397)) 
		
		self._fConst400 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst81)) 
		
		self._fConst401 = (((self._fConst83 + np.float32(-0.15748216)) / self._fConst79) + np.float32(0.9351402)) 
		
		self._fConst402 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.15748216)) / self._fConst79) + np.float32(0.9351402))) 
		
		self._fConst403 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst81)) 
		
		self._fConst404 = (((self._fConst83 + np.float32(-0.74313045)) / self._fConst79) + np.float32(1.4500711)) 
		
		self._fConst405 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.74313045)) / self._fConst79) + np.float32(1.4500711))) 
		
		self._fConst406 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst81)) 
		
		self._fConst407 = (((self._fConst83 + np.float32(-3.1897273)) / self._fConst79) + np.float32(4.0767817)) 
		
		self._fConst408 = (np.float32(1.0) / (((self._fConst83 + np.float32(3.1897273)) / self._fConst79) + np.float32(4.0767817))) 
		
		self._fConst409 = (np.float32(0.0017661728) / self._fConst80) 
		
		self._fConst410 = (self._fConst409 + np.float32(0.0004076782)) 
		
		self._fConst411 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst409)) 
		
		self._fConst412 = (np.float32(11.0520525) / self._fConst80) 
		
		self._fConst413 = (self._fConst412 + np.float32(1.4500711)) 
		
		self._fConst414 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst412)) 
		
		self._fConst415 = (np.float32(50.06381) / self._fConst80) 
		
		self._fConst416 = (self._fConst415 + np.float32(0.9351402)) 
		
		self._fConst417 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst415)) 
		
		self._fConst418 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst94)) 
		
		self._fConst419 = (((self._fConst96 + np.float32(-0.15748216)) / self._fConst92) + np.float32(0.9351402)) 
		
		self._fConst420 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.15748216)) / self._fConst92) + np.float32(0.9351402))) 
		
		self._fConst421 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst94)) 
		
		self._fConst422 = (((self._fConst96 + np.float32(-0.74313045)) / self._fConst92) + np.float32(1.4500711)) 
		
		self._fConst423 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.74313045)) / self._fConst92) + np.float32(1.4500711))) 
		
		self._fConst424 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst94)) 
		
		self._fConst425 = (((self._fConst96 + np.float32(-3.1897273)) / self._fConst92) + np.float32(4.0767817)) 
		
		self._fConst426 = (np.float32(1.0) / (((self._fConst96 + np.float32(3.1897273)) / self._fConst92) + np.float32(4.0767817))) 
		
		self._fConst427 = (np.float32(0.0017661728) / self._fConst93) 
		
		self._fConst428 = (self._fConst427 + np.float32(0.0004076782)) 
		
		self._fConst429 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst427)) 
		
		self._fConst430 = (np.float32(11.0520525) / self._fConst93) 
		
		self._fConst431 = (self._fConst430 + np.float32(1.4500711)) 
		
		self._fConst432 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst430)) 
		
		self._fConst433 = (np.float32(50.06381) / self._fConst93) 
		
		self._fConst434 = (self._fConst433 + np.float32(0.9351402)) 
		
		self._fConst435 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst433)) 
		
		self._fConst436 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst107)) 
		
		self._fConst437 = (((self._fConst109 + np.float32(-0.15748216)) / self._fConst105) + np.float32(0.9351402)) 
		
		self._fConst438 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.15748216)) / self._fConst105) + np.float32(0.9351402))) 
		
		self._fConst439 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst107)) 
		
		self._fConst440 = (((self._fConst109 + np.float32(-0.74313045)) / self._fConst105) + np.float32(1.4500711)) 
		
		self._fConst441 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.74313045)) / self._fConst105) + np.float32(1.4500711))) 
		
		self._fConst442 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst107)) 
		
		self._fConst443 = (((self._fConst109 + np.float32(-3.1897273)) / self._fConst105) + np.float32(4.0767817)) 
		
		self._fConst444 = (np.float32(1.0) / (((self._fConst109 + np.float32(3.1897273)) / self._fConst105) + np.float32(4.0767817))) 
		
		self._fConst445 = (np.float32(0.0017661728) / self._fConst106) 
		
		self._fConst446 = (self._fConst445 + np.float32(0.0004076782)) 
		
		self._fConst447 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst445)) 
		
		self._fConst448 = (np.float32(11.0520525) / self._fConst106) 
		
		self._fConst449 = (self._fConst448 + np.float32(1.4500711)) 
		
		self._fConst450 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst448)) 
		
		self._fConst451 = (np.float32(50.06381) / self._fConst106) 
		
		self._fConst452 = (self._fConst451 + np.float32(0.9351402)) 
		
		self._fConst453 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst451)) 
		
		self._fConst454 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst120)) 
		
		self._fConst455 = (((self._fConst122 + np.float32(-0.15748216)) / self._fConst118) + np.float32(0.9351402)) 
		
		self._fConst456 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.15748216)) / self._fConst118) + np.float32(0.9351402))) 
		
		self._fConst457 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst120)) 
		
		self._fConst458 = (((self._fConst122 + np.float32(-0.74313045)) / self._fConst118) + np.float32(1.4500711)) 
		
		self._fConst459 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.74313045)) / self._fConst118) + np.float32(1.4500711))) 
		
		self._fConst460 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst120)) 
		
		self._fConst461 = (((self._fConst122 + np.float32(-3.1897273)) / self._fConst118) + np.float32(4.0767817)) 
		
		self._fConst462 = (np.float32(1.0) / (((self._fConst122 + np.float32(3.1897273)) / self._fConst118) + np.float32(4.0767817))) 
		
		self._fConst463 = (np.float32(0.0017661728) / self._fConst119) 
		
		self._fConst464 = (self._fConst463 + np.float32(0.0004076782)) 
		
		self._fConst465 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst463)) 
		
		self._fConst466 = (np.float32(11.0520525) / self._fConst119) 
		
		self._fConst467 = (self._fConst466 + np.float32(1.4500711)) 
		
		self._fConst468 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst466)) 
		
		self._fConst469 = (np.float32(50.06381) / self._fConst119) 
		
		self._fConst470 = (self._fConst469 + np.float32(0.9351402)) 
		
		self._fConst471 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst469)) 
		
		self._fConst472 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst133)) 
		
		self._fConst473 = (((self._fConst135 + np.float32(-0.15748216)) / self._fConst131) + np.float32(0.9351402)) 
		
		self._fConst474 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.15748216)) / self._fConst131) + np.float32(0.9351402))) 
		
		self._fConst475 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst133)) 
		
		self._fConst476 = (((self._fConst135 + np.float32(-0.74313045)) / self._fConst131) + np.float32(1.4500711)) 
		
		self._fConst477 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.74313045)) / self._fConst131) + np.float32(1.4500711))) 
		
		self._fConst478 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst133)) 
		
		self._fConst479 = (((self._fConst135 + np.float32(-3.1897273)) / self._fConst131) + np.float32(4.0767817)) 
		
		self._fConst480 = (np.float32(1.0) / (((self._fConst135 + np.float32(3.1897273)) / self._fConst131) + np.float32(4.0767817))) 
		
		self._fConst481 = (np.float32(0.0017661728) / self._fConst132) 
		
		self._fConst482 = (self._fConst481 + np.float32(0.0004076782)) 
		
		self._fConst483 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst481)) 
		
		self._fConst484 = (np.float32(11.0520525) / self._fConst132) 
		
		self._fConst485 = (self._fConst484 + np.float32(1.4500711)) 
		
		self._fConst486 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst484)) 
		
		self._fConst487 = (np.float32(50.06381) / self._fConst132) 
		
		self._fConst488 = (self._fConst487 + np.float32(0.9351402)) 
		
		self._fConst489 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst487)) 
		
		self._fConst490 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst146)) 
		
		self._fConst491 = (((self._fConst148 + np.float32(-0.15748216)) / self._fConst144) + np.float32(0.9351402)) 
		
		self._fConst492 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.15748216)) / self._fConst144) + np.float32(0.9351402))) 
		
		self._fConst493 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst146)) 
		
		self._fConst494 = (((self._fConst148 + np.float32(-0.74313045)) / self._fConst144) + np.float32(1.4500711)) 
		
		self._fConst495 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.74313045)) / self._fConst144) + np.float32(1.4500711))) 
		
		self._fConst496 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst146)) 
		
		self._fConst497 = (((self._fConst148 + np.float32(-3.1897273)) / self._fConst144) + np.float32(4.0767817)) 
		
		self._fConst498 = (np.float32(1.0) / (((self._fConst148 + np.float32(3.1897273)) / self._fConst144) + np.float32(4.0767817))) 
		
		self._fConst499 = (np.float32(0.0017661728) / self._fConst145) 
		
		self._fConst500 = (self._fConst499 + np.float32(0.0004076782)) 
		
		self._fConst501 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst499)) 
		
		self._fConst502 = (np.float32(11.0520525) / self._fConst145) 
		
		self._fConst503 = (self._fConst502 + np.float32(1.4500711)) 
		
		self._fConst504 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst502)) 
		
		self._fConst505 = (np.float32(50.06381) / self._fConst145) 
		
		self._fConst506 = (self._fConst505 + np.float32(0.9351402)) 
		
		self._fConst507 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst505)) 
		
		self._fConst508 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst159)) 
		
		self._fConst509 = (((self._fConst161 + np.float32(-0.15748216)) / self._fConst157) + np.float32(0.9351402)) 
		
		self._fConst510 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.15748216)) / self._fConst157) + np.float32(0.9351402))) 
		
		self._fConst511 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst159)) 
		
		self._fConst512 = (((self._fConst161 + np.float32(-0.74313045)) / self._fConst157) + np.float32(1.4500711)) 
		
		self._fConst513 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.74313045)) / self._fConst157) + np.float32(1.4500711))) 
		
		self._fConst514 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst159)) 
		
		self._fConst515 = (((self._fConst161 + np.float32(-3.1897273)) / self._fConst157) + np.float32(4.0767817)) 
		
		self._fConst516 = (np.float32(1.0) / (((self._fConst161 + np.float32(3.1897273)) / self._fConst157) + np.float32(4.0767817))) 
		
		self._fConst517 = (np.float32(0.0017661728) / self._fConst158) 
		
		self._fConst518 = (self._fConst517 + np.float32(0.0004076782)) 
		
		self._fConst519 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst517)) 
		
		self._fConst520 = (np.float32(11.0520525) / self._fConst158) 
		
		self._fConst521 = (self._fConst520 + np.float32(1.4500711)) 
		
		self._fConst522 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst520)) 
		
		self._fConst523 = (np.float32(50.06381) / self._fConst158) 
		
		self._fConst524 = (self._fConst523 + np.float32(0.9351402)) 
		
		self._fConst525 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst523)) 
		
		self._fConst526 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst172)) 
		
		self._fConst527 = (((self._fConst174 + np.float32(-0.15748216)) / self._fConst170) + np.float32(0.9351402)) 
		
		self._fConst528 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.15748216)) / self._fConst170) + np.float32(0.9351402))) 
		
		self._fConst529 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst172)) 
		
		self._fConst530 = (((self._fConst174 + np.float32(-0.74313045)) / self._fConst170) + np.float32(1.4500711)) 
		
		self._fConst531 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.74313045)) / self._fConst170) + np.float32(1.4500711))) 
		
		self._fConst532 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst172)) 
		
		self._fConst533 = (((self._fConst174 + np.float32(-3.1897273)) / self._fConst170) + np.float32(4.0767817)) 
		
		self._fConst534 = (np.float32(1.0) / (((self._fConst174 + np.float32(3.1897273)) / self._fConst170) + np.float32(4.0767817))) 
		
		self._fConst535 = (np.float32(0.0017661728) / self._fConst171) 
		
		self._fConst536 = (self._fConst535 + np.float32(0.0004076782)) 
		
		self._fConst537 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst535)) 
		
		self._fConst538 = (np.float32(11.0520525) / self._fConst171) 
		
		self._fConst539 = (self._fConst538 + np.float32(1.4500711)) 
		
		self._fConst540 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst538)) 
		
		self._fConst541 = (np.float32(50.06381) / self._fConst171) 
		
		self._fConst542 = (self._fConst541 + np.float32(0.9351402)) 
		
		self._fConst543 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst541)) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec100"] = np.float32(0)
		state["fRec104"] = np.float32(0)
		state["fRec108"] = np.float32(0)
		state["fRec112"] = np.float32(0)
		state["fRec116"] = np.float32(0)
		state["fRec120"] = np.float32(0)
		state["fRec124"] = np.float32(0)
		state["fRec43"] = np.float32(0)
		state["fRec44"] = np.float32(0)
		state["fRec45"] = np.float32(0)
		state["fRec46"] = np.float32(0)
		state["fRec47"] = np.float32(0)
		state["fRec48"] = np.float32(0)
		state["fRec51"] = np.float32(0)
		state["fRec52"] = np.float32(0)
		state["fRec53"] = np.float32(0)
		state["fRec54"] = np.float32(0)
		state["fRec55"] = np.float32(0)
		state["fRec56"] = np.float32(0)
		state["fRec57"] = np.float32(0)
		state["fRec58"] = np.float32(0)
		state["fRec59"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec62"] = np.float32(0)
		state["fRec65"] = np.float32(0)
		state["fRec67"] = np.float32(0)
		state["fRec68"] = np.float32(0)
		state["fRec70"] = np.float32(0)
		state["fRec71"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec76"] = np.float32(0)
		state["fRec80"] = np.float32(0)
		state["fRec84"] = np.float32(0)
		state["fRec88"] = np.float32(0)
		state["fRec92"] = np.float32(0)
		state["fRec96"] = np.float32(0)
		state["fVec1"] = np.float32(0)
		state["fVec10"] = np.float32(0)
		state["fVec11"] = np.float32(0)
		state["fVec13"] = np.float32(0)
		state["fVec14"] = np.float32(0)
		state["fVec16"] = np.float32(0)
		state["fVec17"] = np.float32(0)
		state["fVec18"] = np.float32(0)
		state["fVec19"] = np.float32(0)
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
		state["fVec5"] = np.float32(0)
		state["fVec7"] = np.float32(0)
		state["fVec8"] = np.float32(0)
		state["iRec50"] = np.int32(0)
		# Initialize array delays
		state["iVec0"] = np.zeros((4,), dtype=np.int32)
		state["fRec49"] = np.zeros((4,), dtype=np.float32)
		state["fVec2"] = np.zeros((4096,), dtype=np.float32)
		state["fVec4"] = np.zeros((4096,), dtype=np.float32)
		state["fVec6"] = np.zeros((4096,), dtype=np.float32)
		state["fVec9"] = np.zeros((4096,), dtype=np.float32)
		state["fVec12"] = np.zeros((4096,), dtype=np.float32)
		state["fVec15"] = np.zeros((4096,), dtype=np.float32)
		state["fRec64"] = np.zeros((3,), dtype=np.float32)
		state["fRec63"] = np.zeros((3,), dtype=np.float32)
		state["fRec42"] = np.zeros((3,), dtype=np.float32)
		state["fRec41"] = np.zeros((3,), dtype=np.float32)
		state["fRec40"] = np.zeros((3,), dtype=np.float32)
		state["fRec39"] = np.zeros((3,), dtype=np.float32)
		state["fRec38"] = np.zeros((3,), dtype=np.float32)
		state["fRec37"] = np.zeros((3,), dtype=np.float32)
		state["fRec36"] = np.zeros((3,), dtype=np.float32)
		state["fRec35"] = np.zeros((3,), dtype=np.float32)
		state["fRec34"] = np.zeros((3,), dtype=np.float32)
		state["fRec33"] = np.zeros((3,), dtype=np.float32)
		state["fRec32"] = np.zeros((3,), dtype=np.float32)
		state["fRec31"] = np.zeros((3,), dtype=np.float32)
		state["fRec30"] = np.zeros((3,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fRec28"] = np.zeros((3,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fRec26"] = np.zeros((3,), dtype=np.float32)
		state["fRec25"] = np.zeros((3,), dtype=np.float32)
		state["fRec24"] = np.zeros((3,), dtype=np.float32)
		state["fRec23"] = np.zeros((3,), dtype=np.float32)
		state["fRec22"] = np.zeros((3,), dtype=np.float32)
		state["fRec21"] = np.zeros((3,), dtype=np.float32)
		state["fRec20"] = np.zeros((3,), dtype=np.float32)
		state["fRec19"] = np.zeros((3,), dtype=np.float32)
		state["fRec18"] = np.zeros((3,), dtype=np.float32)
		state["fRec17"] = np.zeros((3,), dtype=np.float32)
		state["fRec16"] = np.zeros((3,), dtype=np.float32)
		state["fRec15"] = np.zeros((3,), dtype=np.float32)
		state["fRec14"] = np.zeros((3,), dtype=np.float32)
		state["fRec13"] = np.zeros((3,), dtype=np.float32)
		state["fRec12"] = np.zeros((3,), dtype=np.float32)
		state["fRec11"] = np.zeros((3,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec75"] = np.zeros((3,), dtype=np.float32)
		state["fRec74"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fRec83"] = np.zeros((3,), dtype=np.float32)
		state["fRec82"] = np.zeros((3,), dtype=np.float32)
		state["fRec81"] = np.zeros((3,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec91"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fRec94"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fRec98"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec103"] = np.zeros((3,), dtype=np.float32)
		state["fRec102"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fRec110"] = np.zeros((3,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec114"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec119"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec123"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec127"] = np.zeros((3,), dtype=np.float32)
		state["fRec126"] = np.zeros((3,), dtype=np.float32)
		state["fRec125"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		iSlow0 = jnp.int32(params["fCheckbox0"]) 
		iSlow1 = jnp.int32(params["fCheckbox1"]) 
		fSlow2 = (jnp.float32(0.44) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fHslider0"] + jnp.float32(-49.0))))) 
		fSlow3 = params["fHslider1"] 
		fSlow4 = (jnp.float32(4.0) * jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.power(fSlow3, jnp.float32(4.0)), jnp.float32(0.999999)))) 
		fSlow5 = params["fVslider0"] 
		fSlow6 = params["fVslider1"] 
		fSlow7 = jnp.where(((fSlow6 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst184 / fSlow6))), jnp.float32(0.0)) 
		fSlow8 = ((jnp.float32(4.4e+02) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fHslider2"] + jnp.float32(-49.0))))) * (jnp.float32(1.0) - fSlow7)) 
		fSlow9 = ((jnp.float32(0.01) * params["fVslider2"]) + jnp.float32(1.0)) 
		fSlow10 = (jnp.float32(0.01) * params["fVslider3"]) 
		fSlow11 = (jnp.float32(1.0) - (jnp.float32(0.01) * params["fVslider4"])) 
		fSlow12 = params["fCheckbox2"] 
		fSlow13 = (jnp.float32(0.083333336) * (jnp.float32(1.0) - fSlow12)) 
		fSlow14 = (self._fConst185 * fSlow12) 
		fSlow15 = (self._fConst0 * params["fVslider5"]) 
		fSlow16 = (self._fConst188 * params["fVslider6"]) 
		fSlow17 = (self._fConst189 * params["fVslider7"]) 
		iSlow18 = jnp.int32((params["fEntry0"] + jnp.float32(-1.0))) 
		iSlow19 = (iSlow18 >= jnp.int32(2)).astype(jnp.int32) 
		iSlow20 = (iSlow18 >= jnp.int32(1)).astype(jnp.int32) 
		iSlow21 = (iSlow18 >= jnp.int32(3)).astype(jnp.int32) 
		fSlow22 = (jnp.float32(0.33333334) * params["fVslider8"]) 
		fSlow23 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider3"]))) 
		fSlow24 = params["fVslider9"] 
		iSlow25 = jnp.int32(params["fCheckbox3"]) 
		fSlow26 = jnp.minimum(jnp.float32(1.4127994), (jnp.float32(1.4142135) * fSlow3)) 
		fSlow27 = (jnp.float32(1.4142135) * fSlow26) 
		fSlow28 = (jnp.float32(2.0) - fSlow27) 
		fSlow29 = jnp.power(fSlow26, jnp.float32(2.0)) 
		fSlow30 = (fSlow27 + jnp.float32(2.0)) 
		fSlow31 = (fSlow27 + fSlow29) 
		fSlow32 = (jnp.float32(1.998) * fSlow3) 
		fSlow33 = (jnp.float32(2.0) - fSlow32) 
		fSlow34 = jnp.power((jnp.float32(1.4127994) * fSlow3), jnp.float32(2.0)) 
		fSlow35 = (fSlow32 + jnp.float32(2.0)) 
		fSlow36 = (fSlow34 + fSlow32) 
		fSlow37 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider4"]))) 
		fSlow38 = params["fHslider5"] 
		fSlow39 = jnp.where((((jnp.float32(0.001) * fSlow38) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst291 / fSlow38))), jnp.float32(0.0)) 
		fSlow40 = (jnp.float32(1.0) - fSlow39) 
		fSlow41 = params["fHslider6"] 
		fRec44_temp = state["fRec44"] 
		iRec50_temp = state["iRec50"] 
		fRec51_temp = state["fRec51"] 
		fRec52_temp = state["fRec52"] 
		fVec1_temp = state["fVec1"] 
		fRec53_temp = state["fRec53"] 
		fRec54_temp = state["fRec54"] 
		fVec3_temp = state["fVec3"] 
		fRec55_temp = state["fRec55"] 
		fVec5_temp = state["fVec5"] 
		fVec7_temp = state["fVec7"] 
		fVec8_temp = state["fVec8"] 
		fVec10_temp = state["fVec10"] 
		fVec11_temp = state["fVec11"] 
		fVec13_temp = state["fVec13"] 
		fVec14_temp = state["fVec14"] 
		fRec56_temp = state["fRec56"] 
		fRec57_temp = state["fRec57"] 
		fRec58_temp = state["fRec58"] 
		fRec59_temp = state["fRec59"] 
		fVec16_temp = state["fVec16"] 
		fVec17_temp = state["fVec17"] 
		fVec18_temp = state["fVec18"] 
		fVec19_temp = state["fVec19"] 
		fVec20_temp = state["fVec20"] 
		fVec21_temp = state["fVec21"] 
		fRec60_temp = state["fRec60"] 
		fVec22_temp = state["fVec22"] 
		fVec23_temp = state["fVec23"] 
		fVec24_temp = state["fVec24"] 
		fVec25_temp = state["fVec25"] 
		fVec26_temp = state["fVec26"] 
		fVec27_temp = state["fVec27"] 
		fRec61_temp = state["fRec61"] 
		fVec28_temp = state["fVec28"] 
		fVec29_temp = state["fVec29"] 
		fVec30_temp = state["fVec30"] 
		fVec31_temp = state["fVec31"] 
		fVec32_temp = state["fVec32"] 
		fVec33_temp = state["fVec33"] 
		fRec62_temp = state["fRec62"] 
		fRec48_temp = state["fRec48"] 
		fRec47_temp = state["fRec47"] 
		fRec46_temp = state["fRec46"] 
		fRec45_temp = state["fRec45"] 
		fRec43_temp = state["fRec43"] 
		fRec70_temp = state["fRec70"] 
		fRec68_temp = state["fRec68"] 
		fRec67_temp = state["fRec67"] 
		fRec65_temp = state["fRec65"] 
		fRec71_temp = state["fRec71"] 
		fRec0_temp = state["fRec0"] 
		fRec72_temp = state["fRec72"] 
		fRec76_temp = state["fRec76"] 
		fRec80_temp = state["fRec80"] 
		fRec84_temp = state["fRec84"] 
		fRec88_temp = state["fRec88"] 
		fRec92_temp = state["fRec92"] 
		fRec96_temp = state["fRec96"] 
		fRec100_temp = state["fRec100"] 
		fRec104_temp = state["fRec104"] 
		fRec108_temp = state["fRec108"] 
		fRec112_temp = state["fRec112"] 
		fRec116_temp = state["fRec116"] 
		fRec120_temp = state["fRec120"] 
		fRec124_temp = state["fRec124"] 
		state["iVec0"] = state["iVec0"].at[0].set(jnp.int32(1)) 
		state["fRec44"] = (fSlow2 + (jnp.float32(0.999) * fRec44_temp)) 
		fTemp0 = (self._fConst183 * state["fRec44"]) 
		fTemp1 = (jnp.float32(1.0) - fTemp0) 
		state["iRec50"] = ((jnp.int32(1103515245) * iRec50_temp) + jnp.int32(12345)) 
		state["fRec49"] = state["fRec49"].at[0].set((((jnp.float32(0.5221894) * state["fRec49"][3]) + ((jnp.float32(4.656613e-10) * (state["iRec50"])) + (jnp.float32(2.494956) * state["fRec49"][1]))) - (jnp.float32(2.0172658) * state["fRec49"][2]))) 
		state["fRec51"] = ((fRec51_temp * fSlow7) + fSlow8) 
		fTemp2 = (fSlow9 * state["fRec51"]) 
		fTemp3 = jnp.maximum(fTemp2, jnp.float32(23.44895)) 
		fTemp4 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp3)) 
		fTemp5 = (fRec52_temp + (self._fConst184 * fTemp4)) 
		state["fRec52"] = (fTemp5 - jnp.floor(fTemp5)) 
		fTemp6 = (jnp.float32(2.0) * state["fRec52"]) 
		fTemp7 = (fTemp6 + jnp.float32(-1.0)) 
		fTemp8 = jnp.power(fTemp7, jnp.float32(2.0)) 
		state["fVec1"] = jnp.float32(fTemp8) 
		fTemp9 = (state["iVec0"][1]) 
		fTemp10 = ((fTemp9 * (fTemp8 - fVec1_temp)) / fTemp4) 
		state["fVec2"] = state["fVec2"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp10) 
		state["fRec53"] = (fSlow10 + (jnp.float32(0.99) * fRec53_temp)) 
		fTemp11 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst0 * (state["fRec53"] / fTemp3)))) 
		iTemp12 = jnp.int32(fTemp11) 
		iTemp13 = (iTemp12 + jnp.int32(1)) 
		fTemp14 = (iTemp12) 
		fTemp15 = (fTemp11 - fTemp14) 
		fTemp16 = (fTemp14 + (jnp.float32(1.0) - fTemp11)) 
		fTemp17 = (fSlow11 * state["fRec51"]) 
		fTemp18 = jnp.maximum(fTemp17, jnp.float32(23.44895)) 
		fTemp19 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp18)) 
		fTemp20 = (fRec54_temp + (self._fConst184 * fTemp19)) 
		state["fRec54"] = (fTemp20 - jnp.floor(fTemp20)) 
		fTemp21 = (jnp.float32(2.0) * state["fRec54"]) 
		fTemp22 = (fTemp21 + jnp.float32(-1.0)) 
		fTemp23 = jnp.power(fTemp22, jnp.float32(2.0)) 
		state["fVec3"] = jnp.float32(fTemp23) 
		fTemp24 = ((fTemp9 * (fTemp23 - fVec3_temp)) / fTemp19) 
		state["fVec4"] = state["fVec4"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp24) 
		fTemp25 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst0 * (state["fRec53"] / fTemp18)))) 
		iTemp26 = jnp.int32(fTemp25) 
		iTemp27 = (iTemp26 + jnp.int32(1)) 
		fTemp28 = (iTemp26) 
		fTemp29 = (fTemp25 - fTemp28) 
		fTemp30 = (fTemp28 + (jnp.float32(1.0) - fTemp25)) 
		fTemp31 = jnp.maximum(state["fRec51"], jnp.float32(23.44895)) 
		fTemp32 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp31)) 
		fTemp33 = (fRec55_temp + (self._fConst184 * fTemp32)) 
		state["fRec55"] = (fTemp33 - jnp.floor(fTemp33)) 
		fTemp34 = (jnp.float32(2.0) * state["fRec55"]) 
		fTemp35 = (fTemp34 + jnp.float32(-1.0)) 
		fTemp36 = jnp.power(fTemp35, jnp.float32(2.0)) 
		state["fVec5"] = jnp.float32(fTemp36) 
		fTemp37 = ((fTemp9 * (fTemp36 - fVec5_temp)) / fTemp32) 
		state["fVec6"] = state["fVec6"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp37) 
		fTemp38 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst0 * (state["fRec53"] / fTemp31)))) 
		iTemp39 = jnp.int32(fTemp38) 
		iTemp40 = (iTemp39 + jnp.int32(1)) 
		fTemp41 = (iTemp39) 
		fTemp42 = (fTemp38 - fTemp41) 
		fTemp43 = (fTemp41 + (jnp.float32(1.0) - fTemp38)) 
		fTemp44 = jnp.power(fTemp7, jnp.float32(3.0)) 
		state["fVec7"] = (fTemp44 + (jnp.float32(1.0) - fTemp6)) 
		fTemp45 = ((fTemp44 + (jnp.float32(1.0) - (fTemp6 + fVec7_temp))) / fTemp4) 
		state["fVec8"] = jnp.float32(fTemp45) 
		fTemp46 = (state["iVec0"][2]) 
		fTemp47 = ((fTemp46 * (fTemp45 - fVec8_temp)) / fTemp4) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp47) 
		fTemp48 = jnp.power(fTemp22, jnp.float32(3.0)) 
		state["fVec10"] = (fTemp48 + (jnp.float32(1.0) - fTemp21)) 
		fTemp49 = ((fTemp48 + (jnp.float32(1.0) - (fTemp21 + fVec10_temp))) / fTemp19) 
		state["fVec11"] = jnp.float32(fTemp49) 
		fTemp50 = ((fTemp46 * (fTemp49 - fVec11_temp)) / fTemp19) 
		state["fVec12"] = state["fVec12"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp50) 
		fTemp51 = jnp.power(fTemp35, jnp.float32(3.0)) 
		state["fVec13"] = (fTemp51 + (jnp.float32(1.0) - fTemp34)) 
		fTemp52 = ((fTemp51 + (jnp.float32(1.0) - (fTemp34 + fVec13_temp))) / fTemp32) 
		state["fVec14"] = jnp.float32(fTemp52) 
		fTemp53 = ((fTemp46 * (fTemp52 - fVec14_temp)) / fTemp32) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp53) 
		fTemp54 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst186 / fTemp3))) 
		iTemp55 = jnp.int32(fTemp54) 
		fTemp56 = (iTemp55) 
		fTemp57 = ((fTemp10 - (state["fVec2"][((state["IOTA0"] - iTemp55) & 4095).astype(jnp.int32)] * (fTemp56 + (jnp.float32(1.0) - fTemp54)))) - ((fTemp54 - fTemp56) * state["fVec2"][((state["IOTA0"] - (iTemp55 + 1)) & 4095).astype(jnp.int32)])) 
		state["fRec56"] = ((self._fConst187 * fTemp57) + (jnp.float32(0.999) * fRec56_temp)) 
		fTemp58 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst186 / fTemp18))) 
		iTemp59 = jnp.int32(fTemp58) 
		fTemp60 = (iTemp59) 
		fTemp61 = ((fTemp24 - (state["fVec4"][((state["IOTA0"] - iTemp59) & 4095).astype(jnp.int32)] * (fTemp60 + (jnp.float32(1.0) - fTemp58)))) - ((fTemp58 - fTemp60) * state["fVec4"][((state["IOTA0"] - (iTemp59 + 1)) & 4095).astype(jnp.int32)])) 
		state["fRec57"] = ((self._fConst187 * fTemp61) + (jnp.float32(0.999) * fRec57_temp)) 
		fTemp62 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst186 / fTemp31))) 
		iTemp63 = jnp.int32(fTemp62) 
		fTemp64 = (iTemp63) 
		fTemp65 = ((fTemp37 - (state["fVec6"][((state["IOTA0"] - iTemp63) & 4095).astype(jnp.int32)] * (fTemp64 + (jnp.float32(1.0) - fTemp62)))) - ((fTemp62 - fTemp64) * state["fVec6"][((state["IOTA0"] - (iTemp63 + 1)) & 4095).astype(jnp.int32)])) 
		state["fRec58"] = ((self._fConst187 * fTemp65) + (jnp.float32(0.999) * fRec58_temp)) 
		fTemp66 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp2)) 
		fTemp67 = (fRec59_temp + (self._fConst184 * fTemp66)) 
		state["fRec59"] = (fTemp67 - jnp.floor(fTemp67)) 
		fTemp68 = (jnp.float32(2.0) * state["fRec59"]) 
		fTemp69 = (fTemp68 + jnp.float32(-1.0)) 
		fTemp70 = jnp.power(fTemp69, jnp.float32(2.0)) 
		state["fVec16"] = jnp.float32(fTemp70) 
		fTemp71 = jnp.power(fTemp69, jnp.float32(3.0)) 
		state["fVec17"] = (fTemp71 + (jnp.float32(1.0) - fTemp68)) 
		fTemp72 = ((fTemp71 + (jnp.float32(1.0) - (fTemp68 + fVec17_temp))) / fTemp66) 
		state["fVec18"] = jnp.float32(fTemp72) 
		fTemp73 = (fTemp70 * (fTemp70 + jnp.float32(-2.0))) 
		state["fVec19"] = jnp.float32(fTemp73) 
		fTemp74 = ((fTemp73 - fVec19_temp) / fTemp66) 
		state["fVec20"] = jnp.float32(fTemp74) 
		fTemp75 = ((fTemp74 - fVec20_temp) / fTemp66) 
		state["fVec21"] = jnp.float32(fTemp75) 
		fTemp76 = (state["iVec0"][3]) 
		fTemp77 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp17)) 
		fTemp78 = (fRec60_temp + (self._fConst184 * fTemp77)) 
		state["fRec60"] = (fTemp78 - jnp.floor(fTemp78)) 
		fTemp79 = (jnp.float32(2.0) * state["fRec60"]) 
		fTemp80 = (fTemp79 + jnp.float32(-1.0)) 
		fTemp81 = jnp.power(fTemp80, jnp.float32(2.0)) 
		state["fVec22"] = jnp.float32(fTemp81) 
		fTemp82 = jnp.power(fTemp80, jnp.float32(3.0)) 
		state["fVec23"] = (fTemp82 + (jnp.float32(1.0) - fTemp79)) 
		fTemp83 = ((fTemp82 + (jnp.float32(1.0) - (fTemp79 + fVec23_temp))) / fTemp77) 
		state["fVec24"] = jnp.float32(fTemp83) 
		fTemp84 = (fTemp81 * (fTemp81 + jnp.float32(-2.0))) 
		state["fVec25"] = jnp.float32(fTemp84) 
		fTemp85 = ((fTemp84 - fVec25_temp) / fTemp77) 
		state["fVec26"] = jnp.float32(fTemp85) 
		fTemp86 = ((fTemp85 - fVec26_temp) / fTemp77) 
		state["fVec27"] = jnp.float32(fTemp86) 
		fTemp87 = jnp.maximum(jnp.float32(2e+01), jnp.abs(state["fRec51"])) 
		fTemp88 = (fRec61_temp + (self._fConst184 * fTemp87)) 
		state["fRec61"] = (fTemp88 - jnp.floor(fTemp88)) 
		fTemp89 = (jnp.float32(2.0) * state["fRec61"]) 
		fTemp90 = (fTemp89 + jnp.float32(-1.0)) 
		fTemp91 = jnp.power(fTemp90, jnp.float32(2.0)) 
		state["fVec28"] = jnp.float32(fTemp91) 
		fTemp92 = jnp.power(fTemp90, jnp.float32(3.0)) 
		state["fVec29"] = (fTemp92 + (jnp.float32(1.0) - fTemp89)) 
		fTemp93 = ((fTemp92 + (jnp.float32(1.0) - (fTemp89 + fVec29_temp))) / fTemp87) 
		state["fVec30"] = jnp.float32(fTemp93) 
		fTemp94 = (fTemp91 * (fTemp91 + jnp.float32(-2.0))) 
		state["fVec31"] = jnp.float32(fTemp94) 
		fTemp95 = ((fTemp94 - fVec31_temp) / fTemp87) 
		state["fVec32"] = jnp.float32(fTemp95) 
		fTemp96 = ((fTemp95 - fVec32_temp) / fTemp87) 
		state["fVec33"] = jnp.float32(fTemp96) 
		state["fRec62"] = (fSlow23 + (jnp.float32(0.999) * fRec62_temp)) 
		fTemp97 = ((fSlow24 * inputs[0]) + (state["fRec62"] * (((((fSlow22 * ((jnp.where((iSlow19 != 0), jnp.where((iSlow21 != 0), (self._fConst191 * ((fTemp76 * (fTemp96 - fVec33_temp)) / fTemp87)), (self._fConst190 * ((fTemp46 * (fTemp93 - fVec30_temp)) / fTemp87))), jnp.where((iSlow20 != 0), (self._fConst187 * ((fTemp9 * (fTemp91 - fVec28_temp)) / fTemp87)), fTemp90)) + jnp.where((iSlow19 != 0), jnp.where((iSlow21 != 0), (self._fConst191 * ((fTemp76 * (fTemp86 - fVec27_temp)) / fTemp77)), (self._fConst190 * ((fTemp46 * (fTemp83 - fVec24_temp)) / fTemp77))), jnp.where((iSlow20 != 0), (self._fConst187 * ((fTemp9 * (fTemp81 - fVec22_temp)) / fTemp77)), fTemp80))) + jnp.where((iSlow19 != 0), jnp.where((iSlow21 != 0), (self._fConst191 * ((fTemp76 * (fTemp75 - fVec21_temp)) / fTemp66)), (self._fConst190 * ((fTemp46 * (fTemp72 - fVec18_temp)) / fTemp66))), jnp.where((iSlow20 != 0), (self._fConst187 * ((fTemp9 * (fTemp70 - fVec16_temp)) / fTemp66)), fTemp69)))) + (fSlow17 * ((fTemp65 + fTemp61) + fTemp57))) + (fSlow16 * (state["fRec51"] * ((state["fRec58"] + (fSlow11 * state["fRec57"])) + (fSlow9 * state["fRec56"]))))) + (fSlow15 * ((fSlow14 * ((((fTemp53 - (state["fVec15"][((state["IOTA0"] - iTemp39) & 4095).astype(jnp.int32)] * fTemp43)) - (fTemp42 * state["fVec15"][((state["IOTA0"] - iTemp40) & 4095).astype(jnp.int32)])) + ((fTemp50 - (state["fVec12"][((state["IOTA0"] - iTemp26) & 4095).astype(jnp.int32)] * fTemp30)) - (fTemp29 * state["fVec12"][((state["IOTA0"] - iTemp27) & 4095).astype(jnp.int32)]))) + ((fTemp47 - (state["fVec9"][((state["IOTA0"] - iTemp12) & 4095).astype(jnp.int32)] * fTemp16)) - (fTemp15 * state["fVec9"][((state["IOTA0"] - iTemp13) & 4095).astype(jnp.int32)])))) + (fSlow13 * ((((fTemp37 - (fTemp43 * state["fVec6"][((state["IOTA0"] - iTemp39) & 4095).astype(jnp.int32)])) - (fTemp42 * state["fVec6"][((state["IOTA0"] - iTemp40) & 4095).astype(jnp.int32)])) + ((fTemp24 - (fTemp30 * state["fVec4"][((state["IOTA0"] - iTemp26) & 4095).astype(jnp.int32)])) - (fTemp29 * state["fVec4"][((state["IOTA0"] - iTemp27) & 4095).astype(jnp.int32)]))) + ((fTemp10 - (fTemp16 * state["fVec2"][((state["IOTA0"] - iTemp12) & 4095).astype(jnp.int32)])) - (fTemp15 * state["fVec2"][((state["IOTA0"] - iTemp13) & 4095).astype(jnp.int32)]))))))) + (fSlow5 * (((jnp.float32(0.049922034) * state["fRec49"][0]) + (jnp.float32(0.0506127) * state["fRec49"][2])) - ((jnp.float32(0.095993534) * state["fRec49"][1]) + (jnp.float32(0.004408786) * state["fRec49"][3]))))))) 
		fTemp98 = jnp.where((iSlow0 != 0), jnp.float32(0.0), fTemp97) 
		state["fRec48"] = (((fTemp1 * fRec48_temp) + fTemp98) - (fSlow4 * fRec43_temp)) 
		state["fRec47"] = (state["fRec48"] + (fTemp1 * fRec47_temp)) 
		state["fRec46"] = (state["fRec47"] + (fTemp1 * fRec46_temp)) 
		state["fRec45"] = (state["fRec46"] + (fRec45_temp * fTemp1)) 
		state["fRec43"] = (state["fRec45"] * jnp.power(fTemp0, jnp.float32(4.0))) 
		fTemp99 = jnp.tan((self._fConst192 * jnp.maximum(jnp.float32(2e+01), jnp.minimum(jnp.float32(1e+04), state["fRec44"])))) 
		fTemp100 = (jnp.float32(1.0) / fTemp99) 
		fTemp101 = (fSlow29 + ((((fSlow28 + fTemp100) / fTemp99) + jnp.float32(1.0)) - fSlow27)) 
		fTemp102 = (jnp.float32(1.0) - (jnp.float32(1.0) / jnp.power(fTemp99, jnp.float32(2.0)))) 
		fTemp103 = (fSlow31 + (((fSlow30 + fTemp100) / fTemp99) + jnp.float32(1.0))) 
		state["fRec64"] = state["fRec64"].at[0].set((fTemp98 - (((state["fRec64"][2] * (fSlow31 + (((fTemp100 - fSlow30) / fTemp99) + jnp.float32(1.0)))) + (jnp.float32(2.0) * (state["fRec64"][1] * (fSlow31 + fTemp102)))) / fTemp103))) 
		state["fRec63"] = state["fRec63"].at[0].set((((state["fRec64"][2] + (state["fRec64"][0] + (jnp.float32(2.0) * state["fRec64"][1]))) / fTemp103) - (((state["fRec63"][2] * (fSlow29 + ((((fTemp100 - fSlow28) / fTemp99) + jnp.float32(1.0)) - fSlow27))) + (jnp.float32(2.0) * (state["fRec63"][1] * (fSlow29 + (fTemp102 - fSlow27))))) / fTemp101))) 
		fTemp104 = jnp.tan((self._fConst192 * jnp.maximum(state["fRec44"], jnp.float32(2e+01)))) 
		fTemp105 = (jnp.float32(1.0) / fTemp104) 
		fTemp106 = (fSlow34 + ((((fSlow33 + fTemp105) / fTemp104) + jnp.float32(1.0)) - fSlow32)) 
		fTemp107 = ((fSlow34 + ((((fTemp105 - fSlow33) / fTemp104) + jnp.float32(1.0)) - fSlow32)) / fTemp106) 
		fTemp108 = (jnp.float32(1.0) - (jnp.float32(1.0) / jnp.power(fTemp104, jnp.float32(2.0)))) 
		fTemp109 = (fSlow34 + (fTemp108 - fSlow32)) 
		fTemp110 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), (jnp.float32(2.0) * (fTemp109 / (fTemp106 * (fTemp107 + jnp.float32(1.0))))))) 
		fTemp111 = (jnp.float32(1.0) - jnp.power(fTemp110, jnp.float32(2.0))) 
		fTemp112 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), fTemp107)) 
		fTemp113 = (jnp.float32(1.0) - jnp.power(fTemp112, jnp.float32(2.0))) 
		fTemp114 = jnp.sqrt(fTemp113) 
		fTemp115 = (jnp.float32(1.0) - (fTemp109 / fTemp106)) 
		fTemp116 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp111)) 
		fTemp117 = (fSlow36 + (((fSlow35 + fTemp105) / fTemp104) + jnp.float32(1.0))) 
		fTemp118 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp113)) 
		fTemp119 = ((fSlow36 + (jnp.float32(1.0) - ((fSlow35 - fTemp105) / fTemp104))) / fTemp117) 
		fTemp120 = (fSlow36 + fTemp108) 
		fTemp121 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), (jnp.float32(2.0) * (fTemp120 / (fTemp117 * (fTemp119 + jnp.float32(1.0))))))) 
		fTemp122 = (jnp.float32(1.0) - jnp.power(fTemp121, jnp.float32(2.0))) 
		fTemp123 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), fTemp119)) 
		fTemp124 = (jnp.float32(1.0) - jnp.power(fTemp123, jnp.float32(2.0))) 
		fTemp125 = jnp.sqrt(fTemp124) 
		fTemp126 = (jnp.float32(1.0) - (fTemp120 / fTemp117)) 
		fTemp127 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp122)) 
		fTemp128 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp124)) 
		fTemp129 = ((fTemp98 * fTemp128) - (fTemp123 * fRec68_temp)) 
		state["fRec70"] = ((fTemp129 * fTemp127) - (fTemp121 * fRec70_temp)) 
		state["fRec68"] = ((fTemp129 * fTemp121) + (fRec70_temp * fTemp127)) 
		fRec69 = state["fRec70"] 
		fTemp130 = ((((fTemp98 * fTemp123) + (fRec68_temp * fTemp128)) + (jnp.float32(2.0) * ((state["fRec68"] * fTemp126) / fTemp125))) + ((fRec69 * ((jnp.float32(1.0) - fTemp119) - (jnp.float32(2.0) * (fTemp121 * fTemp126)))) / (fTemp125 * jnp.sqrt(fTemp122)))) 
		fTemp131 = (((fTemp130 * fTemp118) / fTemp117) - (fTemp112 * fRec65_temp)) 
		state["fRec67"] = ((fTemp131 * fTemp116) - (fTemp110 * fRec67_temp)) 
		state["fRec65"] = ((fTemp131 * fTemp110) + (fRec67_temp * fTemp116)) 
		fRec66 = state["fRec67"] 
		state["fRec71"] = (fSlow37 + (jnp.float32(0.999) * fRec71_temp)) 
		fTemp132 = jnp.where((iSlow0 != 0), fTemp97, (state["fRec71"] * jnp.where((iSlow1 != 0), jnp.where((iSlow25 != 0), ((((((fTemp130 * fTemp112) / fTemp117) + (fRec65_temp * fTemp118)) + (jnp.float32(2.0) * ((state["fRec65"] * fTemp115) / fTemp114))) + ((fRec66 * ((jnp.float32(1.0) - fTemp107) - (jnp.float32(2.0) * (fTemp110 * fTemp115)))) / (fTemp114 * jnp.sqrt(fTemp111)))) / fTemp106), ((state["fRec63"][2] + (state["fRec63"][0] + (jnp.float32(2.0) * state["fRec63"][1]))) / fTemp101)), state["fRec43"]))) 
		state["fRec42"] = state["fRec42"].at[0].set((fTemp132 - (self._fConst182 * ((self._fConst181 * state["fRec42"][2]) + (self._fConst180 * state["fRec42"][1]))))) 
		state["fRec41"] = state["fRec41"].at[0].set(((self._fConst182 * (((self._fConst194 * state["fRec42"][0]) + (self._fConst195 * state["fRec42"][1])) + (self._fConst194 * state["fRec42"][2]))) - (self._fConst179 * ((self._fConst178 * state["fRec41"][2]) + (self._fConst177 * state["fRec41"][1]))))) 
		state["fRec40"] = state["fRec40"].at[0].set(((self._fConst179 * (((self._fConst196 * state["fRec41"][0]) + (self._fConst197 * state["fRec41"][1])) + (self._fConst196 * state["fRec41"][2]))) - (self._fConst176 * ((self._fConst175 * state["fRec40"][2]) + (self._fConst173 * state["fRec40"][1]))))) 
		fTemp133 = (self._fConst176 * (((self._fConst198 * state["fRec40"][0]) + (self._fConst199 * state["fRec40"][1])) + (self._fConst198 * state["fRec40"][2]))) 
		state["fRec39"] = state["fRec39"].at[0].set((fTemp133 - (self._fConst169 * ((self._fConst168 * state["fRec39"][2]) + (self._fConst167 * state["fRec39"][1]))))) 
		state["fRec38"] = state["fRec38"].at[0].set(((self._fConst169 * (((self._fConst201 * state["fRec39"][0]) + (self._fConst202 * state["fRec39"][1])) + (self._fConst201 * state["fRec39"][2]))) - (self._fConst166 * ((self._fConst165 * state["fRec38"][2]) + (self._fConst164 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst166 * (((self._fConst203 * state["fRec38"][0]) + (self._fConst204 * state["fRec38"][1])) + (self._fConst203 * state["fRec38"][2]))) - (self._fConst163 * ((self._fConst162 * state["fRec37"][2]) + (self._fConst160 * state["fRec37"][1]))))) 
		fTemp134 = (self._fConst163 * (((self._fConst205 * state["fRec37"][0]) + (self._fConst206 * state["fRec37"][1])) + (self._fConst205 * state["fRec37"][2]))) 
		state["fRec36"] = state["fRec36"].at[0].set((fTemp134 - (self._fConst156 * ((self._fConst155 * state["fRec36"][2]) + (self._fConst154 * state["fRec36"][1]))))) 
		state["fRec35"] = state["fRec35"].at[0].set(((self._fConst156 * (((self._fConst208 * state["fRec36"][0]) + (self._fConst209 * state["fRec36"][1])) + (self._fConst208 * state["fRec36"][2]))) - (self._fConst153 * ((self._fConst152 * state["fRec35"][2]) + (self._fConst151 * state["fRec35"][1]))))) 
		state["fRec34"] = state["fRec34"].at[0].set(((self._fConst153 * (((self._fConst210 * state["fRec35"][0]) + (self._fConst211 * state["fRec35"][1])) + (self._fConst210 * state["fRec35"][2]))) - (self._fConst150 * ((self._fConst149 * state["fRec34"][2]) + (self._fConst147 * state["fRec34"][1]))))) 
		fTemp135 = (self._fConst150 * (((self._fConst212 * state["fRec34"][0]) + (self._fConst213 * state["fRec34"][1])) + (self._fConst212 * state["fRec34"][2]))) 
		state["fRec33"] = state["fRec33"].at[0].set((fTemp135 - (self._fConst143 * ((self._fConst142 * state["fRec33"][2]) + (self._fConst141 * state["fRec33"][1]))))) 
		state["fRec32"] = state["fRec32"].at[0].set(((self._fConst143 * (((self._fConst215 * state["fRec33"][0]) + (self._fConst216 * state["fRec33"][1])) + (self._fConst215 * state["fRec33"][2]))) - (self._fConst140 * ((self._fConst139 * state["fRec32"][2]) + (self._fConst138 * state["fRec32"][1]))))) 
		state["fRec31"] = state["fRec31"].at[0].set(((self._fConst140 * (((self._fConst217 * state["fRec32"][0]) + (self._fConst218 * state["fRec32"][1])) + (self._fConst217 * state["fRec32"][2]))) - (self._fConst137 * ((self._fConst136 * state["fRec31"][2]) + (self._fConst134 * state["fRec31"][1]))))) 
		fTemp136 = (self._fConst137 * (((self._fConst219 * state["fRec31"][0]) + (self._fConst220 * state["fRec31"][1])) + (self._fConst219 * state["fRec31"][2]))) 
		state["fRec30"] = state["fRec30"].at[0].set((fTemp136 - (self._fConst130 * ((self._fConst129 * state["fRec30"][2]) + (self._fConst128 * state["fRec30"][1]))))) 
		state["fRec29"] = state["fRec29"].at[0].set(((self._fConst130 * (((self._fConst222 * state["fRec30"][0]) + (self._fConst223 * state["fRec30"][1])) + (self._fConst222 * state["fRec30"][2]))) - (self._fConst127 * ((self._fConst126 * state["fRec29"][2]) + (self._fConst125 * state["fRec29"][1]))))) 
		state["fRec28"] = state["fRec28"].at[0].set(((self._fConst127 * (((self._fConst224 * state["fRec29"][0]) + (self._fConst225 * state["fRec29"][1])) + (self._fConst224 * state["fRec29"][2]))) - (self._fConst124 * ((self._fConst123 * state["fRec28"][2]) + (self._fConst121 * state["fRec28"][1]))))) 
		fTemp137 = (self._fConst124 * (((self._fConst226 * state["fRec28"][0]) + (self._fConst227 * state["fRec28"][1])) + (self._fConst226 * state["fRec28"][2]))) 
		state["fRec27"] = state["fRec27"].at[0].set((fTemp137 - (self._fConst117 * ((self._fConst116 * state["fRec27"][2]) + (self._fConst115 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((self._fConst117 * (((self._fConst229 * state["fRec27"][0]) + (self._fConst230 * state["fRec27"][1])) + (self._fConst229 * state["fRec27"][2]))) - (self._fConst114 * ((self._fConst113 * state["fRec26"][2]) + (self._fConst112 * state["fRec26"][1]))))) 
		state["fRec25"] = state["fRec25"].at[0].set(((self._fConst114 * (((self._fConst231 * state["fRec26"][0]) + (self._fConst232 * state["fRec26"][1])) + (self._fConst231 * state["fRec26"][2]))) - (self._fConst111 * ((self._fConst110 * state["fRec25"][2]) + (self._fConst108 * state["fRec25"][1]))))) 
		fTemp138 = (self._fConst111 * (((self._fConst233 * state["fRec25"][0]) + (self._fConst234 * state["fRec25"][1])) + (self._fConst233 * state["fRec25"][2]))) 
		state["fRec24"] = state["fRec24"].at[0].set((fTemp138 - (self._fConst104 * ((self._fConst103 * state["fRec24"][2]) + (self._fConst102 * state["fRec24"][1]))))) 
		state["fRec23"] = state["fRec23"].at[0].set(((self._fConst104 * (((self._fConst236 * state["fRec24"][0]) + (self._fConst237 * state["fRec24"][1])) + (self._fConst236 * state["fRec24"][2]))) - (self._fConst101 * ((self._fConst100 * state["fRec23"][2]) + (self._fConst99 * state["fRec23"][1]))))) 
		state["fRec22"] = state["fRec22"].at[0].set(((self._fConst101 * (((self._fConst238 * state["fRec23"][0]) + (self._fConst239 * state["fRec23"][1])) + (self._fConst238 * state["fRec23"][2]))) - (self._fConst98 * ((self._fConst97 * state["fRec22"][2]) + (self._fConst95 * state["fRec22"][1]))))) 
		fTemp139 = (self._fConst98 * (((self._fConst240 * state["fRec22"][0]) + (self._fConst241 * state["fRec22"][1])) + (self._fConst240 * state["fRec22"][2]))) 
		state["fRec21"] = state["fRec21"].at[0].set((fTemp139 - (self._fConst91 * ((self._fConst90 * state["fRec21"][2]) + (self._fConst89 * state["fRec21"][1]))))) 
		state["fRec20"] = state["fRec20"].at[0].set(((self._fConst91 * (((self._fConst243 * state["fRec21"][0]) + (self._fConst244 * state["fRec21"][1])) + (self._fConst243 * state["fRec21"][2]))) - (self._fConst88 * ((self._fConst87 * state["fRec20"][2]) + (self._fConst86 * state["fRec20"][1]))))) 
		state["fRec19"] = state["fRec19"].at[0].set(((self._fConst88 * (((self._fConst245 * state["fRec20"][0]) + (self._fConst246 * state["fRec20"][1])) + (self._fConst245 * state["fRec20"][2]))) - (self._fConst85 * ((self._fConst84 * state["fRec19"][2]) + (self._fConst82 * state["fRec19"][1]))))) 
		fTemp140 = (self._fConst85 * (((self._fConst247 * state["fRec19"][0]) + (self._fConst248 * state["fRec19"][1])) + (self._fConst247 * state["fRec19"][2]))) 
		state["fRec18"] = state["fRec18"].at[0].set((fTemp140 - (self._fConst78 * ((self._fConst77 * state["fRec18"][2]) + (self._fConst76 * state["fRec18"][1]))))) 
		state["fRec17"] = state["fRec17"].at[0].set(((self._fConst78 * (((self._fConst250 * state["fRec18"][0]) + (self._fConst251 * state["fRec18"][1])) + (self._fConst250 * state["fRec18"][2]))) - (self._fConst75 * ((self._fConst74 * state["fRec17"][2]) + (self._fConst73 * state["fRec17"][1]))))) 
		state["fRec16"] = state["fRec16"].at[0].set(((self._fConst75 * (((self._fConst252 * state["fRec17"][0]) + (self._fConst253 * state["fRec17"][1])) + (self._fConst252 * state["fRec17"][2]))) - (self._fConst72 * ((self._fConst71 * state["fRec16"][2]) + (self._fConst69 * state["fRec16"][1]))))) 
		fTemp141 = (self._fConst72 * (((self._fConst254 * state["fRec16"][0]) + (self._fConst255 * state["fRec16"][1])) + (self._fConst254 * state["fRec16"][2]))) 
		state["fRec15"] = state["fRec15"].at[0].set((fTemp141 - (self._fConst65 * ((self._fConst64 * state["fRec15"][2]) + (self._fConst63 * state["fRec15"][1]))))) 
		state["fRec14"] = state["fRec14"].at[0].set(((self._fConst65 * (((self._fConst257 * state["fRec15"][0]) + (self._fConst258 * state["fRec15"][1])) + (self._fConst257 * state["fRec15"][2]))) - (self._fConst62 * ((self._fConst61 * state["fRec14"][2]) + (self._fConst60 * state["fRec14"][1]))))) 
		state["fRec13"] = state["fRec13"].at[0].set(((self._fConst62 * (((self._fConst259 * state["fRec14"][0]) + (self._fConst260 * state["fRec14"][1])) + (self._fConst259 * state["fRec14"][2]))) - (self._fConst59 * ((self._fConst58 * state["fRec13"][2]) + (self._fConst56 * state["fRec13"][1]))))) 
		fTemp142 = (self._fConst59 * (((self._fConst261 * state["fRec13"][0]) + (self._fConst262 * state["fRec13"][1])) + (self._fConst261 * state["fRec13"][2]))) 
		state["fRec12"] = state["fRec12"].at[0].set((fTemp142 - (self._fConst52 * ((self._fConst51 * state["fRec12"][2]) + (self._fConst50 * state["fRec12"][1]))))) 
		state["fRec11"] = state["fRec11"].at[0].set(((self._fConst52 * (((self._fConst264 * state["fRec12"][0]) + (self._fConst265 * state["fRec12"][1])) + (self._fConst264 * state["fRec12"][2]))) - (self._fConst49 * ((self._fConst48 * state["fRec11"][2]) + (self._fConst47 * state["fRec11"][1]))))) 
		state["fRec10"] = state["fRec10"].at[0].set(((self._fConst49 * (((self._fConst266 * state["fRec11"][0]) + (self._fConst267 * state["fRec11"][1])) + (self._fConst266 * state["fRec11"][2]))) - (self._fConst46 * ((self._fConst45 * state["fRec10"][2]) + (self._fConst43 * state["fRec10"][1]))))) 
		fTemp143 = (self._fConst46 * (((self._fConst268 * state["fRec10"][0]) + (self._fConst269 * state["fRec10"][1])) + (self._fConst268 * state["fRec10"][2]))) 
		state["fRec9"] = state["fRec9"].at[0].set((fTemp143 - (self._fConst39 * ((self._fConst38 * state["fRec9"][2]) + (self._fConst37 * state["fRec9"][1]))))) 
		state["fRec8"] = state["fRec8"].at[0].set(((self._fConst39 * (((self._fConst271 * state["fRec9"][0]) + (self._fConst272 * state["fRec9"][1])) + (self._fConst271 * state["fRec9"][2]))) - (self._fConst36 * ((self._fConst35 * state["fRec8"][2]) + (self._fConst34 * state["fRec8"][1]))))) 
		state["fRec7"] = state["fRec7"].at[0].set(((self._fConst36 * (((self._fConst273 * state["fRec8"][0]) + (self._fConst274 * state["fRec8"][1])) + (self._fConst273 * state["fRec8"][2]))) - (self._fConst33 * ((self._fConst32 * state["fRec7"][2]) + (self._fConst30 * state["fRec7"][1]))))) 
		fTemp144 = (self._fConst33 * (((self._fConst275 * state["fRec7"][0]) + (self._fConst276 * state["fRec7"][1])) + (self._fConst275 * state["fRec7"][2]))) 
		state["fRec6"] = state["fRec6"].at[0].set((fTemp144 - (self._fConst26 * ((self._fConst25 * state["fRec6"][2]) + (self._fConst24 * state["fRec6"][1]))))) 
		state["fRec5"] = state["fRec5"].at[0].set(((self._fConst26 * (((self._fConst278 * state["fRec6"][0]) + (self._fConst279 * state["fRec6"][1])) + (self._fConst278 * state["fRec6"][2]))) - (self._fConst23 * ((self._fConst22 * state["fRec5"][2]) + (self._fConst21 * state["fRec5"][1]))))) 
		state["fRec4"] = state["fRec4"].at[0].set(((self._fConst23 * (((self._fConst280 * state["fRec5"][0]) + (self._fConst281 * state["fRec5"][1])) + (self._fConst280 * state["fRec5"][2]))) - (self._fConst20 * ((self._fConst19 * state["fRec4"][2]) + (self._fConst17 * state["fRec4"][1]))))) 
		fTemp145 = (self._fConst20 * (((self._fConst282 * state["fRec4"][0]) + (self._fConst283 * state["fRec4"][1])) + (self._fConst282 * state["fRec4"][2]))) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp145 - (self._fConst13 * ((self._fConst12 * state["fRec3"][2]) + (self._fConst11 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst13 * (((self._fConst285 * state["fRec3"][0]) + (self._fConst286 * state["fRec3"][1])) + (self._fConst285 * state["fRec3"][2]))) - (self._fConst10 * ((self._fConst9 * state["fRec2"][2]) + (self._fConst8 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst10 * (((self._fConst287 * state["fRec2"][0]) + (self._fConst288 * state["fRec2"][1])) + (self._fConst287 * state["fRec2"][2]))) - (self._fConst7 * ((self._fConst6 * state["fRec1"][2]) + (self._fConst4 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fSlow39 * fRec0_temp) + (fSlow40 * jnp.abs((self._fConst7 * (((self._fConst289 * state["fRec1"][0]) + (self._fConst290 * state["fRec1"][1])) + (self._fConst289 * state["fRec1"][2])))))) 
		fVbargraph0 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec75"] = state["fRec75"].at[0].set((fTemp145 - (self._fConst300 * ((self._fConst299 * state["fRec75"][2]) + (self._fConst298 * state["fRec75"][1]))))) 
		state["fRec74"] = state["fRec74"].at[0].set(((self._fConst300 * (((self._fConst302 * state["fRec75"][0]) + (self._fConst303 * state["fRec75"][1])) + (self._fConst302 * state["fRec75"][2]))) - (self._fConst297 * ((self._fConst296 * state["fRec74"][2]) + (self._fConst295 * state["fRec74"][1]))))) 
		state["fRec73"] = state["fRec73"].at[0].set(((self._fConst297 * (((self._fConst305 * state["fRec74"][0]) + (self._fConst306 * state["fRec74"][1])) + (self._fConst305 * state["fRec74"][2]))) - (self._fConst294 * ((self._fConst293 * state["fRec73"][2]) + (self._fConst292 * state["fRec73"][1]))))) 
		state["fRec72"] = ((fSlow39 * fRec72_temp) + (fSlow40 * jnp.abs((self._fConst294 * (((self._fConst308 * state["fRec73"][0]) + (self._fConst309 * state["fRec73"][1])) + (self._fConst308 * state["fRec73"][2])))))) 
		fVbargraph1 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec72"])))
		self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec79"] = state["fRec79"].at[0].set((fTemp144 - (self._fConst318 * ((self._fConst317 * state["fRec79"][2]) + (self._fConst316 * state["fRec79"][1]))))) 
		state["fRec78"] = state["fRec78"].at[0].set(((self._fConst318 * (((self._fConst320 * state["fRec79"][0]) + (self._fConst321 * state["fRec79"][1])) + (self._fConst320 * state["fRec79"][2]))) - (self._fConst315 * ((self._fConst314 * state["fRec78"][2]) + (self._fConst313 * state["fRec78"][1]))))) 
		state["fRec77"] = state["fRec77"].at[0].set(((self._fConst315 * (((self._fConst323 * state["fRec78"][0]) + (self._fConst324 * state["fRec78"][1])) + (self._fConst323 * state["fRec78"][2]))) - (self._fConst312 * ((self._fConst311 * state["fRec77"][2]) + (self._fConst310 * state["fRec77"][1]))))) 
		state["fRec76"] = ((fSlow39 * fRec76_temp) + (fSlow40 * jnp.abs((self._fConst312 * (((self._fConst326 * state["fRec77"][0]) + (self._fConst327 * state["fRec77"][1])) + (self._fConst326 * state["fRec77"][2])))))) 
		fVbargraph2 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec76"])))
		self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec83"] = state["fRec83"].at[0].set((fTemp143 - (self._fConst336 * ((self._fConst335 * state["fRec83"][2]) + (self._fConst334 * state["fRec83"][1]))))) 
		state["fRec82"] = state["fRec82"].at[0].set(((self._fConst336 * (((self._fConst338 * state["fRec83"][0]) + (self._fConst339 * state["fRec83"][1])) + (self._fConst338 * state["fRec83"][2]))) - (self._fConst333 * ((self._fConst332 * state["fRec82"][2]) + (self._fConst331 * state["fRec82"][1]))))) 
		state["fRec81"] = state["fRec81"].at[0].set(((self._fConst333 * (((self._fConst341 * state["fRec82"][0]) + (self._fConst342 * state["fRec82"][1])) + (self._fConst341 * state["fRec82"][2]))) - (self._fConst330 * ((self._fConst329 * state["fRec81"][2]) + (self._fConst328 * state["fRec81"][1]))))) 
		state["fRec80"] = ((fSlow39 * fRec80_temp) + (fSlow40 * jnp.abs((self._fConst330 * (((self._fConst344 * state["fRec81"][0]) + (self._fConst345 * state["fRec81"][1])) + (self._fConst344 * state["fRec81"][2])))))) 
		fVbargraph3 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec80"])))
		self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec87"] = state["fRec87"].at[0].set((fTemp142 - (self._fConst354 * ((self._fConst353 * state["fRec87"][2]) + (self._fConst352 * state["fRec87"][1]))))) 
		state["fRec86"] = state["fRec86"].at[0].set(((self._fConst354 * (((self._fConst356 * state["fRec87"][0]) + (self._fConst357 * state["fRec87"][1])) + (self._fConst356 * state["fRec87"][2]))) - (self._fConst351 * ((self._fConst350 * state["fRec86"][2]) + (self._fConst349 * state["fRec86"][1]))))) 
		state["fRec85"] = state["fRec85"].at[0].set(((self._fConst351 * (((self._fConst359 * state["fRec86"][0]) + (self._fConst360 * state["fRec86"][1])) + (self._fConst359 * state["fRec86"][2]))) - (self._fConst348 * ((self._fConst347 * state["fRec85"][2]) + (self._fConst346 * state["fRec85"][1]))))) 
		state["fRec84"] = ((fSlow39 * fRec84_temp) + (fSlow40 * jnp.abs((self._fConst348 * (((self._fConst362 * state["fRec85"][0]) + (self._fConst363 * state["fRec85"][1])) + (self._fConst362 * state["fRec85"][2])))))) 
		fVbargraph4 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec84"])))
		self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec91"] = state["fRec91"].at[0].set((fTemp141 - (self._fConst372 * ((self._fConst371 * state["fRec91"][2]) + (self._fConst370 * state["fRec91"][1]))))) 
		state["fRec90"] = state["fRec90"].at[0].set(((self._fConst372 * (((self._fConst374 * state["fRec91"][0]) + (self._fConst375 * state["fRec91"][1])) + (self._fConst374 * state["fRec91"][2]))) - (self._fConst369 * ((self._fConst368 * state["fRec90"][2]) + (self._fConst367 * state["fRec90"][1]))))) 
		state["fRec89"] = state["fRec89"].at[0].set(((self._fConst369 * (((self._fConst377 * state["fRec90"][0]) + (self._fConst378 * state["fRec90"][1])) + (self._fConst377 * state["fRec90"][2]))) - (self._fConst366 * ((self._fConst365 * state["fRec89"][2]) + (self._fConst364 * state["fRec89"][1]))))) 
		state["fRec88"] = ((fSlow39 * fRec88_temp) + (fSlow40 * jnp.abs((self._fConst366 * (((self._fConst380 * state["fRec89"][0]) + (self._fConst381 * state["fRec89"][1])) + (self._fConst380 * state["fRec89"][2])))))) 
		fVbargraph5 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec88"])))
		self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec95"] = state["fRec95"].at[0].set((fTemp140 - (self._fConst390 * ((self._fConst389 * state["fRec95"][2]) + (self._fConst388 * state["fRec95"][1]))))) 
		state["fRec94"] = state["fRec94"].at[0].set(((self._fConst390 * (((self._fConst392 * state["fRec95"][0]) + (self._fConst393 * state["fRec95"][1])) + (self._fConst392 * state["fRec95"][2]))) - (self._fConst387 * ((self._fConst386 * state["fRec94"][2]) + (self._fConst385 * state["fRec94"][1]))))) 
		state["fRec93"] = state["fRec93"].at[0].set(((self._fConst387 * (((self._fConst395 * state["fRec94"][0]) + (self._fConst396 * state["fRec94"][1])) + (self._fConst395 * state["fRec94"][2]))) - (self._fConst384 * ((self._fConst383 * state["fRec93"][2]) + (self._fConst382 * state["fRec93"][1]))))) 
		state["fRec92"] = ((fSlow39 * fRec92_temp) + (fSlow40 * jnp.abs((self._fConst384 * (((self._fConst398 * state["fRec93"][0]) + (self._fConst399 * state["fRec93"][1])) + (self._fConst398 * state["fRec93"][2])))))) 
		fVbargraph6 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec92"])))
		self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec99"] = state["fRec99"].at[0].set((fTemp139 - (self._fConst408 * ((self._fConst407 * state["fRec99"][2]) + (self._fConst406 * state["fRec99"][1]))))) 
		state["fRec98"] = state["fRec98"].at[0].set(((self._fConst408 * (((self._fConst410 * state["fRec99"][0]) + (self._fConst411 * state["fRec99"][1])) + (self._fConst410 * state["fRec99"][2]))) - (self._fConst405 * ((self._fConst404 * state["fRec98"][2]) + (self._fConst403 * state["fRec98"][1]))))) 
		state["fRec97"] = state["fRec97"].at[0].set(((self._fConst405 * (((self._fConst413 * state["fRec98"][0]) + (self._fConst414 * state["fRec98"][1])) + (self._fConst413 * state["fRec98"][2]))) - (self._fConst402 * ((self._fConst401 * state["fRec97"][2]) + (self._fConst400 * state["fRec97"][1]))))) 
		state["fRec96"] = ((fSlow39 * fRec96_temp) + (fSlow40 * jnp.abs((self._fConst402 * (((self._fConst416 * state["fRec97"][0]) + (self._fConst417 * state["fRec97"][1])) + (self._fConst416 * state["fRec97"][2])))))) 
		fVbargraph7 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec96"])))
		self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec103"] = state["fRec103"].at[0].set((fTemp138 - (self._fConst426 * ((self._fConst425 * state["fRec103"][2]) + (self._fConst424 * state["fRec103"][1]))))) 
		state["fRec102"] = state["fRec102"].at[0].set(((self._fConst426 * (((self._fConst428 * state["fRec103"][0]) + (self._fConst429 * state["fRec103"][1])) + (self._fConst428 * state["fRec103"][2]))) - (self._fConst423 * ((self._fConst422 * state["fRec102"][2]) + (self._fConst421 * state["fRec102"][1]))))) 
		state["fRec101"] = state["fRec101"].at[0].set(((self._fConst423 * (((self._fConst431 * state["fRec102"][0]) + (self._fConst432 * state["fRec102"][1])) + (self._fConst431 * state["fRec102"][2]))) - (self._fConst420 * ((self._fConst419 * state["fRec101"][2]) + (self._fConst418 * state["fRec101"][1]))))) 
		state["fRec100"] = ((fSlow39 * fRec100_temp) + (fSlow40 * jnp.abs((self._fConst420 * (((self._fConst434 * state["fRec101"][0]) + (self._fConst435 * state["fRec101"][1])) + (self._fConst434 * state["fRec101"][2])))))) 
		fVbargraph8 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec100"])))
		self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec107"] = state["fRec107"].at[0].set((fTemp137 - (self._fConst444 * ((self._fConst443 * state["fRec107"][2]) + (self._fConst442 * state["fRec107"][1]))))) 
		state["fRec106"] = state["fRec106"].at[0].set(((self._fConst444 * (((self._fConst446 * state["fRec107"][0]) + (self._fConst447 * state["fRec107"][1])) + (self._fConst446 * state["fRec107"][2]))) - (self._fConst441 * ((self._fConst440 * state["fRec106"][2]) + (self._fConst439 * state["fRec106"][1]))))) 
		state["fRec105"] = state["fRec105"].at[0].set(((self._fConst441 * (((self._fConst449 * state["fRec106"][0]) + (self._fConst450 * state["fRec106"][1])) + (self._fConst449 * state["fRec106"][2]))) - (self._fConst438 * ((self._fConst437 * state["fRec105"][2]) + (self._fConst436 * state["fRec105"][1]))))) 
		state["fRec104"] = ((fSlow39 * fRec104_temp) + (fSlow40 * jnp.abs((self._fConst438 * (((self._fConst452 * state["fRec105"][0]) + (self._fConst453 * state["fRec105"][1])) + (self._fConst452 * state["fRec105"][2])))))) 
		fVbargraph9 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec104"])))
		self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec111"] = state["fRec111"].at[0].set((fTemp136 - (self._fConst462 * ((self._fConst461 * state["fRec111"][2]) + (self._fConst460 * state["fRec111"][1]))))) 
		state["fRec110"] = state["fRec110"].at[0].set(((self._fConst462 * (((self._fConst464 * state["fRec111"][0]) + (self._fConst465 * state["fRec111"][1])) + (self._fConst464 * state["fRec111"][2]))) - (self._fConst459 * ((self._fConst458 * state["fRec110"][2]) + (self._fConst457 * state["fRec110"][1]))))) 
		state["fRec109"] = state["fRec109"].at[0].set(((self._fConst459 * (((self._fConst467 * state["fRec110"][0]) + (self._fConst468 * state["fRec110"][1])) + (self._fConst467 * state["fRec110"][2]))) - (self._fConst456 * ((self._fConst455 * state["fRec109"][2]) + (self._fConst454 * state["fRec109"][1]))))) 
		state["fRec108"] = ((fSlow39 * fRec108_temp) + (fSlow40 * jnp.abs((self._fConst456 * (((self._fConst470 * state["fRec109"][0]) + (self._fConst471 * state["fRec109"][1])) + (self._fConst470 * state["fRec109"][2])))))) 
		fVbargraph10 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec108"])))
		self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec115"] = state["fRec115"].at[0].set((fTemp135 - (self._fConst480 * ((self._fConst479 * state["fRec115"][2]) + (self._fConst478 * state["fRec115"][1]))))) 
		state["fRec114"] = state["fRec114"].at[0].set(((self._fConst480 * (((self._fConst482 * state["fRec115"][0]) + (self._fConst483 * state["fRec115"][1])) + (self._fConst482 * state["fRec115"][2]))) - (self._fConst477 * ((self._fConst476 * state["fRec114"][2]) + (self._fConst475 * state["fRec114"][1]))))) 
		state["fRec113"] = state["fRec113"].at[0].set(((self._fConst477 * (((self._fConst485 * state["fRec114"][0]) + (self._fConst486 * state["fRec114"][1])) + (self._fConst485 * state["fRec114"][2]))) - (self._fConst474 * ((self._fConst473 * state["fRec113"][2]) + (self._fConst472 * state["fRec113"][1]))))) 
		state["fRec112"] = ((fSlow39 * fRec112_temp) + (fSlow40 * jnp.abs((self._fConst474 * (((self._fConst488 * state["fRec113"][0]) + (self._fConst489 * state["fRec113"][1])) + (self._fConst488 * state["fRec113"][2])))))) 
		fVbargraph11 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec112"])))
		self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec119"] = state["fRec119"].at[0].set((fTemp134 - (self._fConst498 * ((self._fConst497 * state["fRec119"][2]) + (self._fConst496 * state["fRec119"][1]))))) 
		state["fRec118"] = state["fRec118"].at[0].set(((self._fConst498 * (((self._fConst500 * state["fRec119"][0]) + (self._fConst501 * state["fRec119"][1])) + (self._fConst500 * state["fRec119"][2]))) - (self._fConst495 * ((self._fConst494 * state["fRec118"][2]) + (self._fConst493 * state["fRec118"][1]))))) 
		state["fRec117"] = state["fRec117"].at[0].set(((self._fConst495 * (((self._fConst503 * state["fRec118"][0]) + (self._fConst504 * state["fRec118"][1])) + (self._fConst503 * state["fRec118"][2]))) - (self._fConst492 * ((self._fConst491 * state["fRec117"][2]) + (self._fConst490 * state["fRec117"][1]))))) 
		state["fRec116"] = ((fSlow39 * fRec116_temp) + (fSlow40 * jnp.abs((self._fConst492 * (((self._fConst506 * state["fRec117"][0]) + (self._fConst507 * state["fRec117"][1])) + (self._fConst506 * state["fRec117"][2])))))) 
		fVbargraph12 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec116"])))
		self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec123"] = state["fRec123"].at[0].set((fTemp133 - (self._fConst516 * ((self._fConst515 * state["fRec123"][2]) + (self._fConst514 * state["fRec123"][1]))))) 
		state["fRec122"] = state["fRec122"].at[0].set(((self._fConst516 * (((self._fConst518 * state["fRec123"][0]) + (self._fConst519 * state["fRec123"][1])) + (self._fConst518 * state["fRec123"][2]))) - (self._fConst513 * ((self._fConst512 * state["fRec122"][2]) + (self._fConst511 * state["fRec122"][1]))))) 
		state["fRec121"] = state["fRec121"].at[0].set(((self._fConst513 * (((self._fConst521 * state["fRec122"][0]) + (self._fConst522 * state["fRec122"][1])) + (self._fConst521 * state["fRec122"][2]))) - (self._fConst510 * ((self._fConst509 * state["fRec121"][2]) + (self._fConst508 * state["fRec121"][1]))))) 
		state["fRec120"] = ((fSlow39 * fRec120_temp) + (fSlow40 * jnp.abs((self._fConst510 * (((self._fConst524 * state["fRec121"][0]) + (self._fConst525 * state["fRec121"][1])) + (self._fConst524 * state["fRec121"][2])))))) 
		fVbargraph13 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec120"])))
		self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec127"] = state["fRec127"].at[0].set((fTemp132 - (self._fConst534 * ((self._fConst533 * state["fRec127"][2]) + (self._fConst532 * state["fRec127"][1]))))) 
		state["fRec126"] = state["fRec126"].at[0].set(((self._fConst534 * (((self._fConst536 * state["fRec127"][0]) + (self._fConst537 * state["fRec127"][1])) + (self._fConst536 * state["fRec127"][2]))) - (self._fConst531 * ((self._fConst530 * state["fRec126"][2]) + (self._fConst529 * state["fRec126"][1]))))) 
		state["fRec125"] = state["fRec125"].at[0].set(((self._fConst531 * (((self._fConst539 * state["fRec126"][0]) + (self._fConst540 * state["fRec126"][1])) + (self._fConst539 * state["fRec126"][2]))) - (self._fConst528 * ((self._fConst527 * state["fRec125"][2]) + (self._fConst526 * state["fRec125"][1]))))) 
		state["fRec124"] = ((fRec124_temp * fSlow39) + (jnp.abs((self._fConst528 * (((self._fConst542 * state["fRec125"][0]) + (self._fConst543 * state["fRec125"][1])) + (self._fConst542 * state["fRec125"][2])))) * fSlow40)) 
		fVbargraph14 = (fSlow41 + (jnp.float32(2e+01) * jnp.log10(state["fRec124"])))
		self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		fTemp146 = fTemp132 
		_result0 = fTemp146 
		_result1 = fTemp146 
		state["iVec0"] = jnp.roll(state["iVec0"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec64"] = jnp.roll(state["fRec64"], 1) 
		state["fRec63"] = jnp.roll(state["fRec63"], 1) 
		state["fRec42"] = jnp.roll(state["fRec42"], 1) 
		state["fRec41"] = jnp.roll(state["fRec41"], 1) 
		state["fRec40"] = jnp.roll(state["fRec40"], 1) 
		state["fRec39"] = jnp.roll(state["fRec39"], 1) 
		state["fRec38"] = jnp.roll(state["fRec38"], 1) 
		state["fRec37"] = jnp.roll(state["fRec37"], 1) 
		state["fRec36"] = jnp.roll(state["fRec36"], 1) 
		state["fRec35"] = jnp.roll(state["fRec35"], 1) 
		state["fRec34"] = jnp.roll(state["fRec34"], 1) 
		state["fRec33"] = jnp.roll(state["fRec33"], 1) 
		state["fRec32"] = jnp.roll(state["fRec32"], 1) 
		state["fRec31"] = jnp.roll(state["fRec31"], 1) 
		state["fRec30"] = jnp.roll(state["fRec30"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec28"] = jnp.roll(state["fRec28"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec26"] = jnp.roll(state["fRec26"], 1) 
		state["fRec25"] = jnp.roll(state["fRec25"], 1) 
		state["fRec24"] = jnp.roll(state["fRec24"], 1) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec22"] = jnp.roll(state["fRec22"], 1) 
		state["fRec21"] = jnp.roll(state["fRec21"], 1) 
		state["fRec20"] = jnp.roll(state["fRec20"], 1) 
		state["fRec19"] = jnp.roll(state["fRec19"], 1) 
		state["fRec18"] = jnp.roll(state["fRec18"], 1) 
		state["fRec17"] = jnp.roll(state["fRec17"], 1) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec15"] = jnp.roll(state["fRec15"], 1) 
		state["fRec14"] = jnp.roll(state["fRec14"], 1) 
		state["fRec13"] = jnp.roll(state["fRec13"], 1) 
		state["fRec12"] = jnp.roll(state["fRec12"], 1) 
		state["fRec11"] = jnp.roll(state["fRec11"], 1) 
		state["fRec10"] = jnp.roll(state["fRec10"], 1) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec75"] = jnp.roll(state["fRec75"], 1) 
		state["fRec74"] = jnp.roll(state["fRec74"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec83"] = jnp.roll(state["fRec83"], 1) 
		state["fRec82"] = jnp.roll(state["fRec82"], 1) 
		state["fRec81"] = jnp.roll(state["fRec81"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec91"] = jnp.roll(state["fRec91"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec94"] = jnp.roll(state["fRec94"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec98"] = jnp.roll(state["fRec98"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec103"] = jnp.roll(state["fRec103"], 1) 
		state["fRec102"] = jnp.roll(state["fRec102"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec110"] = jnp.roll(state["fRec110"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec114"] = jnp.roll(state["fRec114"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec119"] = jnp.roll(state["fRec119"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec123"] = jnp.roll(state["fRec123"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec127"] = jnp.roll(state["fRec127"], 1) 
		state["fRec126"] = jnp.roll(state["fRec126"], 1) 
		state["fRec125"] = jnp.roll(state["fRec125"], 1) 
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
