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
		ui_path.append("gate_compressor") 
		ui_path.append("sawtooth") 
		ui_path.append("SAWTOOTH OSCILLATOR") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider4", ui_path, "Amplitude", -2e+01, -1.2e+02, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider1", ui_path, "Frequency", 49.0, 1.0, 88.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider3", ui_path, "Detuning 1", -0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider2", ui_path, "Detuning 2", 0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider0", ui_path, "Portamento", 0.1, 0.001, 1e+01, unnorm_funcs, "log") 
		self.add_nentry("fEntry0", ui_path, "Saw Order", 2.0, 1.0, 4.0, 1.0, unnorm_funcs, "linear") 
		ui_path.append("Alternate Signals") 
		self.add_button("fCheckbox3", ui_path, "Noise (White or Pink - uses only Amplitude control on the left)", unnorm_funcs) 
		self.add_button("fCheckbox4", ui_path, "Pink instead of White Noise (also called 1/f Noise)", unnorm_funcs) 
		self.add_button("fCheckbox2", ui_path, "External Signal Input (overrides Sawtooth/Noise selection above)", unnorm_funcs) 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.append("gate") 
		ui_path.append("GATE") 
		ui_path.append("0x00") 
		self.add_button("fCheckbox1", ui_path, "Bypass", unnorm_funcs) 
		self.add_hbargraph("fHbargraph0", ui_path, "Gate Gain", -5e+01, 1e+01, unnorm_funcs) 
		ui_path.pop()
		ui_path.append("0x00") 
		self.add_hslider("fHslider2", ui_path, "Threshold", -3e+01, -1.2e+02, 0.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider3", ui_path, "Attack", 1e+01, 1e+01, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider5", ui_path, "Hold", 2e+02, 1.0, 1e+03, unnorm_funcs, "log") 
		self.add_hslider("fHslider4", ui_path, "Release", 1e+02, 1.0, 1e+03, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.append("compressor") 
		ui_path.append("COMPRESSOR") 
		ui_path.append("0x00") 
		self.add_button("fCheckbox0", ui_path, "Bypass", unnorm_funcs) 
		self.add_hbargraph("fHbargraph1", ui_path, "Compressor Gain", -5e+01, 1e+01, unnorm_funcs) 
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("Compression Control") 
		self.add_hslider("fHslider7", ui_path, "Ratio", 5.0, 1.0, 2e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider1", ui_path, "Threshold", -3e+01, -1e+02, 1e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Compression Response") 
		self.add_hslider("fHslider0", ui_path, "Attack", 5e+01, 1.0, 1e+03, unnorm_funcs, "log") 
		self.add_hslider("fHslider6", ui_path, "Release", 5e+02, 1.0, 1e+03, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.pop()
		self.add_hslider("fHslider8", ui_path, "Makeup Gain", 4e+01, -96.0, 96.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.append("spectral") 
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
		self.add_hslider("fHslider9", ui_path, "Level Averaging Time", 1e+02, 1.0, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider10", ui_path, "Level dB Offset", 5e+01, 0.0, 1e+02, unnorm_funcs, "linear") 
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
		self._fConst183 = (np.float32(1.0) / self._fConst0) 
		self._fConst184 = (np.float32(2.0) / self._fConst0) 
		self._fConst185 = (np.float32(0.25) * self._fConst0) 
		self._fConst186 = (np.float32(0.041666668) * np.power(self._fConst0, np.float32(2.0))) 
		self._fConst187 = (np.float32(0.0052083335) * np.power(self._fConst0, np.float32(3.0))) 
		self._fConst188 = (np.float32(0.0001) / self._fConst171) 
		self._fConst189 = (self._fConst188 + np.float32(0.0004332272)) 
		self._fConst190 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst188)) 
		self._fConst191 = (self._fConst172 + np.float32(7.6217313)) 
		self._fConst192 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst172)) 
		self._fConst193 = (self._fConst172 + np.float32(53.53615)) 
		self._fConst194 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst172)) 
		self._fConst195 = (np.float32(0.0001) / self._fConst158) 
		self._fConst196 = (self._fConst195 + np.float32(0.0004332272)) 
		self._fConst197 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst195)) 
		self._fConst198 = (self._fConst159 + np.float32(7.6217313)) 
		self._fConst199 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst159)) 
		self._fConst200 = (self._fConst159 + np.float32(53.53615)) 
		self._fConst201 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst159)) 
		self._fConst202 = (np.float32(0.0001) / self._fConst145) 
		self._fConst203 = (self._fConst202 + np.float32(0.0004332272)) 
		self._fConst204 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst202)) 
		self._fConst205 = (self._fConst146 + np.float32(7.6217313)) 
		self._fConst206 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst146)) 
		self._fConst207 = (self._fConst146 + np.float32(53.53615)) 
		self._fConst208 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst146)) 
		self._fConst209 = (np.float32(0.0001) / self._fConst132) 
		self._fConst210 = (self._fConst209 + np.float32(0.0004332272)) 
		self._fConst211 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst209)) 
		self._fConst212 = (self._fConst133 + np.float32(7.6217313)) 
		self._fConst213 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst133)) 
		self._fConst214 = (self._fConst133 + np.float32(53.53615)) 
		self._fConst215 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst133)) 
		self._fConst216 = (np.float32(0.0001) / self._fConst119) 
		self._fConst217 = (self._fConst216 + np.float32(0.0004332272)) 
		self._fConst218 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst216)) 
		self._fConst219 = (self._fConst120 + np.float32(7.6217313)) 
		self._fConst220 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst120)) 
		self._fConst221 = (self._fConst120 + np.float32(53.53615)) 
		self._fConst222 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst120)) 
		self._fConst223 = (np.float32(0.0001) / self._fConst106) 
		self._fConst224 = (self._fConst223 + np.float32(0.0004332272)) 
		self._fConst225 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst223)) 
		self._fConst226 = (self._fConst107 + np.float32(7.6217313)) 
		self._fConst227 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst107)) 
		self._fConst228 = (self._fConst107 + np.float32(53.53615)) 
		self._fConst229 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst107)) 
		self._fConst230 = (np.float32(0.0001) / self._fConst93) 
		self._fConst231 = (self._fConst230 + np.float32(0.0004332272)) 
		self._fConst232 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst230)) 
		self._fConst233 = (self._fConst94 + np.float32(7.6217313)) 
		self._fConst234 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst94)) 
		self._fConst235 = (self._fConst94 + np.float32(53.53615)) 
		self._fConst236 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst94)) 
		self._fConst237 = (np.float32(0.0001) / self._fConst80) 
		self._fConst238 = (self._fConst237 + np.float32(0.0004332272)) 
		self._fConst239 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst237)) 
		self._fConst240 = (self._fConst81 + np.float32(7.6217313)) 
		self._fConst241 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst81)) 
		self._fConst242 = (self._fConst81 + np.float32(53.53615)) 
		self._fConst243 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst81)) 
		self._fConst244 = (np.float32(0.0001) / self._fConst67) 
		self._fConst245 = (self._fConst244 + np.float32(0.0004332272)) 
		self._fConst246 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst244)) 
		self._fConst247 = (self._fConst68 + np.float32(7.6217313)) 
		self._fConst248 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst68)) 
		self._fConst249 = (self._fConst68 + np.float32(53.53615)) 
		self._fConst250 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst68)) 
		self._fConst251 = (np.float32(0.0001) / self._fConst54) 
		self._fConst252 = (self._fConst251 + np.float32(0.0004332272)) 
		self._fConst253 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst251)) 
		self._fConst254 = (self._fConst55 + np.float32(7.6217313)) 
		self._fConst255 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst55)) 
		self._fConst256 = (self._fConst55 + np.float32(53.53615)) 
		self._fConst257 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst55)) 
		self._fConst258 = (np.float32(0.0001) / self._fConst41) 
		self._fConst259 = (self._fConst258 + np.float32(0.0004332272)) 
		self._fConst260 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst258)) 
		self._fConst261 = (self._fConst42 + np.float32(7.6217313)) 
		self._fConst262 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst42)) 
		self._fConst263 = (self._fConst42 + np.float32(53.53615)) 
		self._fConst264 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst42)) 
		self._fConst265 = (np.float32(0.0001) / self._fConst28) 
		self._fConst266 = (self._fConst265 + np.float32(0.0004332272)) 
		self._fConst267 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst265)) 
		self._fConst268 = (self._fConst29 + np.float32(7.6217313)) 
		self._fConst269 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst29)) 
		self._fConst270 = (self._fConst29 + np.float32(53.53615)) 
		self._fConst271 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst29)) 
		self._fConst272 = (np.float32(0.0001) / self._fConst15) 
		self._fConst273 = (self._fConst272 + np.float32(0.0004332272)) 
		self._fConst274 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst272)) 
		self._fConst275 = (self._fConst16 + np.float32(7.6217313)) 
		self._fConst276 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst16)) 
		self._fConst277 = (self._fConst16 + np.float32(53.53615)) 
		self._fConst278 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst16)) 
		self._fConst279 = (np.float32(0.0001) / self._fConst2) 
		self._fConst280 = (self._fConst279 + np.float32(0.0004332272)) 
		self._fConst281 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst279)) 
		self._fConst282 = (self._fConst3 + np.float32(7.6217313)) 
		self._fConst283 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst3)) 
		self._fConst284 = (self._fConst3 + np.float32(53.53615)) 
		self._fConst285 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst3)) 
		self._fConst286 = (np.float32(1e+03) / self._fConst0) 
		self._fConst287 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst3)) 
		self._fConst288 = (((self._fConst5 + np.float32(-0.15748216)) / self._fConst1) + np.float32(0.9351402)) 
		self._fConst289 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.15748216)) / self._fConst1) + np.float32(0.9351402))) 
		self._fConst290 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst3)) 
		self._fConst291 = (((self._fConst5 + np.float32(-0.74313045)) / self._fConst1) + np.float32(1.4500711)) 
		self._fConst292 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.74313045)) / self._fConst1) + np.float32(1.4500711))) 
		self._fConst293 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst3)) 
		self._fConst294 = (((self._fConst5 + np.float32(-3.1897273)) / self._fConst1) + np.float32(4.0767817)) 
		self._fConst295 = (np.float32(1.0) / (((self._fConst5 + np.float32(3.1897273)) / self._fConst1) + np.float32(4.0767817))) 
		self._fConst296 = (np.float32(0.0017661728) / self._fConst2) 
		self._fConst297 = (self._fConst296 + np.float32(0.0004076782)) 
		self._fConst298 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst296)) 
		self._fConst299 = (np.float32(11.0520525) / self._fConst2) 
		self._fConst300 = (self._fConst299 + np.float32(1.4500711)) 
		self._fConst301 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst299)) 
		self._fConst302 = (np.float32(50.06381) / self._fConst2) 
		self._fConst303 = (self._fConst302 + np.float32(0.9351402)) 
		self._fConst304 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst302)) 
		self._fConst305 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst16)) 
		self._fConst306 = (((self._fConst18 + np.float32(-0.15748216)) / self._fConst14) + np.float32(0.9351402)) 
		self._fConst307 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.15748216)) / self._fConst14) + np.float32(0.9351402))) 
		self._fConst308 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst16)) 
		self._fConst309 = (((self._fConst18 + np.float32(-0.74313045)) / self._fConst14) + np.float32(1.4500711)) 
		self._fConst310 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.74313045)) / self._fConst14) + np.float32(1.4500711))) 
		self._fConst311 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst16)) 
		self._fConst312 = (((self._fConst18 + np.float32(-3.1897273)) / self._fConst14) + np.float32(4.0767817)) 
		self._fConst313 = (np.float32(1.0) / (((self._fConst18 + np.float32(3.1897273)) / self._fConst14) + np.float32(4.0767817))) 
		self._fConst314 = (np.float32(0.0017661728) / self._fConst15) 
		self._fConst315 = (self._fConst314 + np.float32(0.0004076782)) 
		self._fConst316 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst314)) 
		self._fConst317 = (np.float32(11.0520525) / self._fConst15) 
		self._fConst318 = (self._fConst317 + np.float32(1.4500711)) 
		self._fConst319 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst317)) 
		self._fConst320 = (np.float32(50.06381) / self._fConst15) 
		self._fConst321 = (self._fConst320 + np.float32(0.9351402)) 
		self._fConst322 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst320)) 
		self._fConst323 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst29)) 
		self._fConst324 = (((self._fConst31 + np.float32(-0.15748216)) / self._fConst27) + np.float32(0.9351402)) 
		self._fConst325 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.15748216)) / self._fConst27) + np.float32(0.9351402))) 
		self._fConst326 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst29)) 
		self._fConst327 = (((self._fConst31 + np.float32(-0.74313045)) / self._fConst27) + np.float32(1.4500711)) 
		self._fConst328 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.74313045)) / self._fConst27) + np.float32(1.4500711))) 
		self._fConst329 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst29)) 
		self._fConst330 = (((self._fConst31 + np.float32(-3.1897273)) / self._fConst27) + np.float32(4.0767817)) 
		self._fConst331 = (np.float32(1.0) / (((self._fConst31 + np.float32(3.1897273)) / self._fConst27) + np.float32(4.0767817))) 
		self._fConst332 = (np.float32(0.0017661728) / self._fConst28) 
		self._fConst333 = (self._fConst332 + np.float32(0.0004076782)) 
		self._fConst334 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst332)) 
		self._fConst335 = (np.float32(11.0520525) / self._fConst28) 
		self._fConst336 = (self._fConst335 + np.float32(1.4500711)) 
		self._fConst337 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst335)) 
		self._fConst338 = (np.float32(50.06381) / self._fConst28) 
		self._fConst339 = (self._fConst338 + np.float32(0.9351402)) 
		self._fConst340 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst338)) 
		self._fConst341 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst42)) 
		self._fConst342 = (((self._fConst44 + np.float32(-0.15748216)) / self._fConst40) + np.float32(0.9351402)) 
		self._fConst343 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.15748216)) / self._fConst40) + np.float32(0.9351402))) 
		self._fConst344 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst42)) 
		self._fConst345 = (((self._fConst44 + np.float32(-0.74313045)) / self._fConst40) + np.float32(1.4500711)) 
		self._fConst346 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.74313045)) / self._fConst40) + np.float32(1.4500711))) 
		self._fConst347 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst42)) 
		self._fConst348 = (((self._fConst44 + np.float32(-3.1897273)) / self._fConst40) + np.float32(4.0767817)) 
		self._fConst349 = (np.float32(1.0) / (((self._fConst44 + np.float32(3.1897273)) / self._fConst40) + np.float32(4.0767817))) 
		self._fConst350 = (np.float32(0.0017661728) / self._fConst41) 
		self._fConst351 = (self._fConst350 + np.float32(0.0004076782)) 
		self._fConst352 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst350)) 
		self._fConst353 = (np.float32(11.0520525) / self._fConst41) 
		self._fConst354 = (self._fConst353 + np.float32(1.4500711)) 
		self._fConst355 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst353)) 
		self._fConst356 = (np.float32(50.06381) / self._fConst41) 
		self._fConst357 = (self._fConst356 + np.float32(0.9351402)) 
		self._fConst358 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst356)) 
		self._fConst359 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst55)) 
		self._fConst360 = (((self._fConst57 + np.float32(-0.15748216)) / self._fConst53) + np.float32(0.9351402)) 
		self._fConst361 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.15748216)) / self._fConst53) + np.float32(0.9351402))) 
		self._fConst362 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst55)) 
		self._fConst363 = (((self._fConst57 + np.float32(-0.74313045)) / self._fConst53) + np.float32(1.4500711)) 
		self._fConst364 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.74313045)) / self._fConst53) + np.float32(1.4500711))) 
		self._fConst365 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst55)) 
		self._fConst366 = (((self._fConst57 + np.float32(-3.1897273)) / self._fConst53) + np.float32(4.0767817)) 
		self._fConst367 = (np.float32(1.0) / (((self._fConst57 + np.float32(3.1897273)) / self._fConst53) + np.float32(4.0767817))) 
		self._fConst368 = (np.float32(0.0017661728) / self._fConst54) 
		self._fConst369 = (self._fConst368 + np.float32(0.0004076782)) 
		self._fConst370 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst368)) 
		self._fConst371 = (np.float32(11.0520525) / self._fConst54) 
		self._fConst372 = (self._fConst371 + np.float32(1.4500711)) 
		self._fConst373 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst371)) 
		self._fConst374 = (np.float32(50.06381) / self._fConst54) 
		self._fConst375 = (self._fConst374 + np.float32(0.9351402)) 
		self._fConst376 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst374)) 
		self._fConst377 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst68)) 
		self._fConst378 = (((self._fConst70 + np.float32(-0.15748216)) / self._fConst66) + np.float32(0.9351402)) 
		self._fConst379 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.15748216)) / self._fConst66) + np.float32(0.9351402))) 
		self._fConst380 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst68)) 
		self._fConst381 = (((self._fConst70 + np.float32(-0.74313045)) / self._fConst66) + np.float32(1.4500711)) 
		self._fConst382 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.74313045)) / self._fConst66) + np.float32(1.4500711))) 
		self._fConst383 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst68)) 
		self._fConst384 = (((self._fConst70 + np.float32(-3.1897273)) / self._fConst66) + np.float32(4.0767817)) 
		self._fConst385 = (np.float32(1.0) / (((self._fConst70 + np.float32(3.1897273)) / self._fConst66) + np.float32(4.0767817))) 
		self._fConst386 = (np.float32(0.0017661728) / self._fConst67) 
		self._fConst387 = (self._fConst386 + np.float32(0.0004076782)) 
		self._fConst388 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst386)) 
		self._fConst389 = (np.float32(11.0520525) / self._fConst67) 
		self._fConst390 = (self._fConst389 + np.float32(1.4500711)) 
		self._fConst391 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst389)) 
		self._fConst392 = (np.float32(50.06381) / self._fConst67) 
		self._fConst393 = (self._fConst392 + np.float32(0.9351402)) 
		self._fConst394 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst392)) 
		self._fConst395 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst81)) 
		self._fConst396 = (((self._fConst83 + np.float32(-0.15748216)) / self._fConst79) + np.float32(0.9351402)) 
		self._fConst397 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.15748216)) / self._fConst79) + np.float32(0.9351402))) 
		self._fConst398 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst81)) 
		self._fConst399 = (((self._fConst83 + np.float32(-0.74313045)) / self._fConst79) + np.float32(1.4500711)) 
		self._fConst400 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.74313045)) / self._fConst79) + np.float32(1.4500711))) 
		self._fConst401 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst81)) 
		self._fConst402 = (((self._fConst83 + np.float32(-3.1897273)) / self._fConst79) + np.float32(4.0767817)) 
		self._fConst403 = (np.float32(1.0) / (((self._fConst83 + np.float32(3.1897273)) / self._fConst79) + np.float32(4.0767817))) 
		self._fConst404 = (np.float32(0.0017661728) / self._fConst80) 
		self._fConst405 = (self._fConst404 + np.float32(0.0004076782)) 
		self._fConst406 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst404)) 
		self._fConst407 = (np.float32(11.0520525) / self._fConst80) 
		self._fConst408 = (self._fConst407 + np.float32(1.4500711)) 
		self._fConst409 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst407)) 
		self._fConst410 = (np.float32(50.06381) / self._fConst80) 
		self._fConst411 = (self._fConst410 + np.float32(0.9351402)) 
		self._fConst412 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst410)) 
		self._fConst413 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst94)) 
		self._fConst414 = (((self._fConst96 + np.float32(-0.15748216)) / self._fConst92) + np.float32(0.9351402)) 
		self._fConst415 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.15748216)) / self._fConst92) + np.float32(0.9351402))) 
		self._fConst416 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst94)) 
		self._fConst417 = (((self._fConst96 + np.float32(-0.74313045)) / self._fConst92) + np.float32(1.4500711)) 
		self._fConst418 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.74313045)) / self._fConst92) + np.float32(1.4500711))) 
		self._fConst419 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst94)) 
		self._fConst420 = (((self._fConst96 + np.float32(-3.1897273)) / self._fConst92) + np.float32(4.0767817)) 
		self._fConst421 = (np.float32(1.0) / (((self._fConst96 + np.float32(3.1897273)) / self._fConst92) + np.float32(4.0767817))) 
		self._fConst422 = (np.float32(0.0017661728) / self._fConst93) 
		self._fConst423 = (self._fConst422 + np.float32(0.0004076782)) 
		self._fConst424 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst422)) 
		self._fConst425 = (np.float32(11.0520525) / self._fConst93) 
		self._fConst426 = (self._fConst425 + np.float32(1.4500711)) 
		self._fConst427 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst425)) 
		self._fConst428 = (np.float32(50.06381) / self._fConst93) 
		self._fConst429 = (self._fConst428 + np.float32(0.9351402)) 
		self._fConst430 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst428)) 
		self._fConst431 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst107)) 
		self._fConst432 = (((self._fConst109 + np.float32(-0.15748216)) / self._fConst105) + np.float32(0.9351402)) 
		self._fConst433 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.15748216)) / self._fConst105) + np.float32(0.9351402))) 
		self._fConst434 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst107)) 
		self._fConst435 = (((self._fConst109 + np.float32(-0.74313045)) / self._fConst105) + np.float32(1.4500711)) 
		self._fConst436 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.74313045)) / self._fConst105) + np.float32(1.4500711))) 
		self._fConst437 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst107)) 
		self._fConst438 = (((self._fConst109 + np.float32(-3.1897273)) / self._fConst105) + np.float32(4.0767817)) 
		self._fConst439 = (np.float32(1.0) / (((self._fConst109 + np.float32(3.1897273)) / self._fConst105) + np.float32(4.0767817))) 
		self._fConst440 = (np.float32(0.0017661728) / self._fConst106) 
		self._fConst441 = (self._fConst440 + np.float32(0.0004076782)) 
		self._fConst442 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst440)) 
		self._fConst443 = (np.float32(11.0520525) / self._fConst106) 
		self._fConst444 = (self._fConst443 + np.float32(1.4500711)) 
		self._fConst445 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst443)) 
		self._fConst446 = (np.float32(50.06381) / self._fConst106) 
		self._fConst447 = (self._fConst446 + np.float32(0.9351402)) 
		self._fConst448 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst446)) 
		self._fConst449 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst120)) 
		self._fConst450 = (((self._fConst122 + np.float32(-0.15748216)) / self._fConst118) + np.float32(0.9351402)) 
		self._fConst451 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.15748216)) / self._fConst118) + np.float32(0.9351402))) 
		self._fConst452 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst120)) 
		self._fConst453 = (((self._fConst122 + np.float32(-0.74313045)) / self._fConst118) + np.float32(1.4500711)) 
		self._fConst454 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.74313045)) / self._fConst118) + np.float32(1.4500711))) 
		self._fConst455 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst120)) 
		self._fConst456 = (((self._fConst122 + np.float32(-3.1897273)) / self._fConst118) + np.float32(4.0767817)) 
		self._fConst457 = (np.float32(1.0) / (((self._fConst122 + np.float32(3.1897273)) / self._fConst118) + np.float32(4.0767817))) 
		self._fConst458 = (np.float32(0.0017661728) / self._fConst119) 
		self._fConst459 = (self._fConst458 + np.float32(0.0004076782)) 
		self._fConst460 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst458)) 
		self._fConst461 = (np.float32(11.0520525) / self._fConst119) 
		self._fConst462 = (self._fConst461 + np.float32(1.4500711)) 
		self._fConst463 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst461)) 
		self._fConst464 = (np.float32(50.06381) / self._fConst119) 
		self._fConst465 = (self._fConst464 + np.float32(0.9351402)) 
		self._fConst466 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst464)) 
		self._fConst467 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst133)) 
		self._fConst468 = (((self._fConst135 + np.float32(-0.15748216)) / self._fConst131) + np.float32(0.9351402)) 
		self._fConst469 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.15748216)) / self._fConst131) + np.float32(0.9351402))) 
		self._fConst470 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst133)) 
		self._fConst471 = (((self._fConst135 + np.float32(-0.74313045)) / self._fConst131) + np.float32(1.4500711)) 
		self._fConst472 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.74313045)) / self._fConst131) + np.float32(1.4500711))) 
		self._fConst473 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst133)) 
		self._fConst474 = (((self._fConst135 + np.float32(-3.1897273)) / self._fConst131) + np.float32(4.0767817)) 
		self._fConst475 = (np.float32(1.0) / (((self._fConst135 + np.float32(3.1897273)) / self._fConst131) + np.float32(4.0767817))) 
		self._fConst476 = (np.float32(0.0017661728) / self._fConst132) 
		self._fConst477 = (self._fConst476 + np.float32(0.0004076782)) 
		self._fConst478 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst476)) 
		self._fConst479 = (np.float32(11.0520525) / self._fConst132) 
		self._fConst480 = (self._fConst479 + np.float32(1.4500711)) 
		self._fConst481 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst479)) 
		self._fConst482 = (np.float32(50.06381) / self._fConst132) 
		self._fConst483 = (self._fConst482 + np.float32(0.9351402)) 
		self._fConst484 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst482)) 
		self._fConst485 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst146)) 
		self._fConst486 = (((self._fConst148 + np.float32(-0.15748216)) / self._fConst144) + np.float32(0.9351402)) 
		self._fConst487 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.15748216)) / self._fConst144) + np.float32(0.9351402))) 
		self._fConst488 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst146)) 
		self._fConst489 = (((self._fConst148 + np.float32(-0.74313045)) / self._fConst144) + np.float32(1.4500711)) 
		self._fConst490 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.74313045)) / self._fConst144) + np.float32(1.4500711))) 
		self._fConst491 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst146)) 
		self._fConst492 = (((self._fConst148 + np.float32(-3.1897273)) / self._fConst144) + np.float32(4.0767817)) 
		self._fConst493 = (np.float32(1.0) / (((self._fConst148 + np.float32(3.1897273)) / self._fConst144) + np.float32(4.0767817))) 
		self._fConst494 = (np.float32(0.0017661728) / self._fConst145) 
		self._fConst495 = (self._fConst494 + np.float32(0.0004076782)) 
		self._fConst496 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst494)) 
		self._fConst497 = (np.float32(11.0520525) / self._fConst145) 
		self._fConst498 = (self._fConst497 + np.float32(1.4500711)) 
		self._fConst499 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst497)) 
		self._fConst500 = (np.float32(50.06381) / self._fConst145) 
		self._fConst501 = (self._fConst500 + np.float32(0.9351402)) 
		self._fConst502 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst500)) 
		self._fConst503 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst159)) 
		self._fConst504 = (((self._fConst161 + np.float32(-0.15748216)) / self._fConst157) + np.float32(0.9351402)) 
		self._fConst505 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.15748216)) / self._fConst157) + np.float32(0.9351402))) 
		self._fConst506 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst159)) 
		self._fConst507 = (((self._fConst161 + np.float32(-0.74313045)) / self._fConst157) + np.float32(1.4500711)) 
		self._fConst508 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.74313045)) / self._fConst157) + np.float32(1.4500711))) 
		self._fConst509 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst159)) 
		self._fConst510 = (((self._fConst161 + np.float32(-3.1897273)) / self._fConst157) + np.float32(4.0767817)) 
		self._fConst511 = (np.float32(1.0) / (((self._fConst161 + np.float32(3.1897273)) / self._fConst157) + np.float32(4.0767817))) 
		self._fConst512 = (np.float32(0.0017661728) / self._fConst158) 
		self._fConst513 = (self._fConst512 + np.float32(0.0004076782)) 
		self._fConst514 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst512)) 
		self._fConst515 = (np.float32(11.0520525) / self._fConst158) 
		self._fConst516 = (self._fConst515 + np.float32(1.4500711)) 
		self._fConst517 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst515)) 
		self._fConst518 = (np.float32(50.06381) / self._fConst158) 
		self._fConst519 = (self._fConst518 + np.float32(0.9351402)) 
		self._fConst520 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst518)) 
		self._fConst521 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst172)) 
		self._fConst522 = (((self._fConst174 + np.float32(-0.15748216)) / self._fConst170) + np.float32(0.9351402)) 
		self._fConst523 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.15748216)) / self._fConst170) + np.float32(0.9351402))) 
		self._fConst524 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst172)) 
		self._fConst525 = (((self._fConst174 + np.float32(-0.74313045)) / self._fConst170) + np.float32(1.4500711)) 
		self._fConst526 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.74313045)) / self._fConst170) + np.float32(1.4500711))) 
		self._fConst527 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst172)) 
		self._fConst528 = (((self._fConst174 + np.float32(-3.1897273)) / self._fConst170) + np.float32(4.0767817)) 
		self._fConst529 = (np.float32(1.0) / (((self._fConst174 + np.float32(3.1897273)) / self._fConst170) + np.float32(4.0767817))) 
		self._fConst530 = (np.float32(0.0017661728) / self._fConst171) 
		self._fConst531 = (self._fConst530 + np.float32(0.0004076782)) 
		self._fConst532 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst530)) 
		self._fConst533 = (np.float32(11.0520525) / self._fConst171) 
		self._fConst534 = (self._fConst533 + np.float32(1.4500711)) 
		self._fConst535 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst533)) 
		self._fConst536 = (np.float32(50.06381) / self._fConst171) 
		self._fConst537 = (self._fConst536 + np.float32(0.9351402)) 
		self._fConst538 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst536)) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec102"] = np.float32(0)
		state["fRec106"] = np.float32(0)
		state["fRec110"] = np.float32(0)
		state["fRec114"] = np.float32(0)
		state["fRec118"] = np.float32(0)
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
		state["fRec56"] = np.float32(0)
		state["fRec57"] = np.float32(0)
		state["fRec59"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec63"] = np.float32(0)
		state["fRec66"] = np.float32(0)
		state["fRec70"] = np.float32(0)
		state["fRec74"] = np.float32(0)
		state["fRec78"] = np.float32(0)
		state["fRec82"] = np.float32(0)
		state["fRec86"] = np.float32(0)
		state["fRec90"] = np.float32(0)
		state["fRec94"] = np.float32(0)
		state["fRec98"] = np.float32(0)
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
		state["fVec2"] = np.float32(0)
		state["fVec3"] = np.float32(0)
		state["fVec4"] = np.float32(0)
		state["fVec5"] = np.float32(0)
		state["fVec6"] = np.float32(0)
		state["fVec7"] = np.float32(0)
		state["fVec8"] = np.float32(0)
		state["fVec9"] = np.float32(0)
		state["iRec54"] = np.int32(0)
		state["iRec64"] = np.int32(0)
		state["iRec65"] = np.int32(0)
		state["iVec19"] = np.int32(0)
		state["iVec20"] = np.int32(0)
		# Initialize array delays
		state["iVec0"] = np.zeros((4,), dtype=np.int32)
		state["fRec55"] = np.zeros((4,), dtype=np.float32)
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
		state["fRec69"] = np.zeros((3,), dtype=np.float32)
		state["fRec68"] = np.zeros((3,), dtype=np.float32)
		state["fRec67"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec72"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fRec76"] = np.zeros((3,), dtype=np.float32)
		state["fRec75"] = np.zeros((3,), dtype=np.float32)
		state["fRec81"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec84"] = np.zeros((3,), dtype=np.float32)
		state["fRec83"] = np.zeros((3,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fRec88"] = np.zeros((3,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec92"] = np.zeros((3,), dtype=np.float32)
		state["fRec91"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec96"] = np.zeros((3,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec103"] = np.zeros((3,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec112"] = np.zeros((3,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec116"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec120"] = np.zeros((3,), dtype=np.float32)
		state["fRec119"] = np.zeros((3,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		iSlow0 = jnp.int32(params["fCheckbox0"]) 
		fSlow1 = jnp.maximum(self._fConst183, (jnp.float32(0.001) * params["fHslider0"])) 
		fSlow2 = jnp.where((((jnp.float32(0.5) * fSlow1) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst184 / fSlow1))), jnp.float32(0.0)) 
		fSlow3 = (jnp.float32(1.0) - fSlow2) 
		fSlow4 = params["fHslider1"] 
		iSlow5 = jnp.int32(params["fCheckbox1"]) 
		iSlow6 = jnp.int32(params["fCheckbox2"]) 
		iSlow7 = jnp.int32(params["fCheckbox3"]) 
		iSlow8 = jnp.int32((params["fEntry0"] + jnp.float32(-1.0))) 
		iSlow9 = (iSlow8 >= jnp.int32(2)).astype(jnp.int32) 
		iSlow10 = (iSlow8 >= jnp.int32(1)).astype(jnp.int32) 
		fSlow11 = params["fVslider0"] 
		fSlow12 = jnp.where(((fSlow11 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst183 / fSlow11))), jnp.float32(0.0)) 
		fSlow13 = ((jnp.float32(4.4e+02) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fVslider1"] + jnp.float32(-49.0))))) * (jnp.float32(1.0) - fSlow12)) 
		fSlow14 = ((jnp.float32(0.01) * params["fVslider2"]) + jnp.float32(1.0)) 
		iSlow15 = (iSlow8 >= jnp.int32(3)).astype(jnp.int32) 
		fSlow16 = ((jnp.float32(0.01) * params["fVslider3"]) + jnp.float32(1.0)) 
		fSlow17 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider4"]))) 
		iSlow18 = jnp.int32(params["fCheckbox4"]) 
		fSlow19 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider2"])) 
		fSlow20 = jnp.maximum(self._fConst183, (jnp.float32(1e-06) * params["fHslider3"])) 
		fSlow21 = jnp.maximum(self._fConst183, (jnp.float32(0.001) * params["fHslider4"])) 
		fSlow22 = jnp.minimum(fSlow20, fSlow21) 
		fSlow23 = jnp.where(((fSlow22 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst183 / fSlow22))), jnp.float32(0.0)) 
		fSlow24 = (jnp.float32(1.0) - fSlow23) 
		iSlow25 = jnp.int32((self._fConst0 * jnp.maximum(self._fConst183, (jnp.float32(0.001) * params["fHslider5"])))) 
		fSlow26 = jnp.where(((fSlow20 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst183 / fSlow20))), jnp.float32(0.0)) 
		fSlow27 = jnp.where(((fSlow21 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst183 / fSlow21))), jnp.float32(0.0)) 
		fSlow28 = jnp.where(((fSlow1 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst183 / fSlow1))), jnp.float32(0.0)) 
		fSlow29 = jnp.maximum(self._fConst183, (jnp.float32(0.001) * params["fHslider6"])) 
		fSlow30 = jnp.where(((fSlow29 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst183 / fSlow29))), jnp.float32(0.0)) 
		fSlow31 = ((jnp.float32(1.0) / params["fHslider7"]) + jnp.float32(-1.0)) 
		fSlow32 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider8"])) 
		fSlow33 = params["fHslider9"] 
		fSlow34 = jnp.where((((jnp.float32(0.001) * fSlow33) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst286 / fSlow33))), jnp.float32(0.0)) 
		fSlow35 = (jnp.float32(1.0) - fSlow34) 
		fSlow36 = params["fHslider10"] 
		fRec50_temp = state["fRec50"] 
		fRec49_temp = state["fRec49"] 
		fVec1_temp = state["fVec1"] 
		fVec2_temp = state["fVec2"] 
		fVec3_temp = state["fVec3"] 
		fVec4_temp = state["fVec4"] 
		fVec5_temp = state["fVec5"] 
		fVec6_temp = state["fVec6"] 
		fRec51_temp = state["fRec51"] 
		fVec7_temp = state["fVec7"] 
		fVec8_temp = state["fVec8"] 
		fVec9_temp = state["fVec9"] 
		fVec10_temp = state["fVec10"] 
		fVec11_temp = state["fVec11"] 
		fVec12_temp = state["fVec12"] 
		fRec52_temp = state["fRec52"] 
		fVec13_temp = state["fVec13"] 
		fVec14_temp = state["fVec14"] 
		fVec15_temp = state["fVec15"] 
		fVec16_temp = state["fVec16"] 
		fVec17_temp = state["fVec17"] 
		fVec18_temp = state["fVec18"] 
		fRec53_temp = state["fRec53"] 
		iRec54_temp = state["iRec54"] 
		fRec63_temp = state["fRec63"] 
		iVec19_temp = state["iVec19"] 
		iRec64_temp = state["iRec64"] 
		fRec61_temp = state["fRec61"] 
		fRec60_temp = state["fRec60"] 
		fRec59_temp = state["fRec59"] 
		iVec20_temp = state["iVec20"] 
		iRec65_temp = state["iRec65"] 
		fRec57_temp = state["fRec57"] 
		fRec56_temp = state["fRec56"] 
		fRec48_temp = state["fRec48"] 
		fRec47_temp = state["fRec47"] 
		fRec46_temp = state["fRec46"] 
		fRec45_temp = state["fRec45"] 
		fRec44_temp = state["fRec44"] 
		fRec43_temp = state["fRec43"] 
		fRec0_temp = state["fRec0"] 
		fRec66_temp = state["fRec66"] 
		fRec70_temp = state["fRec70"] 
		fRec74_temp = state["fRec74"] 
		fRec78_temp = state["fRec78"] 
		fRec82_temp = state["fRec82"] 
		fRec86_temp = state["fRec86"] 
		fRec90_temp = state["fRec90"] 
		fRec94_temp = state["fRec94"] 
		fRec98_temp = state["fRec98"] 
		fRec102_temp = state["fRec102"] 
		fRec106_temp = state["fRec106"] 
		fRec110_temp = state["fRec110"] 
		fRec114_temp = state["fRec114"] 
		fRec118_temp = state["fRec118"] 
		state["iVec0"] = state["iVec0"].at[0].set(jnp.int32(1)) 
		state["fRec50"] = ((fRec50_temp * fSlow12) + fSlow13) 
		fTemp0 = jnp.maximum(jnp.float32(2e+01), jnp.abs((fSlow14 * state["fRec50"]))) 
		fTemp1 = (fRec49_temp + (self._fConst183 * fTemp0)) 
		state["fRec49"] = (fTemp1 - jnp.floor(fTemp1)) 
		fTemp2 = (jnp.float32(2.0) * state["fRec49"]) 
		fTemp3 = (fTemp2 + jnp.float32(-1.0)) 
		fTemp4 = jnp.power(fTemp3, jnp.float32(2.0)) 
		state["fVec1"] = jnp.float32(fTemp4) 
		fTemp5 = (state["iVec0"][1]) 
		fTemp6 = jnp.power(fTemp3, jnp.float32(3.0)) 
		state["fVec2"] = (fTemp6 + (jnp.float32(1.0) - fTemp2)) 
		fTemp7 = ((fTemp6 + (jnp.float32(1.0) - (fTemp2 + fVec2_temp))) / fTemp0) 
		state["fVec3"] = jnp.float32(fTemp7) 
		fTemp8 = (state["iVec0"][2]) 
		fTemp9 = (fTemp4 * (fTemp4 + jnp.float32(-2.0))) 
		state["fVec4"] = jnp.float32(fTemp9) 
		fTemp10 = ((fTemp9 - fVec4_temp) / fTemp0) 
		state["fVec5"] = jnp.float32(fTemp10) 
		fTemp11 = ((fTemp10 - fVec5_temp) / fTemp0) 
		state["fVec6"] = jnp.float32(fTemp11) 
		fTemp12 = (state["iVec0"][3]) 
		fTemp13 = jnp.maximum(jnp.float32(2e+01), jnp.abs((fSlow16 * state["fRec50"]))) 
		fTemp14 = (fRec51_temp + (self._fConst183 * fTemp13)) 
		state["fRec51"] = (fTemp14 - jnp.floor(fTemp14)) 
		fTemp15 = (jnp.float32(2.0) * state["fRec51"]) 
		fTemp16 = (fTemp15 + jnp.float32(-1.0)) 
		fTemp17 = jnp.power(fTemp16, jnp.float32(2.0)) 
		state["fVec7"] = jnp.float32(fTemp17) 
		fTemp18 = jnp.power(fTemp16, jnp.float32(3.0)) 
		state["fVec8"] = (fTemp18 + (jnp.float32(1.0) - fTemp15)) 
		fTemp19 = ((fTemp18 + (jnp.float32(1.0) - (fTemp15 + fVec8_temp))) / fTemp13) 
		state["fVec9"] = jnp.float32(fTemp19) 
		fTemp20 = (fTemp17 * (fTemp17 + jnp.float32(-2.0))) 
		state["fVec10"] = jnp.float32(fTemp20) 
		fTemp21 = ((fTemp20 - fVec10_temp) / fTemp13) 
		state["fVec11"] = jnp.float32(fTemp21) 
		fTemp22 = ((fTemp21 - fVec11_temp) / fTemp13) 
		state["fVec12"] = jnp.float32(fTemp22) 
		fTemp23 = jnp.maximum(jnp.float32(2e+01), jnp.abs(state["fRec50"])) 
		fTemp24 = (fRec52_temp + (self._fConst183 * fTemp23)) 
		state["fRec52"] = (fTemp24 - jnp.floor(fTemp24)) 
		fTemp25 = (jnp.float32(2.0) * state["fRec52"]) 
		fTemp26 = (fTemp25 + jnp.float32(-1.0)) 
		fTemp27 = jnp.power(fTemp26, jnp.float32(2.0)) 
		state["fVec13"] = jnp.float32(fTemp27) 
		fTemp28 = jnp.power(fTemp26, jnp.float32(3.0)) 
		state["fVec14"] = (fTemp28 + (jnp.float32(1.0) - fTemp25)) 
		fTemp29 = ((fTemp28 + (jnp.float32(1.0) - (fTemp25 + fVec14_temp))) / fTemp23) 
		state["fVec15"] = jnp.float32(fTemp29) 
		fTemp30 = (fTemp27 * (fTemp27 + jnp.float32(-2.0))) 
		state["fVec16"] = jnp.float32(fTemp30) 
		fTemp31 = ((fTemp30 - fVec16_temp) / fTemp23) 
		state["fVec17"] = jnp.float32(fTemp31) 
		fTemp32 = ((fTemp31 - fVec17_temp) / fTemp23) 
		state["fVec18"] = jnp.float32(fTemp32) 
		state["fRec53"] = (fSlow17 + (jnp.float32(0.999) * fRec53_temp)) 
		state["iRec54"] = ((jnp.int32(1103515245) * iRec54_temp) + jnp.int32(12345)) 
		fTemp33 = (jnp.float32(4.656613e-10) * (state["iRec54"])) 
		state["fRec55"] = state["fRec55"].at[0].set((((jnp.float32(0.5221894) * state["fRec55"][3]) + (fTemp33 + (jnp.float32(2.494956) * state["fRec55"][1]))) - (jnp.float32(2.0172658) * state["fRec55"][2]))) 
		fTemp34 = (state["fRec53"] * jnp.where((iSlow6 != 0), inputs[0], jnp.where((iSlow7 != 0), jnp.where((iSlow18 != 0), (((jnp.float32(0.049922034) * state["fRec55"][0]) + (jnp.float32(0.0506127) * state["fRec55"][2])) - ((jnp.float32(0.095993534) * state["fRec55"][1]) + (jnp.float32(0.004408786) * state["fRec55"][3]))), fTemp33), (jnp.float32(0.33333334) * (state["fRec53"] * ((jnp.where((iSlow9 != 0), jnp.where((iSlow15 != 0), (self._fConst187 * ((fTemp12 * (fTemp32 - fVec18_temp)) / fTemp23)), (self._fConst186 * ((fTemp8 * (fTemp29 - fVec15_temp)) / fTemp23))), jnp.where((iSlow10 != 0), (self._fConst185 * ((fTemp5 * (fTemp27 - fVec13_temp)) / fTemp23)), fTemp26)) + jnp.where((iSlow9 != 0), jnp.where((iSlow15 != 0), (self._fConst187 * ((fTemp12 * (fTemp22 - fVec12_temp)) / fTemp13)), (self._fConst186 * ((fTemp8 * (fTemp19 - fVec9_temp)) / fTemp13))), jnp.where((iSlow10 != 0), (self._fConst185 * ((fTemp5 * (fTemp17 - fVec7_temp)) / fTemp13)), fTemp16))) + jnp.where((iSlow9 != 0), jnp.where((iSlow15 != 0), (self._fConst187 * ((fTemp12 * (fTemp11 - fVec6_temp)) / fTemp0)), (self._fConst186 * ((fTemp8 * (fTemp7 - fVec3_temp)) / fTemp0))), jnp.where((iSlow10 != 0), (self._fConst185 * ((fTemp5 * (fTemp4 - fVec1_temp)) / fTemp0)), fTemp3)))))))) 
		fTemp35 = jnp.where((iSlow5 != 0), jnp.float32(0.0), fTemp34) 
		fTemp36 = jnp.abs(fTemp35) 
		state["fRec63"] = ((fRec63_temp * fSlow23) + (jnp.abs((jnp.float32(2.0) * fTemp36)) * fSlow24)) 
		fRec62 = state["fRec63"] 
		iTemp37 = (fRec62 > fSlow19).astype(jnp.int32) 
		state["iVec19"] = iTemp37 
		state["iRec64"] = jnp.maximum((iSlow25 * (iTemp37 < iVec19_temp).astype(jnp.int32)), (iRec64_temp + jnp.int32(-1))) 
		fTemp38 = jnp.abs(jnp.maximum((iTemp37), ((state["iRec64"] > jnp.int32(0)).astype(jnp.int32)))) 
		fTemp39 = jnp.where(((fRec60_temp > fTemp38).astype(jnp.int32) != 0), fSlow27, fSlow26) 
		state["fRec61"] = ((fRec61_temp * fTemp39) + (fTemp38 * (jnp.float32(1.0) - fTemp39))) 
		state["fRec60"] = state["fRec61"] 
		fHbargraph0 = (jnp.float32(2e+01) * jnp.log10(state["fRec60"]))
		self.sow("intermediates", "fHbargraph0", fHbargraph0) 
		fTemp40 = fTemp35 
		state["fRec59"] = ((fRec59_temp * fSlow23) + (fSlow24 * jnp.abs((fTemp36 + jnp.abs(fTemp40))))) 
		fRec58 = state["fRec59"] 
		iTemp41 = (fRec58 > fSlow19).astype(jnp.int32) 
		state["iVec20"] = iTemp41 
		state["iRec65"] = jnp.maximum((iSlow25 * (iTemp41 < iVec20_temp).astype(jnp.int32)), (iRec65_temp + jnp.int32(-1))) 
		fTemp42 = jnp.abs(jnp.maximum((iTemp41), ((state["iRec65"] > jnp.int32(0)).astype(jnp.int32)))) 
		fTemp43 = jnp.where(((fRec56_temp > fTemp42).astype(jnp.int32) != 0), fSlow27, fSlow26) 
		state["fRec57"] = ((fRec57_temp * fTemp43) + (fTemp42 * (jnp.float32(1.0) - fTemp43))) 
		state["fRec56"] = state["fRec57"] 
		fTemp44 = jnp.where((iSlow5 != 0), fTemp34, (state["fRec56"] * fTemp35)) 
		fTemp45 = jnp.where((iSlow0 != 0), jnp.float32(0.0), fTemp44) 
		fTemp46 = jnp.where((iSlow5 != 0), fTemp34, (state["fRec56"] * fTemp40)) 
		fTemp47 = jnp.where((iSlow0 != 0), jnp.float32(0.0), fTemp46) 
		fTemp48 = jnp.abs((jnp.abs(fTemp47) + jnp.abs(fTemp45))) 
		fTemp49 = jnp.where(((fRec47_temp > fTemp48).astype(jnp.int32) != 0), fSlow30, fSlow28) 
		state["fRec48"] = ((fRec48_temp * fTemp49) + (fTemp48 * (jnp.float32(1.0) - fTemp49))) 
		state["fRec47"] = state["fRec48"] 
		state["fRec46"] = ((fRec46_temp * fSlow2) + (fSlow31 * (jnp.maximum(((jnp.float32(2e+01) * jnp.log10(state["fRec47"])) - fSlow4), jnp.float32(0.0)) * fSlow3))) 
		fTemp50 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * state["fRec46"])) 
		fTemp51 = (fTemp45 * fTemp50) 
		fTemp52 = (fTemp47 * fTemp50) 
		fTemp53 = jnp.abs((jnp.abs(fTemp52) + jnp.abs(fTemp51))) 
		fTemp54 = jnp.where(((fRec44_temp > fTemp53).astype(jnp.int32) != 0), fSlow30, fSlow28) 
		state["fRec45"] = ((fRec45_temp * fTemp54) + (fTemp53 * (jnp.float32(1.0) - fTemp54))) 
		state["fRec44"] = state["fRec45"] 
		state["fRec43"] = ((fSlow2 * fRec43_temp) + (fSlow31 * (jnp.maximum(((jnp.float32(2e+01) * jnp.log10(state["fRec44"])) - fSlow4), jnp.float32(0.0)) * fSlow3))) 
		fHbargraph1 = (jnp.float32(2e+01) * jnp.log10(jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * state["fRec43"]))))
		self.sow("intermediates", "fHbargraph1", fHbargraph1) 
		fTemp55 = (jnp.where((iSlow0 != 0), fTemp46, (fSlow32 * fTemp52)) + jnp.where((iSlow0 != 0), fTemp44, (fSlow32 * fTemp51))) 
		state["fRec42"] = state["fRec42"].at[0].set((fTemp55 - (self._fConst182 * ((self._fConst181 * state["fRec42"][2]) + (self._fConst180 * state["fRec42"][1]))))) 
		state["fRec41"] = state["fRec41"].at[0].set(((self._fConst182 * (((self._fConst189 * state["fRec42"][0]) + (self._fConst190 * state["fRec42"][1])) + (self._fConst189 * state["fRec42"][2]))) - (self._fConst179 * ((self._fConst178 * state["fRec41"][2]) + (self._fConst177 * state["fRec41"][1]))))) 
		state["fRec40"] = state["fRec40"].at[0].set(((self._fConst179 * (((self._fConst191 * state["fRec41"][0]) + (self._fConst192 * state["fRec41"][1])) + (self._fConst191 * state["fRec41"][2]))) - (self._fConst176 * ((self._fConst175 * state["fRec40"][2]) + (self._fConst173 * state["fRec40"][1]))))) 
		fTemp56 = (self._fConst176 * (((self._fConst193 * state["fRec40"][0]) + (self._fConst194 * state["fRec40"][1])) + (self._fConst193 * state["fRec40"][2]))) 
		state["fRec39"] = state["fRec39"].at[0].set((fTemp56 - (self._fConst169 * ((self._fConst168 * state["fRec39"][2]) + (self._fConst167 * state["fRec39"][1]))))) 
		state["fRec38"] = state["fRec38"].at[0].set(((self._fConst169 * (((self._fConst196 * state["fRec39"][0]) + (self._fConst197 * state["fRec39"][1])) + (self._fConst196 * state["fRec39"][2]))) - (self._fConst166 * ((self._fConst165 * state["fRec38"][2]) + (self._fConst164 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst166 * (((self._fConst198 * state["fRec38"][0]) + (self._fConst199 * state["fRec38"][1])) + (self._fConst198 * state["fRec38"][2]))) - (self._fConst163 * ((self._fConst162 * state["fRec37"][2]) + (self._fConst160 * state["fRec37"][1]))))) 
		fTemp57 = (self._fConst163 * (((self._fConst200 * state["fRec37"][0]) + (self._fConst201 * state["fRec37"][1])) + (self._fConst200 * state["fRec37"][2]))) 
		state["fRec36"] = state["fRec36"].at[0].set((fTemp57 - (self._fConst156 * ((self._fConst155 * state["fRec36"][2]) + (self._fConst154 * state["fRec36"][1]))))) 
		state["fRec35"] = state["fRec35"].at[0].set(((self._fConst156 * (((self._fConst203 * state["fRec36"][0]) + (self._fConst204 * state["fRec36"][1])) + (self._fConst203 * state["fRec36"][2]))) - (self._fConst153 * ((self._fConst152 * state["fRec35"][2]) + (self._fConst151 * state["fRec35"][1]))))) 
		state["fRec34"] = state["fRec34"].at[0].set(((self._fConst153 * (((self._fConst205 * state["fRec35"][0]) + (self._fConst206 * state["fRec35"][1])) + (self._fConst205 * state["fRec35"][2]))) - (self._fConst150 * ((self._fConst149 * state["fRec34"][2]) + (self._fConst147 * state["fRec34"][1]))))) 
		fTemp58 = (self._fConst150 * (((self._fConst207 * state["fRec34"][0]) + (self._fConst208 * state["fRec34"][1])) + (self._fConst207 * state["fRec34"][2]))) 
		state["fRec33"] = state["fRec33"].at[0].set((fTemp58 - (self._fConst143 * ((self._fConst142 * state["fRec33"][2]) + (self._fConst141 * state["fRec33"][1]))))) 
		state["fRec32"] = state["fRec32"].at[0].set(((self._fConst143 * (((self._fConst210 * state["fRec33"][0]) + (self._fConst211 * state["fRec33"][1])) + (self._fConst210 * state["fRec33"][2]))) - (self._fConst140 * ((self._fConst139 * state["fRec32"][2]) + (self._fConst138 * state["fRec32"][1]))))) 
		state["fRec31"] = state["fRec31"].at[0].set(((self._fConst140 * (((self._fConst212 * state["fRec32"][0]) + (self._fConst213 * state["fRec32"][1])) + (self._fConst212 * state["fRec32"][2]))) - (self._fConst137 * ((self._fConst136 * state["fRec31"][2]) + (self._fConst134 * state["fRec31"][1]))))) 
		fTemp59 = (self._fConst137 * (((self._fConst214 * state["fRec31"][0]) + (self._fConst215 * state["fRec31"][1])) + (self._fConst214 * state["fRec31"][2]))) 
		state["fRec30"] = state["fRec30"].at[0].set((fTemp59 - (self._fConst130 * ((self._fConst129 * state["fRec30"][2]) + (self._fConst128 * state["fRec30"][1]))))) 
		state["fRec29"] = state["fRec29"].at[0].set(((self._fConst130 * (((self._fConst217 * state["fRec30"][0]) + (self._fConst218 * state["fRec30"][1])) + (self._fConst217 * state["fRec30"][2]))) - (self._fConst127 * ((self._fConst126 * state["fRec29"][2]) + (self._fConst125 * state["fRec29"][1]))))) 
		state["fRec28"] = state["fRec28"].at[0].set(((self._fConst127 * (((self._fConst219 * state["fRec29"][0]) + (self._fConst220 * state["fRec29"][1])) + (self._fConst219 * state["fRec29"][2]))) - (self._fConst124 * ((self._fConst123 * state["fRec28"][2]) + (self._fConst121 * state["fRec28"][1]))))) 
		fTemp60 = (self._fConst124 * (((self._fConst221 * state["fRec28"][0]) + (self._fConst222 * state["fRec28"][1])) + (self._fConst221 * state["fRec28"][2]))) 
		state["fRec27"] = state["fRec27"].at[0].set((fTemp60 - (self._fConst117 * ((self._fConst116 * state["fRec27"][2]) + (self._fConst115 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((self._fConst117 * (((self._fConst224 * state["fRec27"][0]) + (self._fConst225 * state["fRec27"][1])) + (self._fConst224 * state["fRec27"][2]))) - (self._fConst114 * ((self._fConst113 * state["fRec26"][2]) + (self._fConst112 * state["fRec26"][1]))))) 
		state["fRec25"] = state["fRec25"].at[0].set(((self._fConst114 * (((self._fConst226 * state["fRec26"][0]) + (self._fConst227 * state["fRec26"][1])) + (self._fConst226 * state["fRec26"][2]))) - (self._fConst111 * ((self._fConst110 * state["fRec25"][2]) + (self._fConst108 * state["fRec25"][1]))))) 
		fTemp61 = (self._fConst111 * (((self._fConst228 * state["fRec25"][0]) + (self._fConst229 * state["fRec25"][1])) + (self._fConst228 * state["fRec25"][2]))) 
		state["fRec24"] = state["fRec24"].at[0].set((fTemp61 - (self._fConst104 * ((self._fConst103 * state["fRec24"][2]) + (self._fConst102 * state["fRec24"][1]))))) 
		state["fRec23"] = state["fRec23"].at[0].set(((self._fConst104 * (((self._fConst231 * state["fRec24"][0]) + (self._fConst232 * state["fRec24"][1])) + (self._fConst231 * state["fRec24"][2]))) - (self._fConst101 * ((self._fConst100 * state["fRec23"][2]) + (self._fConst99 * state["fRec23"][1]))))) 
		state["fRec22"] = state["fRec22"].at[0].set(((self._fConst101 * (((self._fConst233 * state["fRec23"][0]) + (self._fConst234 * state["fRec23"][1])) + (self._fConst233 * state["fRec23"][2]))) - (self._fConst98 * ((self._fConst97 * state["fRec22"][2]) + (self._fConst95 * state["fRec22"][1]))))) 
		fTemp62 = (self._fConst98 * (((self._fConst235 * state["fRec22"][0]) + (self._fConst236 * state["fRec22"][1])) + (self._fConst235 * state["fRec22"][2]))) 
		state["fRec21"] = state["fRec21"].at[0].set((fTemp62 - (self._fConst91 * ((self._fConst90 * state["fRec21"][2]) + (self._fConst89 * state["fRec21"][1]))))) 
		state["fRec20"] = state["fRec20"].at[0].set(((self._fConst91 * (((self._fConst238 * state["fRec21"][0]) + (self._fConst239 * state["fRec21"][1])) + (self._fConst238 * state["fRec21"][2]))) - (self._fConst88 * ((self._fConst87 * state["fRec20"][2]) + (self._fConst86 * state["fRec20"][1]))))) 
		state["fRec19"] = state["fRec19"].at[0].set(((self._fConst88 * (((self._fConst240 * state["fRec20"][0]) + (self._fConst241 * state["fRec20"][1])) + (self._fConst240 * state["fRec20"][2]))) - (self._fConst85 * ((self._fConst84 * state["fRec19"][2]) + (self._fConst82 * state["fRec19"][1]))))) 
		fTemp63 = (self._fConst85 * (((self._fConst242 * state["fRec19"][0]) + (self._fConst243 * state["fRec19"][1])) + (self._fConst242 * state["fRec19"][2]))) 
		state["fRec18"] = state["fRec18"].at[0].set((fTemp63 - (self._fConst78 * ((self._fConst77 * state["fRec18"][2]) + (self._fConst76 * state["fRec18"][1]))))) 
		state["fRec17"] = state["fRec17"].at[0].set(((self._fConst78 * (((self._fConst245 * state["fRec18"][0]) + (self._fConst246 * state["fRec18"][1])) + (self._fConst245 * state["fRec18"][2]))) - (self._fConst75 * ((self._fConst74 * state["fRec17"][2]) + (self._fConst73 * state["fRec17"][1]))))) 
		state["fRec16"] = state["fRec16"].at[0].set(((self._fConst75 * (((self._fConst247 * state["fRec17"][0]) + (self._fConst248 * state["fRec17"][1])) + (self._fConst247 * state["fRec17"][2]))) - (self._fConst72 * ((self._fConst71 * state["fRec16"][2]) + (self._fConst69 * state["fRec16"][1]))))) 
		fTemp64 = (self._fConst72 * (((self._fConst249 * state["fRec16"][0]) + (self._fConst250 * state["fRec16"][1])) + (self._fConst249 * state["fRec16"][2]))) 
		state["fRec15"] = state["fRec15"].at[0].set((fTemp64 - (self._fConst65 * ((self._fConst64 * state["fRec15"][2]) + (self._fConst63 * state["fRec15"][1]))))) 
		state["fRec14"] = state["fRec14"].at[0].set(((self._fConst65 * (((self._fConst252 * state["fRec15"][0]) + (self._fConst253 * state["fRec15"][1])) + (self._fConst252 * state["fRec15"][2]))) - (self._fConst62 * ((self._fConst61 * state["fRec14"][2]) + (self._fConst60 * state["fRec14"][1]))))) 
		state["fRec13"] = state["fRec13"].at[0].set(((self._fConst62 * (((self._fConst254 * state["fRec14"][0]) + (self._fConst255 * state["fRec14"][1])) + (self._fConst254 * state["fRec14"][2]))) - (self._fConst59 * ((self._fConst58 * state["fRec13"][2]) + (self._fConst56 * state["fRec13"][1]))))) 
		fTemp65 = (self._fConst59 * (((self._fConst256 * state["fRec13"][0]) + (self._fConst257 * state["fRec13"][1])) + (self._fConst256 * state["fRec13"][2]))) 
		state["fRec12"] = state["fRec12"].at[0].set((fTemp65 - (self._fConst52 * ((self._fConst51 * state["fRec12"][2]) + (self._fConst50 * state["fRec12"][1]))))) 
		state["fRec11"] = state["fRec11"].at[0].set(((self._fConst52 * (((self._fConst259 * state["fRec12"][0]) + (self._fConst260 * state["fRec12"][1])) + (self._fConst259 * state["fRec12"][2]))) - (self._fConst49 * ((self._fConst48 * state["fRec11"][2]) + (self._fConst47 * state["fRec11"][1]))))) 
		state["fRec10"] = state["fRec10"].at[0].set(((self._fConst49 * (((self._fConst261 * state["fRec11"][0]) + (self._fConst262 * state["fRec11"][1])) + (self._fConst261 * state["fRec11"][2]))) - (self._fConst46 * ((self._fConst45 * state["fRec10"][2]) + (self._fConst43 * state["fRec10"][1]))))) 
		fTemp66 = (self._fConst46 * (((self._fConst263 * state["fRec10"][0]) + (self._fConst264 * state["fRec10"][1])) + (self._fConst263 * state["fRec10"][2]))) 
		state["fRec9"] = state["fRec9"].at[0].set((fTemp66 - (self._fConst39 * ((self._fConst38 * state["fRec9"][2]) + (self._fConst37 * state["fRec9"][1]))))) 
		state["fRec8"] = state["fRec8"].at[0].set(((self._fConst39 * (((self._fConst266 * state["fRec9"][0]) + (self._fConst267 * state["fRec9"][1])) + (self._fConst266 * state["fRec9"][2]))) - (self._fConst36 * ((self._fConst35 * state["fRec8"][2]) + (self._fConst34 * state["fRec8"][1]))))) 
		state["fRec7"] = state["fRec7"].at[0].set(((self._fConst36 * (((self._fConst268 * state["fRec8"][0]) + (self._fConst269 * state["fRec8"][1])) + (self._fConst268 * state["fRec8"][2]))) - (self._fConst33 * ((self._fConst32 * state["fRec7"][2]) + (self._fConst30 * state["fRec7"][1]))))) 
		fTemp67 = (self._fConst33 * (((self._fConst270 * state["fRec7"][0]) + (self._fConst271 * state["fRec7"][1])) + (self._fConst270 * state["fRec7"][2]))) 
		state["fRec6"] = state["fRec6"].at[0].set((fTemp67 - (self._fConst26 * ((self._fConst25 * state["fRec6"][2]) + (self._fConst24 * state["fRec6"][1]))))) 
		state["fRec5"] = state["fRec5"].at[0].set(((self._fConst26 * (((self._fConst273 * state["fRec6"][0]) + (self._fConst274 * state["fRec6"][1])) + (self._fConst273 * state["fRec6"][2]))) - (self._fConst23 * ((self._fConst22 * state["fRec5"][2]) + (self._fConst21 * state["fRec5"][1]))))) 
		state["fRec4"] = state["fRec4"].at[0].set(((self._fConst23 * (((self._fConst275 * state["fRec5"][0]) + (self._fConst276 * state["fRec5"][1])) + (self._fConst275 * state["fRec5"][2]))) - (self._fConst20 * ((self._fConst19 * state["fRec4"][2]) + (self._fConst17 * state["fRec4"][1]))))) 
		fTemp68 = (self._fConst20 * (((self._fConst277 * state["fRec4"][0]) + (self._fConst278 * state["fRec4"][1])) + (self._fConst277 * state["fRec4"][2]))) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp68 - (self._fConst13 * ((self._fConst12 * state["fRec3"][2]) + (self._fConst11 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst13 * (((self._fConst280 * state["fRec3"][0]) + (self._fConst281 * state["fRec3"][1])) + (self._fConst280 * state["fRec3"][2]))) - (self._fConst10 * ((self._fConst9 * state["fRec2"][2]) + (self._fConst8 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst10 * (((self._fConst282 * state["fRec2"][0]) + (self._fConst283 * state["fRec2"][1])) + (self._fConst282 * state["fRec2"][2]))) - (self._fConst7 * ((self._fConst6 * state["fRec1"][2]) + (self._fConst4 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fSlow34 * fRec0_temp) + (fSlow35 * jnp.abs((self._fConst7 * (((self._fConst284 * state["fRec1"][0]) + (self._fConst285 * state["fRec1"][1])) + (self._fConst284 * state["fRec1"][2])))))) 
		fVbargraph0 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec69"] = state["fRec69"].at[0].set((fTemp68 - (self._fConst295 * ((self._fConst294 * state["fRec69"][2]) + (self._fConst293 * state["fRec69"][1]))))) 
		state["fRec68"] = state["fRec68"].at[0].set(((self._fConst295 * (((self._fConst297 * state["fRec69"][0]) + (self._fConst298 * state["fRec69"][1])) + (self._fConst297 * state["fRec69"][2]))) - (self._fConst292 * ((self._fConst291 * state["fRec68"][2]) + (self._fConst290 * state["fRec68"][1]))))) 
		state["fRec67"] = state["fRec67"].at[0].set(((self._fConst292 * (((self._fConst300 * state["fRec68"][0]) + (self._fConst301 * state["fRec68"][1])) + (self._fConst300 * state["fRec68"][2]))) - (self._fConst289 * ((self._fConst288 * state["fRec67"][2]) + (self._fConst287 * state["fRec67"][1]))))) 
		state["fRec66"] = ((fSlow34 * fRec66_temp) + (fSlow35 * jnp.abs((self._fConst289 * (((self._fConst303 * state["fRec67"][0]) + (self._fConst304 * state["fRec67"][1])) + (self._fConst303 * state["fRec67"][2])))))) 
		fVbargraph1 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec66"])))
		self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec73"] = state["fRec73"].at[0].set((fTemp67 - (self._fConst313 * ((self._fConst312 * state["fRec73"][2]) + (self._fConst311 * state["fRec73"][1]))))) 
		state["fRec72"] = state["fRec72"].at[0].set(((self._fConst313 * (((self._fConst315 * state["fRec73"][0]) + (self._fConst316 * state["fRec73"][1])) + (self._fConst315 * state["fRec73"][2]))) - (self._fConst310 * ((self._fConst309 * state["fRec72"][2]) + (self._fConst308 * state["fRec72"][1]))))) 
		state["fRec71"] = state["fRec71"].at[0].set(((self._fConst310 * (((self._fConst318 * state["fRec72"][0]) + (self._fConst319 * state["fRec72"][1])) + (self._fConst318 * state["fRec72"][2]))) - (self._fConst307 * ((self._fConst306 * state["fRec71"][2]) + (self._fConst305 * state["fRec71"][1]))))) 
		state["fRec70"] = ((fSlow34 * fRec70_temp) + (fSlow35 * jnp.abs((self._fConst307 * (((self._fConst321 * state["fRec71"][0]) + (self._fConst322 * state["fRec71"][1])) + (self._fConst321 * state["fRec71"][2])))))) 
		fVbargraph2 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec70"])))
		self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec77"] = state["fRec77"].at[0].set((fTemp66 - (self._fConst331 * ((self._fConst330 * state["fRec77"][2]) + (self._fConst329 * state["fRec77"][1]))))) 
		state["fRec76"] = state["fRec76"].at[0].set(((self._fConst331 * (((self._fConst333 * state["fRec77"][0]) + (self._fConst334 * state["fRec77"][1])) + (self._fConst333 * state["fRec77"][2]))) - (self._fConst328 * ((self._fConst327 * state["fRec76"][2]) + (self._fConst326 * state["fRec76"][1]))))) 
		state["fRec75"] = state["fRec75"].at[0].set(((self._fConst328 * (((self._fConst336 * state["fRec76"][0]) + (self._fConst337 * state["fRec76"][1])) + (self._fConst336 * state["fRec76"][2]))) - (self._fConst325 * ((self._fConst324 * state["fRec75"][2]) + (self._fConst323 * state["fRec75"][1]))))) 
		state["fRec74"] = ((fSlow34 * fRec74_temp) + (fSlow35 * jnp.abs((self._fConst325 * (((self._fConst339 * state["fRec75"][0]) + (self._fConst340 * state["fRec75"][1])) + (self._fConst339 * state["fRec75"][2])))))) 
		fVbargraph3 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec74"])))
		self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec81"] = state["fRec81"].at[0].set((fTemp65 - (self._fConst349 * ((self._fConst348 * state["fRec81"][2]) + (self._fConst347 * state["fRec81"][1]))))) 
		state["fRec80"] = state["fRec80"].at[0].set(((self._fConst349 * (((self._fConst351 * state["fRec81"][0]) + (self._fConst352 * state["fRec81"][1])) + (self._fConst351 * state["fRec81"][2]))) - (self._fConst346 * ((self._fConst345 * state["fRec80"][2]) + (self._fConst344 * state["fRec80"][1]))))) 
		state["fRec79"] = state["fRec79"].at[0].set(((self._fConst346 * (((self._fConst354 * state["fRec80"][0]) + (self._fConst355 * state["fRec80"][1])) + (self._fConst354 * state["fRec80"][2]))) - (self._fConst343 * ((self._fConst342 * state["fRec79"][2]) + (self._fConst341 * state["fRec79"][1]))))) 
		state["fRec78"] = ((fSlow34 * fRec78_temp) + (fSlow35 * jnp.abs((self._fConst343 * (((self._fConst357 * state["fRec79"][0]) + (self._fConst358 * state["fRec79"][1])) + (self._fConst357 * state["fRec79"][2])))))) 
		fVbargraph4 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec78"])))
		self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec85"] = state["fRec85"].at[0].set((fTemp64 - (self._fConst367 * ((self._fConst366 * state["fRec85"][2]) + (self._fConst365 * state["fRec85"][1]))))) 
		state["fRec84"] = state["fRec84"].at[0].set(((self._fConst367 * (((self._fConst369 * state["fRec85"][0]) + (self._fConst370 * state["fRec85"][1])) + (self._fConst369 * state["fRec85"][2]))) - (self._fConst364 * ((self._fConst363 * state["fRec84"][2]) + (self._fConst362 * state["fRec84"][1]))))) 
		state["fRec83"] = state["fRec83"].at[0].set(((self._fConst364 * (((self._fConst372 * state["fRec84"][0]) + (self._fConst373 * state["fRec84"][1])) + (self._fConst372 * state["fRec84"][2]))) - (self._fConst361 * ((self._fConst360 * state["fRec83"][2]) + (self._fConst359 * state["fRec83"][1]))))) 
		state["fRec82"] = ((fSlow34 * fRec82_temp) + (fSlow35 * jnp.abs((self._fConst361 * (((self._fConst375 * state["fRec83"][0]) + (self._fConst376 * state["fRec83"][1])) + (self._fConst375 * state["fRec83"][2])))))) 
		fVbargraph5 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec82"])))
		self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec89"] = state["fRec89"].at[0].set((fTemp63 - (self._fConst385 * ((self._fConst384 * state["fRec89"][2]) + (self._fConst383 * state["fRec89"][1]))))) 
		state["fRec88"] = state["fRec88"].at[0].set(((self._fConst385 * (((self._fConst387 * state["fRec89"][0]) + (self._fConst388 * state["fRec89"][1])) + (self._fConst387 * state["fRec89"][2]))) - (self._fConst382 * ((self._fConst381 * state["fRec88"][2]) + (self._fConst380 * state["fRec88"][1]))))) 
		state["fRec87"] = state["fRec87"].at[0].set(((self._fConst382 * (((self._fConst390 * state["fRec88"][0]) + (self._fConst391 * state["fRec88"][1])) + (self._fConst390 * state["fRec88"][2]))) - (self._fConst379 * ((self._fConst378 * state["fRec87"][2]) + (self._fConst377 * state["fRec87"][1]))))) 
		state["fRec86"] = ((fSlow34 * fRec86_temp) + (fSlow35 * jnp.abs((self._fConst379 * (((self._fConst393 * state["fRec87"][0]) + (self._fConst394 * state["fRec87"][1])) + (self._fConst393 * state["fRec87"][2])))))) 
		fVbargraph6 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec86"])))
		self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec93"] = state["fRec93"].at[0].set((fTemp62 - (self._fConst403 * ((self._fConst402 * state["fRec93"][2]) + (self._fConst401 * state["fRec93"][1]))))) 
		state["fRec92"] = state["fRec92"].at[0].set(((self._fConst403 * (((self._fConst405 * state["fRec93"][0]) + (self._fConst406 * state["fRec93"][1])) + (self._fConst405 * state["fRec93"][2]))) - (self._fConst400 * ((self._fConst399 * state["fRec92"][2]) + (self._fConst398 * state["fRec92"][1]))))) 
		state["fRec91"] = state["fRec91"].at[0].set(((self._fConst400 * (((self._fConst408 * state["fRec92"][0]) + (self._fConst409 * state["fRec92"][1])) + (self._fConst408 * state["fRec92"][2]))) - (self._fConst397 * ((self._fConst396 * state["fRec91"][2]) + (self._fConst395 * state["fRec91"][1]))))) 
		state["fRec90"] = ((fSlow34 * fRec90_temp) + (fSlow35 * jnp.abs((self._fConst397 * (((self._fConst411 * state["fRec91"][0]) + (self._fConst412 * state["fRec91"][1])) + (self._fConst411 * state["fRec91"][2])))))) 
		fVbargraph7 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec90"])))
		self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec97"] = state["fRec97"].at[0].set((fTemp61 - (self._fConst421 * ((self._fConst420 * state["fRec97"][2]) + (self._fConst419 * state["fRec97"][1]))))) 
		state["fRec96"] = state["fRec96"].at[0].set(((self._fConst421 * (((self._fConst423 * state["fRec97"][0]) + (self._fConst424 * state["fRec97"][1])) + (self._fConst423 * state["fRec97"][2]))) - (self._fConst418 * ((self._fConst417 * state["fRec96"][2]) + (self._fConst416 * state["fRec96"][1]))))) 
		state["fRec95"] = state["fRec95"].at[0].set(((self._fConst418 * (((self._fConst426 * state["fRec96"][0]) + (self._fConst427 * state["fRec96"][1])) + (self._fConst426 * state["fRec96"][2]))) - (self._fConst415 * ((self._fConst414 * state["fRec95"][2]) + (self._fConst413 * state["fRec95"][1]))))) 
		state["fRec94"] = ((fSlow34 * fRec94_temp) + (fSlow35 * jnp.abs((self._fConst415 * (((self._fConst429 * state["fRec95"][0]) + (self._fConst430 * state["fRec95"][1])) + (self._fConst429 * state["fRec95"][2])))))) 
		fVbargraph8 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec94"])))
		self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec101"] = state["fRec101"].at[0].set((fTemp60 - (self._fConst439 * ((self._fConst438 * state["fRec101"][2]) + (self._fConst437 * state["fRec101"][1]))))) 
		state["fRec100"] = state["fRec100"].at[0].set(((self._fConst439 * (((self._fConst441 * state["fRec101"][0]) + (self._fConst442 * state["fRec101"][1])) + (self._fConst441 * state["fRec101"][2]))) - (self._fConst436 * ((self._fConst435 * state["fRec100"][2]) + (self._fConst434 * state["fRec100"][1]))))) 
		state["fRec99"] = state["fRec99"].at[0].set(((self._fConst436 * (((self._fConst444 * state["fRec100"][0]) + (self._fConst445 * state["fRec100"][1])) + (self._fConst444 * state["fRec100"][2]))) - (self._fConst433 * ((self._fConst432 * state["fRec99"][2]) + (self._fConst431 * state["fRec99"][1]))))) 
		state["fRec98"] = ((fSlow34 * fRec98_temp) + (fSlow35 * jnp.abs((self._fConst433 * (((self._fConst447 * state["fRec99"][0]) + (self._fConst448 * state["fRec99"][1])) + (self._fConst447 * state["fRec99"][2])))))) 
		fVbargraph9 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec98"])))
		self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec105"] = state["fRec105"].at[0].set((fTemp59 - (self._fConst457 * ((self._fConst456 * state["fRec105"][2]) + (self._fConst455 * state["fRec105"][1]))))) 
		state["fRec104"] = state["fRec104"].at[0].set(((self._fConst457 * (((self._fConst459 * state["fRec105"][0]) + (self._fConst460 * state["fRec105"][1])) + (self._fConst459 * state["fRec105"][2]))) - (self._fConst454 * ((self._fConst453 * state["fRec104"][2]) + (self._fConst452 * state["fRec104"][1]))))) 
		state["fRec103"] = state["fRec103"].at[0].set(((self._fConst454 * (((self._fConst462 * state["fRec104"][0]) + (self._fConst463 * state["fRec104"][1])) + (self._fConst462 * state["fRec104"][2]))) - (self._fConst451 * ((self._fConst450 * state["fRec103"][2]) + (self._fConst449 * state["fRec103"][1]))))) 
		state["fRec102"] = ((fSlow34 * fRec102_temp) + (fSlow35 * jnp.abs((self._fConst451 * (((self._fConst465 * state["fRec103"][0]) + (self._fConst466 * state["fRec103"][1])) + (self._fConst465 * state["fRec103"][2])))))) 
		fVbargraph10 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec102"])))
		self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec109"] = state["fRec109"].at[0].set((fTemp58 - (self._fConst475 * ((self._fConst474 * state["fRec109"][2]) + (self._fConst473 * state["fRec109"][1]))))) 
		state["fRec108"] = state["fRec108"].at[0].set(((self._fConst475 * (((self._fConst477 * state["fRec109"][0]) + (self._fConst478 * state["fRec109"][1])) + (self._fConst477 * state["fRec109"][2]))) - (self._fConst472 * ((self._fConst471 * state["fRec108"][2]) + (self._fConst470 * state["fRec108"][1]))))) 
		state["fRec107"] = state["fRec107"].at[0].set(((self._fConst472 * (((self._fConst480 * state["fRec108"][0]) + (self._fConst481 * state["fRec108"][1])) + (self._fConst480 * state["fRec108"][2]))) - (self._fConst469 * ((self._fConst468 * state["fRec107"][2]) + (self._fConst467 * state["fRec107"][1]))))) 
		state["fRec106"] = ((fSlow34 * fRec106_temp) + (fSlow35 * jnp.abs((self._fConst469 * (((self._fConst483 * state["fRec107"][0]) + (self._fConst484 * state["fRec107"][1])) + (self._fConst483 * state["fRec107"][2])))))) 
		fVbargraph11 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec106"])))
		self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec113"] = state["fRec113"].at[0].set((fTemp57 - (self._fConst493 * ((self._fConst492 * state["fRec113"][2]) + (self._fConst491 * state["fRec113"][1]))))) 
		state["fRec112"] = state["fRec112"].at[0].set(((self._fConst493 * (((self._fConst495 * state["fRec113"][0]) + (self._fConst496 * state["fRec113"][1])) + (self._fConst495 * state["fRec113"][2]))) - (self._fConst490 * ((self._fConst489 * state["fRec112"][2]) + (self._fConst488 * state["fRec112"][1]))))) 
		state["fRec111"] = state["fRec111"].at[0].set(((self._fConst490 * (((self._fConst498 * state["fRec112"][0]) + (self._fConst499 * state["fRec112"][1])) + (self._fConst498 * state["fRec112"][2]))) - (self._fConst487 * ((self._fConst486 * state["fRec111"][2]) + (self._fConst485 * state["fRec111"][1]))))) 
		state["fRec110"] = ((fSlow34 * fRec110_temp) + (fSlow35 * jnp.abs((self._fConst487 * (((self._fConst501 * state["fRec111"][0]) + (self._fConst502 * state["fRec111"][1])) + (self._fConst501 * state["fRec111"][2])))))) 
		fVbargraph12 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec110"])))
		self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec117"] = state["fRec117"].at[0].set((fTemp56 - (self._fConst511 * ((self._fConst510 * state["fRec117"][2]) + (self._fConst509 * state["fRec117"][1]))))) 
		state["fRec116"] = state["fRec116"].at[0].set(((self._fConst511 * (((self._fConst513 * state["fRec117"][0]) + (self._fConst514 * state["fRec117"][1])) + (self._fConst513 * state["fRec117"][2]))) - (self._fConst508 * ((self._fConst507 * state["fRec116"][2]) + (self._fConst506 * state["fRec116"][1]))))) 
		state["fRec115"] = state["fRec115"].at[0].set(((self._fConst508 * (((self._fConst516 * state["fRec116"][0]) + (self._fConst517 * state["fRec116"][1])) + (self._fConst516 * state["fRec116"][2]))) - (self._fConst505 * ((self._fConst504 * state["fRec115"][2]) + (self._fConst503 * state["fRec115"][1]))))) 
		state["fRec114"] = ((fSlow34 * fRec114_temp) + (fSlow35 * jnp.abs((self._fConst505 * (((self._fConst519 * state["fRec115"][0]) + (self._fConst520 * state["fRec115"][1])) + (self._fConst519 * state["fRec115"][2])))))) 
		fVbargraph13 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec114"])))
		self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec121"] = state["fRec121"].at[0].set((fTemp55 - (self._fConst529 * ((self._fConst528 * state["fRec121"][2]) + (self._fConst527 * state["fRec121"][1]))))) 
		state["fRec120"] = state["fRec120"].at[0].set(((self._fConst529 * (((self._fConst531 * state["fRec121"][0]) + (self._fConst532 * state["fRec121"][1])) + (self._fConst531 * state["fRec121"][2]))) - (self._fConst526 * ((self._fConst525 * state["fRec120"][2]) + (self._fConst524 * state["fRec120"][1]))))) 
		state["fRec119"] = state["fRec119"].at[0].set(((self._fConst526 * (((self._fConst534 * state["fRec120"][0]) + (self._fConst535 * state["fRec120"][1])) + (self._fConst534 * state["fRec120"][2]))) - (self._fConst523 * ((self._fConst522 * state["fRec119"][2]) + (self._fConst521 * state["fRec119"][1]))))) 
		state["fRec118"] = ((fRec118_temp * fSlow34) + (jnp.abs((self._fConst523 * (((self._fConst537 * state["fRec119"][0]) + (self._fConst538 * state["fRec119"][1])) + (self._fConst537 * state["fRec119"][2])))) * fSlow35)) 
		fVbargraph14 = (fSlow36 + (jnp.float32(2e+01) * jnp.log10(state["fRec118"])))
		self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		fTemp69 = fTemp55 
		_result0 = fTemp69 
		_result1 = fTemp69 
		state["iVec0"] = jnp.roll(state["iVec0"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
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
		state["fRec69"] = jnp.roll(state["fRec69"], 1) 
		state["fRec68"] = jnp.roll(state["fRec68"], 1) 
		state["fRec67"] = jnp.roll(state["fRec67"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec72"] = jnp.roll(state["fRec72"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec76"] = jnp.roll(state["fRec76"], 1) 
		state["fRec75"] = jnp.roll(state["fRec75"], 1) 
		state["fRec81"] = jnp.roll(state["fRec81"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec84"] = jnp.roll(state["fRec84"], 1) 
		state["fRec83"] = jnp.roll(state["fRec83"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec88"] = jnp.roll(state["fRec88"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec92"] = jnp.roll(state["fRec92"], 1) 
		state["fRec91"] = jnp.roll(state["fRec91"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec96"] = jnp.roll(state["fRec96"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec103"] = jnp.roll(state["fRec103"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec112"] = jnp.roll(state["fRec112"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec116"] = jnp.roll(state["fRec116"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec120"] = jnp.roll(state["fRec120"], 1) 
		state["fRec119"] = jnp.roll(state["fRec119"], 1) 
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
