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
		ui_path.append("parametric_eq") 
		ui_path.append("0x00") 
		ui_path.append("SAWTOOTH OSCILLATOR") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider4", ui_path, "Amplitude", -2e+01, -1.2e+02, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider1", ui_path, "Frequency", 49.0, 1.0, 88.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider3", ui_path, "Detuning 1", -0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider2", ui_path, "Detuning 2", 0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider0", ui_path, "Portamento", 0.1, 0.001, 1e+01, unnorm_funcs, "log") 
		self.add_nentry("fEntry0", ui_path, "Saw Order", 2.0, 1.0, 4.0, 1.0, unnorm_funcs, "linear") 
		ui_path.append("Alternate Signals") 
		self.add_button("fCheckbox1", ui_path, "Noise (White or Pink - uses only Amplitude control on the left)", unnorm_funcs) 
		self.add_button("fCheckbox2", ui_path, "Pink instead of White Noise (also called 1/f Noise)", unnorm_funcs) 
		self.add_button("fCheckbox0", ui_path, "External Signal Input (overrides Sawtooth/Noise selection above)", unnorm_funcs) 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("PARAMETRIC EQ SECTIONS") 
		ui_path.append("Low Shelf") 
		self.add_hslider("fHslider5", ui_path, "Low Boost|Cut", 0.0, -4e+01, 4e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider4", ui_path, "Transition Frequency", 2e+02, 1.0, 5e+03, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Peaking Equalizer") 
		self.add_hslider("fHslider2", ui_path, "Peak Boost|Cut", 0.0, -4e+01, 4e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider1", ui_path, "Peak Frequency", 49.0, 1.0, 1e+02, unnorm_funcs, "linear") 
		self.add_hslider("fHslider3", ui_path, "Peak Q", 4e+01, 1.0, 1e+03, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("High Shelf") 
		self.add_hslider("fHslider6", ui_path, "High Boost|Cut", 0.0, -4e+01, 4e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider0", ui_path, "Transition Frequency", 8e+03, 2e+01, 1e+04, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("CONSTANT-Q SPECTRUM ANALYZER (6E), 20 bands spanning LP, 9 octaves below 16000 Hz, HP") 
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
		self.add_vbargraph("fVbargraph15", ui_path, "vbargraph15", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph16", ui_path, "vbargraph16", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph17", ui_path, "vbargraph17", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph18", ui_path, "vbargraph18", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph19", ui_path, "vbargraph19", -5e+01, 1e+01, unnorm_funcs) 
		ui_path.pop()
		ui_path.append("SPECTRUM ANALYZER CONTROLS") 
		self.add_hslider("fHslider7", ui_path, "Level Averaging Time", 1e+02, 1.0, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider8", ui_path, "Level dB Offset", 5e+01, 0.0, 1e+02, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate))) 
		self._fConst1 = np.tan((np.float32(98.174774) / self._fConst0)) 
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
		self._fConst14 = np.tan((np.float32(138.84009) / self._fConst0)) 
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
		self._fConst27 = np.tan((np.float32(196.34955) / self._fConst0)) 
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
		self._fConst40 = np.tan((np.float32(277.68018) / self._fConst0)) 
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
		self._fConst53 = np.tan((np.float32(392.6991) / self._fConst0)) 
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
		self._fConst66 = np.tan((np.float32(555.36035) / self._fConst0)) 
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
		self._fConst79 = np.tan((np.float32(785.3982) / self._fConst0)) 
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
		self._fConst92 = np.tan((np.float32(1110.7207) / self._fConst0)) 
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
		self._fConst105 = np.tan((np.float32(1570.7964) / self._fConst0)) 
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
		self._fConst118 = np.tan((np.float32(2221.4414) / self._fConst0)) 
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
		self._fConst131 = np.tan((np.float32(3141.5928) / self._fConst0)) 
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
		self._fConst144 = np.tan((np.float32(4442.883) / self._fConst0)) 
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
		self._fConst157 = np.tan((np.float32(6283.1855) / self._fConst0)) 
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
		self._fConst170 = np.tan((np.float32(8885.766) / self._fConst0)) 
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
		self._fConst183 = np.tan((np.float32(12566.371) / self._fConst0)) 
		self._fConst184 = np.power(self._fConst183, np.float32(2.0)) 
		self._fConst185 = (np.float32(1.0) / self._fConst184) 
		self._fConst186 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst185)) 
		self._fConst187 = (np.float32(1.0) / self._fConst183) 
		self._fConst188 = (((self._fConst187 + np.float32(-0.16840488)) / self._fConst183) + np.float32(1.0693583)) 
		self._fConst189 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.16840488)) / self._fConst183) + np.float32(1.0693583))) 
		self._fConst190 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst185)) 
		self._fConst191 = (((self._fConst187 + np.float32(-0.51247865)) / self._fConst183) + np.float32(0.6896214)) 
		self._fConst192 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.51247865)) / self._fConst183) + np.float32(0.6896214))) 
		self._fConst193 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst185)) 
		self._fConst194 = (((self._fConst187 + np.float32(-0.78241307)) / self._fConst183) + np.float32(0.2452915)) 
		self._fConst195 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.78241307)) / self._fConst183) + np.float32(0.2452915))) 
		self._fConst196 = np.tan((np.float32(17771.531) / self._fConst0)) 
		self._fConst197 = np.power(self._fConst196, np.float32(2.0)) 
		self._fConst198 = (np.float32(1.0) / self._fConst197) 
		self._fConst199 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst198)) 
		self._fConst200 = (np.float32(1.0) / self._fConst196) 
		self._fConst201 = (((self._fConst200 + np.float32(-0.16840488)) / self._fConst196) + np.float32(1.0693583)) 
		self._fConst202 = (np.float32(1.0) / (((self._fConst200 + np.float32(0.16840488)) / self._fConst196) + np.float32(1.0693583))) 
		self._fConst203 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst198)) 
		self._fConst204 = (((self._fConst200 + np.float32(-0.51247865)) / self._fConst196) + np.float32(0.6896214)) 
		self._fConst205 = (np.float32(1.0) / (((self._fConst200 + np.float32(0.51247865)) / self._fConst196) + np.float32(0.6896214))) 
		self._fConst206 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst198)) 
		self._fConst207 = (((self._fConst200 + np.float32(-0.78241307)) / self._fConst196) + np.float32(0.2452915)) 
		self._fConst208 = (np.float32(1.0) / (((self._fConst200 + np.float32(0.78241307)) / self._fConst196) + np.float32(0.2452915))) 
		self._fConst209 = np.tan((np.float32(25132.742) / self._fConst0)) 
		self._fConst210 = np.power(self._fConst209, np.float32(2.0)) 
		self._fConst211 = (np.float32(1.0) / self._fConst210) 
		self._fConst212 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst211)) 
		self._fConst213 = (np.float32(1.0) / self._fConst209) 
		self._fConst214 = (((self._fConst213 + np.float32(-0.16840488)) / self._fConst209) + np.float32(1.0693583)) 
		self._fConst215 = (np.float32(1.0) / (((self._fConst213 + np.float32(0.16840488)) / self._fConst209) + np.float32(1.0693583))) 
		self._fConst216 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst211)) 
		self._fConst217 = (((self._fConst213 + np.float32(-0.51247865)) / self._fConst209) + np.float32(0.6896214)) 
		self._fConst218 = (np.float32(1.0) / (((self._fConst213 + np.float32(0.51247865)) / self._fConst209) + np.float32(0.6896214))) 
		self._fConst219 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst211)) 
		self._fConst220 = (((self._fConst213 + np.float32(-0.78241307)) / self._fConst209) + np.float32(0.2452915)) 
		self._fConst221 = (np.float32(1.0) / (((self._fConst213 + np.float32(0.78241307)) / self._fConst209) + np.float32(0.2452915))) 
		self._fConst222 = np.tan((np.float32(35543.062) / self._fConst0)) 
		self._fConst223 = np.power(self._fConst222, np.float32(2.0)) 
		self._fConst224 = (np.float32(1.0) / self._fConst223) 
		self._fConst225 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst224)) 
		self._fConst226 = (np.float32(1.0) / self._fConst222) 
		self._fConst227 = (((self._fConst226 + np.float32(-0.16840488)) / self._fConst222) + np.float32(1.0693583)) 
		self._fConst228 = (np.float32(1.0) / (((self._fConst226 + np.float32(0.16840488)) / self._fConst222) + np.float32(1.0693583))) 
		self._fConst229 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst224)) 
		self._fConst230 = (((self._fConst226 + np.float32(-0.51247865)) / self._fConst222) + np.float32(0.6896214)) 
		self._fConst231 = (np.float32(1.0) / (((self._fConst226 + np.float32(0.51247865)) / self._fConst222) + np.float32(0.6896214))) 
		self._fConst232 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst224)) 
		self._fConst233 = (((self._fConst226 + np.float32(-0.78241307)) / self._fConst222) + np.float32(0.2452915)) 
		self._fConst234 = (np.float32(1.0) / (((self._fConst226 + np.float32(0.78241307)) / self._fConst222) + np.float32(0.2452915))) 
		self._fConst235 = np.tan((np.float32(50265.484) / self._fConst0)) 
		self._fConst236 = np.power(self._fConst235, np.float32(2.0)) 
		self._fConst237 = (np.float32(1.0) / self._fConst236) 
		self._fConst238 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst237)) 
		self._fConst239 = (np.float32(1.0) / self._fConst235) 
		self._fConst240 = (((self._fConst239 + np.float32(-0.16840488)) / self._fConst235) + np.float32(1.0693583)) 
		self._fConst241 = (np.float32(1.0) / (((self._fConst239 + np.float32(0.16840488)) / self._fConst235) + np.float32(1.0693583))) 
		self._fConst242 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst237)) 
		self._fConst243 = (((self._fConst239 + np.float32(-0.51247865)) / self._fConst235) + np.float32(0.6896214)) 
		self._fConst244 = (np.float32(1.0) / (((self._fConst239 + np.float32(0.51247865)) / self._fConst235) + np.float32(0.6896214))) 
		self._fConst245 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst237)) 
		self._fConst246 = (((self._fConst239 + np.float32(-0.78241307)) / self._fConst235) + np.float32(0.2452915)) 
		self._fConst247 = (np.float32(1.0) / (((self._fConst239 + np.float32(0.78241307)) / self._fConst235) + np.float32(0.2452915))) 
		self._fConst248 = (np.float32(3.1415927) / self._fConst0) 
		self._fConst249 = (np.float32(1382.3008) / self._fConst0) 
		self._fConst250 = (np.float32(2764.6016) / self._fConst0) 
		self._fConst251 = (np.float32(1.0) / self._fConst0) 
		self._fConst252 = (np.float32(0.25) * self._fConst0) 
		self._fConst253 = (np.float32(0.041666668) * np.power(self._fConst0, np.float32(2.0))) 
		self._fConst254 = (np.float32(0.0052083335) * np.power(self._fConst0, np.float32(3.0))) 
		self._fConst255 = (np.float32(0.0001) / self._fConst236) 
		self._fConst256 = (self._fConst255 + np.float32(0.0004332272)) 
		self._fConst257 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst255)) 
		self._fConst258 = (self._fConst237 + np.float32(7.6217313)) 
		self._fConst259 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst237)) 
		self._fConst260 = (self._fConst237 + np.float32(53.53615)) 
		self._fConst261 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst237)) 
		self._fConst262 = (np.float32(0.0001) / self._fConst223) 
		self._fConst263 = (self._fConst262 + np.float32(0.0004332272)) 
		self._fConst264 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst262)) 
		self._fConst265 = (self._fConst224 + np.float32(7.6217313)) 
		self._fConst266 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst224)) 
		self._fConst267 = (self._fConst224 + np.float32(53.53615)) 
		self._fConst268 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst224)) 
		self._fConst269 = (np.float32(0.0001) / self._fConst210) 
		self._fConst270 = (self._fConst269 + np.float32(0.0004332272)) 
		self._fConst271 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst269)) 
		self._fConst272 = (self._fConst211 + np.float32(7.6217313)) 
		self._fConst273 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst211)) 
		self._fConst274 = (self._fConst211 + np.float32(53.53615)) 
		self._fConst275 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst211)) 
		self._fConst276 = (np.float32(0.0001) / self._fConst197) 
		self._fConst277 = (self._fConst276 + np.float32(0.0004332272)) 
		self._fConst278 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst276)) 
		self._fConst279 = (self._fConst198 + np.float32(7.6217313)) 
		self._fConst280 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst198)) 
		self._fConst281 = (self._fConst198 + np.float32(53.53615)) 
		self._fConst282 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst198)) 
		self._fConst283 = (np.float32(0.0001) / self._fConst184) 
		self._fConst284 = (self._fConst283 + np.float32(0.0004332272)) 
		self._fConst285 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst283)) 
		self._fConst286 = (self._fConst185 + np.float32(7.6217313)) 
		self._fConst287 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst185)) 
		self._fConst288 = (self._fConst185 + np.float32(53.53615)) 
		self._fConst289 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst185)) 
		self._fConst290 = (np.float32(0.0001) / self._fConst171) 
		self._fConst291 = (self._fConst290 + np.float32(0.0004332272)) 
		self._fConst292 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst290)) 
		self._fConst293 = (self._fConst172 + np.float32(7.6217313)) 
		self._fConst294 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst172)) 
		self._fConst295 = (self._fConst172 + np.float32(53.53615)) 
		self._fConst296 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst172)) 
		self._fConst297 = (np.float32(0.0001) / self._fConst158) 
		self._fConst298 = (self._fConst297 + np.float32(0.0004332272)) 
		self._fConst299 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst297)) 
		self._fConst300 = (self._fConst159 + np.float32(7.6217313)) 
		self._fConst301 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst159)) 
		self._fConst302 = (self._fConst159 + np.float32(53.53615)) 
		self._fConst303 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst159)) 
		self._fConst304 = (np.float32(0.0001) / self._fConst145) 
		self._fConst305 = (self._fConst304 + np.float32(0.0004332272)) 
		self._fConst306 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst304)) 
		self._fConst307 = (self._fConst146 + np.float32(7.6217313)) 
		self._fConst308 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst146)) 
		self._fConst309 = (self._fConst146 + np.float32(53.53615)) 
		self._fConst310 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst146)) 
		self._fConst311 = (np.float32(0.0001) / self._fConst132) 
		self._fConst312 = (self._fConst311 + np.float32(0.0004332272)) 
		self._fConst313 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst311)) 
		self._fConst314 = (self._fConst133 + np.float32(7.6217313)) 
		self._fConst315 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst133)) 
		self._fConst316 = (self._fConst133 + np.float32(53.53615)) 
		self._fConst317 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst133)) 
		self._fConst318 = (np.float32(0.0001) / self._fConst119) 
		self._fConst319 = (self._fConst318 + np.float32(0.0004332272)) 
		self._fConst320 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst318)) 
		self._fConst321 = (self._fConst120 + np.float32(7.6217313)) 
		self._fConst322 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst120)) 
		self._fConst323 = (self._fConst120 + np.float32(53.53615)) 
		self._fConst324 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst120)) 
		self._fConst325 = (np.float32(0.0001) / self._fConst106) 
		self._fConst326 = (self._fConst325 + np.float32(0.0004332272)) 
		self._fConst327 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst325)) 
		self._fConst328 = (self._fConst107 + np.float32(7.6217313)) 
		self._fConst329 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst107)) 
		self._fConst330 = (self._fConst107 + np.float32(53.53615)) 
		self._fConst331 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst107)) 
		self._fConst332 = (np.float32(0.0001) / self._fConst93) 
		self._fConst333 = (self._fConst332 + np.float32(0.0004332272)) 
		self._fConst334 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst332)) 
		self._fConst335 = (self._fConst94 + np.float32(7.6217313)) 
		self._fConst336 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst94)) 
		self._fConst337 = (self._fConst94 + np.float32(53.53615)) 
		self._fConst338 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst94)) 
		self._fConst339 = (np.float32(0.0001) / self._fConst80) 
		self._fConst340 = (self._fConst339 + np.float32(0.0004332272)) 
		self._fConst341 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst339)) 
		self._fConst342 = (self._fConst81 + np.float32(7.6217313)) 
		self._fConst343 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst81)) 
		self._fConst344 = (self._fConst81 + np.float32(53.53615)) 
		self._fConst345 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst81)) 
		self._fConst346 = (np.float32(0.0001) / self._fConst67) 
		self._fConst347 = (self._fConst346 + np.float32(0.0004332272)) 
		self._fConst348 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst346)) 
		self._fConst349 = (self._fConst68 + np.float32(7.6217313)) 
		self._fConst350 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst68)) 
		self._fConst351 = (self._fConst68 + np.float32(53.53615)) 
		self._fConst352 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst68)) 
		self._fConst353 = (np.float32(0.0001) / self._fConst54) 
		self._fConst354 = (self._fConst353 + np.float32(0.0004332272)) 
		self._fConst355 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst353)) 
		self._fConst356 = (self._fConst55 + np.float32(7.6217313)) 
		self._fConst357 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst55)) 
		self._fConst358 = (self._fConst55 + np.float32(53.53615)) 
		self._fConst359 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst55)) 
		self._fConst360 = (np.float32(0.0001) / self._fConst41) 
		self._fConst361 = (self._fConst360 + np.float32(0.0004332272)) 
		self._fConst362 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst360)) 
		self._fConst363 = (self._fConst42 + np.float32(7.6217313)) 
		self._fConst364 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst42)) 
		self._fConst365 = (self._fConst42 + np.float32(53.53615)) 
		self._fConst366 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst42)) 
		self._fConst367 = (np.float32(0.0001) / self._fConst28) 
		self._fConst368 = (self._fConst367 + np.float32(0.0004332272)) 
		self._fConst369 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst367)) 
		self._fConst370 = (self._fConst29 + np.float32(7.6217313)) 
		self._fConst371 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst29)) 
		self._fConst372 = (self._fConst29 + np.float32(53.53615)) 
		self._fConst373 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst29)) 
		self._fConst374 = (np.float32(0.0001) / self._fConst15) 
		self._fConst375 = (self._fConst374 + np.float32(0.0004332272)) 
		self._fConst376 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst374)) 
		self._fConst377 = (self._fConst16 + np.float32(7.6217313)) 
		self._fConst378 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst16)) 
		self._fConst379 = (self._fConst16 + np.float32(53.53615)) 
		self._fConst380 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst16)) 
		self._fConst381 = (np.float32(0.0001) / self._fConst2) 
		self._fConst382 = (self._fConst381 + np.float32(0.0004332272)) 
		self._fConst383 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst381)) 
		self._fConst384 = (self._fConst3 + np.float32(7.6217313)) 
		self._fConst385 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst3)) 
		self._fConst386 = (self._fConst3 + np.float32(53.53615)) 
		self._fConst387 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst3)) 
		self._fConst388 = (np.float32(1e+03) / self._fConst0) 
		self._fConst389 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst3)) 
		self._fConst390 = (((self._fConst5 + np.float32(-0.15748216)) / self._fConst1) + np.float32(0.9351402)) 
		self._fConst391 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.15748216)) / self._fConst1) + np.float32(0.9351402))) 
		self._fConst392 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst3)) 
		self._fConst393 = (((self._fConst5 + np.float32(-0.74313045)) / self._fConst1) + np.float32(1.4500711)) 
		self._fConst394 = (np.float32(1.0) / (((self._fConst5 + np.float32(0.74313045)) / self._fConst1) + np.float32(1.4500711))) 
		self._fConst395 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst3)) 
		self._fConst396 = (((self._fConst5 + np.float32(-3.1897273)) / self._fConst1) + np.float32(4.0767817)) 
		self._fConst397 = (np.float32(1.0) / (((self._fConst5 + np.float32(3.1897273)) / self._fConst1) + np.float32(4.0767817))) 
		self._fConst398 = (np.float32(0.0017661728) / self._fConst2) 
		self._fConst399 = (self._fConst398 + np.float32(0.0004076782)) 
		self._fConst400 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst398)) 
		self._fConst401 = (np.float32(11.0520525) / self._fConst2) 
		self._fConst402 = (self._fConst401 + np.float32(1.4500711)) 
		self._fConst403 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst401)) 
		self._fConst404 = (np.float32(50.06381) / self._fConst2) 
		self._fConst405 = (self._fConst404 + np.float32(0.9351402)) 
		self._fConst406 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst404)) 
		self._fConst407 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst16)) 
		self._fConst408 = (((self._fConst18 + np.float32(-0.15748216)) / self._fConst14) + np.float32(0.9351402)) 
		self._fConst409 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.15748216)) / self._fConst14) + np.float32(0.9351402))) 
		self._fConst410 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst16)) 
		self._fConst411 = (((self._fConst18 + np.float32(-0.74313045)) / self._fConst14) + np.float32(1.4500711)) 
		self._fConst412 = (np.float32(1.0) / (((self._fConst18 + np.float32(0.74313045)) / self._fConst14) + np.float32(1.4500711))) 
		self._fConst413 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst16)) 
		self._fConst414 = (((self._fConst18 + np.float32(-3.1897273)) / self._fConst14) + np.float32(4.0767817)) 
		self._fConst415 = (np.float32(1.0) / (((self._fConst18 + np.float32(3.1897273)) / self._fConst14) + np.float32(4.0767817))) 
		self._fConst416 = (np.float32(0.0017661728) / self._fConst15) 
		self._fConst417 = (self._fConst416 + np.float32(0.0004076782)) 
		self._fConst418 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst416)) 
		self._fConst419 = (np.float32(11.0520525) / self._fConst15) 
		self._fConst420 = (self._fConst419 + np.float32(1.4500711)) 
		self._fConst421 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst419)) 
		self._fConst422 = (np.float32(50.06381) / self._fConst15) 
		self._fConst423 = (self._fConst422 + np.float32(0.9351402)) 
		self._fConst424 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst422)) 
		self._fConst425 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst29)) 
		self._fConst426 = (((self._fConst31 + np.float32(-0.15748216)) / self._fConst27) + np.float32(0.9351402)) 
		self._fConst427 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.15748216)) / self._fConst27) + np.float32(0.9351402))) 
		self._fConst428 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst29)) 
		self._fConst429 = (((self._fConst31 + np.float32(-0.74313045)) / self._fConst27) + np.float32(1.4500711)) 
		self._fConst430 = (np.float32(1.0) / (((self._fConst31 + np.float32(0.74313045)) / self._fConst27) + np.float32(1.4500711))) 
		self._fConst431 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst29)) 
		self._fConst432 = (((self._fConst31 + np.float32(-3.1897273)) / self._fConst27) + np.float32(4.0767817)) 
		self._fConst433 = (np.float32(1.0) / (((self._fConst31 + np.float32(3.1897273)) / self._fConst27) + np.float32(4.0767817))) 
		self._fConst434 = (np.float32(0.0017661728) / self._fConst28) 
		self._fConst435 = (self._fConst434 + np.float32(0.0004076782)) 
		self._fConst436 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst434)) 
		self._fConst437 = (np.float32(11.0520525) / self._fConst28) 
		self._fConst438 = (self._fConst437 + np.float32(1.4500711)) 
		self._fConst439 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst437)) 
		self._fConst440 = (np.float32(50.06381) / self._fConst28) 
		self._fConst441 = (self._fConst440 + np.float32(0.9351402)) 
		self._fConst442 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst440)) 
		self._fConst443 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst42)) 
		self._fConst444 = (((self._fConst44 + np.float32(-0.15748216)) / self._fConst40) + np.float32(0.9351402)) 
		self._fConst445 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.15748216)) / self._fConst40) + np.float32(0.9351402))) 
		self._fConst446 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst42)) 
		self._fConst447 = (((self._fConst44 + np.float32(-0.74313045)) / self._fConst40) + np.float32(1.4500711)) 
		self._fConst448 = (np.float32(1.0) / (((self._fConst44 + np.float32(0.74313045)) / self._fConst40) + np.float32(1.4500711))) 
		self._fConst449 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst42)) 
		self._fConst450 = (((self._fConst44 + np.float32(-3.1897273)) / self._fConst40) + np.float32(4.0767817)) 
		self._fConst451 = (np.float32(1.0) / (((self._fConst44 + np.float32(3.1897273)) / self._fConst40) + np.float32(4.0767817))) 
		self._fConst452 = (np.float32(0.0017661728) / self._fConst41) 
		self._fConst453 = (self._fConst452 + np.float32(0.0004076782)) 
		self._fConst454 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst452)) 
		self._fConst455 = (np.float32(11.0520525) / self._fConst41) 
		self._fConst456 = (self._fConst455 + np.float32(1.4500711)) 
		self._fConst457 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst455)) 
		self._fConst458 = (np.float32(50.06381) / self._fConst41) 
		self._fConst459 = (self._fConst458 + np.float32(0.9351402)) 
		self._fConst460 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst458)) 
		self._fConst461 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst55)) 
		self._fConst462 = (((self._fConst57 + np.float32(-0.15748216)) / self._fConst53) + np.float32(0.9351402)) 
		self._fConst463 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.15748216)) / self._fConst53) + np.float32(0.9351402))) 
		self._fConst464 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst55)) 
		self._fConst465 = (((self._fConst57 + np.float32(-0.74313045)) / self._fConst53) + np.float32(1.4500711)) 
		self._fConst466 = (np.float32(1.0) / (((self._fConst57 + np.float32(0.74313045)) / self._fConst53) + np.float32(1.4500711))) 
		self._fConst467 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst55)) 
		self._fConst468 = (((self._fConst57 + np.float32(-3.1897273)) / self._fConst53) + np.float32(4.0767817)) 
		self._fConst469 = (np.float32(1.0) / (((self._fConst57 + np.float32(3.1897273)) / self._fConst53) + np.float32(4.0767817))) 
		self._fConst470 = (np.float32(0.0017661728) / self._fConst54) 
		self._fConst471 = (self._fConst470 + np.float32(0.0004076782)) 
		self._fConst472 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst470)) 
		self._fConst473 = (np.float32(11.0520525) / self._fConst54) 
		self._fConst474 = (self._fConst473 + np.float32(1.4500711)) 
		self._fConst475 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst473)) 
		self._fConst476 = (np.float32(50.06381) / self._fConst54) 
		self._fConst477 = (self._fConst476 + np.float32(0.9351402)) 
		self._fConst478 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst476)) 
		self._fConst479 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst68)) 
		self._fConst480 = (((self._fConst70 + np.float32(-0.15748216)) / self._fConst66) + np.float32(0.9351402)) 
		self._fConst481 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.15748216)) / self._fConst66) + np.float32(0.9351402))) 
		self._fConst482 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst68)) 
		self._fConst483 = (((self._fConst70 + np.float32(-0.74313045)) / self._fConst66) + np.float32(1.4500711)) 
		self._fConst484 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.74313045)) / self._fConst66) + np.float32(1.4500711))) 
		self._fConst485 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst68)) 
		self._fConst486 = (((self._fConst70 + np.float32(-3.1897273)) / self._fConst66) + np.float32(4.0767817)) 
		self._fConst487 = (np.float32(1.0) / (((self._fConst70 + np.float32(3.1897273)) / self._fConst66) + np.float32(4.0767817))) 
		self._fConst488 = (np.float32(0.0017661728) / self._fConst67) 
		self._fConst489 = (self._fConst488 + np.float32(0.0004076782)) 
		self._fConst490 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst488)) 
		self._fConst491 = (np.float32(11.0520525) / self._fConst67) 
		self._fConst492 = (self._fConst491 + np.float32(1.4500711)) 
		self._fConst493 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst491)) 
		self._fConst494 = (np.float32(50.06381) / self._fConst67) 
		self._fConst495 = (self._fConst494 + np.float32(0.9351402)) 
		self._fConst496 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst494)) 
		self._fConst497 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst81)) 
		self._fConst498 = (((self._fConst83 + np.float32(-0.15748216)) / self._fConst79) + np.float32(0.9351402)) 
		self._fConst499 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.15748216)) / self._fConst79) + np.float32(0.9351402))) 
		self._fConst500 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst81)) 
		self._fConst501 = (((self._fConst83 + np.float32(-0.74313045)) / self._fConst79) + np.float32(1.4500711)) 
		self._fConst502 = (np.float32(1.0) / (((self._fConst83 + np.float32(0.74313045)) / self._fConst79) + np.float32(1.4500711))) 
		self._fConst503 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst81)) 
		self._fConst504 = (((self._fConst83 + np.float32(-3.1897273)) / self._fConst79) + np.float32(4.0767817)) 
		self._fConst505 = (np.float32(1.0) / (((self._fConst83 + np.float32(3.1897273)) / self._fConst79) + np.float32(4.0767817))) 
		self._fConst506 = (np.float32(0.0017661728) / self._fConst80) 
		self._fConst507 = (self._fConst506 + np.float32(0.0004076782)) 
		self._fConst508 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst506)) 
		self._fConst509 = (np.float32(11.0520525) / self._fConst80) 
		self._fConst510 = (self._fConst509 + np.float32(1.4500711)) 
		self._fConst511 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst509)) 
		self._fConst512 = (np.float32(50.06381) / self._fConst80) 
		self._fConst513 = (self._fConst512 + np.float32(0.9351402)) 
		self._fConst514 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst512)) 
		self._fConst515 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst94)) 
		self._fConst516 = (((self._fConst96 + np.float32(-0.15748216)) / self._fConst92) + np.float32(0.9351402)) 
		self._fConst517 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.15748216)) / self._fConst92) + np.float32(0.9351402))) 
		self._fConst518 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst94)) 
		self._fConst519 = (((self._fConst96 + np.float32(-0.74313045)) / self._fConst92) + np.float32(1.4500711)) 
		self._fConst520 = (np.float32(1.0) / (((self._fConst96 + np.float32(0.74313045)) / self._fConst92) + np.float32(1.4500711))) 
		self._fConst521 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst94)) 
		self._fConst522 = (((self._fConst96 + np.float32(-3.1897273)) / self._fConst92) + np.float32(4.0767817)) 
		self._fConst523 = (np.float32(1.0) / (((self._fConst96 + np.float32(3.1897273)) / self._fConst92) + np.float32(4.0767817))) 
		self._fConst524 = (np.float32(0.0017661728) / self._fConst93) 
		self._fConst525 = (self._fConst524 + np.float32(0.0004076782)) 
		self._fConst526 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst524)) 
		self._fConst527 = (np.float32(11.0520525) / self._fConst93) 
		self._fConst528 = (self._fConst527 + np.float32(1.4500711)) 
		self._fConst529 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst527)) 
		self._fConst530 = (np.float32(50.06381) / self._fConst93) 
		self._fConst531 = (self._fConst530 + np.float32(0.9351402)) 
		self._fConst532 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst530)) 
		self._fConst533 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst107)) 
		self._fConst534 = (((self._fConst109 + np.float32(-0.15748216)) / self._fConst105) + np.float32(0.9351402)) 
		self._fConst535 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.15748216)) / self._fConst105) + np.float32(0.9351402))) 
		self._fConst536 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst107)) 
		self._fConst537 = (((self._fConst109 + np.float32(-0.74313045)) / self._fConst105) + np.float32(1.4500711)) 
		self._fConst538 = (np.float32(1.0) / (((self._fConst109 + np.float32(0.74313045)) / self._fConst105) + np.float32(1.4500711))) 
		self._fConst539 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst107)) 
		self._fConst540 = (((self._fConst109 + np.float32(-3.1897273)) / self._fConst105) + np.float32(4.0767817)) 
		self._fConst541 = (np.float32(1.0) / (((self._fConst109 + np.float32(3.1897273)) / self._fConst105) + np.float32(4.0767817))) 
		self._fConst542 = (np.float32(0.0017661728) / self._fConst106) 
		self._fConst543 = (self._fConst542 + np.float32(0.0004076782)) 
		self._fConst544 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst542)) 
		self._fConst545 = (np.float32(11.0520525) / self._fConst106) 
		self._fConst546 = (self._fConst545 + np.float32(1.4500711)) 
		self._fConst547 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst545)) 
		self._fConst548 = (np.float32(50.06381) / self._fConst106) 
		self._fConst549 = (self._fConst548 + np.float32(0.9351402)) 
		self._fConst550 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst548)) 
		self._fConst551 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst120)) 
		self._fConst552 = (((self._fConst122 + np.float32(-0.15748216)) / self._fConst118) + np.float32(0.9351402)) 
		self._fConst553 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.15748216)) / self._fConst118) + np.float32(0.9351402))) 
		self._fConst554 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst120)) 
		self._fConst555 = (((self._fConst122 + np.float32(-0.74313045)) / self._fConst118) + np.float32(1.4500711)) 
		self._fConst556 = (np.float32(1.0) / (((self._fConst122 + np.float32(0.74313045)) / self._fConst118) + np.float32(1.4500711))) 
		self._fConst557 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst120)) 
		self._fConst558 = (((self._fConst122 + np.float32(-3.1897273)) / self._fConst118) + np.float32(4.0767817)) 
		self._fConst559 = (np.float32(1.0) / (((self._fConst122 + np.float32(3.1897273)) / self._fConst118) + np.float32(4.0767817))) 
		self._fConst560 = (np.float32(0.0017661728) / self._fConst119) 
		self._fConst561 = (self._fConst560 + np.float32(0.0004076782)) 
		self._fConst562 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst560)) 
		self._fConst563 = (np.float32(11.0520525) / self._fConst119) 
		self._fConst564 = (self._fConst563 + np.float32(1.4500711)) 
		self._fConst565 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst563)) 
		self._fConst566 = (np.float32(50.06381) / self._fConst119) 
		self._fConst567 = (self._fConst566 + np.float32(0.9351402)) 
		self._fConst568 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst566)) 
		self._fConst569 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst133)) 
		self._fConst570 = (((self._fConst135 + np.float32(-0.15748216)) / self._fConst131) + np.float32(0.9351402)) 
		self._fConst571 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.15748216)) / self._fConst131) + np.float32(0.9351402))) 
		self._fConst572 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst133)) 
		self._fConst573 = (((self._fConst135 + np.float32(-0.74313045)) / self._fConst131) + np.float32(1.4500711)) 
		self._fConst574 = (np.float32(1.0) / (((self._fConst135 + np.float32(0.74313045)) / self._fConst131) + np.float32(1.4500711))) 
		self._fConst575 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst133)) 
		self._fConst576 = (((self._fConst135 + np.float32(-3.1897273)) / self._fConst131) + np.float32(4.0767817)) 
		self._fConst577 = (np.float32(1.0) / (((self._fConst135 + np.float32(3.1897273)) / self._fConst131) + np.float32(4.0767817))) 
		self._fConst578 = (np.float32(0.0017661728) / self._fConst132) 
		self._fConst579 = (self._fConst578 + np.float32(0.0004076782)) 
		self._fConst580 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst578)) 
		self._fConst581 = (np.float32(11.0520525) / self._fConst132) 
		self._fConst582 = (self._fConst581 + np.float32(1.4500711)) 
		self._fConst583 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst581)) 
		self._fConst584 = (np.float32(50.06381) / self._fConst132) 
		self._fConst585 = (self._fConst584 + np.float32(0.9351402)) 
		self._fConst586 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst584)) 
		self._fConst587 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst146)) 
		self._fConst588 = (((self._fConst148 + np.float32(-0.15748216)) / self._fConst144) + np.float32(0.9351402)) 
		self._fConst589 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.15748216)) / self._fConst144) + np.float32(0.9351402))) 
		self._fConst590 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst146)) 
		self._fConst591 = (((self._fConst148 + np.float32(-0.74313045)) / self._fConst144) + np.float32(1.4500711)) 
		self._fConst592 = (np.float32(1.0) / (((self._fConst148 + np.float32(0.74313045)) / self._fConst144) + np.float32(1.4500711))) 
		self._fConst593 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst146)) 
		self._fConst594 = (((self._fConst148 + np.float32(-3.1897273)) / self._fConst144) + np.float32(4.0767817)) 
		self._fConst595 = (np.float32(1.0) / (((self._fConst148 + np.float32(3.1897273)) / self._fConst144) + np.float32(4.0767817))) 
		self._fConst596 = (np.float32(0.0017661728) / self._fConst145) 
		self._fConst597 = (self._fConst596 + np.float32(0.0004076782)) 
		self._fConst598 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst596)) 
		self._fConst599 = (np.float32(11.0520525) / self._fConst145) 
		self._fConst600 = (self._fConst599 + np.float32(1.4500711)) 
		self._fConst601 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst599)) 
		self._fConst602 = (np.float32(50.06381) / self._fConst145) 
		self._fConst603 = (self._fConst602 + np.float32(0.9351402)) 
		self._fConst604 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst602)) 
		self._fConst605 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst159)) 
		self._fConst606 = (((self._fConst161 + np.float32(-0.15748216)) / self._fConst157) + np.float32(0.9351402)) 
		self._fConst607 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.15748216)) / self._fConst157) + np.float32(0.9351402))) 
		self._fConst608 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst159)) 
		self._fConst609 = (((self._fConst161 + np.float32(-0.74313045)) / self._fConst157) + np.float32(1.4500711)) 
		self._fConst610 = (np.float32(1.0) / (((self._fConst161 + np.float32(0.74313045)) / self._fConst157) + np.float32(1.4500711))) 
		self._fConst611 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst159)) 
		self._fConst612 = (((self._fConst161 + np.float32(-3.1897273)) / self._fConst157) + np.float32(4.0767817)) 
		self._fConst613 = (np.float32(1.0) / (((self._fConst161 + np.float32(3.1897273)) / self._fConst157) + np.float32(4.0767817))) 
		self._fConst614 = (np.float32(0.0017661728) / self._fConst158) 
		self._fConst615 = (self._fConst614 + np.float32(0.0004076782)) 
		self._fConst616 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst614)) 
		self._fConst617 = (np.float32(11.0520525) / self._fConst158) 
		self._fConst618 = (self._fConst617 + np.float32(1.4500711)) 
		self._fConst619 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst617)) 
		self._fConst620 = (np.float32(50.06381) / self._fConst158) 
		self._fConst621 = (self._fConst620 + np.float32(0.9351402)) 
		self._fConst622 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst620)) 
		self._fConst623 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst172)) 
		self._fConst624 = (((self._fConst174 + np.float32(-0.15748216)) / self._fConst170) + np.float32(0.9351402)) 
		self._fConst625 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.15748216)) / self._fConst170) + np.float32(0.9351402))) 
		self._fConst626 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst172)) 
		self._fConst627 = (((self._fConst174 + np.float32(-0.74313045)) / self._fConst170) + np.float32(1.4500711)) 
		self._fConst628 = (np.float32(1.0) / (((self._fConst174 + np.float32(0.74313045)) / self._fConst170) + np.float32(1.4500711))) 
		self._fConst629 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst172)) 
		self._fConst630 = (((self._fConst174 + np.float32(-3.1897273)) / self._fConst170) + np.float32(4.0767817)) 
		self._fConst631 = (np.float32(1.0) / (((self._fConst174 + np.float32(3.1897273)) / self._fConst170) + np.float32(4.0767817))) 
		self._fConst632 = (np.float32(0.0017661728) / self._fConst171) 
		self._fConst633 = (self._fConst632 + np.float32(0.0004076782)) 
		self._fConst634 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst632)) 
		self._fConst635 = (np.float32(11.0520525) / self._fConst171) 
		self._fConst636 = (self._fConst635 + np.float32(1.4500711)) 
		self._fConst637 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst635)) 
		self._fConst638 = (np.float32(50.06381) / self._fConst171) 
		self._fConst639 = (self._fConst638 + np.float32(0.9351402)) 
		self._fConst640 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst638)) 
		self._fConst641 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst185)) 
		self._fConst642 = (((self._fConst187 + np.float32(-0.15748216)) / self._fConst183) + np.float32(0.9351402)) 
		self._fConst643 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.15748216)) / self._fConst183) + np.float32(0.9351402))) 
		self._fConst644 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst185)) 
		self._fConst645 = (((self._fConst187 + np.float32(-0.74313045)) / self._fConst183) + np.float32(1.4500711)) 
		self._fConst646 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.74313045)) / self._fConst183) + np.float32(1.4500711))) 
		self._fConst647 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst185)) 
		self._fConst648 = (((self._fConst187 + np.float32(-3.1897273)) / self._fConst183) + np.float32(4.0767817)) 
		self._fConst649 = (np.float32(1.0) / (((self._fConst187 + np.float32(3.1897273)) / self._fConst183) + np.float32(4.0767817))) 
		self._fConst650 = (np.float32(0.0017661728) / self._fConst184) 
		self._fConst651 = (self._fConst650 + np.float32(0.0004076782)) 
		self._fConst652 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst650)) 
		self._fConst653 = (np.float32(11.0520525) / self._fConst184) 
		self._fConst654 = (self._fConst653 + np.float32(1.4500711)) 
		self._fConst655 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst653)) 
		self._fConst656 = (np.float32(50.06381) / self._fConst184) 
		self._fConst657 = (self._fConst656 + np.float32(0.9351402)) 
		self._fConst658 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst656)) 
		self._fConst659 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst198)) 
		self._fConst660 = (((self._fConst200 + np.float32(-0.15748216)) / self._fConst196) + np.float32(0.9351402)) 
		self._fConst661 = (np.float32(1.0) / (((self._fConst200 + np.float32(0.15748216)) / self._fConst196) + np.float32(0.9351402))) 
		self._fConst662 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst198)) 
		self._fConst663 = (((self._fConst200 + np.float32(-0.74313045)) / self._fConst196) + np.float32(1.4500711)) 
		self._fConst664 = (np.float32(1.0) / (((self._fConst200 + np.float32(0.74313045)) / self._fConst196) + np.float32(1.4500711))) 
		self._fConst665 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst198)) 
		self._fConst666 = (((self._fConst200 + np.float32(-3.1897273)) / self._fConst196) + np.float32(4.0767817)) 
		self._fConst667 = (np.float32(1.0) / (((self._fConst200 + np.float32(3.1897273)) / self._fConst196) + np.float32(4.0767817))) 
		self._fConst668 = (np.float32(0.0017661728) / self._fConst197) 
		self._fConst669 = (self._fConst668 + np.float32(0.0004076782)) 
		self._fConst670 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst668)) 
		self._fConst671 = (np.float32(11.0520525) / self._fConst197) 
		self._fConst672 = (self._fConst671 + np.float32(1.4500711)) 
		self._fConst673 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst671)) 
		self._fConst674 = (np.float32(50.06381) / self._fConst197) 
		self._fConst675 = (self._fConst674 + np.float32(0.9351402)) 
		self._fConst676 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst674)) 
		self._fConst677 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst211)) 
		self._fConst678 = (((self._fConst213 + np.float32(-0.15748216)) / self._fConst209) + np.float32(0.9351402)) 
		self._fConst679 = (np.float32(1.0) / (((self._fConst213 + np.float32(0.15748216)) / self._fConst209) + np.float32(0.9351402))) 
		self._fConst680 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst211)) 
		self._fConst681 = (((self._fConst213 + np.float32(-0.74313045)) / self._fConst209) + np.float32(1.4500711)) 
		self._fConst682 = (np.float32(1.0) / (((self._fConst213 + np.float32(0.74313045)) / self._fConst209) + np.float32(1.4500711))) 
		self._fConst683 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst211)) 
		self._fConst684 = (((self._fConst213 + np.float32(-3.1897273)) / self._fConst209) + np.float32(4.0767817)) 
		self._fConst685 = (np.float32(1.0) / (((self._fConst213 + np.float32(3.1897273)) / self._fConst209) + np.float32(4.0767817))) 
		self._fConst686 = (np.float32(0.0017661728) / self._fConst210) 
		self._fConst687 = (self._fConst686 + np.float32(0.0004076782)) 
		self._fConst688 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst686)) 
		self._fConst689 = (np.float32(11.0520525) / self._fConst210) 
		self._fConst690 = (self._fConst689 + np.float32(1.4500711)) 
		self._fConst691 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst689)) 
		self._fConst692 = (np.float32(50.06381) / self._fConst210) 
		self._fConst693 = (self._fConst692 + np.float32(0.9351402)) 
		self._fConst694 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst692)) 
		self._fConst695 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst224)) 
		self._fConst696 = (((self._fConst226 + np.float32(-0.15748216)) / self._fConst222) + np.float32(0.9351402)) 
		self._fConst697 = (np.float32(1.0) / (((self._fConst226 + np.float32(0.15748216)) / self._fConst222) + np.float32(0.9351402))) 
		self._fConst698 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst224)) 
		self._fConst699 = (((self._fConst226 + np.float32(-0.74313045)) / self._fConst222) + np.float32(1.4500711)) 
		self._fConst700 = (np.float32(1.0) / (((self._fConst226 + np.float32(0.74313045)) / self._fConst222) + np.float32(1.4500711))) 
		self._fConst701 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst224)) 
		self._fConst702 = (((self._fConst226 + np.float32(-3.1897273)) / self._fConst222) + np.float32(4.0767817)) 
		self._fConst703 = (np.float32(1.0) / (((self._fConst226 + np.float32(3.1897273)) / self._fConst222) + np.float32(4.0767817))) 
		self._fConst704 = (np.float32(0.0017661728) / self._fConst223) 
		self._fConst705 = (self._fConst704 + np.float32(0.0004076782)) 
		self._fConst706 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst704)) 
		self._fConst707 = (np.float32(11.0520525) / self._fConst223) 
		self._fConst708 = (self._fConst707 + np.float32(1.4500711)) 
		self._fConst709 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst707)) 
		self._fConst710 = (np.float32(50.06381) / self._fConst223) 
		self._fConst711 = (self._fConst710 + np.float32(0.9351402)) 
		self._fConst712 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst710)) 
		self._fConst713 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst237)) 
		self._fConst714 = (((self._fConst239 + np.float32(-0.15748216)) / self._fConst235) + np.float32(0.9351402)) 
		self._fConst715 = (np.float32(1.0) / (((self._fConst239 + np.float32(0.15748216)) / self._fConst235) + np.float32(0.9351402))) 
		self._fConst716 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst237)) 
		self._fConst717 = (((self._fConst239 + np.float32(-0.74313045)) / self._fConst235) + np.float32(1.4500711)) 
		self._fConst718 = (np.float32(1.0) / (((self._fConst239 + np.float32(0.74313045)) / self._fConst235) + np.float32(1.4500711))) 
		self._fConst719 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst237)) 
		self._fConst720 = (((self._fConst239 + np.float32(-3.1897273)) / self._fConst235) + np.float32(4.0767817)) 
		self._fConst721 = (np.float32(1.0) / (((self._fConst239 + np.float32(3.1897273)) / self._fConst235) + np.float32(4.0767817))) 
		self._fConst722 = (np.float32(0.0017661728) / self._fConst236) 
		self._fConst723 = (self._fConst722 + np.float32(0.0004076782)) 
		self._fConst724 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst722)) 
		self._fConst725 = (np.float32(11.0520525) / self._fConst236) 
		self._fConst726 = (self._fConst725 + np.float32(1.4500711)) 
		self._fConst727 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst725)) 
		self._fConst728 = (np.float32(50.06381) / self._fConst236) 
		self._fConst729 = (self._fConst728 + np.float32(0.9351402)) 
		self._fConst730 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst728)) 
		
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec103"] = np.float32(0)
		state["fRec107"] = np.float32(0)
		state["fRec111"] = np.float32(0)
		state["fRec115"] = np.float32(0)
		state["fRec119"] = np.float32(0)
		state["fRec123"] = np.float32(0)
		state["fRec127"] = np.float32(0)
		state["fRec131"] = np.float32(0)
		state["fRec135"] = np.float32(0)
		state["fRec139"] = np.float32(0)
		state["fRec143"] = np.float32(0)
		state["fRec147"] = np.float32(0)
		state["fRec59"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec63"] = np.float32(0)
		state["fRec64"] = np.float32(0)
		state["fRec65"] = np.float32(0)
		state["fRec66"] = np.float32(0)
		state["fRec67"] = np.float32(0)
		state["fRec68"] = np.float32(0)
		state["fRec72"] = np.float32(0)
		state["fRec74"] = np.float32(0)
		state["fRec75"] = np.float32(0)
		state["fRec79"] = np.float32(0)
		state["fRec83"] = np.float32(0)
		state["fRec87"] = np.float32(0)
		state["fRec91"] = np.float32(0)
		state["fRec95"] = np.float32(0)
		state["fRec99"] = np.float32(0)
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
		state["fVec3"] = np.float32(0)
		state["fVec4"] = np.float32(0)
		state["fVec5"] = np.float32(0)
		state["fVec6"] = np.float32(0)
		state["fVec7"] = np.float32(0)
		state["fVec8"] = np.float32(0)
		state["fVec9"] = np.float32(0)
		state["iRec69"] = np.int32(0)
		# Initialize array delays
		state["iVec0"] = np.zeros((4,), dtype=np.int32)
		state["fRec70"] = np.zeros((4,), dtype=np.float32)
		state["fRec62"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec61"] = np.zeros((3,), dtype=np.float32)
		state["fRec58"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec57"] = np.zeros((3,), dtype=np.float32)
		state["fRec56"] = np.zeros((3,), dtype=np.float32)
		state["fRec55"] = np.zeros((3,), dtype=np.float32)
		state["fRec54"] = np.zeros((3,), dtype=np.float32)
		state["fRec53"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec50"] = np.zeros((3,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fRec48"] = np.zeros((3,), dtype=np.float32)
		state["fRec47"] = np.zeros((3,), dtype=np.float32)
		state["fRec46"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		state["fRec43"] = np.zeros((3,), dtype=np.float32)
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
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fRec76"] = np.zeros((3,), dtype=np.float32)
		state["fRec82"] = np.zeros((3,), dtype=np.float32)
		state["fRec81"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec84"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fRec88"] = np.zeros((3,), dtype=np.float32)
		state["fRec94"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec92"] = np.zeros((3,), dtype=np.float32)
		state["fRec98"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec96"] = np.zeros((3,), dtype=np.float32)
		state["fRec102"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec110"] = np.zeros((3,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec114"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec112"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec116"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec120"] = np.zeros((3,), dtype=np.float32)
		state["fRec126"] = np.zeros((3,), dtype=np.float32)
		state["fRec125"] = np.zeros((3,), dtype=np.float32)
		state["fRec124"] = np.zeros((3,), dtype=np.float32)
		state["fRec130"] = np.zeros((3,), dtype=np.float32)
		state["fRec129"] = np.zeros((3,), dtype=np.float32)
		state["fRec128"] = np.zeros((3,), dtype=np.float32)
		state["fRec134"] = np.zeros((3,), dtype=np.float32)
		state["fRec133"] = np.zeros((3,), dtype=np.float32)
		state["fRec132"] = np.zeros((3,), dtype=np.float32)
		state["fRec138"] = np.zeros((3,), dtype=np.float32)
		state["fRec137"] = np.zeros((3,), dtype=np.float32)
		state["fRec136"] = np.zeros((3,), dtype=np.float32)
		state["fRec142"] = np.zeros((3,), dtype=np.float32)
		state["fRec141"] = np.zeros((3,), dtype=np.float32)
		state["fRec140"] = np.zeros((3,), dtype=np.float32)
		state["fRec146"] = np.zeros((3,), dtype=np.float32)
		state["fRec145"] = np.zeros((3,), dtype=np.float32)
		state["fRec144"] = np.zeros((3,), dtype=np.float32)
		state["fRec150"] = np.zeros((3,), dtype=np.float32)
		state["fRec149"] = np.zeros((3,), dtype=np.float32)
		state["fRec148"] = np.zeros((3,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = jnp.tan((self._fConst248 * params["fHslider0"])) 
		fSlow1 = jnp.power(fSlow0, jnp.float32(2.0)) 
		fSlow2 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow1))) 
		fSlow3 = (jnp.float32(1.0) / fSlow0) 
		fSlow4 = (((fSlow3 + jnp.float32(-1.0)) / fSlow0) + jnp.float32(1.0)) 
		fSlow5 = (jnp.float32(1.0) / (((fSlow3 + jnp.float32(1.0)) / fSlow0) + jnp.float32(1.0))) 
		fSlow6 = (jnp.float32(0.001) * params["fHslider1"]) 
		fSlow7 = params["fHslider2"] 
		iSlow8 = (fSlow7 > jnp.float32(0.0)).astype(jnp.int32) 
		fSlow9 = params["fHslider3"] 
		fSlow10 = (self._fConst249 * (jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * jnp.abs(fSlow7))) / fSlow9)) 
		fSlow11 = (self._fConst249 / fSlow9) 
		fSlow12 = jnp.tan((self._fConst248 * params["fHslider4"])) 
		fSlow13 = (jnp.float32(1.0) / jnp.power(fSlow12, jnp.float32(2.0))) 
		fSlow14 = (jnp.float32(2.0) * (jnp.float32(1.0) - fSlow13)) 
		fSlow15 = (jnp.float32(1.0) / fSlow12) 
		fSlow16 = (((fSlow15 + jnp.float32(-1.0)) / fSlow12) + jnp.float32(1.0)) 
		fSlow17 = (jnp.float32(1.0) / (((fSlow15 + jnp.float32(1.0)) / fSlow12) + jnp.float32(1.0))) 
		iSlow18 = jnp.int32(params["fCheckbox0"]) 
		iSlow19 = jnp.int32(params["fCheckbox1"]) 
		iSlow20 = jnp.int32((params["fEntry0"] + jnp.float32(-1.0))) 
		iSlow21 = (iSlow20 >= jnp.int32(2)).astype(jnp.int32) 
		iSlow22 = (iSlow20 >= jnp.int32(1)).astype(jnp.int32) 
		fSlow23 = params["fVslider0"] 
		fSlow24 = jnp.where(((fSlow23 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst251 / fSlow23))), jnp.float32(0.0)) 
		fSlow25 = ((jnp.float32(4.4e+02) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fVslider1"] + jnp.float32(-49.0))))) * (jnp.float32(1.0) - fSlow24)) 
		fSlow26 = ((jnp.float32(0.01) * params["fVslider2"]) + jnp.float32(1.0)) 
		iSlow27 = (iSlow20 >= jnp.int32(3)).astype(jnp.int32) 
		fSlow28 = ((jnp.float32(0.01) * params["fVslider3"]) + jnp.float32(1.0)) 
		fSlow29 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider4"]))) 
		iSlow30 = jnp.int32(params["fCheckbox2"]) 
		fSlow31 = (jnp.float32(1.0) - fSlow15) 
		fSlow32 = (jnp.float32(1.0) / (fSlow15 + jnp.float32(1.0))) 
		fSlow33 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider5"])) 
		fSlow34 = (jnp.float32(1.0) - fSlow3) 
		fSlow35 = (jnp.float32(1.0) / (fSlow3 + jnp.float32(1.0))) 
		fSlow36 = (jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider6"])) / fSlow1) 
		fSlow37 = params["fHslider7"] 
		fSlow38 = jnp.where((((jnp.float32(0.001) * fSlow37) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst388 / fSlow37))), jnp.float32(0.0)) 
		fSlow39 = (jnp.float32(1.0) - fSlow38) 
		fSlow40 = params["fHslider8"] 
		fRec60_temp = state["fRec60"] 
		fRec65_temp = state["fRec65"] 
		fRec64_temp = state["fRec64"] 
		fVec1_temp = state["fVec1"] 
		fVec2_temp = state["fVec2"] 
		fVec3_temp = state["fVec3"] 
		fVec4_temp = state["fVec4"] 
		fVec5_temp = state["fVec5"] 
		fVec6_temp = state["fVec6"] 
		fRec66_temp = state["fRec66"] 
		fVec7_temp = state["fVec7"] 
		fVec8_temp = state["fVec8"] 
		fVec9_temp = state["fVec9"] 
		fVec10_temp = state["fVec10"] 
		fVec11_temp = state["fVec11"] 
		fVec12_temp = state["fVec12"] 
		fRec67_temp = state["fRec67"] 
		fVec13_temp = state["fVec13"] 
		fVec14_temp = state["fVec14"] 
		fVec15_temp = state["fVec15"] 
		fVec16_temp = state["fVec16"] 
		fVec17_temp = state["fVec17"] 
		fVec18_temp = state["fVec18"] 
		fRec68_temp = state["fRec68"] 
		iRec69_temp = state["iRec69"] 
		fVec19_temp = state["fVec19"] 
		fRec63_temp = state["fRec63"] 
		fRec72_temp = state["fRec72"] 
		fVec20_temp = state["fVec20"] 
		fRec59_temp = state["fRec59"] 
		fRec74_temp = state["fRec74"] 
		fRec0_temp = state["fRec0"] 
		fRec75_temp = state["fRec75"] 
		fRec79_temp = state["fRec79"] 
		fRec83_temp = state["fRec83"] 
		fRec87_temp = state["fRec87"] 
		fRec91_temp = state["fRec91"] 
		fRec95_temp = state["fRec95"] 
		fRec99_temp = state["fRec99"] 
		fRec103_temp = state["fRec103"] 
		fRec107_temp = state["fRec107"] 
		fRec111_temp = state["fRec111"] 
		fRec115_temp = state["fRec115"] 
		fRec119_temp = state["fRec119"] 
		fRec123_temp = state["fRec123"] 
		fRec127_temp = state["fRec127"] 
		fRec131_temp = state["fRec131"] 
		fRec135_temp = state["fRec135"] 
		fRec139_temp = state["fRec139"] 
		fRec143_temp = state["fRec143"] 
		fRec147_temp = state["fRec147"] 
		state["iVec0"] = state["iVec0"].at[0].set(jnp.int32(1)) 
		state["fRec60"] = (fSlow6 + (jnp.float32(0.999) * fRec60_temp)) 
		fTemp0 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (state["fRec60"] + jnp.float32(-49.0)))) 
		fTemp1 = jnp.tan((self._fConst249 * fTemp0)) 
		fTemp2 = (fTemp0 / jnp.sin((self._fConst250 * fTemp0))) 
		fTemp3 = (fSlow10 * fTemp2) 
		fTemp4 = (fSlow11 * fTemp2) 
		fTemp5 = jnp.where((iSlow8 != 0), fTemp4, fTemp3) 
		fTemp6 = (jnp.float32(1.0) / fTemp1) 
		fTemp7 = (((fTemp6 + fTemp5) / fTemp1) + jnp.float32(1.0)) 
		fTemp8 = jnp.where((iSlow8 != 0), fTemp3, fTemp4) 
		fTemp9 = (jnp.float32(2.0) * (state["fRec61"][1] * (jnp.float32(1.0) - (jnp.float32(1.0) / jnp.power(fTemp1, jnp.float32(2.0)))))) 
		state["fRec65"] = ((fRec65_temp * fSlow24) + fSlow25) 
		fTemp10 = jnp.maximum(jnp.float32(2e+01), jnp.abs((fSlow26 * state["fRec65"]))) 
		fTemp11 = (fRec64_temp + (self._fConst251 * fTemp10)) 
		state["fRec64"] = (fTemp11 - jnp.floor(fTemp11)) 
		fTemp12 = (jnp.float32(2.0) * state["fRec64"]) 
		fTemp13 = (fTemp12 + jnp.float32(-1.0)) 
		fTemp14 = jnp.power(fTemp13, jnp.float32(2.0)) 
		state["fVec1"] = jnp.float32(fTemp14) 
		fTemp15 = (state["iVec0"][1]) 
		fTemp16 = jnp.power(fTemp13, jnp.float32(3.0)) 
		state["fVec2"] = (fTemp16 + (jnp.float32(1.0) - fTemp12)) 
		fTemp17 = ((fTemp16 + (jnp.float32(1.0) - (fTemp12 + fVec2_temp))) / fTemp10) 
		state["fVec3"] = jnp.float32(fTemp17) 
		fTemp18 = (state["iVec0"][2]) 
		fTemp19 = (fTemp14 * (fTemp14 + jnp.float32(-2.0))) 
		state["fVec4"] = jnp.float32(fTemp19) 
		fTemp20 = ((fTemp19 - fVec4_temp) / fTemp10) 
		state["fVec5"] = jnp.float32(fTemp20) 
		fTemp21 = ((fTemp20 - fVec5_temp) / fTemp10) 
		state["fVec6"] = jnp.float32(fTemp21) 
		fTemp22 = (state["iVec0"][3]) 
		fTemp23 = jnp.maximum(jnp.float32(2e+01), jnp.abs((fSlow28 * state["fRec65"]))) 
		fTemp24 = (fRec66_temp + (self._fConst251 * fTemp23)) 
		state["fRec66"] = (fTemp24 - jnp.floor(fTemp24)) 
		fTemp25 = (jnp.float32(2.0) * state["fRec66"]) 
		fTemp26 = (fTemp25 + jnp.float32(-1.0)) 
		fTemp27 = jnp.power(fTemp26, jnp.float32(2.0)) 
		state["fVec7"] = jnp.float32(fTemp27) 
		fTemp28 = jnp.power(fTemp26, jnp.float32(3.0)) 
		state["fVec8"] = (fTemp28 + (jnp.float32(1.0) - fTemp25)) 
		fTemp29 = ((fTemp28 + (jnp.float32(1.0) - (fTemp25 + fVec8_temp))) / fTemp23) 
		state["fVec9"] = jnp.float32(fTemp29) 
		fTemp30 = (fTemp27 * (fTemp27 + jnp.float32(-2.0))) 
		state["fVec10"] = jnp.float32(fTemp30) 
		fTemp31 = ((fTemp30 - fVec10_temp) / fTemp23) 
		state["fVec11"] = jnp.float32(fTemp31) 
		fTemp32 = ((fTemp31 - fVec11_temp) / fTemp23) 
		state["fVec12"] = jnp.float32(fTemp32) 
		fTemp33 = jnp.maximum(jnp.float32(2e+01), jnp.abs(state["fRec65"])) 
		fTemp34 = (fRec67_temp + (self._fConst251 * fTemp33)) 
		state["fRec67"] = (fTemp34 - jnp.floor(fTemp34)) 
		fTemp35 = (jnp.float32(2.0) * state["fRec67"]) 
		fTemp36 = (fTemp35 + jnp.float32(-1.0)) 
		fTemp37 = jnp.power(fTemp36, jnp.float32(2.0)) 
		state["fVec13"] = jnp.float32(fTemp37) 
		fTemp38 = jnp.power(fTemp36, jnp.float32(3.0)) 
		state["fVec14"] = (fTemp38 + (jnp.float32(1.0) - fTemp35)) 
		fTemp39 = ((fTemp38 + (jnp.float32(1.0) - (fTemp35 + fVec14_temp))) / fTemp33) 
		state["fVec15"] = jnp.float32(fTemp39) 
		fTemp40 = (fTemp37 * (fTemp37 + jnp.float32(-2.0))) 
		state["fVec16"] = jnp.float32(fTemp40) 
		fTemp41 = ((fTemp40 - fVec16_temp) / fTemp33) 
		state["fVec17"] = jnp.float32(fTemp41) 
		fTemp42 = ((fTemp41 - fVec17_temp) / fTemp33) 
		state["fVec18"] = jnp.float32(fTemp42) 
		state["fRec68"] = (fSlow29 + (jnp.float32(0.999) * fRec68_temp)) 
		state["iRec69"] = ((jnp.int32(1103515245) * iRec69_temp) + jnp.int32(12345)) 
		fTemp43 = (jnp.float32(4.656613e-10) * (state["iRec69"])) 
		state["fRec70"] = state["fRec70"].at[0].set((((jnp.float32(0.5221894) * state["fRec70"][3]) + (fTemp43 + (jnp.float32(2.494956) * state["fRec70"][1]))) - (jnp.float32(2.0172658) * state["fRec70"][2]))) 
		fTemp44 = (state["fRec68"] * jnp.where((iSlow18 != 0), inputs[0], jnp.where((iSlow19 != 0), jnp.where((iSlow30 != 0), (((jnp.float32(0.049922034) * state["fRec70"][0]) + (jnp.float32(0.0506127) * state["fRec70"][2])) - ((jnp.float32(0.095993534) * state["fRec70"][1]) + (jnp.float32(0.004408786) * state["fRec70"][3]))), fTemp43), (jnp.float32(0.33333334) * (state["fRec68"] * ((jnp.where((iSlow21 != 0), jnp.where((iSlow27 != 0), (self._fConst254 * ((fTemp22 * (fTemp42 - fVec18_temp)) / fTemp33)), (self._fConst253 * ((fTemp18 * (fTemp39 - fVec15_temp)) / fTemp33))), jnp.where((iSlow22 != 0), (self._fConst252 * ((fTemp15 * (fTemp37 - fVec13_temp)) / fTemp33)), fTemp36)) + jnp.where((iSlow21 != 0), jnp.where((iSlow27 != 0), (self._fConst254 * ((fTemp22 * (fTemp32 - fVec12_temp)) / fTemp23)), (self._fConst253 * ((fTemp18 * (fTemp29 - fVec9_temp)) / fTemp23))), jnp.where((iSlow22 != 0), (self._fConst252 * ((fTemp15 * (fTemp27 - fVec7_temp)) / fTemp23)), fTemp26))) + jnp.where((iSlow21 != 0), jnp.where((iSlow27 != 0), (self._fConst254 * ((fTemp22 * (fTemp21 - fVec6_temp)) / fTemp10)), (self._fConst253 * ((fTemp18 * (fTemp17 - fVec3_temp)) / fTemp10))), jnp.where((iSlow22 != 0), (self._fConst252 * ((fTemp15 * (fTemp14 - fVec1_temp)) / fTemp10)), fTemp13)))))))) 
		state["fVec19"] = jnp.float32(fTemp44) 
		state["fRec63"] = -((fSlow32 * ((fSlow31 * fRec63_temp) - (fTemp44 + fVec19_temp)))) 
		state["fRec62"] = state["fRec62"].at[0].set((state["fRec63"] - (fSlow17 * ((fSlow16 * state["fRec62"][2]) + (fSlow14 * state["fRec62"][1]))))) 
		state["fRec72"] = -((fSlow32 * ((fSlow31 * fRec72_temp) - (fSlow15 * (fTemp44 - fVec19_temp))))) 
		state["fRec71"] = state["fRec71"].at[0].set((state["fRec72"] - (fSlow17 * ((fSlow16 * state["fRec71"][2]) + (fSlow14 * state["fRec71"][1]))))) 
		state["fRec61"] = state["fRec61"].at[0].set(((fSlow17 * ((fSlow13 * (state["fRec71"][2] + (state["fRec71"][0] - (jnp.float32(2.0) * state["fRec71"][1])))) + (fSlow33 * (state["fRec62"][2] + (state["fRec62"][0] + (jnp.float32(2.0) * state["fRec62"][1])))))) - (((state["fRec61"][2] * (((fTemp6 - fTemp5) / fTemp1) + jnp.float32(1.0))) + fTemp9) / fTemp7))) 
		fTemp45 = (((fTemp9 + (state["fRec61"][0] * (((fTemp6 + fTemp8) / fTemp1) + jnp.float32(1.0)))) + (state["fRec61"][2] * (((fTemp6 - fTemp8) / fTemp1) + jnp.float32(1.0)))) / fTemp7) 
		state["fVec20"] = jnp.float32(fTemp45) 
		state["fRec59"] = -((fSlow35 * ((fSlow34 * fRec59_temp) - (fSlow3 * (fTemp45 - fVec20_temp))))) 
		state["fRec58"] = state["fRec58"].at[0].set((state["fRec59"] - (fSlow5 * ((fSlow4 * state["fRec58"][2]) + (fSlow2 * state["fRec58"][1]))))) 
		state["fRec74"] = -((fSlow35 * ((fSlow34 * fRec74_temp) - (fTemp45 + fVec20_temp)))) 
		state["fRec73"] = state["fRec73"].at[0].set((state["fRec74"] - (fSlow5 * ((fSlow4 * state["fRec73"][2]) + (fSlow2 * state["fRec73"][1]))))) 
		fTemp46 = (fSlow5 * ((state["fRec73"][2] + (state["fRec73"][0] + (jnp.float32(2.0) * state["fRec73"][1]))) + (fSlow36 * (state["fRec58"][2] + (state["fRec58"][0] - (jnp.float32(2.0) * state["fRec58"][1])))))) 
		state["fRec57"] = state["fRec57"].at[0].set((fTemp46 - (self._fConst247 * ((self._fConst246 * state["fRec57"][2]) + (self._fConst245 * state["fRec57"][1]))))) 
		state["fRec56"] = state["fRec56"].at[0].set(((self._fConst247 * (((self._fConst256 * state["fRec57"][0]) + (self._fConst257 * state["fRec57"][1])) + (self._fConst256 * state["fRec57"][2]))) - (self._fConst244 * ((self._fConst243 * state["fRec56"][2]) + (self._fConst242 * state["fRec56"][1]))))) 
		state["fRec55"] = state["fRec55"].at[0].set(((self._fConst244 * (((self._fConst258 * state["fRec56"][0]) + (self._fConst259 * state["fRec56"][1])) + (self._fConst258 * state["fRec56"][2]))) - (self._fConst241 * ((self._fConst240 * state["fRec55"][2]) + (self._fConst238 * state["fRec55"][1]))))) 
		fTemp47 = (self._fConst241 * (((self._fConst260 * state["fRec55"][0]) + (self._fConst261 * state["fRec55"][1])) + (self._fConst260 * state["fRec55"][2]))) 
		state["fRec54"] = state["fRec54"].at[0].set((fTemp47 - (self._fConst234 * ((self._fConst233 * state["fRec54"][2]) + (self._fConst232 * state["fRec54"][1]))))) 
		state["fRec53"] = state["fRec53"].at[0].set(((self._fConst234 * (((self._fConst263 * state["fRec54"][0]) + (self._fConst264 * state["fRec54"][1])) + (self._fConst263 * state["fRec54"][2]))) - (self._fConst231 * ((self._fConst230 * state["fRec53"][2]) + (self._fConst229 * state["fRec53"][1]))))) 
		state["fRec52"] = state["fRec52"].at[0].set(((self._fConst231 * (((self._fConst265 * state["fRec53"][0]) + (self._fConst266 * state["fRec53"][1])) + (self._fConst265 * state["fRec53"][2]))) - (self._fConst228 * ((self._fConst227 * state["fRec52"][2]) + (self._fConst225 * state["fRec52"][1]))))) 
		fTemp48 = (self._fConst228 * (((self._fConst267 * state["fRec52"][0]) + (self._fConst268 * state["fRec52"][1])) + (self._fConst267 * state["fRec52"][2]))) 
		state["fRec51"] = state["fRec51"].at[0].set((fTemp48 - (self._fConst221 * ((self._fConst220 * state["fRec51"][2]) + (self._fConst219 * state["fRec51"][1]))))) 
		state["fRec50"] = state["fRec50"].at[0].set(((self._fConst221 * (((self._fConst270 * state["fRec51"][0]) + (self._fConst271 * state["fRec51"][1])) + (self._fConst270 * state["fRec51"][2]))) - (self._fConst218 * ((self._fConst217 * state["fRec50"][2]) + (self._fConst216 * state["fRec50"][1]))))) 
		state["fRec49"] = state["fRec49"].at[0].set(((self._fConst218 * (((self._fConst272 * state["fRec50"][0]) + (self._fConst273 * state["fRec50"][1])) + (self._fConst272 * state["fRec50"][2]))) - (self._fConst215 * ((self._fConst214 * state["fRec49"][2]) + (self._fConst212 * state["fRec49"][1]))))) 
		fTemp49 = (self._fConst215 * (((self._fConst274 * state["fRec49"][0]) + (self._fConst275 * state["fRec49"][1])) + (self._fConst274 * state["fRec49"][2]))) 
		state["fRec48"] = state["fRec48"].at[0].set((fTemp49 - (self._fConst208 * ((self._fConst207 * state["fRec48"][2]) + (self._fConst206 * state["fRec48"][1]))))) 
		state["fRec47"] = state["fRec47"].at[0].set(((self._fConst208 * (((self._fConst277 * state["fRec48"][0]) + (self._fConst278 * state["fRec48"][1])) + (self._fConst277 * state["fRec48"][2]))) - (self._fConst205 * ((self._fConst204 * state["fRec47"][2]) + (self._fConst203 * state["fRec47"][1]))))) 
		state["fRec46"] = state["fRec46"].at[0].set(((self._fConst205 * (((self._fConst279 * state["fRec47"][0]) + (self._fConst280 * state["fRec47"][1])) + (self._fConst279 * state["fRec47"][2]))) - (self._fConst202 * ((self._fConst201 * state["fRec46"][2]) + (self._fConst199 * state["fRec46"][1]))))) 
		fTemp50 = (self._fConst202 * (((self._fConst281 * state["fRec46"][0]) + (self._fConst282 * state["fRec46"][1])) + (self._fConst281 * state["fRec46"][2]))) 
		state["fRec45"] = state["fRec45"].at[0].set((fTemp50 - (self._fConst195 * ((self._fConst194 * state["fRec45"][2]) + (self._fConst193 * state["fRec45"][1]))))) 
		state["fRec44"] = state["fRec44"].at[0].set(((self._fConst195 * (((self._fConst284 * state["fRec45"][0]) + (self._fConst285 * state["fRec45"][1])) + (self._fConst284 * state["fRec45"][2]))) - (self._fConst192 * ((self._fConst191 * state["fRec44"][2]) + (self._fConst190 * state["fRec44"][1]))))) 
		state["fRec43"] = state["fRec43"].at[0].set(((self._fConst192 * (((self._fConst286 * state["fRec44"][0]) + (self._fConst287 * state["fRec44"][1])) + (self._fConst286 * state["fRec44"][2]))) - (self._fConst189 * ((self._fConst188 * state["fRec43"][2]) + (self._fConst186 * state["fRec43"][1]))))) 
		fTemp51 = (self._fConst189 * (((self._fConst288 * state["fRec43"][0]) + (self._fConst289 * state["fRec43"][1])) + (self._fConst288 * state["fRec43"][2]))) 
		state["fRec42"] = state["fRec42"].at[0].set((fTemp51 - (self._fConst182 * ((self._fConst181 * state["fRec42"][2]) + (self._fConst180 * state["fRec42"][1]))))) 
		state["fRec41"] = state["fRec41"].at[0].set(((self._fConst182 * (((self._fConst291 * state["fRec42"][0]) + (self._fConst292 * state["fRec42"][1])) + (self._fConst291 * state["fRec42"][2]))) - (self._fConst179 * ((self._fConst178 * state["fRec41"][2]) + (self._fConst177 * state["fRec41"][1]))))) 
		state["fRec40"] = state["fRec40"].at[0].set(((self._fConst179 * (((self._fConst293 * state["fRec41"][0]) + (self._fConst294 * state["fRec41"][1])) + (self._fConst293 * state["fRec41"][2]))) - (self._fConst176 * ((self._fConst175 * state["fRec40"][2]) + (self._fConst173 * state["fRec40"][1]))))) 
		fTemp52 = (self._fConst176 * (((self._fConst295 * state["fRec40"][0]) + (self._fConst296 * state["fRec40"][1])) + (self._fConst295 * state["fRec40"][2]))) 
		state["fRec39"] = state["fRec39"].at[0].set((fTemp52 - (self._fConst169 * ((self._fConst168 * state["fRec39"][2]) + (self._fConst167 * state["fRec39"][1]))))) 
		state["fRec38"] = state["fRec38"].at[0].set(((self._fConst169 * (((self._fConst298 * state["fRec39"][0]) + (self._fConst299 * state["fRec39"][1])) + (self._fConst298 * state["fRec39"][2]))) - (self._fConst166 * ((self._fConst165 * state["fRec38"][2]) + (self._fConst164 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst166 * (((self._fConst300 * state["fRec38"][0]) + (self._fConst301 * state["fRec38"][1])) + (self._fConst300 * state["fRec38"][2]))) - (self._fConst163 * ((self._fConst162 * state["fRec37"][2]) + (self._fConst160 * state["fRec37"][1]))))) 
		fTemp53 = (self._fConst163 * (((self._fConst302 * state["fRec37"][0]) + (self._fConst303 * state["fRec37"][1])) + (self._fConst302 * state["fRec37"][2]))) 
		state["fRec36"] = state["fRec36"].at[0].set((fTemp53 - (self._fConst156 * ((self._fConst155 * state["fRec36"][2]) + (self._fConst154 * state["fRec36"][1]))))) 
		state["fRec35"] = state["fRec35"].at[0].set(((self._fConst156 * (((self._fConst305 * state["fRec36"][0]) + (self._fConst306 * state["fRec36"][1])) + (self._fConst305 * state["fRec36"][2]))) - (self._fConst153 * ((self._fConst152 * state["fRec35"][2]) + (self._fConst151 * state["fRec35"][1]))))) 
		state["fRec34"] = state["fRec34"].at[0].set(((self._fConst153 * (((self._fConst307 * state["fRec35"][0]) + (self._fConst308 * state["fRec35"][1])) + (self._fConst307 * state["fRec35"][2]))) - (self._fConst150 * ((self._fConst149 * state["fRec34"][2]) + (self._fConst147 * state["fRec34"][1]))))) 
		fTemp54 = (self._fConst150 * (((self._fConst309 * state["fRec34"][0]) + (self._fConst310 * state["fRec34"][1])) + (self._fConst309 * state["fRec34"][2]))) 
		state["fRec33"] = state["fRec33"].at[0].set((fTemp54 - (self._fConst143 * ((self._fConst142 * state["fRec33"][2]) + (self._fConst141 * state["fRec33"][1]))))) 
		state["fRec32"] = state["fRec32"].at[0].set(((self._fConst143 * (((self._fConst312 * state["fRec33"][0]) + (self._fConst313 * state["fRec33"][1])) + (self._fConst312 * state["fRec33"][2]))) - (self._fConst140 * ((self._fConst139 * state["fRec32"][2]) + (self._fConst138 * state["fRec32"][1]))))) 
		state["fRec31"] = state["fRec31"].at[0].set(((self._fConst140 * (((self._fConst314 * state["fRec32"][0]) + (self._fConst315 * state["fRec32"][1])) + (self._fConst314 * state["fRec32"][2]))) - (self._fConst137 * ((self._fConst136 * state["fRec31"][2]) + (self._fConst134 * state["fRec31"][1]))))) 
		fTemp55 = (self._fConst137 * (((self._fConst316 * state["fRec31"][0]) + (self._fConst317 * state["fRec31"][1])) + (self._fConst316 * state["fRec31"][2]))) 
		state["fRec30"] = state["fRec30"].at[0].set((fTemp55 - (self._fConst130 * ((self._fConst129 * state["fRec30"][2]) + (self._fConst128 * state["fRec30"][1]))))) 
		state["fRec29"] = state["fRec29"].at[0].set(((self._fConst130 * (((self._fConst319 * state["fRec30"][0]) + (self._fConst320 * state["fRec30"][1])) + (self._fConst319 * state["fRec30"][2]))) - (self._fConst127 * ((self._fConst126 * state["fRec29"][2]) + (self._fConst125 * state["fRec29"][1]))))) 
		state["fRec28"] = state["fRec28"].at[0].set(((self._fConst127 * (((self._fConst321 * state["fRec29"][0]) + (self._fConst322 * state["fRec29"][1])) + (self._fConst321 * state["fRec29"][2]))) - (self._fConst124 * ((self._fConst123 * state["fRec28"][2]) + (self._fConst121 * state["fRec28"][1]))))) 
		fTemp56 = (self._fConst124 * (((self._fConst323 * state["fRec28"][0]) + (self._fConst324 * state["fRec28"][1])) + (self._fConst323 * state["fRec28"][2]))) 
		state["fRec27"] = state["fRec27"].at[0].set((fTemp56 - (self._fConst117 * ((self._fConst116 * state["fRec27"][2]) + (self._fConst115 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((self._fConst117 * (((self._fConst326 * state["fRec27"][0]) + (self._fConst327 * state["fRec27"][1])) + (self._fConst326 * state["fRec27"][2]))) - (self._fConst114 * ((self._fConst113 * state["fRec26"][2]) + (self._fConst112 * state["fRec26"][1]))))) 
		state["fRec25"] = state["fRec25"].at[0].set(((self._fConst114 * (((self._fConst328 * state["fRec26"][0]) + (self._fConst329 * state["fRec26"][1])) + (self._fConst328 * state["fRec26"][2]))) - (self._fConst111 * ((self._fConst110 * state["fRec25"][2]) + (self._fConst108 * state["fRec25"][1]))))) 
		fTemp57 = (self._fConst111 * (((self._fConst330 * state["fRec25"][0]) + (self._fConst331 * state["fRec25"][1])) + (self._fConst330 * state["fRec25"][2]))) 
		state["fRec24"] = state["fRec24"].at[0].set((fTemp57 - (self._fConst104 * ((self._fConst103 * state["fRec24"][2]) + (self._fConst102 * state["fRec24"][1]))))) 
		state["fRec23"] = state["fRec23"].at[0].set(((self._fConst104 * (((self._fConst333 * state["fRec24"][0]) + (self._fConst334 * state["fRec24"][1])) + (self._fConst333 * state["fRec24"][2]))) - (self._fConst101 * ((self._fConst100 * state["fRec23"][2]) + (self._fConst99 * state["fRec23"][1]))))) 
		state["fRec22"] = state["fRec22"].at[0].set(((self._fConst101 * (((self._fConst335 * state["fRec23"][0]) + (self._fConst336 * state["fRec23"][1])) + (self._fConst335 * state["fRec23"][2]))) - (self._fConst98 * ((self._fConst97 * state["fRec22"][2]) + (self._fConst95 * state["fRec22"][1]))))) 
		fTemp58 = (self._fConst98 * (((self._fConst337 * state["fRec22"][0]) + (self._fConst338 * state["fRec22"][1])) + (self._fConst337 * state["fRec22"][2]))) 
		state["fRec21"] = state["fRec21"].at[0].set((fTemp58 - (self._fConst91 * ((self._fConst90 * state["fRec21"][2]) + (self._fConst89 * state["fRec21"][1]))))) 
		state["fRec20"] = state["fRec20"].at[0].set(((self._fConst91 * (((self._fConst340 * state["fRec21"][0]) + (self._fConst341 * state["fRec21"][1])) + (self._fConst340 * state["fRec21"][2]))) - (self._fConst88 * ((self._fConst87 * state["fRec20"][2]) + (self._fConst86 * state["fRec20"][1]))))) 
		state["fRec19"] = state["fRec19"].at[0].set(((self._fConst88 * (((self._fConst342 * state["fRec20"][0]) + (self._fConst343 * state["fRec20"][1])) + (self._fConst342 * state["fRec20"][2]))) - (self._fConst85 * ((self._fConst84 * state["fRec19"][2]) + (self._fConst82 * state["fRec19"][1]))))) 
		fTemp59 = (self._fConst85 * (((self._fConst344 * state["fRec19"][0]) + (self._fConst345 * state["fRec19"][1])) + (self._fConst344 * state["fRec19"][2]))) 
		state["fRec18"] = state["fRec18"].at[0].set((fTemp59 - (self._fConst78 * ((self._fConst77 * state["fRec18"][2]) + (self._fConst76 * state["fRec18"][1]))))) 
		state["fRec17"] = state["fRec17"].at[0].set(((self._fConst78 * (((self._fConst347 * state["fRec18"][0]) + (self._fConst348 * state["fRec18"][1])) + (self._fConst347 * state["fRec18"][2]))) - (self._fConst75 * ((self._fConst74 * state["fRec17"][2]) + (self._fConst73 * state["fRec17"][1]))))) 
		state["fRec16"] = state["fRec16"].at[0].set(((self._fConst75 * (((self._fConst349 * state["fRec17"][0]) + (self._fConst350 * state["fRec17"][1])) + (self._fConst349 * state["fRec17"][2]))) - (self._fConst72 * ((self._fConst71 * state["fRec16"][2]) + (self._fConst69 * state["fRec16"][1]))))) 
		fTemp60 = (self._fConst72 * (((self._fConst351 * state["fRec16"][0]) + (self._fConst352 * state["fRec16"][1])) + (self._fConst351 * state["fRec16"][2]))) 
		state["fRec15"] = state["fRec15"].at[0].set((fTemp60 - (self._fConst65 * ((self._fConst64 * state["fRec15"][2]) + (self._fConst63 * state["fRec15"][1]))))) 
		state["fRec14"] = state["fRec14"].at[0].set(((self._fConst65 * (((self._fConst354 * state["fRec15"][0]) + (self._fConst355 * state["fRec15"][1])) + (self._fConst354 * state["fRec15"][2]))) - (self._fConst62 * ((self._fConst61 * state["fRec14"][2]) + (self._fConst60 * state["fRec14"][1]))))) 
		state["fRec13"] = state["fRec13"].at[0].set(((self._fConst62 * (((self._fConst356 * state["fRec14"][0]) + (self._fConst357 * state["fRec14"][1])) + (self._fConst356 * state["fRec14"][2]))) - (self._fConst59 * ((self._fConst58 * state["fRec13"][2]) + (self._fConst56 * state["fRec13"][1]))))) 
		fTemp61 = (self._fConst59 * (((self._fConst358 * state["fRec13"][0]) + (self._fConst359 * state["fRec13"][1])) + (self._fConst358 * state["fRec13"][2]))) 
		state["fRec12"] = state["fRec12"].at[0].set((fTemp61 - (self._fConst52 * ((self._fConst51 * state["fRec12"][2]) + (self._fConst50 * state["fRec12"][1]))))) 
		state["fRec11"] = state["fRec11"].at[0].set(((self._fConst52 * (((self._fConst361 * state["fRec12"][0]) + (self._fConst362 * state["fRec12"][1])) + (self._fConst361 * state["fRec12"][2]))) - (self._fConst49 * ((self._fConst48 * state["fRec11"][2]) + (self._fConst47 * state["fRec11"][1]))))) 
		state["fRec10"] = state["fRec10"].at[0].set(((self._fConst49 * (((self._fConst363 * state["fRec11"][0]) + (self._fConst364 * state["fRec11"][1])) + (self._fConst363 * state["fRec11"][2]))) - (self._fConst46 * ((self._fConst45 * state["fRec10"][2]) + (self._fConst43 * state["fRec10"][1]))))) 
		fTemp62 = (self._fConst46 * (((self._fConst365 * state["fRec10"][0]) + (self._fConst366 * state["fRec10"][1])) + (self._fConst365 * state["fRec10"][2]))) 
		state["fRec9"] = state["fRec9"].at[0].set((fTemp62 - (self._fConst39 * ((self._fConst38 * state["fRec9"][2]) + (self._fConst37 * state["fRec9"][1]))))) 
		state["fRec8"] = state["fRec8"].at[0].set(((self._fConst39 * (((self._fConst368 * state["fRec9"][0]) + (self._fConst369 * state["fRec9"][1])) + (self._fConst368 * state["fRec9"][2]))) - (self._fConst36 * ((self._fConst35 * state["fRec8"][2]) + (self._fConst34 * state["fRec8"][1]))))) 
		state["fRec7"] = state["fRec7"].at[0].set(((self._fConst36 * (((self._fConst370 * state["fRec8"][0]) + (self._fConst371 * state["fRec8"][1])) + (self._fConst370 * state["fRec8"][2]))) - (self._fConst33 * ((self._fConst32 * state["fRec7"][2]) + (self._fConst30 * state["fRec7"][1]))))) 
		fTemp63 = (self._fConst33 * (((self._fConst372 * state["fRec7"][0]) + (self._fConst373 * state["fRec7"][1])) + (self._fConst372 * state["fRec7"][2]))) 
		state["fRec6"] = state["fRec6"].at[0].set((fTemp63 - (self._fConst26 * ((self._fConst25 * state["fRec6"][2]) + (self._fConst24 * state["fRec6"][1]))))) 
		state["fRec5"] = state["fRec5"].at[0].set(((self._fConst26 * (((self._fConst375 * state["fRec6"][0]) + (self._fConst376 * state["fRec6"][1])) + (self._fConst375 * state["fRec6"][2]))) - (self._fConst23 * ((self._fConst22 * state["fRec5"][2]) + (self._fConst21 * state["fRec5"][1]))))) 
		state["fRec4"] = state["fRec4"].at[0].set(((self._fConst23 * (((self._fConst377 * state["fRec5"][0]) + (self._fConst378 * state["fRec5"][1])) + (self._fConst377 * state["fRec5"][2]))) - (self._fConst20 * ((self._fConst19 * state["fRec4"][2]) + (self._fConst17 * state["fRec4"][1]))))) 
		fTemp64 = (self._fConst20 * (((self._fConst379 * state["fRec4"][0]) + (self._fConst380 * state["fRec4"][1])) + (self._fConst379 * state["fRec4"][2]))) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp64 - (self._fConst13 * ((self._fConst12 * state["fRec3"][2]) + (self._fConst11 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst13 * (((self._fConst382 * state["fRec3"][0]) + (self._fConst383 * state["fRec3"][1])) + (self._fConst382 * state["fRec3"][2]))) - (self._fConst10 * ((self._fConst9 * state["fRec2"][2]) + (self._fConst8 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst10 * (((self._fConst384 * state["fRec2"][0]) + (self._fConst385 * state["fRec2"][1])) + (self._fConst384 * state["fRec2"][2]))) - (self._fConst7 * ((self._fConst6 * state["fRec1"][2]) + (self._fConst4 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fSlow38 * fRec0_temp) + (fSlow39 * jnp.abs((self._fConst7 * (((self._fConst386 * state["fRec1"][0]) + (self._fConst387 * state["fRec1"][1])) + (self._fConst386 * state["fRec1"][2])))))) 
		fVbargraph0 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec78"] = state["fRec78"].at[0].set((fTemp64 - (self._fConst397 * ((self._fConst396 * state["fRec78"][2]) + (self._fConst395 * state["fRec78"][1]))))) 
		state["fRec77"] = state["fRec77"].at[0].set(((self._fConst397 * (((self._fConst399 * state["fRec78"][0]) + (self._fConst400 * state["fRec78"][1])) + (self._fConst399 * state["fRec78"][2]))) - (self._fConst394 * ((self._fConst393 * state["fRec77"][2]) + (self._fConst392 * state["fRec77"][1]))))) 
		state["fRec76"] = state["fRec76"].at[0].set(((self._fConst394 * (((self._fConst402 * state["fRec77"][0]) + (self._fConst403 * state["fRec77"][1])) + (self._fConst402 * state["fRec77"][2]))) - (self._fConst391 * ((self._fConst390 * state["fRec76"][2]) + (self._fConst389 * state["fRec76"][1]))))) 
		state["fRec75"] = ((fSlow38 * fRec75_temp) + (fSlow39 * jnp.abs((self._fConst391 * (((self._fConst405 * state["fRec76"][0]) + (self._fConst406 * state["fRec76"][1])) + (self._fConst405 * state["fRec76"][2])))))) 
		fVbargraph1 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec75"])))
		self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec82"] = state["fRec82"].at[0].set((fTemp63 - (self._fConst415 * ((self._fConst414 * state["fRec82"][2]) + (self._fConst413 * state["fRec82"][1]))))) 
		state["fRec81"] = state["fRec81"].at[0].set(((self._fConst415 * (((self._fConst417 * state["fRec82"][0]) + (self._fConst418 * state["fRec82"][1])) + (self._fConst417 * state["fRec82"][2]))) - (self._fConst412 * ((self._fConst411 * state["fRec81"][2]) + (self._fConst410 * state["fRec81"][1]))))) 
		state["fRec80"] = state["fRec80"].at[0].set(((self._fConst412 * (((self._fConst420 * state["fRec81"][0]) + (self._fConst421 * state["fRec81"][1])) + (self._fConst420 * state["fRec81"][2]))) - (self._fConst409 * ((self._fConst408 * state["fRec80"][2]) + (self._fConst407 * state["fRec80"][1]))))) 
		state["fRec79"] = ((fSlow38 * fRec79_temp) + (fSlow39 * jnp.abs((self._fConst409 * (((self._fConst423 * state["fRec80"][0]) + (self._fConst424 * state["fRec80"][1])) + (self._fConst423 * state["fRec80"][2])))))) 
		fVbargraph2 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec79"])))
		self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec86"] = state["fRec86"].at[0].set((fTemp62 - (self._fConst433 * ((self._fConst432 * state["fRec86"][2]) + (self._fConst431 * state["fRec86"][1]))))) 
		state["fRec85"] = state["fRec85"].at[0].set(((self._fConst433 * (((self._fConst435 * state["fRec86"][0]) + (self._fConst436 * state["fRec86"][1])) + (self._fConst435 * state["fRec86"][2]))) - (self._fConst430 * ((self._fConst429 * state["fRec85"][2]) + (self._fConst428 * state["fRec85"][1]))))) 
		state["fRec84"] = state["fRec84"].at[0].set(((self._fConst430 * (((self._fConst438 * state["fRec85"][0]) + (self._fConst439 * state["fRec85"][1])) + (self._fConst438 * state["fRec85"][2]))) - (self._fConst427 * ((self._fConst426 * state["fRec84"][2]) + (self._fConst425 * state["fRec84"][1]))))) 
		state["fRec83"] = ((fSlow38 * fRec83_temp) + (fSlow39 * jnp.abs((self._fConst427 * (((self._fConst441 * state["fRec84"][0]) + (self._fConst442 * state["fRec84"][1])) + (self._fConst441 * state["fRec84"][2])))))) 
		fVbargraph3 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec83"])))
		self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec90"] = state["fRec90"].at[0].set((fTemp61 - (self._fConst451 * ((self._fConst450 * state["fRec90"][2]) + (self._fConst449 * state["fRec90"][1]))))) 
		state["fRec89"] = state["fRec89"].at[0].set(((self._fConst451 * (((self._fConst453 * state["fRec90"][0]) + (self._fConst454 * state["fRec90"][1])) + (self._fConst453 * state["fRec90"][2]))) - (self._fConst448 * ((self._fConst447 * state["fRec89"][2]) + (self._fConst446 * state["fRec89"][1]))))) 
		state["fRec88"] = state["fRec88"].at[0].set(((self._fConst448 * (((self._fConst456 * state["fRec89"][0]) + (self._fConst457 * state["fRec89"][1])) + (self._fConst456 * state["fRec89"][2]))) - (self._fConst445 * ((self._fConst444 * state["fRec88"][2]) + (self._fConst443 * state["fRec88"][1]))))) 
		state["fRec87"] = ((fSlow38 * fRec87_temp) + (fSlow39 * jnp.abs((self._fConst445 * (((self._fConst459 * state["fRec88"][0]) + (self._fConst460 * state["fRec88"][1])) + (self._fConst459 * state["fRec88"][2])))))) 
		fVbargraph4 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec87"])))
		self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec94"] = state["fRec94"].at[0].set((fTemp60 - (self._fConst469 * ((self._fConst468 * state["fRec94"][2]) + (self._fConst467 * state["fRec94"][1]))))) 
		state["fRec93"] = state["fRec93"].at[0].set(((self._fConst469 * (((self._fConst471 * state["fRec94"][0]) + (self._fConst472 * state["fRec94"][1])) + (self._fConst471 * state["fRec94"][2]))) - (self._fConst466 * ((self._fConst465 * state["fRec93"][2]) + (self._fConst464 * state["fRec93"][1]))))) 
		state["fRec92"] = state["fRec92"].at[0].set(((self._fConst466 * (((self._fConst474 * state["fRec93"][0]) + (self._fConst475 * state["fRec93"][1])) + (self._fConst474 * state["fRec93"][2]))) - (self._fConst463 * ((self._fConst462 * state["fRec92"][2]) + (self._fConst461 * state["fRec92"][1]))))) 
		state["fRec91"] = ((fSlow38 * fRec91_temp) + (fSlow39 * jnp.abs((self._fConst463 * (((self._fConst477 * state["fRec92"][0]) + (self._fConst478 * state["fRec92"][1])) + (self._fConst477 * state["fRec92"][2])))))) 
		fVbargraph5 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec91"])))
		self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec98"] = state["fRec98"].at[0].set((fTemp59 - (self._fConst487 * ((self._fConst486 * state["fRec98"][2]) + (self._fConst485 * state["fRec98"][1]))))) 
		state["fRec97"] = state["fRec97"].at[0].set(((self._fConst487 * (((self._fConst489 * state["fRec98"][0]) + (self._fConst490 * state["fRec98"][1])) + (self._fConst489 * state["fRec98"][2]))) - (self._fConst484 * ((self._fConst483 * state["fRec97"][2]) + (self._fConst482 * state["fRec97"][1]))))) 
		state["fRec96"] = state["fRec96"].at[0].set(((self._fConst484 * (((self._fConst492 * state["fRec97"][0]) + (self._fConst493 * state["fRec97"][1])) + (self._fConst492 * state["fRec97"][2]))) - (self._fConst481 * ((self._fConst480 * state["fRec96"][2]) + (self._fConst479 * state["fRec96"][1]))))) 
		state["fRec95"] = ((fSlow38 * fRec95_temp) + (fSlow39 * jnp.abs((self._fConst481 * (((self._fConst495 * state["fRec96"][0]) + (self._fConst496 * state["fRec96"][1])) + (self._fConst495 * state["fRec96"][2])))))) 
		fVbargraph6 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec95"])))
		self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec102"] = state["fRec102"].at[0].set((fTemp58 - (self._fConst505 * ((self._fConst504 * state["fRec102"][2]) + (self._fConst503 * state["fRec102"][1]))))) 
		state["fRec101"] = state["fRec101"].at[0].set(((self._fConst505 * (((self._fConst507 * state["fRec102"][0]) + (self._fConst508 * state["fRec102"][1])) + (self._fConst507 * state["fRec102"][2]))) - (self._fConst502 * ((self._fConst501 * state["fRec101"][2]) + (self._fConst500 * state["fRec101"][1]))))) 
		state["fRec100"] = state["fRec100"].at[0].set(((self._fConst502 * (((self._fConst510 * state["fRec101"][0]) + (self._fConst511 * state["fRec101"][1])) + (self._fConst510 * state["fRec101"][2]))) - (self._fConst499 * ((self._fConst498 * state["fRec100"][2]) + (self._fConst497 * state["fRec100"][1]))))) 
		state["fRec99"] = ((fSlow38 * fRec99_temp) + (fSlow39 * jnp.abs((self._fConst499 * (((self._fConst513 * state["fRec100"][0]) + (self._fConst514 * state["fRec100"][1])) + (self._fConst513 * state["fRec100"][2])))))) 
		fVbargraph7 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec99"])))
		self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec106"] = state["fRec106"].at[0].set((fTemp57 - (self._fConst523 * ((self._fConst522 * state["fRec106"][2]) + (self._fConst521 * state["fRec106"][1]))))) 
		state["fRec105"] = state["fRec105"].at[0].set(((self._fConst523 * (((self._fConst525 * state["fRec106"][0]) + (self._fConst526 * state["fRec106"][1])) + (self._fConst525 * state["fRec106"][2]))) - (self._fConst520 * ((self._fConst519 * state["fRec105"][2]) + (self._fConst518 * state["fRec105"][1]))))) 
		state["fRec104"] = state["fRec104"].at[0].set(((self._fConst520 * (((self._fConst528 * state["fRec105"][0]) + (self._fConst529 * state["fRec105"][1])) + (self._fConst528 * state["fRec105"][2]))) - (self._fConst517 * ((self._fConst516 * state["fRec104"][2]) + (self._fConst515 * state["fRec104"][1]))))) 
		state["fRec103"] = ((fSlow38 * fRec103_temp) + (fSlow39 * jnp.abs((self._fConst517 * (((self._fConst531 * state["fRec104"][0]) + (self._fConst532 * state["fRec104"][1])) + (self._fConst531 * state["fRec104"][2])))))) 
		fVbargraph8 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec103"])))
		self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec110"] = state["fRec110"].at[0].set((fTemp56 - (self._fConst541 * ((self._fConst540 * state["fRec110"][2]) + (self._fConst539 * state["fRec110"][1]))))) 
		state["fRec109"] = state["fRec109"].at[0].set(((self._fConst541 * (((self._fConst543 * state["fRec110"][0]) + (self._fConst544 * state["fRec110"][1])) + (self._fConst543 * state["fRec110"][2]))) - (self._fConst538 * ((self._fConst537 * state["fRec109"][2]) + (self._fConst536 * state["fRec109"][1]))))) 
		state["fRec108"] = state["fRec108"].at[0].set(((self._fConst538 * (((self._fConst546 * state["fRec109"][0]) + (self._fConst547 * state["fRec109"][1])) + (self._fConst546 * state["fRec109"][2]))) - (self._fConst535 * ((self._fConst534 * state["fRec108"][2]) + (self._fConst533 * state["fRec108"][1]))))) 
		state["fRec107"] = ((fSlow38 * fRec107_temp) + (fSlow39 * jnp.abs((self._fConst535 * (((self._fConst549 * state["fRec108"][0]) + (self._fConst550 * state["fRec108"][1])) + (self._fConst549 * state["fRec108"][2])))))) 
		fVbargraph9 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec107"])))
		self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec114"] = state["fRec114"].at[0].set((fTemp55 - (self._fConst559 * ((self._fConst558 * state["fRec114"][2]) + (self._fConst557 * state["fRec114"][1]))))) 
		state["fRec113"] = state["fRec113"].at[0].set(((self._fConst559 * (((self._fConst561 * state["fRec114"][0]) + (self._fConst562 * state["fRec114"][1])) + (self._fConst561 * state["fRec114"][2]))) - (self._fConst556 * ((self._fConst555 * state["fRec113"][2]) + (self._fConst554 * state["fRec113"][1]))))) 
		state["fRec112"] = state["fRec112"].at[0].set(((self._fConst556 * (((self._fConst564 * state["fRec113"][0]) + (self._fConst565 * state["fRec113"][1])) + (self._fConst564 * state["fRec113"][2]))) - (self._fConst553 * ((self._fConst552 * state["fRec112"][2]) + (self._fConst551 * state["fRec112"][1]))))) 
		state["fRec111"] = ((fSlow38 * fRec111_temp) + (fSlow39 * jnp.abs((self._fConst553 * (((self._fConst567 * state["fRec112"][0]) + (self._fConst568 * state["fRec112"][1])) + (self._fConst567 * state["fRec112"][2])))))) 
		fVbargraph10 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec111"])))
		self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec118"] = state["fRec118"].at[0].set((fTemp54 - (self._fConst577 * ((self._fConst576 * state["fRec118"][2]) + (self._fConst575 * state["fRec118"][1]))))) 
		state["fRec117"] = state["fRec117"].at[0].set(((self._fConst577 * (((self._fConst579 * state["fRec118"][0]) + (self._fConst580 * state["fRec118"][1])) + (self._fConst579 * state["fRec118"][2]))) - (self._fConst574 * ((self._fConst573 * state["fRec117"][2]) + (self._fConst572 * state["fRec117"][1]))))) 
		state["fRec116"] = state["fRec116"].at[0].set(((self._fConst574 * (((self._fConst582 * state["fRec117"][0]) + (self._fConst583 * state["fRec117"][1])) + (self._fConst582 * state["fRec117"][2]))) - (self._fConst571 * ((self._fConst570 * state["fRec116"][2]) + (self._fConst569 * state["fRec116"][1]))))) 
		state["fRec115"] = ((fSlow38 * fRec115_temp) + (fSlow39 * jnp.abs((self._fConst571 * (((self._fConst585 * state["fRec116"][0]) + (self._fConst586 * state["fRec116"][1])) + (self._fConst585 * state["fRec116"][2])))))) 
		fVbargraph11 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec115"])))
		self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec122"] = state["fRec122"].at[0].set((fTemp53 - (self._fConst595 * ((self._fConst594 * state["fRec122"][2]) + (self._fConst593 * state["fRec122"][1]))))) 
		state["fRec121"] = state["fRec121"].at[0].set(((self._fConst595 * (((self._fConst597 * state["fRec122"][0]) + (self._fConst598 * state["fRec122"][1])) + (self._fConst597 * state["fRec122"][2]))) - (self._fConst592 * ((self._fConst591 * state["fRec121"][2]) + (self._fConst590 * state["fRec121"][1]))))) 
		state["fRec120"] = state["fRec120"].at[0].set(((self._fConst592 * (((self._fConst600 * state["fRec121"][0]) + (self._fConst601 * state["fRec121"][1])) + (self._fConst600 * state["fRec121"][2]))) - (self._fConst589 * ((self._fConst588 * state["fRec120"][2]) + (self._fConst587 * state["fRec120"][1]))))) 
		state["fRec119"] = ((fSlow38 * fRec119_temp) + (fSlow39 * jnp.abs((self._fConst589 * (((self._fConst603 * state["fRec120"][0]) + (self._fConst604 * state["fRec120"][1])) + (self._fConst603 * state["fRec120"][2])))))) 
		fVbargraph12 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec119"])))
		self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec126"] = state["fRec126"].at[0].set((fTemp52 - (self._fConst613 * ((self._fConst612 * state["fRec126"][2]) + (self._fConst611 * state["fRec126"][1]))))) 
		state["fRec125"] = state["fRec125"].at[0].set(((self._fConst613 * (((self._fConst615 * state["fRec126"][0]) + (self._fConst616 * state["fRec126"][1])) + (self._fConst615 * state["fRec126"][2]))) - (self._fConst610 * ((self._fConst609 * state["fRec125"][2]) + (self._fConst608 * state["fRec125"][1]))))) 
		state["fRec124"] = state["fRec124"].at[0].set(((self._fConst610 * (((self._fConst618 * state["fRec125"][0]) + (self._fConst619 * state["fRec125"][1])) + (self._fConst618 * state["fRec125"][2]))) - (self._fConst607 * ((self._fConst606 * state["fRec124"][2]) + (self._fConst605 * state["fRec124"][1]))))) 
		state["fRec123"] = ((fSlow38 * fRec123_temp) + (fSlow39 * jnp.abs((self._fConst607 * (((self._fConst621 * state["fRec124"][0]) + (self._fConst622 * state["fRec124"][1])) + (self._fConst621 * state["fRec124"][2])))))) 
		fVbargraph13 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec123"])))
		self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec130"] = state["fRec130"].at[0].set((fTemp51 - (self._fConst631 * ((self._fConst630 * state["fRec130"][2]) + (self._fConst629 * state["fRec130"][1]))))) 
		state["fRec129"] = state["fRec129"].at[0].set(((self._fConst631 * (((self._fConst633 * state["fRec130"][0]) + (self._fConst634 * state["fRec130"][1])) + (self._fConst633 * state["fRec130"][2]))) - (self._fConst628 * ((self._fConst627 * state["fRec129"][2]) + (self._fConst626 * state["fRec129"][1]))))) 
		state["fRec128"] = state["fRec128"].at[0].set(((self._fConst628 * (((self._fConst636 * state["fRec129"][0]) + (self._fConst637 * state["fRec129"][1])) + (self._fConst636 * state["fRec129"][2]))) - (self._fConst625 * ((self._fConst624 * state["fRec128"][2]) + (self._fConst623 * state["fRec128"][1]))))) 
		state["fRec127"] = ((fSlow38 * fRec127_temp) + (fSlow39 * jnp.abs((self._fConst625 * (((self._fConst639 * state["fRec128"][0]) + (self._fConst640 * state["fRec128"][1])) + (self._fConst639 * state["fRec128"][2])))))) 
		fVbargraph14 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec127"])))
		self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		state["fRec134"] = state["fRec134"].at[0].set((fTemp50 - (self._fConst649 * ((self._fConst648 * state["fRec134"][2]) + (self._fConst647 * state["fRec134"][1]))))) 
		state["fRec133"] = state["fRec133"].at[0].set(((self._fConst649 * (((self._fConst651 * state["fRec134"][0]) + (self._fConst652 * state["fRec134"][1])) + (self._fConst651 * state["fRec134"][2]))) - (self._fConst646 * ((self._fConst645 * state["fRec133"][2]) + (self._fConst644 * state["fRec133"][1]))))) 
		state["fRec132"] = state["fRec132"].at[0].set(((self._fConst646 * (((self._fConst654 * state["fRec133"][0]) + (self._fConst655 * state["fRec133"][1])) + (self._fConst654 * state["fRec133"][2]))) - (self._fConst643 * ((self._fConst642 * state["fRec132"][2]) + (self._fConst641 * state["fRec132"][1]))))) 
		state["fRec131"] = ((fSlow38 * fRec131_temp) + (fSlow39 * jnp.abs((self._fConst643 * (((self._fConst657 * state["fRec132"][0]) + (self._fConst658 * state["fRec132"][1])) + (self._fConst657 * state["fRec132"][2])))))) 
		fVbargraph15 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec131"])))
		self.sow("intermediates", "fVbargraph15", fVbargraph15) 
		state["fRec138"] = state["fRec138"].at[0].set((fTemp49 - (self._fConst667 * ((self._fConst666 * state["fRec138"][2]) + (self._fConst665 * state["fRec138"][1]))))) 
		state["fRec137"] = state["fRec137"].at[0].set(((self._fConst667 * (((self._fConst669 * state["fRec138"][0]) + (self._fConst670 * state["fRec138"][1])) + (self._fConst669 * state["fRec138"][2]))) - (self._fConst664 * ((self._fConst663 * state["fRec137"][2]) + (self._fConst662 * state["fRec137"][1]))))) 
		state["fRec136"] = state["fRec136"].at[0].set(((self._fConst664 * (((self._fConst672 * state["fRec137"][0]) + (self._fConst673 * state["fRec137"][1])) + (self._fConst672 * state["fRec137"][2]))) - (self._fConst661 * ((self._fConst660 * state["fRec136"][2]) + (self._fConst659 * state["fRec136"][1]))))) 
		state["fRec135"] = ((fSlow38 * fRec135_temp) + (fSlow39 * jnp.abs((self._fConst661 * (((self._fConst675 * state["fRec136"][0]) + (self._fConst676 * state["fRec136"][1])) + (self._fConst675 * state["fRec136"][2])))))) 
		fVbargraph16 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec135"])))
		self.sow("intermediates", "fVbargraph16", fVbargraph16) 
		state["fRec142"] = state["fRec142"].at[0].set((fTemp48 - (self._fConst685 * ((self._fConst684 * state["fRec142"][2]) + (self._fConst683 * state["fRec142"][1]))))) 
		state["fRec141"] = state["fRec141"].at[0].set(((self._fConst685 * (((self._fConst687 * state["fRec142"][0]) + (self._fConst688 * state["fRec142"][1])) + (self._fConst687 * state["fRec142"][2]))) - (self._fConst682 * ((self._fConst681 * state["fRec141"][2]) + (self._fConst680 * state["fRec141"][1]))))) 
		state["fRec140"] = state["fRec140"].at[0].set(((self._fConst682 * (((self._fConst690 * state["fRec141"][0]) + (self._fConst691 * state["fRec141"][1])) + (self._fConst690 * state["fRec141"][2]))) - (self._fConst679 * ((self._fConst678 * state["fRec140"][2]) + (self._fConst677 * state["fRec140"][1]))))) 
		state["fRec139"] = ((fSlow38 * fRec139_temp) + (fSlow39 * jnp.abs((self._fConst679 * (((self._fConst693 * state["fRec140"][0]) + (self._fConst694 * state["fRec140"][1])) + (self._fConst693 * state["fRec140"][2])))))) 
		fVbargraph17 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec139"])))
		self.sow("intermediates", "fVbargraph17", fVbargraph17) 
		state["fRec146"] = state["fRec146"].at[0].set((fTemp47 - (self._fConst703 * ((self._fConst702 * state["fRec146"][2]) + (self._fConst701 * state["fRec146"][1]))))) 
		state["fRec145"] = state["fRec145"].at[0].set(((self._fConst703 * (((self._fConst705 * state["fRec146"][0]) + (self._fConst706 * state["fRec146"][1])) + (self._fConst705 * state["fRec146"][2]))) - (self._fConst700 * ((self._fConst699 * state["fRec145"][2]) + (self._fConst698 * state["fRec145"][1]))))) 
		state["fRec144"] = state["fRec144"].at[0].set(((self._fConst700 * (((self._fConst708 * state["fRec145"][0]) + (self._fConst709 * state["fRec145"][1])) + (self._fConst708 * state["fRec145"][2]))) - (self._fConst697 * ((self._fConst696 * state["fRec144"][2]) + (self._fConst695 * state["fRec144"][1]))))) 
		state["fRec143"] = ((fSlow38 * fRec143_temp) + (fSlow39 * jnp.abs((self._fConst697 * (((self._fConst711 * state["fRec144"][0]) + (self._fConst712 * state["fRec144"][1])) + (self._fConst711 * state["fRec144"][2])))))) 
		fVbargraph18 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec143"])))
		self.sow("intermediates", "fVbargraph18", fVbargraph18) 
		state["fRec150"] = state["fRec150"].at[0].set((fTemp46 - (self._fConst721 * ((self._fConst720 * state["fRec150"][2]) + (self._fConst719 * state["fRec150"][1]))))) 
		state["fRec149"] = state["fRec149"].at[0].set(((self._fConst721 * (((self._fConst723 * state["fRec150"][0]) + (self._fConst724 * state["fRec150"][1])) + (self._fConst723 * state["fRec150"][2]))) - (self._fConst718 * ((self._fConst717 * state["fRec149"][2]) + (self._fConst716 * state["fRec149"][1]))))) 
		state["fRec148"] = state["fRec148"].at[0].set(((self._fConst718 * (((self._fConst726 * state["fRec149"][0]) + (self._fConst727 * state["fRec149"][1])) + (self._fConst726 * state["fRec149"][2]))) - (self._fConst715 * ((self._fConst714 * state["fRec148"][2]) + (self._fConst713 * state["fRec148"][1]))))) 
		state["fRec147"] = ((fRec147_temp * fSlow38) + (jnp.abs((self._fConst715 * (((self._fConst729 * state["fRec148"][0]) + (self._fConst730 * state["fRec148"][1])) + (self._fConst729 * state["fRec148"][2])))) * fSlow39)) 
		fVbargraph19 = (fSlow40 + (jnp.float32(2e+01) * jnp.log10(state["fRec147"])))
		self.sow("intermediates", "fVbargraph19", fVbargraph19) 
		fTemp65 = fTemp46 
		_result0 = fTemp65 
		_result1 = fTemp65 
		state["iVec0"] = jnp.roll(state["iVec0"], 1) 
		state["fRec70"] = jnp.roll(state["fRec70"], 1) 
		state["fRec62"] = jnp.roll(state["fRec62"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec61"] = jnp.roll(state["fRec61"], 1) 
		state["fRec58"] = jnp.roll(state["fRec58"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec57"] = jnp.roll(state["fRec57"], 1) 
		state["fRec56"] = jnp.roll(state["fRec56"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
		state["fRec54"] = jnp.roll(state["fRec54"], 1) 
		state["fRec53"] = jnp.roll(state["fRec53"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec50"] = jnp.roll(state["fRec50"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec48"] = jnp.roll(state["fRec48"], 1) 
		state["fRec47"] = jnp.roll(state["fRec47"], 1) 
		state["fRec46"] = jnp.roll(state["fRec46"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
		state["fRec43"] = jnp.roll(state["fRec43"], 1) 
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
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec76"] = jnp.roll(state["fRec76"], 1) 
		state["fRec82"] = jnp.roll(state["fRec82"], 1) 
		state["fRec81"] = jnp.roll(state["fRec81"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec84"] = jnp.roll(state["fRec84"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec88"] = jnp.roll(state["fRec88"], 1) 
		state["fRec94"] = jnp.roll(state["fRec94"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec92"] = jnp.roll(state["fRec92"], 1) 
		state["fRec98"] = jnp.roll(state["fRec98"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec96"] = jnp.roll(state["fRec96"], 1) 
		state["fRec102"] = jnp.roll(state["fRec102"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec110"] = jnp.roll(state["fRec110"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec114"] = jnp.roll(state["fRec114"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec112"] = jnp.roll(state["fRec112"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec116"] = jnp.roll(state["fRec116"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec120"] = jnp.roll(state["fRec120"], 1) 
		state["fRec126"] = jnp.roll(state["fRec126"], 1) 
		state["fRec125"] = jnp.roll(state["fRec125"], 1) 
		state["fRec124"] = jnp.roll(state["fRec124"], 1) 
		state["fRec130"] = jnp.roll(state["fRec130"], 1) 
		state["fRec129"] = jnp.roll(state["fRec129"], 1) 
		state["fRec128"] = jnp.roll(state["fRec128"], 1) 
		state["fRec134"] = jnp.roll(state["fRec134"], 1) 
		state["fRec133"] = jnp.roll(state["fRec133"], 1) 
		state["fRec132"] = jnp.roll(state["fRec132"], 1) 
		state["fRec138"] = jnp.roll(state["fRec138"], 1) 
		state["fRec137"] = jnp.roll(state["fRec137"], 1) 
		state["fRec136"] = jnp.roll(state["fRec136"], 1) 
		state["fRec142"] = jnp.roll(state["fRec142"], 1) 
		state["fRec141"] = jnp.roll(state["fRec141"], 1) 
		state["fRec140"] = jnp.roll(state["fRec140"], 1) 
		state["fRec146"] = jnp.roll(state["fRec146"], 1) 
		state["fRec145"] = jnp.roll(state["fRec145"], 1) 
		state["fRec144"] = jnp.roll(state["fRec144"], 1) 
		state["fRec150"] = jnp.roll(state["fRec150"], 1) 
		state["fRec149"] = jnp.roll(state["fRec149"], 1) 
		state["fRec148"] = jnp.roll(state["fRec148"], 1) 
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
