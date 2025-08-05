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
		ui_path.append("parametric_eq") 
		ui_path.append("0x00") 
		ui_path.append("SAWTOOTH OSCILLATOR") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider0", ui_path, "Amplitude", -2e+01, -1.2e+02, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider2", ui_path, "Frequency", 49.0, 1.0, 88.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider3", ui_path, "Detuning 1", -0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider4", ui_path, "Detuning 2", 0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider1", ui_path, "Portamento", 0.1, 0.001, 1e+01, unnorm_funcs, "log") 
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
		self.add_hslider("fHslider4", ui_path, "Low Boost|Cut", 0.0, -4e+01, 4e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider3", ui_path, "Transition Frequency", 2e+02, 1.0, 5e+03, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("Peaking Equalizer") 
		self.add_hslider("fHslider6", ui_path, "Peak Boost|Cut", 0.0, -4e+01, 4e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider5", ui_path, "Peak Frequency", 49.0, 1.0, 1e+02, unnorm_funcs, "linear") 
		self.add_hslider("fHslider7", ui_path, "Peak Q", 4e+01, 1.0, 1e+03, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.append("High Shelf") 
		self.add_hslider("fHslider8", ui_path, "High Boost|Cut", 0.0, -4e+01, 4e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "Transition Frequency", 8e+03, 2e+01, 1e+04, unnorm_funcs, "log") 
		ui_path.pop()
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("CONSTANT-Q SPECTRUM ANALYZER (6E), 20 bands spanning LP, 9 octaves below 16000 Hz, HP") 
		self.add_vbargraph("fVbargraph19", ui_path, "vbargraph0", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph18", ui_path, "vbargraph1", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph17", ui_path, "vbargraph2", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph16", ui_path, "vbargraph3", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph15", ui_path, "vbargraph4", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph14", ui_path, "vbargraph5", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph13", ui_path, "vbargraph6", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph12", ui_path, "vbargraph7", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph11", ui_path, "vbargraph8", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph10", ui_path, "vbargraph9", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph9", ui_path, "vbargraph10", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph8", ui_path, "vbargraph11", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph7", ui_path, "vbargraph12", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph6", ui_path, "vbargraph13", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph5", ui_path, "vbargraph14", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph4", ui_path, "vbargraph15", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph3", ui_path, "vbargraph16", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph2", ui_path, "vbargraph17", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph1", ui_path, "vbargraph18", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph0", ui_path, "vbargraph19", -5e+01, 1e+01, unnorm_funcs) 
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
		self._fConst14 = (np.float32(3.1415927) / self._fConst0) 
		self._fConst15 = (np.float32(1.0) / self._fConst0) 
		self._fConst16 = (np.float32(0.25) * self._fConst0) 
		self._fConst17 = (np.float32(0.041666668) * np.power(self._fConst0, np.float32(2.0))) 
		self._fConst18 = (np.float32(0.0052083335) * np.power(self._fConst0, np.float32(3.0))) 
		self._fConst19 = (np.float32(1382.3008) / self._fConst0) 
		self._fConst20 = (np.float32(2764.6016) / self._fConst0) 
		self._fConst21 = (((self._fConst3 + np.float32(-3.1897273)) / self._fConst2) + np.float32(4.0767817)) 
		self._fConst22 = (np.float32(1.0) / self._fConst5) 
		self._fConst23 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst22)) 
		self._fConst24 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst12)) 
		self._fConst25 = (((self._fConst3 + np.float32(-0.74313045)) / self._fConst2) + np.float32(1.4500711)) 
		self._fConst26 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst22)) 
		self._fConst27 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst9)) 
		self._fConst28 = (((self._fConst3 + np.float32(-0.15748216)) / self._fConst2) + np.float32(0.9351402)) 
		self._fConst29 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst22)) 
		self._fConst30 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst6)) 
		self._fConst31 = np.tan((np.float32(35543.062) / self._fConst0)) 
		self._fConst32 = (np.float32(1.0) / self._fConst31) 
		self._fConst33 = (np.float32(1.0) / (((self._fConst32 + np.float32(0.15748216)) / self._fConst31) + np.float32(0.9351402))) 
		self._fConst34 = np.power(self._fConst31, np.float32(2.0)) 
		self._fConst35 = (np.float32(50.06381) / self._fConst34) 
		self._fConst36 = (self._fConst35 + np.float32(0.9351402)) 
		self._fConst37 = (np.float32(1.0) / (((self._fConst32 + np.float32(0.74313045)) / self._fConst31) + np.float32(1.4500711))) 
		self._fConst38 = (np.float32(11.0520525) / self._fConst34) 
		self._fConst39 = (self._fConst38 + np.float32(1.4500711)) 
		self._fConst40 = (np.float32(1.0) / (((self._fConst32 + np.float32(3.1897273)) / self._fConst31) + np.float32(4.0767817))) 
		self._fConst41 = (np.float32(0.0017661728) / self._fConst34) 
		self._fConst42 = (self._fConst41 + np.float32(0.0004076782)) 
		self._fConst43 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.16840488)) / self._fConst2) + np.float32(1.0693583))) 
		self._fConst44 = (self._fConst22 + np.float32(53.53615)) 
		self._fConst45 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.51247865)) / self._fConst2) + np.float32(0.6896214))) 
		self._fConst46 = (self._fConst22 + np.float32(7.6217313)) 
		self._fConst47 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.78241307)) / self._fConst2) + np.float32(0.2452915))) 
		self._fConst48 = (np.float32(0.0001) / self._fConst5) 
		self._fConst49 = (self._fConst48 + np.float32(0.0004332272)) 
		self._fConst50 = (((self._fConst3 + np.float32(-0.78241307)) / self._fConst2) + np.float32(0.2452915)) 
		self._fConst51 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst22)) 
		self._fConst52 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst48)) 
		self._fConst53 = (((self._fConst3 + np.float32(-0.51247865)) / self._fConst2) + np.float32(0.6896214)) 
		self._fConst54 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst22)) 
		self._fConst55 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst22)) 
		self._fConst56 = (((self._fConst3 + np.float32(-0.16840488)) / self._fConst2) + np.float32(1.0693583)) 
		self._fConst57 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst22)) 
		self._fConst58 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst22)) 
		self._fConst59 = (((self._fConst32 + np.float32(-3.1897273)) / self._fConst31) + np.float32(4.0767817)) 
		self._fConst60 = (np.float32(1.0) / self._fConst34) 
		self._fConst61 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst60)) 
		self._fConst62 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst41)) 
		self._fConst63 = (((self._fConst32 + np.float32(-0.74313045)) / self._fConst31) + np.float32(1.4500711)) 
		self._fConst64 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst60)) 
		self._fConst65 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst38)) 
		self._fConst66 = (((self._fConst32 + np.float32(-0.15748216)) / self._fConst31) + np.float32(0.9351402)) 
		self._fConst67 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst60)) 
		self._fConst68 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst35)) 
		self._fConst69 = np.tan((np.float32(25132.742) / self._fConst0)) 
		self._fConst70 = (np.float32(1.0) / self._fConst69) 
		self._fConst71 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.15748216)) / self._fConst69) + np.float32(0.9351402))) 
		self._fConst72 = np.power(self._fConst69, np.float32(2.0)) 
		self._fConst73 = (np.float32(50.06381) / self._fConst72) 
		self._fConst74 = (self._fConst73 + np.float32(0.9351402)) 
		self._fConst75 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.74313045)) / self._fConst69) + np.float32(1.4500711))) 
		self._fConst76 = (np.float32(11.0520525) / self._fConst72) 
		self._fConst77 = (self._fConst76 + np.float32(1.4500711)) 
		self._fConst78 = (np.float32(1.0) / (((self._fConst70 + np.float32(3.1897273)) / self._fConst69) + np.float32(4.0767817))) 
		self._fConst79 = (np.float32(0.0017661728) / self._fConst72) 
		self._fConst80 = (self._fConst79 + np.float32(0.0004076782)) 
		self._fConst81 = (np.float32(1.0) / (((self._fConst32 + np.float32(0.16840488)) / self._fConst31) + np.float32(1.0693583))) 
		self._fConst82 = (self._fConst60 + np.float32(53.53615)) 
		self._fConst83 = (np.float32(1.0) / (((self._fConst32 + np.float32(0.51247865)) / self._fConst31) + np.float32(0.6896214))) 
		self._fConst84 = (self._fConst60 + np.float32(7.6217313)) 
		self._fConst85 = (np.float32(1.0) / (((self._fConst32 + np.float32(0.78241307)) / self._fConst31) + np.float32(0.2452915))) 
		self._fConst86 = (np.float32(0.0001) / self._fConst34) 
		self._fConst87 = (self._fConst86 + np.float32(0.0004332272)) 
		self._fConst88 = (((self._fConst32 + np.float32(-0.78241307)) / self._fConst31) + np.float32(0.2452915)) 
		self._fConst89 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst60)) 
		self._fConst90 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst86)) 
		self._fConst91 = (((self._fConst32 + np.float32(-0.51247865)) / self._fConst31) + np.float32(0.6896214)) 
		self._fConst92 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst60)) 
		self._fConst93 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst60)) 
		self._fConst94 = (((self._fConst32 + np.float32(-0.16840488)) / self._fConst31) + np.float32(1.0693583)) 
		self._fConst95 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst60)) 
		self._fConst96 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst60)) 
		self._fConst97 = (((self._fConst70 + np.float32(-3.1897273)) / self._fConst69) + np.float32(4.0767817)) 
		self._fConst98 = (np.float32(1.0) / self._fConst72) 
		self._fConst99 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst98)) 
		self._fConst100 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst79)) 
		self._fConst101 = (((self._fConst70 + np.float32(-0.74313045)) / self._fConst69) + np.float32(1.4500711)) 
		self._fConst102 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst98)) 
		self._fConst103 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst76)) 
		self._fConst104 = (((self._fConst70 + np.float32(-0.15748216)) / self._fConst69) + np.float32(0.9351402)) 
		self._fConst105 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst98)) 
		self._fConst106 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst73)) 
		self._fConst107 = np.tan((np.float32(17771.531) / self._fConst0)) 
		self._fConst108 = (np.float32(1.0) / self._fConst107) 
		self._fConst109 = (np.float32(1.0) / (((self._fConst108 + np.float32(0.15748216)) / self._fConst107) + np.float32(0.9351402))) 
		self._fConst110 = np.power(self._fConst107, np.float32(2.0)) 
		self._fConst111 = (np.float32(50.06381) / self._fConst110) 
		self._fConst112 = (self._fConst111 + np.float32(0.9351402)) 
		self._fConst113 = (np.float32(1.0) / (((self._fConst108 + np.float32(0.74313045)) / self._fConst107) + np.float32(1.4500711))) 
		self._fConst114 = (np.float32(11.0520525) / self._fConst110) 
		self._fConst115 = (self._fConst114 + np.float32(1.4500711)) 
		self._fConst116 = (np.float32(1.0) / (((self._fConst108 + np.float32(3.1897273)) / self._fConst107) + np.float32(4.0767817))) 
		self._fConst117 = (np.float32(0.0017661728) / self._fConst110) 
		self._fConst118 = (self._fConst117 + np.float32(0.0004076782)) 
		self._fConst119 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.16840488)) / self._fConst69) + np.float32(1.0693583))) 
		self._fConst120 = (self._fConst98 + np.float32(53.53615)) 
		self._fConst121 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.51247865)) / self._fConst69) + np.float32(0.6896214))) 
		self._fConst122 = (self._fConst98 + np.float32(7.6217313)) 
		self._fConst123 = (np.float32(1.0) / (((self._fConst70 + np.float32(0.78241307)) / self._fConst69) + np.float32(0.2452915))) 
		self._fConst124 = (np.float32(0.0001) / self._fConst72) 
		self._fConst125 = (self._fConst124 + np.float32(0.0004332272)) 
		self._fConst126 = (((self._fConst70 + np.float32(-0.78241307)) / self._fConst69) + np.float32(0.2452915)) 
		self._fConst127 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst98)) 
		self._fConst128 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst124)) 
		self._fConst129 = (((self._fConst70 + np.float32(-0.51247865)) / self._fConst69) + np.float32(0.6896214)) 
		self._fConst130 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst98)) 
		self._fConst131 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst98)) 
		self._fConst132 = (((self._fConst70 + np.float32(-0.16840488)) / self._fConst69) + np.float32(1.0693583)) 
		self._fConst133 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst98)) 
		self._fConst134 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst98)) 
		self._fConst135 = (((self._fConst108 + np.float32(-3.1897273)) / self._fConst107) + np.float32(4.0767817)) 
		self._fConst136 = (np.float32(1.0) / self._fConst110) 
		self._fConst137 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst136)) 
		self._fConst138 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst117)) 
		self._fConst139 = (((self._fConst108 + np.float32(-0.74313045)) / self._fConst107) + np.float32(1.4500711)) 
		self._fConst140 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst136)) 
		self._fConst141 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst114)) 
		self._fConst142 = (((self._fConst108 + np.float32(-0.15748216)) / self._fConst107) + np.float32(0.9351402)) 
		self._fConst143 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst136)) 
		self._fConst144 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst111)) 
		self._fConst145 = np.tan((np.float32(12566.371) / self._fConst0)) 
		self._fConst146 = (np.float32(1.0) / self._fConst145) 
		self._fConst147 = (np.float32(1.0) / (((self._fConst146 + np.float32(0.15748216)) / self._fConst145) + np.float32(0.9351402))) 
		self._fConst148 = np.power(self._fConst145, np.float32(2.0)) 
		self._fConst149 = (np.float32(50.06381) / self._fConst148) 
		self._fConst150 = (self._fConst149 + np.float32(0.9351402)) 
		self._fConst151 = (np.float32(1.0) / (((self._fConst146 + np.float32(0.74313045)) / self._fConst145) + np.float32(1.4500711))) 
		self._fConst152 = (np.float32(11.0520525) / self._fConst148) 
		self._fConst153 = (self._fConst152 + np.float32(1.4500711)) 
		self._fConst154 = (np.float32(1.0) / (((self._fConst146 + np.float32(3.1897273)) / self._fConst145) + np.float32(4.0767817))) 
		self._fConst155 = (np.float32(0.0017661728) / self._fConst148) 
		self._fConst156 = (self._fConst155 + np.float32(0.0004076782)) 
		self._fConst157 = (np.float32(1.0) / (((self._fConst108 + np.float32(0.16840488)) / self._fConst107) + np.float32(1.0693583))) 
		self._fConst158 = (self._fConst136 + np.float32(53.53615)) 
		self._fConst159 = (np.float32(1.0) / (((self._fConst108 + np.float32(0.51247865)) / self._fConst107) + np.float32(0.6896214))) 
		self._fConst160 = (self._fConst136 + np.float32(7.6217313)) 
		self._fConst161 = (np.float32(1.0) / (((self._fConst108 + np.float32(0.78241307)) / self._fConst107) + np.float32(0.2452915))) 
		self._fConst162 = (np.float32(0.0001) / self._fConst110) 
		self._fConst163 = (self._fConst162 + np.float32(0.0004332272)) 
		self._fConst164 = (((self._fConst108 + np.float32(-0.78241307)) / self._fConst107) + np.float32(0.2452915)) 
		self._fConst165 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst136)) 
		self._fConst166 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst162)) 
		self._fConst167 = (((self._fConst108 + np.float32(-0.51247865)) / self._fConst107) + np.float32(0.6896214)) 
		self._fConst168 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst136)) 
		self._fConst169 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst136)) 
		self._fConst170 = (((self._fConst108 + np.float32(-0.16840488)) / self._fConst107) + np.float32(1.0693583)) 
		self._fConst171 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst136)) 
		self._fConst172 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst136)) 
		self._fConst173 = (((self._fConst146 + np.float32(-3.1897273)) / self._fConst145) + np.float32(4.0767817)) 
		self._fConst174 = (np.float32(1.0) / self._fConst148) 
		self._fConst175 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst174)) 
		self._fConst176 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst155)) 
		self._fConst177 = (((self._fConst146 + np.float32(-0.74313045)) / self._fConst145) + np.float32(1.4500711)) 
		self._fConst178 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst174)) 
		self._fConst179 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst152)) 
		self._fConst180 = (((self._fConst146 + np.float32(-0.15748216)) / self._fConst145) + np.float32(0.9351402)) 
		self._fConst181 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst174)) 
		self._fConst182 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst149)) 
		self._fConst183 = np.tan((np.float32(8885.766) / self._fConst0)) 
		self._fConst184 = (np.float32(1.0) / self._fConst183) 
		self._fConst185 = (np.float32(1.0) / (((self._fConst184 + np.float32(0.15748216)) / self._fConst183) + np.float32(0.9351402))) 
		self._fConst186 = np.power(self._fConst183, np.float32(2.0)) 
		self._fConst187 = (np.float32(50.06381) / self._fConst186) 
		self._fConst188 = (self._fConst187 + np.float32(0.9351402)) 
		self._fConst189 = (np.float32(1.0) / (((self._fConst184 + np.float32(0.74313045)) / self._fConst183) + np.float32(1.4500711))) 
		self._fConst190 = (np.float32(11.0520525) / self._fConst186) 
		self._fConst191 = (self._fConst190 + np.float32(1.4500711)) 
		self._fConst192 = (np.float32(1.0) / (((self._fConst184 + np.float32(3.1897273)) / self._fConst183) + np.float32(4.0767817))) 
		self._fConst193 = (np.float32(0.0017661728) / self._fConst186) 
		self._fConst194 = (self._fConst193 + np.float32(0.0004076782)) 
		self._fConst195 = (np.float32(1.0) / (((self._fConst146 + np.float32(0.16840488)) / self._fConst145) + np.float32(1.0693583))) 
		self._fConst196 = (self._fConst174 + np.float32(53.53615)) 
		self._fConst197 = (np.float32(1.0) / (((self._fConst146 + np.float32(0.51247865)) / self._fConst145) + np.float32(0.6896214))) 
		self._fConst198 = (self._fConst174 + np.float32(7.6217313)) 
		self._fConst199 = (np.float32(1.0) / (((self._fConst146 + np.float32(0.78241307)) / self._fConst145) + np.float32(0.2452915))) 
		self._fConst200 = (np.float32(0.0001) / self._fConst148) 
		self._fConst201 = (self._fConst200 + np.float32(0.0004332272)) 
		self._fConst202 = (((self._fConst146 + np.float32(-0.78241307)) / self._fConst145) + np.float32(0.2452915)) 
		self._fConst203 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst174)) 
		self._fConst204 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst200)) 
		self._fConst205 = (((self._fConst146 + np.float32(-0.51247865)) / self._fConst145) + np.float32(0.6896214)) 
		self._fConst206 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst174)) 
		self._fConst207 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst174)) 
		self._fConst208 = (((self._fConst146 + np.float32(-0.16840488)) / self._fConst145) + np.float32(1.0693583)) 
		self._fConst209 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst174)) 
		self._fConst210 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst174)) 
		self._fConst211 = (((self._fConst184 + np.float32(-3.1897273)) / self._fConst183) + np.float32(4.0767817)) 
		self._fConst212 = (np.float32(1.0) / self._fConst186) 
		self._fConst213 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst212)) 
		self._fConst214 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst193)) 
		self._fConst215 = (((self._fConst184 + np.float32(-0.74313045)) / self._fConst183) + np.float32(1.4500711)) 
		self._fConst216 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst212)) 
		self._fConst217 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst190)) 
		self._fConst218 = (((self._fConst184 + np.float32(-0.15748216)) / self._fConst183) + np.float32(0.9351402)) 
		self._fConst219 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst212)) 
		self._fConst220 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst187)) 
		self._fConst221 = np.tan((np.float32(6283.1855) / self._fConst0)) 
		self._fConst222 = (np.float32(1.0) / self._fConst221) 
		self._fConst223 = (np.float32(1.0) / (((self._fConst222 + np.float32(0.15748216)) / self._fConst221) + np.float32(0.9351402))) 
		self._fConst224 = np.power(self._fConst221, np.float32(2.0)) 
		self._fConst225 = (np.float32(50.06381) / self._fConst224) 
		self._fConst226 = (self._fConst225 + np.float32(0.9351402)) 
		self._fConst227 = (np.float32(1.0) / (((self._fConst222 + np.float32(0.74313045)) / self._fConst221) + np.float32(1.4500711))) 
		self._fConst228 = (np.float32(11.0520525) / self._fConst224) 
		self._fConst229 = (self._fConst228 + np.float32(1.4500711)) 
		self._fConst230 = (np.float32(1.0) / (((self._fConst222 + np.float32(3.1897273)) / self._fConst221) + np.float32(4.0767817))) 
		self._fConst231 = (np.float32(0.0017661728) / self._fConst224) 
		self._fConst232 = (self._fConst231 + np.float32(0.0004076782)) 
		self._fConst233 = (np.float32(1.0) / (((self._fConst184 + np.float32(0.16840488)) / self._fConst183) + np.float32(1.0693583))) 
		self._fConst234 = (self._fConst212 + np.float32(53.53615)) 
		self._fConst235 = (np.float32(1.0) / (((self._fConst184 + np.float32(0.51247865)) / self._fConst183) + np.float32(0.6896214))) 
		self._fConst236 = (self._fConst212 + np.float32(7.6217313)) 
		self._fConst237 = (np.float32(1.0) / (((self._fConst184 + np.float32(0.78241307)) / self._fConst183) + np.float32(0.2452915))) 
		self._fConst238 = (np.float32(0.0001) / self._fConst186) 
		self._fConst239 = (self._fConst238 + np.float32(0.0004332272)) 
		self._fConst240 = (((self._fConst184 + np.float32(-0.78241307)) / self._fConst183) + np.float32(0.2452915)) 
		self._fConst241 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst212)) 
		self._fConst242 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst238)) 
		self._fConst243 = (((self._fConst184 + np.float32(-0.51247865)) / self._fConst183) + np.float32(0.6896214)) 
		self._fConst244 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst212)) 
		self._fConst245 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst212)) 
		self._fConst246 = (((self._fConst184 + np.float32(-0.16840488)) / self._fConst183) + np.float32(1.0693583)) 
		self._fConst247 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst212)) 
		self._fConst248 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst212)) 
		self._fConst249 = (((self._fConst222 + np.float32(-3.1897273)) / self._fConst221) + np.float32(4.0767817)) 
		self._fConst250 = (np.float32(1.0) / self._fConst224) 
		self._fConst251 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst250)) 
		self._fConst252 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst231)) 
		self._fConst253 = (((self._fConst222 + np.float32(-0.74313045)) / self._fConst221) + np.float32(1.4500711)) 
		self._fConst254 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst250)) 
		self._fConst255 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst228)) 
		self._fConst256 = (((self._fConst222 + np.float32(-0.15748216)) / self._fConst221) + np.float32(0.9351402)) 
		self._fConst257 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst250)) 
		self._fConst258 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst225)) 
		self._fConst259 = np.tan((np.float32(4442.883) / self._fConst0)) 
		self._fConst260 = (np.float32(1.0) / self._fConst259) 
		self._fConst261 = (np.float32(1.0) / (((self._fConst260 + np.float32(0.15748216)) / self._fConst259) + np.float32(0.9351402))) 
		self._fConst262 = np.power(self._fConst259, np.float32(2.0)) 
		self._fConst263 = (np.float32(50.06381) / self._fConst262) 
		self._fConst264 = (self._fConst263 + np.float32(0.9351402)) 
		self._fConst265 = (np.float32(1.0) / (((self._fConst260 + np.float32(0.74313045)) / self._fConst259) + np.float32(1.4500711))) 
		self._fConst266 = (np.float32(11.0520525) / self._fConst262) 
		self._fConst267 = (self._fConst266 + np.float32(1.4500711)) 
		self._fConst268 = (np.float32(1.0) / (((self._fConst260 + np.float32(3.1897273)) / self._fConst259) + np.float32(4.0767817))) 
		self._fConst269 = (np.float32(0.0017661728) / self._fConst262) 
		self._fConst270 = (self._fConst269 + np.float32(0.0004076782)) 
		self._fConst271 = (np.float32(1.0) / (((self._fConst222 + np.float32(0.16840488)) / self._fConst221) + np.float32(1.0693583))) 
		self._fConst272 = (self._fConst250 + np.float32(53.53615)) 
		self._fConst273 = (np.float32(1.0) / (((self._fConst222 + np.float32(0.51247865)) / self._fConst221) + np.float32(0.6896214))) 
		self._fConst274 = (self._fConst250 + np.float32(7.6217313)) 
		self._fConst275 = (np.float32(1.0) / (((self._fConst222 + np.float32(0.78241307)) / self._fConst221) + np.float32(0.2452915))) 
		self._fConst276 = (np.float32(0.0001) / self._fConst224) 
		self._fConst277 = (self._fConst276 + np.float32(0.0004332272)) 
		self._fConst278 = (((self._fConst222 + np.float32(-0.78241307)) / self._fConst221) + np.float32(0.2452915)) 
		self._fConst279 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst250)) 
		self._fConst280 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst276)) 
		self._fConst281 = (((self._fConst222 + np.float32(-0.51247865)) / self._fConst221) + np.float32(0.6896214)) 
		self._fConst282 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst250)) 
		self._fConst283 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst250)) 
		self._fConst284 = (((self._fConst222 + np.float32(-0.16840488)) / self._fConst221) + np.float32(1.0693583)) 
		self._fConst285 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst250)) 
		self._fConst286 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst250)) 
		self._fConst287 = (((self._fConst260 + np.float32(-3.1897273)) / self._fConst259) + np.float32(4.0767817)) 
		self._fConst288 = (np.float32(1.0) / self._fConst262) 
		self._fConst289 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst288)) 
		self._fConst290 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst269)) 
		self._fConst291 = (((self._fConst260 + np.float32(-0.74313045)) / self._fConst259) + np.float32(1.4500711)) 
		self._fConst292 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst288)) 
		self._fConst293 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst266)) 
		self._fConst294 = (((self._fConst260 + np.float32(-0.15748216)) / self._fConst259) + np.float32(0.9351402)) 
		self._fConst295 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst288)) 
		self._fConst296 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst263)) 
		self._fConst297 = np.tan((np.float32(3141.5928) / self._fConst0)) 
		self._fConst298 = (np.float32(1.0) / self._fConst297) 
		self._fConst299 = (np.float32(1.0) / (((self._fConst298 + np.float32(0.15748216)) / self._fConst297) + np.float32(0.9351402))) 
		self._fConst300 = np.power(self._fConst297, np.float32(2.0)) 
		self._fConst301 = (np.float32(50.06381) / self._fConst300) 
		self._fConst302 = (self._fConst301 + np.float32(0.9351402)) 
		self._fConst303 = (np.float32(1.0) / (((self._fConst298 + np.float32(0.74313045)) / self._fConst297) + np.float32(1.4500711))) 
		self._fConst304 = (np.float32(11.0520525) / self._fConst300) 
		self._fConst305 = (self._fConst304 + np.float32(1.4500711)) 
		self._fConst306 = (np.float32(1.0) / (((self._fConst298 + np.float32(3.1897273)) / self._fConst297) + np.float32(4.0767817))) 
		self._fConst307 = (np.float32(0.0017661728) / self._fConst300) 
		self._fConst308 = (self._fConst307 + np.float32(0.0004076782)) 
		self._fConst309 = (np.float32(1.0) / (((self._fConst260 + np.float32(0.16840488)) / self._fConst259) + np.float32(1.0693583))) 
		self._fConst310 = (self._fConst288 + np.float32(53.53615)) 
		self._fConst311 = (np.float32(1.0) / (((self._fConst260 + np.float32(0.51247865)) / self._fConst259) + np.float32(0.6896214))) 
		self._fConst312 = (self._fConst288 + np.float32(7.6217313)) 
		self._fConst313 = (np.float32(1.0) / (((self._fConst260 + np.float32(0.78241307)) / self._fConst259) + np.float32(0.2452915))) 
		self._fConst314 = (np.float32(0.0001) / self._fConst262) 
		self._fConst315 = (self._fConst314 + np.float32(0.0004332272)) 
		self._fConst316 = (((self._fConst260 + np.float32(-0.78241307)) / self._fConst259) + np.float32(0.2452915)) 
		self._fConst317 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst288)) 
		self._fConst318 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst314)) 
		self._fConst319 = (((self._fConst260 + np.float32(-0.51247865)) / self._fConst259) + np.float32(0.6896214)) 
		self._fConst320 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst288)) 
		self._fConst321 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst288)) 
		self._fConst322 = (((self._fConst260 + np.float32(-0.16840488)) / self._fConst259) + np.float32(1.0693583)) 
		self._fConst323 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst288)) 
		self._fConst324 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst288)) 
		self._fConst325 = (((self._fConst298 + np.float32(-3.1897273)) / self._fConst297) + np.float32(4.0767817)) 
		self._fConst326 = (np.float32(1.0) / self._fConst300) 
		self._fConst327 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst326)) 
		self._fConst328 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst307)) 
		self._fConst329 = (((self._fConst298 + np.float32(-0.74313045)) / self._fConst297) + np.float32(1.4500711)) 
		self._fConst330 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst326)) 
		self._fConst331 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst304)) 
		self._fConst332 = (((self._fConst298 + np.float32(-0.15748216)) / self._fConst297) + np.float32(0.9351402)) 
		self._fConst333 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst326)) 
		self._fConst334 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst301)) 
		self._fConst335 = np.tan((np.float32(2221.4414) / self._fConst0)) 
		self._fConst336 = (np.float32(1.0) / self._fConst335) 
		self._fConst337 = (np.float32(1.0) / (((self._fConst336 + np.float32(0.15748216)) / self._fConst335) + np.float32(0.9351402))) 
		self._fConst338 = np.power(self._fConst335, np.float32(2.0)) 
		self._fConst339 = (np.float32(50.06381) / self._fConst338) 
		self._fConst340 = (self._fConst339 + np.float32(0.9351402)) 
		self._fConst341 = (np.float32(1.0) / (((self._fConst336 + np.float32(0.74313045)) / self._fConst335) + np.float32(1.4500711))) 
		self._fConst342 = (np.float32(11.0520525) / self._fConst338) 
		self._fConst343 = (self._fConst342 + np.float32(1.4500711)) 
		self._fConst344 = (np.float32(1.0) / (((self._fConst336 + np.float32(3.1897273)) / self._fConst335) + np.float32(4.0767817))) 
		self._fConst345 = (np.float32(0.0017661728) / self._fConst338) 
		self._fConst346 = (self._fConst345 + np.float32(0.0004076782)) 
		self._fConst347 = (np.float32(1.0) / (((self._fConst298 + np.float32(0.16840488)) / self._fConst297) + np.float32(1.0693583))) 
		self._fConst348 = (self._fConst326 + np.float32(53.53615)) 
		self._fConst349 = (np.float32(1.0) / (((self._fConst298 + np.float32(0.51247865)) / self._fConst297) + np.float32(0.6896214))) 
		self._fConst350 = (self._fConst326 + np.float32(7.6217313)) 
		self._fConst351 = (np.float32(1.0) / (((self._fConst298 + np.float32(0.78241307)) / self._fConst297) + np.float32(0.2452915))) 
		self._fConst352 = (np.float32(0.0001) / self._fConst300) 
		self._fConst353 = (self._fConst352 + np.float32(0.0004332272)) 
		self._fConst354 = (((self._fConst298 + np.float32(-0.78241307)) / self._fConst297) + np.float32(0.2452915)) 
		self._fConst355 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst326)) 
		self._fConst356 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst352)) 
		self._fConst357 = (((self._fConst298 + np.float32(-0.51247865)) / self._fConst297) + np.float32(0.6896214)) 
		self._fConst358 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst326)) 
		self._fConst359 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst326)) 
		self._fConst360 = (((self._fConst298 + np.float32(-0.16840488)) / self._fConst297) + np.float32(1.0693583)) 
		self._fConst361 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst326)) 
		self._fConst362 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst326)) 
		self._fConst363 = (((self._fConst336 + np.float32(-3.1897273)) / self._fConst335) + np.float32(4.0767817)) 
		self._fConst364 = (np.float32(1.0) / self._fConst338) 
		self._fConst365 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst364)) 
		self._fConst366 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst345)) 
		self._fConst367 = (((self._fConst336 + np.float32(-0.74313045)) / self._fConst335) + np.float32(1.4500711)) 
		self._fConst368 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst364)) 
		self._fConst369 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst342)) 
		self._fConst370 = (((self._fConst336 + np.float32(-0.15748216)) / self._fConst335) + np.float32(0.9351402)) 
		self._fConst371 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst364)) 
		self._fConst372 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst339)) 
		self._fConst373 = np.tan((np.float32(1570.7964) / self._fConst0)) 
		self._fConst374 = (np.float32(1.0) / self._fConst373) 
		self._fConst375 = (np.float32(1.0) / (((self._fConst374 + np.float32(0.15748216)) / self._fConst373) + np.float32(0.9351402))) 
		self._fConst376 = np.power(self._fConst373, np.float32(2.0)) 
		self._fConst377 = (np.float32(50.06381) / self._fConst376) 
		self._fConst378 = (self._fConst377 + np.float32(0.9351402)) 
		self._fConst379 = (np.float32(1.0) / (((self._fConst374 + np.float32(0.74313045)) / self._fConst373) + np.float32(1.4500711))) 
		self._fConst380 = (np.float32(11.0520525) / self._fConst376) 
		self._fConst381 = (self._fConst380 + np.float32(1.4500711)) 
		self._fConst382 = (np.float32(1.0) / (((self._fConst374 + np.float32(3.1897273)) / self._fConst373) + np.float32(4.0767817))) 
		self._fConst383 = (np.float32(0.0017661728) / self._fConst376) 
		self._fConst384 = (self._fConst383 + np.float32(0.0004076782)) 
		self._fConst385 = (np.float32(1.0) / (((self._fConst336 + np.float32(0.16840488)) / self._fConst335) + np.float32(1.0693583))) 
		self._fConst386 = (self._fConst364 + np.float32(53.53615)) 
		self._fConst387 = (np.float32(1.0) / (((self._fConst336 + np.float32(0.51247865)) / self._fConst335) + np.float32(0.6896214))) 
		self._fConst388 = (self._fConst364 + np.float32(7.6217313)) 
		self._fConst389 = (np.float32(1.0) / (((self._fConst336 + np.float32(0.78241307)) / self._fConst335) + np.float32(0.2452915))) 
		self._fConst390 = (np.float32(0.0001) / self._fConst338) 
		self._fConst391 = (self._fConst390 + np.float32(0.0004332272)) 
		self._fConst392 = (((self._fConst336 + np.float32(-0.78241307)) / self._fConst335) + np.float32(0.2452915)) 
		self._fConst393 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst364)) 
		self._fConst394 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst390)) 
		self._fConst395 = (((self._fConst336 + np.float32(-0.51247865)) / self._fConst335) + np.float32(0.6896214)) 
		self._fConst396 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst364)) 
		self._fConst397 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst364)) 
		self._fConst398 = (((self._fConst336 + np.float32(-0.16840488)) / self._fConst335) + np.float32(1.0693583)) 
		self._fConst399 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst364)) 
		self._fConst400 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst364)) 
		self._fConst401 = (((self._fConst374 + np.float32(-3.1897273)) / self._fConst373) + np.float32(4.0767817)) 
		self._fConst402 = (np.float32(1.0) / self._fConst376) 
		self._fConst403 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst402)) 
		self._fConst404 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst383)) 
		self._fConst405 = (((self._fConst374 + np.float32(-0.74313045)) / self._fConst373) + np.float32(1.4500711)) 
		self._fConst406 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst402)) 
		self._fConst407 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst380)) 
		self._fConst408 = (((self._fConst374 + np.float32(-0.15748216)) / self._fConst373) + np.float32(0.9351402)) 
		self._fConst409 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst402)) 
		self._fConst410 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst377)) 
		self._fConst411 = np.tan((np.float32(1110.7207) / self._fConst0)) 
		self._fConst412 = (np.float32(1.0) / self._fConst411) 
		self._fConst413 = (np.float32(1.0) / (((self._fConst412 + np.float32(0.15748216)) / self._fConst411) + np.float32(0.9351402))) 
		self._fConst414 = np.power(self._fConst411, np.float32(2.0)) 
		self._fConst415 = (np.float32(50.06381) / self._fConst414) 
		self._fConst416 = (self._fConst415 + np.float32(0.9351402)) 
		self._fConst417 = (np.float32(1.0) / (((self._fConst412 + np.float32(0.74313045)) / self._fConst411) + np.float32(1.4500711))) 
		self._fConst418 = (np.float32(11.0520525) / self._fConst414) 
		self._fConst419 = (self._fConst418 + np.float32(1.4500711)) 
		self._fConst420 = (np.float32(1.0) / (((self._fConst412 + np.float32(3.1897273)) / self._fConst411) + np.float32(4.0767817))) 
		self._fConst421 = (np.float32(0.0017661728) / self._fConst414) 
		self._fConst422 = (self._fConst421 + np.float32(0.0004076782)) 
		self._fConst423 = (np.float32(1.0) / (((self._fConst374 + np.float32(0.16840488)) / self._fConst373) + np.float32(1.0693583))) 
		self._fConst424 = (self._fConst402 + np.float32(53.53615)) 
		self._fConst425 = (np.float32(1.0) / (((self._fConst374 + np.float32(0.51247865)) / self._fConst373) + np.float32(0.6896214))) 
		self._fConst426 = (self._fConst402 + np.float32(7.6217313)) 
		self._fConst427 = (np.float32(1.0) / (((self._fConst374 + np.float32(0.78241307)) / self._fConst373) + np.float32(0.2452915))) 
		self._fConst428 = (np.float32(0.0001) / self._fConst376) 
		self._fConst429 = (self._fConst428 + np.float32(0.0004332272)) 
		self._fConst430 = (((self._fConst374 + np.float32(-0.78241307)) / self._fConst373) + np.float32(0.2452915)) 
		self._fConst431 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst402)) 
		self._fConst432 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst428)) 
		self._fConst433 = (((self._fConst374 + np.float32(-0.51247865)) / self._fConst373) + np.float32(0.6896214)) 
		self._fConst434 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst402)) 
		self._fConst435 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst402)) 
		self._fConst436 = (((self._fConst374 + np.float32(-0.16840488)) / self._fConst373) + np.float32(1.0693583)) 
		self._fConst437 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst402)) 
		self._fConst438 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst402)) 
		self._fConst439 = (((self._fConst412 + np.float32(-3.1897273)) / self._fConst411) + np.float32(4.0767817)) 
		self._fConst440 = (np.float32(1.0) / self._fConst414) 
		self._fConst441 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst440)) 
		self._fConst442 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst421)) 
		self._fConst443 = (((self._fConst412 + np.float32(-0.74313045)) / self._fConst411) + np.float32(1.4500711)) 
		self._fConst444 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst440)) 
		self._fConst445 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst418)) 
		self._fConst446 = (((self._fConst412 + np.float32(-0.15748216)) / self._fConst411) + np.float32(0.9351402)) 
		self._fConst447 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst440)) 
		self._fConst448 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst415)) 
		self._fConst449 = np.tan((np.float32(785.3982) / self._fConst0)) 
		self._fConst450 = (np.float32(1.0) / self._fConst449) 
		self._fConst451 = (np.float32(1.0) / (((self._fConst450 + np.float32(0.15748216)) / self._fConst449) + np.float32(0.9351402))) 
		self._fConst452 = np.power(self._fConst449, np.float32(2.0)) 
		self._fConst453 = (np.float32(50.06381) / self._fConst452) 
		self._fConst454 = (self._fConst453 + np.float32(0.9351402)) 
		self._fConst455 = (np.float32(1.0) / (((self._fConst450 + np.float32(0.74313045)) / self._fConst449) + np.float32(1.4500711))) 
		self._fConst456 = (np.float32(11.0520525) / self._fConst452) 
		self._fConst457 = (self._fConst456 + np.float32(1.4500711)) 
		self._fConst458 = (np.float32(1.0) / (((self._fConst450 + np.float32(3.1897273)) / self._fConst449) + np.float32(4.0767817))) 
		self._fConst459 = (np.float32(0.0017661728) / self._fConst452) 
		self._fConst460 = (self._fConst459 + np.float32(0.0004076782)) 
		self._fConst461 = (np.float32(1.0) / (((self._fConst412 + np.float32(0.16840488)) / self._fConst411) + np.float32(1.0693583))) 
		self._fConst462 = (self._fConst440 + np.float32(53.53615)) 
		self._fConst463 = (np.float32(1.0) / (((self._fConst412 + np.float32(0.51247865)) / self._fConst411) + np.float32(0.6896214))) 
		self._fConst464 = (self._fConst440 + np.float32(7.6217313)) 
		self._fConst465 = (np.float32(1.0) / (((self._fConst412 + np.float32(0.78241307)) / self._fConst411) + np.float32(0.2452915))) 
		self._fConst466 = (np.float32(0.0001) / self._fConst414) 
		self._fConst467 = (self._fConst466 + np.float32(0.0004332272)) 
		self._fConst468 = (((self._fConst412 + np.float32(-0.78241307)) / self._fConst411) + np.float32(0.2452915)) 
		self._fConst469 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst440)) 
		self._fConst470 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst466)) 
		self._fConst471 = (((self._fConst412 + np.float32(-0.51247865)) / self._fConst411) + np.float32(0.6896214)) 
		self._fConst472 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst440)) 
		self._fConst473 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst440)) 
		self._fConst474 = (((self._fConst412 + np.float32(-0.16840488)) / self._fConst411) + np.float32(1.0693583)) 
		self._fConst475 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst440)) 
		self._fConst476 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst440)) 
		self._fConst477 = (((self._fConst450 + np.float32(-3.1897273)) / self._fConst449) + np.float32(4.0767817)) 
		self._fConst478 = (np.float32(1.0) / self._fConst452) 
		self._fConst479 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst478)) 
		self._fConst480 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst459)) 
		self._fConst481 = (((self._fConst450 + np.float32(-0.74313045)) / self._fConst449) + np.float32(1.4500711)) 
		self._fConst482 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst478)) 
		self._fConst483 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst456)) 
		self._fConst484 = (((self._fConst450 + np.float32(-0.15748216)) / self._fConst449) + np.float32(0.9351402)) 
		self._fConst485 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst478)) 
		self._fConst486 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst453)) 
		self._fConst487 = np.tan((np.float32(555.36035) / self._fConst0)) 
		self._fConst488 = (np.float32(1.0) / self._fConst487) 
		self._fConst489 = (np.float32(1.0) / (((self._fConst488 + np.float32(0.15748216)) / self._fConst487) + np.float32(0.9351402))) 
		self._fConst490 = np.power(self._fConst487, np.float32(2.0)) 
		self._fConst491 = (np.float32(50.06381) / self._fConst490) 
		self._fConst492 = (self._fConst491 + np.float32(0.9351402)) 
		self._fConst493 = (np.float32(1.0) / (((self._fConst488 + np.float32(0.74313045)) / self._fConst487) + np.float32(1.4500711))) 
		self._fConst494 = (np.float32(11.0520525) / self._fConst490) 
		self._fConst495 = (self._fConst494 + np.float32(1.4500711)) 
		self._fConst496 = (np.float32(1.0) / (((self._fConst488 + np.float32(3.1897273)) / self._fConst487) + np.float32(4.0767817))) 
		self._fConst497 = (np.float32(0.0017661728) / self._fConst490) 
		self._fConst498 = (self._fConst497 + np.float32(0.0004076782)) 
		self._fConst499 = (np.float32(1.0) / (((self._fConst450 + np.float32(0.16840488)) / self._fConst449) + np.float32(1.0693583))) 
		self._fConst500 = (self._fConst478 + np.float32(53.53615)) 
		self._fConst501 = (np.float32(1.0) / (((self._fConst450 + np.float32(0.51247865)) / self._fConst449) + np.float32(0.6896214))) 
		self._fConst502 = (self._fConst478 + np.float32(7.6217313)) 
		self._fConst503 = (np.float32(1.0) / (((self._fConst450 + np.float32(0.78241307)) / self._fConst449) + np.float32(0.2452915))) 
		self._fConst504 = (np.float32(0.0001) / self._fConst452) 
		self._fConst505 = (self._fConst504 + np.float32(0.0004332272)) 
		self._fConst506 = (((self._fConst450 + np.float32(-0.78241307)) / self._fConst449) + np.float32(0.2452915)) 
		self._fConst507 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst478)) 
		self._fConst508 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst504)) 
		self._fConst509 = (((self._fConst450 + np.float32(-0.51247865)) / self._fConst449) + np.float32(0.6896214)) 
		self._fConst510 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst478)) 
		self._fConst511 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst478)) 
		self._fConst512 = (((self._fConst450 + np.float32(-0.16840488)) / self._fConst449) + np.float32(1.0693583)) 
		self._fConst513 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst478)) 
		self._fConst514 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst478)) 
		self._fConst515 = (((self._fConst488 + np.float32(-3.1897273)) / self._fConst487) + np.float32(4.0767817)) 
		self._fConst516 = (np.float32(1.0) / self._fConst490) 
		self._fConst517 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst516)) 
		self._fConst518 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst497)) 
		self._fConst519 = (((self._fConst488 + np.float32(-0.74313045)) / self._fConst487) + np.float32(1.4500711)) 
		self._fConst520 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst516)) 
		self._fConst521 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst494)) 
		self._fConst522 = (((self._fConst488 + np.float32(-0.15748216)) / self._fConst487) + np.float32(0.9351402)) 
		self._fConst523 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst516)) 
		self._fConst524 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst491)) 
		self._fConst525 = np.tan((np.float32(392.6991) / self._fConst0)) 
		self._fConst526 = (np.float32(1.0) / self._fConst525) 
		self._fConst527 = (np.float32(1.0) / (((self._fConst526 + np.float32(0.15748216)) / self._fConst525) + np.float32(0.9351402))) 
		self._fConst528 = np.power(self._fConst525, np.float32(2.0)) 
		self._fConst529 = (np.float32(50.06381) / self._fConst528) 
		self._fConst530 = (self._fConst529 + np.float32(0.9351402)) 
		self._fConst531 = (np.float32(1.0) / (((self._fConst526 + np.float32(0.74313045)) / self._fConst525) + np.float32(1.4500711))) 
		self._fConst532 = (np.float32(11.0520525) / self._fConst528) 
		self._fConst533 = (self._fConst532 + np.float32(1.4500711)) 
		self._fConst534 = (np.float32(1.0) / (((self._fConst526 + np.float32(3.1897273)) / self._fConst525) + np.float32(4.0767817))) 
		self._fConst535 = (np.float32(0.0017661728) / self._fConst528) 
		self._fConst536 = (self._fConst535 + np.float32(0.0004076782)) 
		self._fConst537 = (np.float32(1.0) / (((self._fConst488 + np.float32(0.16840488)) / self._fConst487) + np.float32(1.0693583))) 
		self._fConst538 = (self._fConst516 + np.float32(53.53615)) 
		self._fConst539 = (np.float32(1.0) / (((self._fConst488 + np.float32(0.51247865)) / self._fConst487) + np.float32(0.6896214))) 
		self._fConst540 = (self._fConst516 + np.float32(7.6217313)) 
		self._fConst541 = (np.float32(1.0) / (((self._fConst488 + np.float32(0.78241307)) / self._fConst487) + np.float32(0.2452915))) 
		self._fConst542 = (np.float32(0.0001) / self._fConst490) 
		self._fConst543 = (self._fConst542 + np.float32(0.0004332272)) 
		self._fConst544 = (((self._fConst488 + np.float32(-0.78241307)) / self._fConst487) + np.float32(0.2452915)) 
		self._fConst545 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst516)) 
		self._fConst546 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst542)) 
		self._fConst547 = (((self._fConst488 + np.float32(-0.51247865)) / self._fConst487) + np.float32(0.6896214)) 
		self._fConst548 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst516)) 
		self._fConst549 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst516)) 
		self._fConst550 = (((self._fConst488 + np.float32(-0.16840488)) / self._fConst487) + np.float32(1.0693583)) 
		self._fConst551 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst516)) 
		self._fConst552 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst516)) 
		self._fConst553 = (((self._fConst526 + np.float32(-3.1897273)) / self._fConst525) + np.float32(4.0767817)) 
		self._fConst554 = (np.float32(1.0) / self._fConst528) 
		self._fConst555 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst554)) 
		self._fConst556 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst535)) 
		self._fConst557 = (((self._fConst526 + np.float32(-0.74313045)) / self._fConst525) + np.float32(1.4500711)) 
		self._fConst558 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst554)) 
		self._fConst559 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst532)) 
		self._fConst560 = (((self._fConst526 + np.float32(-0.15748216)) / self._fConst525) + np.float32(0.9351402)) 
		self._fConst561 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst554)) 
		self._fConst562 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst529)) 
		self._fConst563 = np.tan((np.float32(277.68018) / self._fConst0)) 
		self._fConst564 = (np.float32(1.0) / self._fConst563) 
		self._fConst565 = (np.float32(1.0) / (((self._fConst564 + np.float32(0.15748216)) / self._fConst563) + np.float32(0.9351402))) 
		self._fConst566 = np.power(self._fConst563, np.float32(2.0)) 
		self._fConst567 = (np.float32(50.06381) / self._fConst566) 
		self._fConst568 = (self._fConst567 + np.float32(0.9351402)) 
		self._fConst569 = (np.float32(1.0) / (((self._fConst564 + np.float32(0.74313045)) / self._fConst563) + np.float32(1.4500711))) 
		self._fConst570 = (np.float32(11.0520525) / self._fConst566) 
		self._fConst571 = (self._fConst570 + np.float32(1.4500711)) 
		self._fConst572 = (np.float32(1.0) / (((self._fConst564 + np.float32(3.1897273)) / self._fConst563) + np.float32(4.0767817))) 
		self._fConst573 = (np.float32(0.0017661728) / self._fConst566) 
		self._fConst574 = (self._fConst573 + np.float32(0.0004076782)) 
		self._fConst575 = (np.float32(1.0) / (((self._fConst526 + np.float32(0.16840488)) / self._fConst525) + np.float32(1.0693583))) 
		self._fConst576 = (self._fConst554 + np.float32(53.53615)) 
		self._fConst577 = (np.float32(1.0) / (((self._fConst526 + np.float32(0.51247865)) / self._fConst525) + np.float32(0.6896214))) 
		self._fConst578 = (self._fConst554 + np.float32(7.6217313)) 
		self._fConst579 = (np.float32(1.0) / (((self._fConst526 + np.float32(0.78241307)) / self._fConst525) + np.float32(0.2452915))) 
		self._fConst580 = (np.float32(0.0001) / self._fConst528) 
		self._fConst581 = (self._fConst580 + np.float32(0.0004332272)) 
		self._fConst582 = (((self._fConst526 + np.float32(-0.78241307)) / self._fConst525) + np.float32(0.2452915)) 
		self._fConst583 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst554)) 
		self._fConst584 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst580)) 
		self._fConst585 = (((self._fConst526 + np.float32(-0.51247865)) / self._fConst525) + np.float32(0.6896214)) 
		self._fConst586 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst554)) 
		self._fConst587 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst554)) 
		self._fConst588 = (((self._fConst526 + np.float32(-0.16840488)) / self._fConst525) + np.float32(1.0693583)) 
		self._fConst589 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst554)) 
		self._fConst590 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst554)) 
		self._fConst591 = (((self._fConst564 + np.float32(-3.1897273)) / self._fConst563) + np.float32(4.0767817)) 
		self._fConst592 = (np.float32(1.0) / self._fConst566) 
		self._fConst593 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst592)) 
		self._fConst594 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst573)) 
		self._fConst595 = (((self._fConst564 + np.float32(-0.74313045)) / self._fConst563) + np.float32(1.4500711)) 
		self._fConst596 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst592)) 
		self._fConst597 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst570)) 
		self._fConst598 = (((self._fConst564 + np.float32(-0.15748216)) / self._fConst563) + np.float32(0.9351402)) 
		self._fConst599 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst592)) 
		self._fConst600 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst567)) 
		self._fConst601 = np.tan((np.float32(196.34955) / self._fConst0)) 
		self._fConst602 = (np.float32(1.0) / self._fConst601) 
		self._fConst603 = (np.float32(1.0) / (((self._fConst602 + np.float32(0.15748216)) / self._fConst601) + np.float32(0.9351402))) 
		self._fConst604 = np.power(self._fConst601, np.float32(2.0)) 
		self._fConst605 = (np.float32(50.06381) / self._fConst604) 
		self._fConst606 = (self._fConst605 + np.float32(0.9351402)) 
		self._fConst607 = (np.float32(1.0) / (((self._fConst602 + np.float32(0.74313045)) / self._fConst601) + np.float32(1.4500711))) 
		self._fConst608 = (np.float32(11.0520525) / self._fConst604) 
		self._fConst609 = (self._fConst608 + np.float32(1.4500711)) 
		self._fConst610 = (np.float32(1.0) / (((self._fConst602 + np.float32(3.1897273)) / self._fConst601) + np.float32(4.0767817))) 
		self._fConst611 = (np.float32(0.0017661728) / self._fConst604) 
		self._fConst612 = (self._fConst611 + np.float32(0.0004076782)) 
		self._fConst613 = (np.float32(1.0) / (((self._fConst564 + np.float32(0.16840488)) / self._fConst563) + np.float32(1.0693583))) 
		self._fConst614 = (self._fConst592 + np.float32(53.53615)) 
		self._fConst615 = (np.float32(1.0) / (((self._fConst564 + np.float32(0.51247865)) / self._fConst563) + np.float32(0.6896214))) 
		self._fConst616 = (self._fConst592 + np.float32(7.6217313)) 
		self._fConst617 = (np.float32(1.0) / (((self._fConst564 + np.float32(0.78241307)) / self._fConst563) + np.float32(0.2452915))) 
		self._fConst618 = (np.float32(0.0001) / self._fConst566) 
		self._fConst619 = (self._fConst618 + np.float32(0.0004332272)) 
		self._fConst620 = (((self._fConst564 + np.float32(-0.78241307)) / self._fConst563) + np.float32(0.2452915)) 
		self._fConst621 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst592)) 
		self._fConst622 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst618)) 
		self._fConst623 = (((self._fConst564 + np.float32(-0.51247865)) / self._fConst563) + np.float32(0.6896214)) 
		self._fConst624 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst592)) 
		self._fConst625 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst592)) 
		self._fConst626 = (((self._fConst564 + np.float32(-0.16840488)) / self._fConst563) + np.float32(1.0693583)) 
		self._fConst627 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst592)) 
		self._fConst628 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst592)) 
		self._fConst629 = (((self._fConst602 + np.float32(-3.1897273)) / self._fConst601) + np.float32(4.0767817)) 
		self._fConst630 = (np.float32(1.0) / self._fConst604) 
		self._fConst631 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst630)) 
		self._fConst632 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst611)) 
		self._fConst633 = (((self._fConst602 + np.float32(-0.74313045)) / self._fConst601) + np.float32(1.4500711)) 
		self._fConst634 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst630)) 
		self._fConst635 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst608)) 
		self._fConst636 = (((self._fConst602 + np.float32(-0.15748216)) / self._fConst601) + np.float32(0.9351402)) 
		self._fConst637 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst630)) 
		self._fConst638 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst605)) 
		self._fConst639 = np.tan((np.float32(138.84009) / self._fConst0)) 
		self._fConst640 = (np.float32(1.0) / self._fConst639) 
		self._fConst641 = (np.float32(1.0) / (((self._fConst640 + np.float32(0.15748216)) / self._fConst639) + np.float32(0.9351402))) 
		self._fConst642 = np.power(self._fConst639, np.float32(2.0)) 
		self._fConst643 = (np.float32(50.06381) / self._fConst642) 
		self._fConst644 = (self._fConst643 + np.float32(0.9351402)) 
		self._fConst645 = (np.float32(1.0) / (((self._fConst640 + np.float32(0.74313045)) / self._fConst639) + np.float32(1.4500711))) 
		self._fConst646 = (np.float32(11.0520525) / self._fConst642) 
		self._fConst647 = (self._fConst646 + np.float32(1.4500711)) 
		self._fConst648 = (np.float32(1.0) / (((self._fConst640 + np.float32(3.1897273)) / self._fConst639) + np.float32(4.0767817))) 
		self._fConst649 = (np.float32(0.0017661728) / self._fConst642) 
		self._fConst650 = (self._fConst649 + np.float32(0.0004076782)) 
		self._fConst651 = (np.float32(1.0) / (((self._fConst602 + np.float32(0.16840488)) / self._fConst601) + np.float32(1.0693583))) 
		self._fConst652 = (self._fConst630 + np.float32(53.53615)) 
		self._fConst653 = (np.float32(1.0) / (((self._fConst602 + np.float32(0.51247865)) / self._fConst601) + np.float32(0.6896214))) 
		self._fConst654 = (self._fConst630 + np.float32(7.6217313)) 
		self._fConst655 = (np.float32(1.0) / (((self._fConst602 + np.float32(0.78241307)) / self._fConst601) + np.float32(0.2452915))) 
		self._fConst656 = (np.float32(0.0001) / self._fConst604) 
		self._fConst657 = (self._fConst656 + np.float32(0.0004332272)) 
		self._fConst658 = (((self._fConst602 + np.float32(-0.78241307)) / self._fConst601) + np.float32(0.2452915)) 
		self._fConst659 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst630)) 
		self._fConst660 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst656)) 
		self._fConst661 = (((self._fConst602 + np.float32(-0.51247865)) / self._fConst601) + np.float32(0.6896214)) 
		self._fConst662 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst630)) 
		self._fConst663 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst630)) 
		self._fConst664 = (((self._fConst602 + np.float32(-0.16840488)) / self._fConst601) + np.float32(1.0693583)) 
		self._fConst665 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst630)) 
		self._fConst666 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst630)) 
		self._fConst667 = (((self._fConst640 + np.float32(-3.1897273)) / self._fConst639) + np.float32(4.0767817)) 
		self._fConst668 = (np.float32(1.0) / self._fConst642) 
		self._fConst669 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst668)) 
		self._fConst670 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst649)) 
		self._fConst671 = (((self._fConst640 + np.float32(-0.74313045)) / self._fConst639) + np.float32(1.4500711)) 
		self._fConst672 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst668)) 
		self._fConst673 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst646)) 
		self._fConst674 = (((self._fConst640 + np.float32(-0.15748216)) / self._fConst639) + np.float32(0.9351402)) 
		self._fConst675 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst668)) 
		self._fConst676 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst643)) 
		self._fConst677 = np.tan((np.float32(98.174774) / self._fConst0)) 
		self._fConst678 = (np.float32(1.0) / self._fConst677) 
		self._fConst679 = (np.float32(1.0) / (((self._fConst678 + np.float32(0.15748216)) / self._fConst677) + np.float32(0.9351402))) 
		self._fConst680 = np.power(self._fConst677, np.float32(2.0)) 
		self._fConst681 = (np.float32(50.06381) / self._fConst680) 
		self._fConst682 = (self._fConst681 + np.float32(0.9351402)) 
		self._fConst683 = (np.float32(1.0) / (((self._fConst678 + np.float32(0.74313045)) / self._fConst677) + np.float32(1.4500711))) 
		self._fConst684 = (np.float32(11.0520525) / self._fConst680) 
		self._fConst685 = (self._fConst684 + np.float32(1.4500711)) 
		self._fConst686 = (np.float32(1.0) / (((self._fConst678 + np.float32(3.1897273)) / self._fConst677) + np.float32(4.0767817))) 
		self._fConst687 = (np.float32(0.0017661728) / self._fConst680) 
		self._fConst688 = (self._fConst687 + np.float32(0.0004076782)) 
		self._fConst689 = (np.float32(1.0) / (((self._fConst640 + np.float32(0.16840488)) / self._fConst639) + np.float32(1.0693583))) 
		self._fConst690 = (self._fConst668 + np.float32(53.53615)) 
		self._fConst691 = (np.float32(1.0) / (((self._fConst640 + np.float32(0.51247865)) / self._fConst639) + np.float32(0.6896214))) 
		self._fConst692 = (self._fConst668 + np.float32(7.6217313)) 
		self._fConst693 = (np.float32(1.0) / (((self._fConst640 + np.float32(0.78241307)) / self._fConst639) + np.float32(0.2452915))) 
		self._fConst694 = (np.float32(0.0001) / self._fConst642) 
		self._fConst695 = (self._fConst694 + np.float32(0.0004332272)) 
		self._fConst696 = (((self._fConst640 + np.float32(-0.78241307)) / self._fConst639) + np.float32(0.2452915)) 
		self._fConst697 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst668)) 
		self._fConst698 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst694)) 
		self._fConst699 = (((self._fConst640 + np.float32(-0.51247865)) / self._fConst639) + np.float32(0.6896214)) 
		self._fConst700 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst668)) 
		self._fConst701 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst668)) 
		self._fConst702 = (((self._fConst640 + np.float32(-0.16840488)) / self._fConst639) + np.float32(1.0693583)) 
		self._fConst703 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst668)) 
		self._fConst704 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst668)) 
		self._fConst705 = (((self._fConst678 + np.float32(-3.1897273)) / self._fConst677) + np.float32(4.0767817)) 
		self._fConst706 = (np.float32(1.0) / self._fConst680) 
		self._fConst707 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst706)) 
		self._fConst708 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst687)) 
		self._fConst709 = (((self._fConst678 + np.float32(-0.74313045)) / self._fConst677) + np.float32(1.4500711)) 
		self._fConst710 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst706)) 
		self._fConst711 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst684)) 
		self._fConst712 = (((self._fConst678 + np.float32(-0.15748216)) / self._fConst677) + np.float32(0.9351402)) 
		self._fConst713 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst706)) 
		self._fConst714 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst681)) 
		self._fConst715 = (np.float32(1.0) / (((self._fConst678 + np.float32(0.16840488)) / self._fConst677) + np.float32(1.0693583))) 
		self._fConst716 = (self._fConst706 + np.float32(53.53615)) 
		self._fConst717 = (np.float32(1.0) / (((self._fConst678 + np.float32(0.51247865)) / self._fConst677) + np.float32(0.6896214))) 
		self._fConst718 = (self._fConst706 + np.float32(7.6217313)) 
		self._fConst719 = (np.float32(1.0) / (((self._fConst678 + np.float32(0.78241307)) / self._fConst677) + np.float32(0.2452915))) 
		self._fConst720 = (np.float32(0.0001) / self._fConst680) 
		self._fConst721 = (self._fConst720 + np.float32(0.0004332272)) 
		self._fConst722 = (((self._fConst678 + np.float32(-0.78241307)) / self._fConst677) + np.float32(0.2452915)) 
		self._fConst723 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst706)) 
		self._fConst724 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst720)) 
		self._fConst725 = (((self._fConst678 + np.float32(-0.51247865)) / self._fConst677) + np.float32(0.6896214)) 
		self._fConst726 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst706)) 
		self._fConst727 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst706)) 
		self._fConst728 = (((self._fConst678 + np.float32(-0.16840488)) / self._fConst677) + np.float32(1.0693583)) 
		self._fConst729 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst706)) 
		self._fConst730 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst706)) 
		
	@property
	def num_inputs(self):
		return 1
	
	@property
	def num_outputs(self):
		return 2
	
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec10"] = np.float32(0)
		state["fRec105"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec112"] = np.float32(0)
		state["fRec119"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec126"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec133"] = np.float32(0)
		state["fRec140"] = np.float32(0)
		state["fRec147"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec20"] = np.float32(0)
		state["fRec21"] = np.float32(0)
		state["fRec28"] = np.float32(0)
		state["fRec35"] = np.float32(0)
		state["fRec42"] = np.float32(0)
		state["fRec49"] = np.float32(0)
		state["fRec5"] = np.float32(0)
		state["fRec56"] = np.float32(0)
		state["fRec63"] = np.float32(0)
		state["fRec70"] = np.float32(0)
		state["fRec77"] = np.float32(0)
		state["fRec8"] = np.float32(0)
		state["fRec84"] = np.float32(0)
		state["fRec9"] = np.float32(0)
		state["fRec91"] = np.float32(0)
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
		state["iRec14"] = np.int32(0)
		# Initialize array delays
		state["iVec0"] = np.zeros((4,), dtype=np.int32)
		state["fRec15"] = np.zeros((4,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec16"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fRec19"] = np.zeros((3,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec27"] = np.zeros((3,), dtype=np.float32)
		state["fRec26"] = np.zeros((3,), dtype=np.float32)
		state["fRec25"] = np.zeros((3,), dtype=np.float32)
		state["fRec24"] = np.zeros((3,), dtype=np.float32)
		state["fRec23"] = np.zeros((3,), dtype=np.float32)
		state["fRec22"] = np.zeros((3,), dtype=np.float32)
		state["fRec34"] = np.zeros((3,), dtype=np.float32)
		state["fRec33"] = np.zeros((3,), dtype=np.float32)
		state["fRec32"] = np.zeros((3,), dtype=np.float32)
		state["fRec31"] = np.zeros((3,), dtype=np.float32)
		state["fRec30"] = np.zeros((3,), dtype=np.float32)
		state["fRec29"] = np.zeros((3,), dtype=np.float32)
		state["fRec41"] = np.zeros((3,), dtype=np.float32)
		state["fRec40"] = np.zeros((3,), dtype=np.float32)
		state["fRec39"] = np.zeros((3,), dtype=np.float32)
		state["fRec38"] = np.zeros((3,), dtype=np.float32)
		state["fRec37"] = np.zeros((3,), dtype=np.float32)
		state["fRec36"] = np.zeros((3,), dtype=np.float32)
		state["fRec48"] = np.zeros((3,), dtype=np.float32)
		state["fRec47"] = np.zeros((3,), dtype=np.float32)
		state["fRec46"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		state["fRec43"] = np.zeros((3,), dtype=np.float32)
		state["fRec55"] = np.zeros((3,), dtype=np.float32)
		state["fRec54"] = np.zeros((3,), dtype=np.float32)
		state["fRec53"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec50"] = np.zeros((3,), dtype=np.float32)
		state["fRec62"] = np.zeros((3,), dtype=np.float32)
		state["fRec61"] = np.zeros((3,), dtype=np.float32)
		state["fRec60"] = np.zeros((3,), dtype=np.float32)
		state["fRec59"] = np.zeros((3,), dtype=np.float32)
		state["fRec58"] = np.zeros((3,), dtype=np.float32)
		state["fRec57"] = np.zeros((3,), dtype=np.float32)
		state["fRec69"] = np.zeros((3,), dtype=np.float32)
		state["fRec68"] = np.zeros((3,), dtype=np.float32)
		state["fRec67"] = np.zeros((3,), dtype=np.float32)
		state["fRec66"] = np.zeros((3,), dtype=np.float32)
		state["fRec65"] = np.zeros((3,), dtype=np.float32)
		state["fRec64"] = np.zeros((3,), dtype=np.float32)
		state["fRec76"] = np.zeros((3,), dtype=np.float32)
		state["fRec75"] = np.zeros((3,), dtype=np.float32)
		state["fRec74"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec72"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec83"] = np.zeros((3,), dtype=np.float32)
		state["fRec82"] = np.zeros((3,), dtype=np.float32)
		state["fRec81"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec89"] = np.zeros((3,), dtype=np.float32)
		state["fRec88"] = np.zeros((3,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec96"] = np.zeros((3,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fRec94"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec92"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec103"] = np.zeros((3,), dtype=np.float32)
		state["fRec102"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fRec110"] = np.zeros((3,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec116"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec114"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec125"] = np.zeros((3,), dtype=np.float32)
		state["fRec124"] = np.zeros((3,), dtype=np.float32)
		state["fRec123"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec120"] = np.zeros((3,), dtype=np.float32)
		state["fRec132"] = np.zeros((3,), dtype=np.float32)
		state["fRec131"] = np.zeros((3,), dtype=np.float32)
		state["fRec130"] = np.zeros((3,), dtype=np.float32)
		state["fRec129"] = np.zeros((3,), dtype=np.float32)
		state["fRec128"] = np.zeros((3,), dtype=np.float32)
		state["fRec127"] = np.zeros((3,), dtype=np.float32)
		state["fRec139"] = np.zeros((3,), dtype=np.float32)
		state["fRec138"] = np.zeros((3,), dtype=np.float32)
		state["fRec137"] = np.zeros((3,), dtype=np.float32)
		state["fRec136"] = np.zeros((3,), dtype=np.float32)
		state["fRec135"] = np.zeros((3,), dtype=np.float32)
		state["fRec134"] = np.zeros((3,), dtype=np.float32)
		state["fRec146"] = np.zeros((3,), dtype=np.float32)
		state["fRec145"] = np.zeros((3,), dtype=np.float32)
		state["fRec144"] = np.zeros((3,), dtype=np.float32)
		state["fRec143"] = np.zeros((3,), dtype=np.float32)
		state["fRec142"] = np.zeros((3,), dtype=np.float32)
		state["fRec141"] = np.zeros((3,), dtype=np.float32)
		state["fRec150"] = np.zeros((3,), dtype=np.float32)
		state["fRec149"] = np.zeros((3,), dtype=np.float32)
		state["fRec148"] = np.zeros((3,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = params["fHslider0"] 
		fSlow1 = params["fHslider1"] 
		fSlow2 = jnp.where((((jnp.float32(0.001) * fSlow1) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst1 / fSlow1))), jnp.float32(0.0)) 
		fSlow3 = jnp.tan((self._fConst14 * params["fHslider2"])) 
		fSlow4 = (jnp.float32(1.0) / fSlow3) 
		fSlow5 = (jnp.float32(1.0) / (((fSlow4 + jnp.float32(1.0)) / fSlow3) + jnp.float32(1.0))) 
		fSlow6 = (jnp.float32(1.0) / (fSlow4 + jnp.float32(1.0))) 
		fSlow7 = (jnp.float32(1.0) - fSlow4) 
		fSlow8 = jnp.tan((self._fConst14 * params["fHslider3"])) 
		fSlow9 = (jnp.float32(1.0) / fSlow8) 
		fSlow10 = (jnp.float32(1.0) / (((fSlow9 + jnp.float32(1.0)) / fSlow8) + jnp.float32(1.0))) 
		fSlow11 = (jnp.float32(1.0) / jnp.power(fSlow8, jnp.float32(2.0))) 
		fSlow12 = (jnp.float32(1.0) / (fSlow9 + jnp.float32(1.0))) 
		fSlow13 = (jnp.float32(1.0) - fSlow9) 
		fSlow14 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider0"]))) 
		iSlow15 = jnp.int32(params["fCheckbox0"]) 
		iSlow16 = jnp.int32(params["fCheckbox1"]) 
		iSlow17 = jnp.int32((params["fEntry0"] + jnp.float32(-1.0))) 
		iSlow18 = (iSlow17 >= jnp.int32(2)).astype(jnp.int32) 
		iSlow19 = (iSlow17 >= jnp.int32(1)).astype(jnp.int32) 
		fSlow20 = params["fVslider1"] 
		fSlow21 = jnp.where(((fSlow20 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst15 / fSlow20))), jnp.float32(0.0)) 
		fSlow22 = ((jnp.float32(4.4e+02) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fVslider2"] + jnp.float32(-49.0))))) * (jnp.float32(1.0) - fSlow21)) 
		iSlow23 = (iSlow17 >= jnp.int32(3)).astype(jnp.int32) 
		fSlow24 = ((jnp.float32(0.01) * params["fVslider3"]) + jnp.float32(1.0)) 
		fSlow25 = ((jnp.float32(0.01) * params["fVslider4"]) + jnp.float32(1.0)) 
		iSlow26 = jnp.int32(params["fCheckbox2"]) 
		fSlow27 = (((fSlow9 + jnp.float32(-1.0)) / fSlow8) + jnp.float32(1.0)) 
		fSlow28 = (jnp.float32(2.0) * (jnp.float32(1.0) - fSlow11)) 
		fSlow29 = jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider4"])) 
		fSlow30 = (jnp.float32(0.001) * params["fHslider5"]) 
		fSlow31 = params["fHslider6"] 
		iSlow32 = (fSlow31 > jnp.float32(0.0)).astype(jnp.int32) 
		fSlow33 = params["fHslider7"] 
		fSlow34 = (self._fConst19 * (jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * jnp.abs(fSlow31))) / fSlow33)) 
		fSlow35 = (self._fConst19 / fSlow33) 
		fSlow36 = (((fSlow4 + jnp.float32(-1.0)) / fSlow3) + jnp.float32(1.0)) 
		fSlow37 = jnp.power(fSlow3, jnp.float32(2.0)) 
		fSlow38 = (jnp.float32(2.0) * (jnp.float32(1.0) - (jnp.float32(1.0) / fSlow37))) 
		fSlow39 = (jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider8"])) / fSlow37) 
		fSlow40 = (jnp.float32(1.0) - fSlow2) 
		fRec9_temp = state["fRec9"] 
		fRec11_temp = state["fRec11"] 
		fRec10_temp = state["fRec10"] 
		fVec1_temp = state["fVec1"] 
		fVec2_temp = state["fVec2"] 
		fVec3_temp = state["fVec3"] 
		fVec4_temp = state["fVec4"] 
		fVec5_temp = state["fVec5"] 
		fVec6_temp = state["fVec6"] 
		fRec12_temp = state["fRec12"] 
		fVec7_temp = state["fVec7"] 
		fVec8_temp = state["fVec8"] 
		fVec9_temp = state["fVec9"] 
		fVec10_temp = state["fVec10"] 
		fVec11_temp = state["fVec11"] 
		fVec12_temp = state["fVec12"] 
		fRec13_temp = state["fRec13"] 
		fVec13_temp = state["fVec13"] 
		fVec14_temp = state["fVec14"] 
		fVec15_temp = state["fVec15"] 
		fVec16_temp = state["fVec16"] 
		fVec17_temp = state["fVec17"] 
		fVec18_temp = state["fVec18"] 
		iRec14_temp = state["iRec14"] 
		fVec19_temp = state["fVec19"] 
		fRec8_temp = state["fRec8"] 
		fRec17_temp = state["fRec17"] 
		fRec18_temp = state["fRec18"] 
		fVec20_temp = state["fVec20"] 
		fRec5_temp = state["fRec5"] 
		fRec20_temp = state["fRec20"] 
		fRec0_temp = state["fRec0"] 
		fRec21_temp = state["fRec21"] 
		fRec28_temp = state["fRec28"] 
		fRec35_temp = state["fRec35"] 
		fRec42_temp = state["fRec42"] 
		fRec49_temp = state["fRec49"] 
		fRec56_temp = state["fRec56"] 
		fRec63_temp = state["fRec63"] 
		fRec70_temp = state["fRec70"] 
		fRec77_temp = state["fRec77"] 
		fRec84_temp = state["fRec84"] 
		fRec91_temp = state["fRec91"] 
		fRec98_temp = state["fRec98"] 
		fRec105_temp = state["fRec105"] 
		fRec112_temp = state["fRec112"] 
		fRec119_temp = state["fRec119"] 
		fRec126_temp = state["fRec126"] 
		fRec133_temp = state["fRec133"] 
		fRec140_temp = state["fRec140"] 
		fRec147_temp = state["fRec147"] 
		state["fRec9"] = (fSlow14 + (jnp.float32(0.999) * fRec9_temp)) 
		state["iVec0"] = state["iVec0"].at[0].set(jnp.int32(1)) 
		state["fRec11"] = ((fRec11_temp * fSlow21) + fSlow22) 
		fTemp0 = jnp.maximum(jnp.float32(2e+01), jnp.abs(state["fRec11"])) 
		fTemp1 = (fRec10_temp + (self._fConst15 * fTemp0)) 
		state["fRec10"] = (fTemp1 - jnp.floor(fTemp1)) 
		fTemp2 = (jnp.float32(2.0) * state["fRec10"]) 
		fTemp3 = (fTemp2 + jnp.float32(-1.0)) 
		fTemp4 = (state["iVec0"][1]) 
		fTemp5 = jnp.power(fTemp3, jnp.float32(2.0)) 
		state["fVec1"] = jnp.float32(fTemp5) 
		fTemp6 = (state["iVec0"][2]) 
		fTemp7 = jnp.power(fTemp3, jnp.float32(3.0)) 
		state["fVec2"] = (fTemp7 + (jnp.float32(1.0) - fTemp2)) 
		fTemp8 = ((fTemp7 + (jnp.float32(1.0) - (fTemp2 + fVec2_temp))) / fTemp0) 
		state["fVec3"] = jnp.float32(fTemp8) 
		fTemp9 = (state["iVec0"][3]) 
		fTemp10 = (fTemp5 * (fTemp5 + jnp.float32(-2.0))) 
		state["fVec4"] = jnp.float32(fTemp10) 
		fTemp11 = ((fTemp10 - fVec4_temp) / fTemp0) 
		state["fVec5"] = jnp.float32(fTemp11) 
		fTemp12 = ((fTemp11 - fVec5_temp) / fTemp0) 
		state["fVec6"] = jnp.float32(fTemp12) 
		fTemp13 = jnp.maximum(jnp.float32(2e+01), jnp.abs((fSlow24 * state["fRec11"]))) 
		fTemp14 = (fRec12_temp + (self._fConst15 * fTemp13)) 
		state["fRec12"] = (fTemp14 - jnp.floor(fTemp14)) 
		fTemp15 = (jnp.float32(2.0) * state["fRec12"]) 
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
		fTemp23 = jnp.maximum(jnp.float32(2e+01), jnp.abs((fSlow25 * state["fRec11"]))) 
		fTemp24 = (fRec13_temp + (self._fConst15 * fTemp23)) 
		state["fRec13"] = (fTemp24 - jnp.floor(fTemp24)) 
		fTemp25 = (jnp.float32(2.0) * state["fRec13"]) 
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
		state["iRec14"] = ((jnp.int32(1103515245) * iRec14_temp) + jnp.int32(12345)) 
		fTemp33 = (jnp.float32(4.656613e-10) * (state["iRec14"])) 
		state["fRec15"] = state["fRec15"].at[0].set((((jnp.float32(0.5221894) * state["fRec15"][3]) + (fTemp33 + (jnp.float32(2.494956) * state["fRec15"][1]))) - (jnp.float32(2.0172658) * state["fRec15"][2]))) 
		fTemp34 = (state["fRec9"] * jnp.where((iSlow15 != 0), inputs[0], jnp.where((iSlow16 != 0), jnp.where((iSlow26 != 0), (((jnp.float32(0.049922034) * state["fRec15"][0]) + (jnp.float32(0.0506127) * state["fRec15"][2])) - ((jnp.float32(0.095993534) * state["fRec15"][1]) + (jnp.float32(0.004408786) * state["fRec15"][3]))), fTemp33), (jnp.float32(0.33333334) * (state["fRec9"] * ((jnp.where((iSlow18 != 0), jnp.where((iSlow23 != 0), (self._fConst18 * ((fTemp9 * (fTemp12 - fVec6_temp)) / fTemp0)), (self._fConst17 * ((fTemp6 * (fTemp8 - fVec3_temp)) / fTemp0))), jnp.where((iSlow19 != 0), (self._fConst16 * ((fTemp4 * (fTemp5 - fVec1_temp)) / fTemp0)), fTemp3)) + jnp.where((iSlow18 != 0), jnp.where((iSlow23 != 0), (self._fConst18 * ((fTemp9 * (fTemp22 - fVec12_temp)) / fTemp13)), (self._fConst17 * ((fTemp6 * (fTemp19 - fVec9_temp)) / fTemp13))), jnp.where((iSlow19 != 0), (self._fConst16 * ((fTemp4 * (fTemp17 - fVec7_temp)) / fTemp13)), fTemp16))) + jnp.where((iSlow18 != 0), jnp.where((iSlow23 != 0), (self._fConst18 * ((fTemp9 * (fTemp32 - fVec18_temp)) / fTemp23)), (self._fConst17 * ((fTemp6 * (fTemp29 - fVec15_temp)) / fTemp23))), jnp.where((iSlow19 != 0), (self._fConst16 * ((fTemp4 * (fTemp27 - fVec13_temp)) / fTemp23)), fTemp26)))))))) 
		state["fVec19"] = jnp.float32(fTemp34) 
		state["fRec8"] = -((fSlow12 * ((fSlow13 * fRec8_temp) - (fSlow9 * (fTemp34 - fVec19_temp))))) 
		state["fRec7"] = state["fRec7"].at[0].set((state["fRec8"] - (fSlow10 * ((fSlow27 * state["fRec7"][2]) + (fSlow28 * state["fRec7"][1]))))) 
		state["fRec17"] = -((fSlow12 * ((fSlow13 * fRec17_temp) - (fTemp34 + fVec19_temp)))) 
		state["fRec16"] = state["fRec16"].at[0].set((state["fRec17"] - (fSlow10 * ((fSlow27 * state["fRec16"][2]) + (fSlow28 * state["fRec16"][1]))))) 
		state["fRec18"] = (fSlow30 + (jnp.float32(0.999) * fRec18_temp)) 
		fTemp35 = jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (state["fRec18"] + jnp.float32(-49.0)))) 
		fTemp36 = jnp.tan((self._fConst19 * fTemp35)) 
		fTemp37 = (jnp.float32(1.0) / fTemp36) 
		fTemp38 = (fTemp35 / jnp.sin((self._fConst20 * fTemp35))) 
		fTemp39 = (fSlow34 * fTemp38) 
		fTemp40 = (fSlow35 * fTemp38) 
		fTemp41 = jnp.where((iSlow32 != 0), fTemp40, fTemp39) 
		fTemp42 = (jnp.float32(2.0) * (state["fRec6"][1] * (jnp.float32(1.0) - (jnp.float32(1.0) / jnp.power(fTemp36, jnp.float32(2.0)))))) 
		fTemp43 = (((fTemp37 + fTemp41) / fTemp36) + jnp.float32(1.0)) 
		state["fRec6"] = state["fRec6"].at[0].set(((fSlow10 * ((fSlow11 * (state["fRec7"][2] + (state["fRec7"][0] - (jnp.float32(2.0) * state["fRec7"][1])))) + (fSlow29 * (state["fRec16"][2] + (state["fRec16"][0] + (jnp.float32(2.0) * state["fRec16"][1])))))) - (((state["fRec6"][2] * (((fTemp37 - fTemp41) / fTemp36) + jnp.float32(1.0))) + fTemp42) / fTemp43))) 
		fTemp44 = jnp.where((iSlow32 != 0), fTemp39, fTemp40) 
		fTemp45 = (((fTemp42 + (state["fRec6"][0] * (((fTemp37 + fTemp44) / fTemp36) + jnp.float32(1.0)))) + (state["fRec6"][2] * (((fTemp37 - fTemp44) / fTemp36) + jnp.float32(1.0)))) / fTemp43) 
		state["fVec20"] = jnp.float32(fTemp45) 
		state["fRec5"] = -((fSlow6 * ((fSlow7 * fRec5_temp) - (fTemp45 + fVec20_temp)))) 
		state["fRec4"] = state["fRec4"].at[0].set((state["fRec5"] - (fSlow5 * ((fSlow36 * state["fRec4"][2]) + (fSlow38 * state["fRec4"][1]))))) 
		state["fRec20"] = -((fSlow6 * ((fSlow7 * fRec20_temp) - (fSlow4 * (fTemp45 - fVec20_temp))))) 
		state["fRec19"] = state["fRec19"].at[0].set((state["fRec20"] - (fSlow5 * ((fSlow36 * state["fRec19"][2]) + (fSlow38 * state["fRec19"][1]))))) 
		fTemp46 = (fSlow5 * ((state["fRec4"][2] + (state["fRec4"][0] + (jnp.float32(2.0) * state["fRec4"][1]))) + (fSlow39 * (state["fRec19"][2] + (state["fRec19"][0] - (jnp.float32(2.0) * state["fRec19"][1])))))) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp46 - (self._fConst11 * ((self._fConst21 * state["fRec3"][2]) + (self._fConst23 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst11 * (((self._fConst13 * state["fRec3"][0]) + (self._fConst24 * state["fRec3"][1])) + (self._fConst13 * state["fRec3"][2]))) - (self._fConst8 * ((self._fConst25 * state["fRec2"][2]) + (self._fConst26 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst8 * (((self._fConst10 * state["fRec2"][0]) + (self._fConst27 * state["fRec2"][1])) + (self._fConst10 * state["fRec2"][2]))) - (self._fConst4 * ((self._fConst28 * state["fRec1"][2]) + (self._fConst29 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fRec0_temp * fSlow2) + (jnp.abs((self._fConst4 * (((self._fConst7 * state["fRec1"][0]) + (self._fConst30 * state["fRec1"][1])) + (self._fConst7 * state["fRec1"][2])))) * fSlow40)) 
		fVbargraph0 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		# self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec27"] = state["fRec27"].at[0].set((fTemp46 - (self._fConst47 * ((self._fConst50 * state["fRec27"][2]) + (self._fConst51 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((self._fConst47 * (((self._fConst49 * state["fRec27"][0]) + (self._fConst52 * state["fRec27"][1])) + (self._fConst49 * state["fRec27"][2]))) - (self._fConst45 * ((self._fConst53 * state["fRec26"][2]) + (self._fConst54 * state["fRec26"][1]))))) 
		state["fRec25"] = state["fRec25"].at[0].set(((self._fConst45 * (((self._fConst46 * state["fRec26"][0]) + (self._fConst55 * state["fRec26"][1])) + (self._fConst46 * state["fRec26"][2]))) - (self._fConst43 * ((self._fConst56 * state["fRec25"][2]) + (self._fConst57 * state["fRec25"][1]))))) 
		fTemp47 = (self._fConst43 * (((self._fConst44 * state["fRec25"][0]) + (self._fConst58 * state["fRec25"][1])) + (self._fConst44 * state["fRec25"][2]))) 
		state["fRec24"] = state["fRec24"].at[0].set((fTemp47 - (self._fConst40 * ((self._fConst59 * state["fRec24"][2]) + (self._fConst61 * state["fRec24"][1]))))) 
		state["fRec23"] = state["fRec23"].at[0].set(((self._fConst40 * (((self._fConst42 * state["fRec24"][0]) + (self._fConst62 * state["fRec24"][1])) + (self._fConst42 * state["fRec24"][2]))) - (self._fConst37 * ((self._fConst63 * state["fRec23"][2]) + (self._fConst64 * state["fRec23"][1]))))) 
		state["fRec22"] = state["fRec22"].at[0].set(((self._fConst37 * (((self._fConst39 * state["fRec23"][0]) + (self._fConst65 * state["fRec23"][1])) + (self._fConst39 * state["fRec23"][2]))) - (self._fConst33 * ((self._fConst66 * state["fRec22"][2]) + (self._fConst67 * state["fRec22"][1]))))) 
		state["fRec21"] = ((fSlow2 * fRec21_temp) + (fSlow40 * jnp.abs((self._fConst33 * (((self._fConst36 * state["fRec22"][0]) + (self._fConst68 * state["fRec22"][1])) + (self._fConst36 * state["fRec22"][2])))))) 
		fVbargraph1 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec21"])))
		# self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec34"] = state["fRec34"].at[0].set((fTemp47 - (self._fConst85 * ((self._fConst88 * state["fRec34"][2]) + (self._fConst89 * state["fRec34"][1]))))) 
		state["fRec33"] = state["fRec33"].at[0].set(((self._fConst85 * (((self._fConst87 * state["fRec34"][0]) + (self._fConst90 * state["fRec34"][1])) + (self._fConst87 * state["fRec34"][2]))) - (self._fConst83 * ((self._fConst91 * state["fRec33"][2]) + (self._fConst92 * state["fRec33"][1]))))) 
		state["fRec32"] = state["fRec32"].at[0].set(((self._fConst83 * (((self._fConst84 * state["fRec33"][0]) + (self._fConst93 * state["fRec33"][1])) + (self._fConst84 * state["fRec33"][2]))) - (self._fConst81 * ((self._fConst94 * state["fRec32"][2]) + (self._fConst95 * state["fRec32"][1]))))) 
		fTemp48 = (self._fConst81 * (((self._fConst82 * state["fRec32"][0]) + (self._fConst96 * state["fRec32"][1])) + (self._fConst82 * state["fRec32"][2]))) 
		state["fRec31"] = state["fRec31"].at[0].set((fTemp48 - (self._fConst78 * ((self._fConst97 * state["fRec31"][2]) + (self._fConst99 * state["fRec31"][1]))))) 
		state["fRec30"] = state["fRec30"].at[0].set(((self._fConst78 * (((self._fConst80 * state["fRec31"][0]) + (self._fConst100 * state["fRec31"][1])) + (self._fConst80 * state["fRec31"][2]))) - (self._fConst75 * ((self._fConst101 * state["fRec30"][2]) + (self._fConst102 * state["fRec30"][1]))))) 
		state["fRec29"] = state["fRec29"].at[0].set(((self._fConst75 * (((self._fConst77 * state["fRec30"][0]) + (self._fConst103 * state["fRec30"][1])) + (self._fConst77 * state["fRec30"][2]))) - (self._fConst71 * ((self._fConst104 * state["fRec29"][2]) + (self._fConst105 * state["fRec29"][1]))))) 
		state["fRec28"] = ((fSlow2 * fRec28_temp) + (fSlow40 * jnp.abs((self._fConst71 * (((self._fConst74 * state["fRec29"][0]) + (self._fConst106 * state["fRec29"][1])) + (self._fConst74 * state["fRec29"][2])))))) 
		fVbargraph2 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec28"])))
		# self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec41"] = state["fRec41"].at[0].set((fTemp48 - (self._fConst123 * ((self._fConst126 * state["fRec41"][2]) + (self._fConst127 * state["fRec41"][1]))))) 
		state["fRec40"] = state["fRec40"].at[0].set(((self._fConst123 * (((self._fConst125 * state["fRec41"][0]) + (self._fConst128 * state["fRec41"][1])) + (self._fConst125 * state["fRec41"][2]))) - (self._fConst121 * ((self._fConst129 * state["fRec40"][2]) + (self._fConst130 * state["fRec40"][1]))))) 
		state["fRec39"] = state["fRec39"].at[0].set(((self._fConst121 * (((self._fConst122 * state["fRec40"][0]) + (self._fConst131 * state["fRec40"][1])) + (self._fConst122 * state["fRec40"][2]))) - (self._fConst119 * ((self._fConst132 * state["fRec39"][2]) + (self._fConst133 * state["fRec39"][1]))))) 
		fTemp49 = (self._fConst119 * (((self._fConst120 * state["fRec39"][0]) + (self._fConst134 * state["fRec39"][1])) + (self._fConst120 * state["fRec39"][2]))) 
		state["fRec38"] = state["fRec38"].at[0].set((fTemp49 - (self._fConst116 * ((self._fConst135 * state["fRec38"][2]) + (self._fConst137 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst116 * (((self._fConst118 * state["fRec38"][0]) + (self._fConst138 * state["fRec38"][1])) + (self._fConst118 * state["fRec38"][2]))) - (self._fConst113 * ((self._fConst139 * state["fRec37"][2]) + (self._fConst140 * state["fRec37"][1]))))) 
		state["fRec36"] = state["fRec36"].at[0].set(((self._fConst113 * (((self._fConst115 * state["fRec37"][0]) + (self._fConst141 * state["fRec37"][1])) + (self._fConst115 * state["fRec37"][2]))) - (self._fConst109 * ((self._fConst142 * state["fRec36"][2]) + (self._fConst143 * state["fRec36"][1]))))) 
		state["fRec35"] = ((fSlow2 * fRec35_temp) + (fSlow40 * jnp.abs((self._fConst109 * (((self._fConst112 * state["fRec36"][0]) + (self._fConst144 * state["fRec36"][1])) + (self._fConst112 * state["fRec36"][2])))))) 
		fVbargraph3 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec35"])))
		# self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec48"] = state["fRec48"].at[0].set((fTemp49 - (self._fConst161 * ((self._fConst164 * state["fRec48"][2]) + (self._fConst165 * state["fRec48"][1]))))) 
		state["fRec47"] = state["fRec47"].at[0].set(((self._fConst161 * (((self._fConst163 * state["fRec48"][0]) + (self._fConst166 * state["fRec48"][1])) + (self._fConst163 * state["fRec48"][2]))) - (self._fConst159 * ((self._fConst167 * state["fRec47"][2]) + (self._fConst168 * state["fRec47"][1]))))) 
		state["fRec46"] = state["fRec46"].at[0].set(((self._fConst159 * (((self._fConst160 * state["fRec47"][0]) + (self._fConst169 * state["fRec47"][1])) + (self._fConst160 * state["fRec47"][2]))) - (self._fConst157 * ((self._fConst170 * state["fRec46"][2]) + (self._fConst171 * state["fRec46"][1]))))) 
		fTemp50 = (self._fConst157 * (((self._fConst158 * state["fRec46"][0]) + (self._fConst172 * state["fRec46"][1])) + (self._fConst158 * state["fRec46"][2]))) 
		state["fRec45"] = state["fRec45"].at[0].set((fTemp50 - (self._fConst154 * ((self._fConst173 * state["fRec45"][2]) + (self._fConst175 * state["fRec45"][1]))))) 
		state["fRec44"] = state["fRec44"].at[0].set(((self._fConst154 * (((self._fConst156 * state["fRec45"][0]) + (self._fConst176 * state["fRec45"][1])) + (self._fConst156 * state["fRec45"][2]))) - (self._fConst151 * ((self._fConst177 * state["fRec44"][2]) + (self._fConst178 * state["fRec44"][1]))))) 
		state["fRec43"] = state["fRec43"].at[0].set(((self._fConst151 * (((self._fConst153 * state["fRec44"][0]) + (self._fConst179 * state["fRec44"][1])) + (self._fConst153 * state["fRec44"][2]))) - (self._fConst147 * ((self._fConst180 * state["fRec43"][2]) + (self._fConst181 * state["fRec43"][1]))))) 
		state["fRec42"] = ((fSlow2 * fRec42_temp) + (fSlow40 * jnp.abs((self._fConst147 * (((self._fConst150 * state["fRec43"][0]) + (self._fConst182 * state["fRec43"][1])) + (self._fConst150 * state["fRec43"][2])))))) 
		fVbargraph4 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec42"])))
		# self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec55"] = state["fRec55"].at[0].set((fTemp50 - (self._fConst199 * ((self._fConst202 * state["fRec55"][2]) + (self._fConst203 * state["fRec55"][1]))))) 
		state["fRec54"] = state["fRec54"].at[0].set(((self._fConst199 * (((self._fConst201 * state["fRec55"][0]) + (self._fConst204 * state["fRec55"][1])) + (self._fConst201 * state["fRec55"][2]))) - (self._fConst197 * ((self._fConst205 * state["fRec54"][2]) + (self._fConst206 * state["fRec54"][1]))))) 
		state["fRec53"] = state["fRec53"].at[0].set(((self._fConst197 * (((self._fConst198 * state["fRec54"][0]) + (self._fConst207 * state["fRec54"][1])) + (self._fConst198 * state["fRec54"][2]))) - (self._fConst195 * ((self._fConst208 * state["fRec53"][2]) + (self._fConst209 * state["fRec53"][1]))))) 
		fTemp51 = (self._fConst195 * (((self._fConst196 * state["fRec53"][0]) + (self._fConst210 * state["fRec53"][1])) + (self._fConst196 * state["fRec53"][2]))) 
		state["fRec52"] = state["fRec52"].at[0].set((fTemp51 - (self._fConst192 * ((self._fConst211 * state["fRec52"][2]) + (self._fConst213 * state["fRec52"][1]))))) 
		state["fRec51"] = state["fRec51"].at[0].set(((self._fConst192 * (((self._fConst194 * state["fRec52"][0]) + (self._fConst214 * state["fRec52"][1])) + (self._fConst194 * state["fRec52"][2]))) - (self._fConst189 * ((self._fConst215 * state["fRec51"][2]) + (self._fConst216 * state["fRec51"][1]))))) 
		state["fRec50"] = state["fRec50"].at[0].set(((self._fConst189 * (((self._fConst191 * state["fRec51"][0]) + (self._fConst217 * state["fRec51"][1])) + (self._fConst191 * state["fRec51"][2]))) - (self._fConst185 * ((self._fConst218 * state["fRec50"][2]) + (self._fConst219 * state["fRec50"][1]))))) 
		state["fRec49"] = ((fSlow2 * fRec49_temp) + (fSlow40 * jnp.abs((self._fConst185 * (((self._fConst188 * state["fRec50"][0]) + (self._fConst220 * state["fRec50"][1])) + (self._fConst188 * state["fRec50"][2])))))) 
		fVbargraph5 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec49"])))
		# self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec62"] = state["fRec62"].at[0].set((fTemp51 - (self._fConst237 * ((self._fConst240 * state["fRec62"][2]) + (self._fConst241 * state["fRec62"][1]))))) 
		state["fRec61"] = state["fRec61"].at[0].set(((self._fConst237 * (((self._fConst239 * state["fRec62"][0]) + (self._fConst242 * state["fRec62"][1])) + (self._fConst239 * state["fRec62"][2]))) - (self._fConst235 * ((self._fConst243 * state["fRec61"][2]) + (self._fConst244 * state["fRec61"][1]))))) 
		state["fRec60"] = state["fRec60"].at[0].set(((self._fConst235 * (((self._fConst236 * state["fRec61"][0]) + (self._fConst245 * state["fRec61"][1])) + (self._fConst236 * state["fRec61"][2]))) - (self._fConst233 * ((self._fConst246 * state["fRec60"][2]) + (self._fConst247 * state["fRec60"][1]))))) 
		fTemp52 = (self._fConst233 * (((self._fConst234 * state["fRec60"][0]) + (self._fConst248 * state["fRec60"][1])) + (self._fConst234 * state["fRec60"][2]))) 
		state["fRec59"] = state["fRec59"].at[0].set((fTemp52 - (self._fConst230 * ((self._fConst249 * state["fRec59"][2]) + (self._fConst251 * state["fRec59"][1]))))) 
		state["fRec58"] = state["fRec58"].at[0].set(((self._fConst230 * (((self._fConst232 * state["fRec59"][0]) + (self._fConst252 * state["fRec59"][1])) + (self._fConst232 * state["fRec59"][2]))) - (self._fConst227 * ((self._fConst253 * state["fRec58"][2]) + (self._fConst254 * state["fRec58"][1]))))) 
		state["fRec57"] = state["fRec57"].at[0].set(((self._fConst227 * (((self._fConst229 * state["fRec58"][0]) + (self._fConst255 * state["fRec58"][1])) + (self._fConst229 * state["fRec58"][2]))) - (self._fConst223 * ((self._fConst256 * state["fRec57"][2]) + (self._fConst257 * state["fRec57"][1]))))) 
		state["fRec56"] = ((fSlow2 * fRec56_temp) + (fSlow40 * jnp.abs((self._fConst223 * (((self._fConst226 * state["fRec57"][0]) + (self._fConst258 * state["fRec57"][1])) + (self._fConst226 * state["fRec57"][2])))))) 
		fVbargraph6 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec56"])))
		# self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec69"] = state["fRec69"].at[0].set((fTemp52 - (self._fConst275 * ((self._fConst278 * state["fRec69"][2]) + (self._fConst279 * state["fRec69"][1]))))) 
		state["fRec68"] = state["fRec68"].at[0].set(((self._fConst275 * (((self._fConst277 * state["fRec69"][0]) + (self._fConst280 * state["fRec69"][1])) + (self._fConst277 * state["fRec69"][2]))) - (self._fConst273 * ((self._fConst281 * state["fRec68"][2]) + (self._fConst282 * state["fRec68"][1]))))) 
		state["fRec67"] = state["fRec67"].at[0].set(((self._fConst273 * (((self._fConst274 * state["fRec68"][0]) + (self._fConst283 * state["fRec68"][1])) + (self._fConst274 * state["fRec68"][2]))) - (self._fConst271 * ((self._fConst284 * state["fRec67"][2]) + (self._fConst285 * state["fRec67"][1]))))) 
		fTemp53 = (self._fConst271 * (((self._fConst272 * state["fRec67"][0]) + (self._fConst286 * state["fRec67"][1])) + (self._fConst272 * state["fRec67"][2]))) 
		state["fRec66"] = state["fRec66"].at[0].set((fTemp53 - (self._fConst268 * ((self._fConst287 * state["fRec66"][2]) + (self._fConst289 * state["fRec66"][1]))))) 
		state["fRec65"] = state["fRec65"].at[0].set(((self._fConst268 * (((self._fConst270 * state["fRec66"][0]) + (self._fConst290 * state["fRec66"][1])) + (self._fConst270 * state["fRec66"][2]))) - (self._fConst265 * ((self._fConst291 * state["fRec65"][2]) + (self._fConst292 * state["fRec65"][1]))))) 
		state["fRec64"] = state["fRec64"].at[0].set(((self._fConst265 * (((self._fConst267 * state["fRec65"][0]) + (self._fConst293 * state["fRec65"][1])) + (self._fConst267 * state["fRec65"][2]))) - (self._fConst261 * ((self._fConst294 * state["fRec64"][2]) + (self._fConst295 * state["fRec64"][1]))))) 
		state["fRec63"] = ((fSlow2 * fRec63_temp) + (fSlow40 * jnp.abs((self._fConst261 * (((self._fConst264 * state["fRec64"][0]) + (self._fConst296 * state["fRec64"][1])) + (self._fConst264 * state["fRec64"][2])))))) 
		fVbargraph7 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec63"])))
		# self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec76"] = state["fRec76"].at[0].set((fTemp53 - (self._fConst313 * ((self._fConst316 * state["fRec76"][2]) + (self._fConst317 * state["fRec76"][1]))))) 
		state["fRec75"] = state["fRec75"].at[0].set(((self._fConst313 * (((self._fConst315 * state["fRec76"][0]) + (self._fConst318 * state["fRec76"][1])) + (self._fConst315 * state["fRec76"][2]))) - (self._fConst311 * ((self._fConst319 * state["fRec75"][2]) + (self._fConst320 * state["fRec75"][1]))))) 
		state["fRec74"] = state["fRec74"].at[0].set(((self._fConst311 * (((self._fConst312 * state["fRec75"][0]) + (self._fConst321 * state["fRec75"][1])) + (self._fConst312 * state["fRec75"][2]))) - (self._fConst309 * ((self._fConst322 * state["fRec74"][2]) + (self._fConst323 * state["fRec74"][1]))))) 
		fTemp54 = (self._fConst309 * (((self._fConst310 * state["fRec74"][0]) + (self._fConst324 * state["fRec74"][1])) + (self._fConst310 * state["fRec74"][2]))) 
		state["fRec73"] = state["fRec73"].at[0].set((fTemp54 - (self._fConst306 * ((self._fConst325 * state["fRec73"][2]) + (self._fConst327 * state["fRec73"][1]))))) 
		state["fRec72"] = state["fRec72"].at[0].set(((self._fConst306 * (((self._fConst308 * state["fRec73"][0]) + (self._fConst328 * state["fRec73"][1])) + (self._fConst308 * state["fRec73"][2]))) - (self._fConst303 * ((self._fConst329 * state["fRec72"][2]) + (self._fConst330 * state["fRec72"][1]))))) 
		state["fRec71"] = state["fRec71"].at[0].set(((self._fConst303 * (((self._fConst305 * state["fRec72"][0]) + (self._fConst331 * state["fRec72"][1])) + (self._fConst305 * state["fRec72"][2]))) - (self._fConst299 * ((self._fConst332 * state["fRec71"][2]) + (self._fConst333 * state["fRec71"][1]))))) 
		state["fRec70"] = ((fSlow2 * fRec70_temp) + (fSlow40 * jnp.abs((self._fConst299 * (((self._fConst302 * state["fRec71"][0]) + (self._fConst334 * state["fRec71"][1])) + (self._fConst302 * state["fRec71"][2])))))) 
		fVbargraph8 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec70"])))
		# self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec83"] = state["fRec83"].at[0].set((fTemp54 - (self._fConst351 * ((self._fConst354 * state["fRec83"][2]) + (self._fConst355 * state["fRec83"][1]))))) 
		state["fRec82"] = state["fRec82"].at[0].set(((self._fConst351 * (((self._fConst353 * state["fRec83"][0]) + (self._fConst356 * state["fRec83"][1])) + (self._fConst353 * state["fRec83"][2]))) - (self._fConst349 * ((self._fConst357 * state["fRec82"][2]) + (self._fConst358 * state["fRec82"][1]))))) 
		state["fRec81"] = state["fRec81"].at[0].set(((self._fConst349 * (((self._fConst350 * state["fRec82"][0]) + (self._fConst359 * state["fRec82"][1])) + (self._fConst350 * state["fRec82"][2]))) - (self._fConst347 * ((self._fConst360 * state["fRec81"][2]) + (self._fConst361 * state["fRec81"][1]))))) 
		fTemp55 = (self._fConst347 * (((self._fConst348 * state["fRec81"][0]) + (self._fConst362 * state["fRec81"][1])) + (self._fConst348 * state["fRec81"][2]))) 
		state["fRec80"] = state["fRec80"].at[0].set((fTemp55 - (self._fConst344 * ((self._fConst363 * state["fRec80"][2]) + (self._fConst365 * state["fRec80"][1]))))) 
		state["fRec79"] = state["fRec79"].at[0].set(((self._fConst344 * (((self._fConst346 * state["fRec80"][0]) + (self._fConst366 * state["fRec80"][1])) + (self._fConst346 * state["fRec80"][2]))) - (self._fConst341 * ((self._fConst367 * state["fRec79"][2]) + (self._fConst368 * state["fRec79"][1]))))) 
		state["fRec78"] = state["fRec78"].at[0].set(((self._fConst341 * (((self._fConst343 * state["fRec79"][0]) + (self._fConst369 * state["fRec79"][1])) + (self._fConst343 * state["fRec79"][2]))) - (self._fConst337 * ((self._fConst370 * state["fRec78"][2]) + (self._fConst371 * state["fRec78"][1]))))) 
		state["fRec77"] = ((fSlow2 * fRec77_temp) + (fSlow40 * jnp.abs((self._fConst337 * (((self._fConst340 * state["fRec78"][0]) + (self._fConst372 * state["fRec78"][1])) + (self._fConst340 * state["fRec78"][2])))))) 
		fVbargraph9 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec77"])))
		# self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec90"] = state["fRec90"].at[0].set((fTemp55 - (self._fConst389 * ((self._fConst392 * state["fRec90"][2]) + (self._fConst393 * state["fRec90"][1]))))) 
		state["fRec89"] = state["fRec89"].at[0].set(((self._fConst389 * (((self._fConst391 * state["fRec90"][0]) + (self._fConst394 * state["fRec90"][1])) + (self._fConst391 * state["fRec90"][2]))) - (self._fConst387 * ((self._fConst395 * state["fRec89"][2]) + (self._fConst396 * state["fRec89"][1]))))) 
		state["fRec88"] = state["fRec88"].at[0].set(((self._fConst387 * (((self._fConst388 * state["fRec89"][0]) + (self._fConst397 * state["fRec89"][1])) + (self._fConst388 * state["fRec89"][2]))) - (self._fConst385 * ((self._fConst398 * state["fRec88"][2]) + (self._fConst399 * state["fRec88"][1]))))) 
		fTemp56 = (self._fConst385 * (((self._fConst386 * state["fRec88"][0]) + (self._fConst400 * state["fRec88"][1])) + (self._fConst386 * state["fRec88"][2]))) 
		state["fRec87"] = state["fRec87"].at[0].set((fTemp56 - (self._fConst382 * ((self._fConst401 * state["fRec87"][2]) + (self._fConst403 * state["fRec87"][1]))))) 
		state["fRec86"] = state["fRec86"].at[0].set(((self._fConst382 * (((self._fConst384 * state["fRec87"][0]) + (self._fConst404 * state["fRec87"][1])) + (self._fConst384 * state["fRec87"][2]))) - (self._fConst379 * ((self._fConst405 * state["fRec86"][2]) + (self._fConst406 * state["fRec86"][1]))))) 
		state["fRec85"] = state["fRec85"].at[0].set(((self._fConst379 * (((self._fConst381 * state["fRec86"][0]) + (self._fConst407 * state["fRec86"][1])) + (self._fConst381 * state["fRec86"][2]))) - (self._fConst375 * ((self._fConst408 * state["fRec85"][2]) + (self._fConst409 * state["fRec85"][1]))))) 
		state["fRec84"] = ((fSlow2 * fRec84_temp) + (fSlow40 * jnp.abs((self._fConst375 * (((self._fConst378 * state["fRec85"][0]) + (self._fConst410 * state["fRec85"][1])) + (self._fConst378 * state["fRec85"][2])))))) 
		fVbargraph10 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec84"])))
		# self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec97"] = state["fRec97"].at[0].set((fTemp56 - (self._fConst427 * ((self._fConst430 * state["fRec97"][2]) + (self._fConst431 * state["fRec97"][1]))))) 
		state["fRec96"] = state["fRec96"].at[0].set(((self._fConst427 * (((self._fConst429 * state["fRec97"][0]) + (self._fConst432 * state["fRec97"][1])) + (self._fConst429 * state["fRec97"][2]))) - (self._fConst425 * ((self._fConst433 * state["fRec96"][2]) + (self._fConst434 * state["fRec96"][1]))))) 
		state["fRec95"] = state["fRec95"].at[0].set(((self._fConst425 * (((self._fConst426 * state["fRec96"][0]) + (self._fConst435 * state["fRec96"][1])) + (self._fConst426 * state["fRec96"][2]))) - (self._fConst423 * ((self._fConst436 * state["fRec95"][2]) + (self._fConst437 * state["fRec95"][1]))))) 
		fTemp57 = (self._fConst423 * (((self._fConst424 * state["fRec95"][0]) + (self._fConst438 * state["fRec95"][1])) + (self._fConst424 * state["fRec95"][2]))) 
		state["fRec94"] = state["fRec94"].at[0].set((fTemp57 - (self._fConst420 * ((self._fConst439 * state["fRec94"][2]) + (self._fConst441 * state["fRec94"][1]))))) 
		state["fRec93"] = state["fRec93"].at[0].set(((self._fConst420 * (((self._fConst422 * state["fRec94"][0]) + (self._fConst442 * state["fRec94"][1])) + (self._fConst422 * state["fRec94"][2]))) - (self._fConst417 * ((self._fConst443 * state["fRec93"][2]) + (self._fConst444 * state["fRec93"][1]))))) 
		state["fRec92"] = state["fRec92"].at[0].set(((self._fConst417 * (((self._fConst419 * state["fRec93"][0]) + (self._fConst445 * state["fRec93"][1])) + (self._fConst419 * state["fRec93"][2]))) - (self._fConst413 * ((self._fConst446 * state["fRec92"][2]) + (self._fConst447 * state["fRec92"][1]))))) 
		state["fRec91"] = ((fSlow2 * fRec91_temp) + (fSlow40 * jnp.abs((self._fConst413 * (((self._fConst416 * state["fRec92"][0]) + (self._fConst448 * state["fRec92"][1])) + (self._fConst416 * state["fRec92"][2])))))) 
		fVbargraph11 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec91"])))
		# self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec104"] = state["fRec104"].at[0].set((fTemp57 - (self._fConst465 * ((self._fConst468 * state["fRec104"][2]) + (self._fConst469 * state["fRec104"][1]))))) 
		state["fRec103"] = state["fRec103"].at[0].set(((self._fConst465 * (((self._fConst467 * state["fRec104"][0]) + (self._fConst470 * state["fRec104"][1])) + (self._fConst467 * state["fRec104"][2]))) - (self._fConst463 * ((self._fConst471 * state["fRec103"][2]) + (self._fConst472 * state["fRec103"][1]))))) 
		state["fRec102"] = state["fRec102"].at[0].set(((self._fConst463 * (((self._fConst464 * state["fRec103"][0]) + (self._fConst473 * state["fRec103"][1])) + (self._fConst464 * state["fRec103"][2]))) - (self._fConst461 * ((self._fConst474 * state["fRec102"][2]) + (self._fConst475 * state["fRec102"][1]))))) 
		fTemp58 = (self._fConst461 * (((self._fConst462 * state["fRec102"][0]) + (self._fConst476 * state["fRec102"][1])) + (self._fConst462 * state["fRec102"][2]))) 
		state["fRec101"] = state["fRec101"].at[0].set((fTemp58 - (self._fConst458 * ((self._fConst477 * state["fRec101"][2]) + (self._fConst479 * state["fRec101"][1]))))) 
		state["fRec100"] = state["fRec100"].at[0].set(((self._fConst458 * (((self._fConst460 * state["fRec101"][0]) + (self._fConst480 * state["fRec101"][1])) + (self._fConst460 * state["fRec101"][2]))) - (self._fConst455 * ((self._fConst481 * state["fRec100"][2]) + (self._fConst482 * state["fRec100"][1]))))) 
		state["fRec99"] = state["fRec99"].at[0].set(((self._fConst455 * (((self._fConst457 * state["fRec100"][0]) + (self._fConst483 * state["fRec100"][1])) + (self._fConst457 * state["fRec100"][2]))) - (self._fConst451 * ((self._fConst484 * state["fRec99"][2]) + (self._fConst485 * state["fRec99"][1]))))) 
		state["fRec98"] = ((fSlow2 * fRec98_temp) + (fSlow40 * jnp.abs((self._fConst451 * (((self._fConst454 * state["fRec99"][0]) + (self._fConst486 * state["fRec99"][1])) + (self._fConst454 * state["fRec99"][2])))))) 
		fVbargraph12 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec98"])))
		# self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec111"] = state["fRec111"].at[0].set((fTemp58 - (self._fConst503 * ((self._fConst506 * state["fRec111"][2]) + (self._fConst507 * state["fRec111"][1]))))) 
		state["fRec110"] = state["fRec110"].at[0].set(((self._fConst503 * (((self._fConst505 * state["fRec111"][0]) + (self._fConst508 * state["fRec111"][1])) + (self._fConst505 * state["fRec111"][2]))) - (self._fConst501 * ((self._fConst509 * state["fRec110"][2]) + (self._fConst510 * state["fRec110"][1]))))) 
		state["fRec109"] = state["fRec109"].at[0].set(((self._fConst501 * (((self._fConst502 * state["fRec110"][0]) + (self._fConst511 * state["fRec110"][1])) + (self._fConst502 * state["fRec110"][2]))) - (self._fConst499 * ((self._fConst512 * state["fRec109"][2]) + (self._fConst513 * state["fRec109"][1]))))) 
		fTemp59 = (self._fConst499 * (((self._fConst500 * state["fRec109"][0]) + (self._fConst514 * state["fRec109"][1])) + (self._fConst500 * state["fRec109"][2]))) 
		state["fRec108"] = state["fRec108"].at[0].set((fTemp59 - (self._fConst496 * ((self._fConst515 * state["fRec108"][2]) + (self._fConst517 * state["fRec108"][1]))))) 
		state["fRec107"] = state["fRec107"].at[0].set(((self._fConst496 * (((self._fConst498 * state["fRec108"][0]) + (self._fConst518 * state["fRec108"][1])) + (self._fConst498 * state["fRec108"][2]))) - (self._fConst493 * ((self._fConst519 * state["fRec107"][2]) + (self._fConst520 * state["fRec107"][1]))))) 
		state["fRec106"] = state["fRec106"].at[0].set(((self._fConst493 * (((self._fConst495 * state["fRec107"][0]) + (self._fConst521 * state["fRec107"][1])) + (self._fConst495 * state["fRec107"][2]))) - (self._fConst489 * ((self._fConst522 * state["fRec106"][2]) + (self._fConst523 * state["fRec106"][1]))))) 
		state["fRec105"] = ((fSlow2 * fRec105_temp) + (fSlow40 * jnp.abs((self._fConst489 * (((self._fConst492 * state["fRec106"][0]) + (self._fConst524 * state["fRec106"][1])) + (self._fConst492 * state["fRec106"][2])))))) 
		fVbargraph13 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec105"])))
		# self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec118"] = state["fRec118"].at[0].set((fTemp59 - (self._fConst541 * ((self._fConst544 * state["fRec118"][2]) + (self._fConst545 * state["fRec118"][1]))))) 
		state["fRec117"] = state["fRec117"].at[0].set(((self._fConst541 * (((self._fConst543 * state["fRec118"][0]) + (self._fConst546 * state["fRec118"][1])) + (self._fConst543 * state["fRec118"][2]))) - (self._fConst539 * ((self._fConst547 * state["fRec117"][2]) + (self._fConst548 * state["fRec117"][1]))))) 
		state["fRec116"] = state["fRec116"].at[0].set(((self._fConst539 * (((self._fConst540 * state["fRec117"][0]) + (self._fConst549 * state["fRec117"][1])) + (self._fConst540 * state["fRec117"][2]))) - (self._fConst537 * ((self._fConst550 * state["fRec116"][2]) + (self._fConst551 * state["fRec116"][1]))))) 
		fTemp60 = (self._fConst537 * (((self._fConst538 * state["fRec116"][0]) + (self._fConst552 * state["fRec116"][1])) + (self._fConst538 * state["fRec116"][2]))) 
		state["fRec115"] = state["fRec115"].at[0].set((fTemp60 - (self._fConst534 * ((self._fConst553 * state["fRec115"][2]) + (self._fConst555 * state["fRec115"][1]))))) 
		state["fRec114"] = state["fRec114"].at[0].set(((self._fConst534 * (((self._fConst536 * state["fRec115"][0]) + (self._fConst556 * state["fRec115"][1])) + (self._fConst536 * state["fRec115"][2]))) - (self._fConst531 * ((self._fConst557 * state["fRec114"][2]) + (self._fConst558 * state["fRec114"][1]))))) 
		state["fRec113"] = state["fRec113"].at[0].set(((self._fConst531 * (((self._fConst533 * state["fRec114"][0]) + (self._fConst559 * state["fRec114"][1])) + (self._fConst533 * state["fRec114"][2]))) - (self._fConst527 * ((self._fConst560 * state["fRec113"][2]) + (self._fConst561 * state["fRec113"][1]))))) 
		state["fRec112"] = ((fSlow2 * fRec112_temp) + (fSlow40 * jnp.abs((self._fConst527 * (((self._fConst530 * state["fRec113"][0]) + (self._fConst562 * state["fRec113"][1])) + (self._fConst530 * state["fRec113"][2])))))) 
		fVbargraph14 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec112"])))
		# self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		state["fRec125"] = state["fRec125"].at[0].set((fTemp60 - (self._fConst579 * ((self._fConst582 * state["fRec125"][2]) + (self._fConst583 * state["fRec125"][1]))))) 
		state["fRec124"] = state["fRec124"].at[0].set(((self._fConst579 * (((self._fConst581 * state["fRec125"][0]) + (self._fConst584 * state["fRec125"][1])) + (self._fConst581 * state["fRec125"][2]))) - (self._fConst577 * ((self._fConst585 * state["fRec124"][2]) + (self._fConst586 * state["fRec124"][1]))))) 
		state["fRec123"] = state["fRec123"].at[0].set(((self._fConst577 * (((self._fConst578 * state["fRec124"][0]) + (self._fConst587 * state["fRec124"][1])) + (self._fConst578 * state["fRec124"][2]))) - (self._fConst575 * ((self._fConst588 * state["fRec123"][2]) + (self._fConst589 * state["fRec123"][1]))))) 
		fTemp61 = (self._fConst575 * (((self._fConst576 * state["fRec123"][0]) + (self._fConst590 * state["fRec123"][1])) + (self._fConst576 * state["fRec123"][2]))) 
		state["fRec122"] = state["fRec122"].at[0].set((fTemp61 - (self._fConst572 * ((self._fConst591 * state["fRec122"][2]) + (self._fConst593 * state["fRec122"][1]))))) 
		state["fRec121"] = state["fRec121"].at[0].set(((self._fConst572 * (((self._fConst574 * state["fRec122"][0]) + (self._fConst594 * state["fRec122"][1])) + (self._fConst574 * state["fRec122"][2]))) - (self._fConst569 * ((self._fConst595 * state["fRec121"][2]) + (self._fConst596 * state["fRec121"][1]))))) 
		state["fRec120"] = state["fRec120"].at[0].set(((self._fConst569 * (((self._fConst571 * state["fRec121"][0]) + (self._fConst597 * state["fRec121"][1])) + (self._fConst571 * state["fRec121"][2]))) - (self._fConst565 * ((self._fConst598 * state["fRec120"][2]) + (self._fConst599 * state["fRec120"][1]))))) 
		state["fRec119"] = ((fSlow2 * fRec119_temp) + (fSlow40 * jnp.abs((self._fConst565 * (((self._fConst568 * state["fRec120"][0]) + (self._fConst600 * state["fRec120"][1])) + (self._fConst568 * state["fRec120"][2])))))) 
		fVbargraph15 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec119"])))
		# self.sow("intermediates", "fVbargraph15", fVbargraph15) 
		state["fRec132"] = state["fRec132"].at[0].set((fTemp61 - (self._fConst617 * ((self._fConst620 * state["fRec132"][2]) + (self._fConst621 * state["fRec132"][1]))))) 
		state["fRec131"] = state["fRec131"].at[0].set(((self._fConst617 * (((self._fConst619 * state["fRec132"][0]) + (self._fConst622 * state["fRec132"][1])) + (self._fConst619 * state["fRec132"][2]))) - (self._fConst615 * ((self._fConst623 * state["fRec131"][2]) + (self._fConst624 * state["fRec131"][1]))))) 
		state["fRec130"] = state["fRec130"].at[0].set(((self._fConst615 * (((self._fConst616 * state["fRec131"][0]) + (self._fConst625 * state["fRec131"][1])) + (self._fConst616 * state["fRec131"][2]))) - (self._fConst613 * ((self._fConst626 * state["fRec130"][2]) + (self._fConst627 * state["fRec130"][1]))))) 
		fTemp62 = (self._fConst613 * (((self._fConst614 * state["fRec130"][0]) + (self._fConst628 * state["fRec130"][1])) + (self._fConst614 * state["fRec130"][2]))) 
		state["fRec129"] = state["fRec129"].at[0].set((fTemp62 - (self._fConst610 * ((self._fConst629 * state["fRec129"][2]) + (self._fConst631 * state["fRec129"][1]))))) 
		state["fRec128"] = state["fRec128"].at[0].set(((self._fConst610 * (((self._fConst612 * state["fRec129"][0]) + (self._fConst632 * state["fRec129"][1])) + (self._fConst612 * state["fRec129"][2]))) - (self._fConst607 * ((self._fConst633 * state["fRec128"][2]) + (self._fConst634 * state["fRec128"][1]))))) 
		state["fRec127"] = state["fRec127"].at[0].set(((self._fConst607 * (((self._fConst609 * state["fRec128"][0]) + (self._fConst635 * state["fRec128"][1])) + (self._fConst609 * state["fRec128"][2]))) - (self._fConst603 * ((self._fConst636 * state["fRec127"][2]) + (self._fConst637 * state["fRec127"][1]))))) 
		state["fRec126"] = ((fSlow2 * fRec126_temp) + (fSlow40 * jnp.abs((self._fConst603 * (((self._fConst606 * state["fRec127"][0]) + (self._fConst638 * state["fRec127"][1])) + (self._fConst606 * state["fRec127"][2])))))) 
		fVbargraph16 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec126"])))
		# self.sow("intermediates", "fVbargraph16", fVbargraph16) 
		state["fRec139"] = state["fRec139"].at[0].set((fTemp62 - (self._fConst655 * ((self._fConst658 * state["fRec139"][2]) + (self._fConst659 * state["fRec139"][1]))))) 
		state["fRec138"] = state["fRec138"].at[0].set(((self._fConst655 * (((self._fConst657 * state["fRec139"][0]) + (self._fConst660 * state["fRec139"][1])) + (self._fConst657 * state["fRec139"][2]))) - (self._fConst653 * ((self._fConst661 * state["fRec138"][2]) + (self._fConst662 * state["fRec138"][1]))))) 
		state["fRec137"] = state["fRec137"].at[0].set(((self._fConst653 * (((self._fConst654 * state["fRec138"][0]) + (self._fConst663 * state["fRec138"][1])) + (self._fConst654 * state["fRec138"][2]))) - (self._fConst651 * ((self._fConst664 * state["fRec137"][2]) + (self._fConst665 * state["fRec137"][1]))))) 
		fTemp63 = (self._fConst651 * (((self._fConst652 * state["fRec137"][0]) + (self._fConst666 * state["fRec137"][1])) + (self._fConst652 * state["fRec137"][2]))) 
		state["fRec136"] = state["fRec136"].at[0].set((fTemp63 - (self._fConst648 * ((self._fConst667 * state["fRec136"][2]) + (self._fConst669 * state["fRec136"][1]))))) 
		state["fRec135"] = state["fRec135"].at[0].set(((self._fConst648 * (((self._fConst650 * state["fRec136"][0]) + (self._fConst670 * state["fRec136"][1])) + (self._fConst650 * state["fRec136"][2]))) - (self._fConst645 * ((self._fConst671 * state["fRec135"][2]) + (self._fConst672 * state["fRec135"][1]))))) 
		state["fRec134"] = state["fRec134"].at[0].set(((self._fConst645 * (((self._fConst647 * state["fRec135"][0]) + (self._fConst673 * state["fRec135"][1])) + (self._fConst647 * state["fRec135"][2]))) - (self._fConst641 * ((self._fConst674 * state["fRec134"][2]) + (self._fConst675 * state["fRec134"][1]))))) 
		state["fRec133"] = ((fSlow2 * fRec133_temp) + (fSlow40 * jnp.abs((self._fConst641 * (((self._fConst644 * state["fRec134"][0]) + (self._fConst676 * state["fRec134"][1])) + (self._fConst644 * state["fRec134"][2])))))) 
		fVbargraph17 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec133"])))
		# self.sow("intermediates", "fVbargraph17", fVbargraph17) 
		state["fRec146"] = state["fRec146"].at[0].set((fTemp63 - (self._fConst693 * ((self._fConst696 * state["fRec146"][2]) + (self._fConst697 * state["fRec146"][1]))))) 
		state["fRec145"] = state["fRec145"].at[0].set(((self._fConst693 * (((self._fConst695 * state["fRec146"][0]) + (self._fConst698 * state["fRec146"][1])) + (self._fConst695 * state["fRec146"][2]))) - (self._fConst691 * ((self._fConst699 * state["fRec145"][2]) + (self._fConst700 * state["fRec145"][1]))))) 
		state["fRec144"] = state["fRec144"].at[0].set(((self._fConst691 * (((self._fConst692 * state["fRec145"][0]) + (self._fConst701 * state["fRec145"][1])) + (self._fConst692 * state["fRec145"][2]))) - (self._fConst689 * ((self._fConst702 * state["fRec144"][2]) + (self._fConst703 * state["fRec144"][1]))))) 
		fTemp64 = (self._fConst689 * (((self._fConst690 * state["fRec144"][0]) + (self._fConst704 * state["fRec144"][1])) + (self._fConst690 * state["fRec144"][2]))) 
		state["fRec143"] = state["fRec143"].at[0].set((fTemp64 - (self._fConst686 * ((self._fConst705 * state["fRec143"][2]) + (self._fConst707 * state["fRec143"][1]))))) 
		state["fRec142"] = state["fRec142"].at[0].set(((self._fConst686 * (((self._fConst688 * state["fRec143"][0]) + (self._fConst708 * state["fRec143"][1])) + (self._fConst688 * state["fRec143"][2]))) - (self._fConst683 * ((self._fConst709 * state["fRec142"][2]) + (self._fConst710 * state["fRec142"][1]))))) 
		state["fRec141"] = state["fRec141"].at[0].set(((self._fConst683 * (((self._fConst685 * state["fRec142"][0]) + (self._fConst711 * state["fRec142"][1])) + (self._fConst685 * state["fRec142"][2]))) - (self._fConst679 * ((self._fConst712 * state["fRec141"][2]) + (self._fConst713 * state["fRec141"][1]))))) 
		state["fRec140"] = ((fSlow2 * fRec140_temp) + (fSlow40 * jnp.abs((self._fConst679 * (((self._fConst682 * state["fRec141"][0]) + (self._fConst714 * state["fRec141"][1])) + (self._fConst682 * state["fRec141"][2])))))) 
		fVbargraph18 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec140"])))
		# self.sow("intermediates", "fVbargraph18", fVbargraph18) 
		state["fRec150"] = state["fRec150"].at[0].set((fTemp64 - (self._fConst719 * ((self._fConst722 * state["fRec150"][2]) + (self._fConst723 * state["fRec150"][1]))))) 
		state["fRec149"] = state["fRec149"].at[0].set(((self._fConst719 * (((self._fConst721 * state["fRec150"][0]) + (self._fConst724 * state["fRec150"][1])) + (self._fConst721 * state["fRec150"][2]))) - (self._fConst717 * ((self._fConst725 * state["fRec149"][2]) + (self._fConst726 * state["fRec149"][1]))))) 
		state["fRec148"] = state["fRec148"].at[0].set(((self._fConst717 * (((self._fConst718 * state["fRec149"][0]) + (self._fConst727 * state["fRec149"][1])) + (self._fConst718 * state["fRec149"][2]))) - (self._fConst715 * ((self._fConst728 * state["fRec148"][2]) + (self._fConst729 * state["fRec148"][1]))))) 
		state["fRec147"] = ((fSlow2 * fRec147_temp) + (fSlow40 * jnp.abs((self._fConst715 * (((self._fConst716 * state["fRec148"][0]) + (self._fConst730 * state["fRec148"][1])) + (self._fConst716 * state["fRec148"][2])))))) 
		fVbargraph19 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec147"])))
		# self.sow("intermediates", "fVbargraph19", fVbargraph19) 
		fTemp65 = fTemp46 
		_result0 = fTemp65 
		_result1 = fTemp65 
		state["iVec0"] = jnp.roll(state["iVec0"], 1) 
		state["fRec15"] = jnp.roll(state["fRec15"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec19"] = jnp.roll(state["fRec19"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec27"] = jnp.roll(state["fRec27"], 1) 
		state["fRec26"] = jnp.roll(state["fRec26"], 1) 
		state["fRec25"] = jnp.roll(state["fRec25"], 1) 
		state["fRec24"] = jnp.roll(state["fRec24"], 1) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec22"] = jnp.roll(state["fRec22"], 1) 
		state["fRec34"] = jnp.roll(state["fRec34"], 1) 
		state["fRec33"] = jnp.roll(state["fRec33"], 1) 
		state["fRec32"] = jnp.roll(state["fRec32"], 1) 
		state["fRec31"] = jnp.roll(state["fRec31"], 1) 
		state["fRec30"] = jnp.roll(state["fRec30"], 1) 
		state["fRec29"] = jnp.roll(state["fRec29"], 1) 
		state["fRec41"] = jnp.roll(state["fRec41"], 1) 
		state["fRec40"] = jnp.roll(state["fRec40"], 1) 
		state["fRec39"] = jnp.roll(state["fRec39"], 1) 
		state["fRec38"] = jnp.roll(state["fRec38"], 1) 
		state["fRec37"] = jnp.roll(state["fRec37"], 1) 
		state["fRec36"] = jnp.roll(state["fRec36"], 1) 
		state["fRec48"] = jnp.roll(state["fRec48"], 1) 
		state["fRec47"] = jnp.roll(state["fRec47"], 1) 
		state["fRec46"] = jnp.roll(state["fRec46"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
		state["fRec43"] = jnp.roll(state["fRec43"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
		state["fRec54"] = jnp.roll(state["fRec54"], 1) 
		state["fRec53"] = jnp.roll(state["fRec53"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec50"] = jnp.roll(state["fRec50"], 1) 
		state["fRec62"] = jnp.roll(state["fRec62"], 1) 
		state["fRec61"] = jnp.roll(state["fRec61"], 1) 
		state["fRec60"] = jnp.roll(state["fRec60"], 1) 
		state["fRec59"] = jnp.roll(state["fRec59"], 1) 
		state["fRec58"] = jnp.roll(state["fRec58"], 1) 
		state["fRec57"] = jnp.roll(state["fRec57"], 1) 
		state["fRec69"] = jnp.roll(state["fRec69"], 1) 
		state["fRec68"] = jnp.roll(state["fRec68"], 1) 
		state["fRec67"] = jnp.roll(state["fRec67"], 1) 
		state["fRec66"] = jnp.roll(state["fRec66"], 1) 
		state["fRec65"] = jnp.roll(state["fRec65"], 1) 
		state["fRec64"] = jnp.roll(state["fRec64"], 1) 
		state["fRec76"] = jnp.roll(state["fRec76"], 1) 
		state["fRec75"] = jnp.roll(state["fRec75"], 1) 
		state["fRec74"] = jnp.roll(state["fRec74"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec72"] = jnp.roll(state["fRec72"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec83"] = jnp.roll(state["fRec83"], 1) 
		state["fRec82"] = jnp.roll(state["fRec82"], 1) 
		state["fRec81"] = jnp.roll(state["fRec81"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec89"] = jnp.roll(state["fRec89"], 1) 
		state["fRec88"] = jnp.roll(state["fRec88"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec96"] = jnp.roll(state["fRec96"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec94"] = jnp.roll(state["fRec94"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec92"] = jnp.roll(state["fRec92"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec103"] = jnp.roll(state["fRec103"], 1) 
		state["fRec102"] = jnp.roll(state["fRec102"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec110"] = jnp.roll(state["fRec110"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec116"] = jnp.roll(state["fRec116"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec114"] = jnp.roll(state["fRec114"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec125"] = jnp.roll(state["fRec125"], 1) 
		state["fRec124"] = jnp.roll(state["fRec124"], 1) 
		state["fRec123"] = jnp.roll(state["fRec123"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec120"] = jnp.roll(state["fRec120"], 1) 
		state["fRec132"] = jnp.roll(state["fRec132"], 1) 
		state["fRec131"] = jnp.roll(state["fRec131"], 1) 
		state["fRec130"] = jnp.roll(state["fRec130"], 1) 
		state["fRec129"] = jnp.roll(state["fRec129"], 1) 
		state["fRec128"] = jnp.roll(state["fRec128"], 1) 
		state["fRec127"] = jnp.roll(state["fRec127"], 1) 
		state["fRec139"] = jnp.roll(state["fRec139"], 1) 
		state["fRec138"] = jnp.roll(state["fRec138"], 1) 
		state["fRec137"] = jnp.roll(state["fRec137"], 1) 
		state["fRec136"] = jnp.roll(state["fRec136"], 1) 
		state["fRec135"] = jnp.roll(state["fRec135"], 1) 
		state["fRec134"] = jnp.roll(state["fRec134"], 1) 
		state["fRec146"] = jnp.roll(state["fRec146"], 1) 
		state["fRec145"] = jnp.roll(state["fRec145"], 1) 
		state["fRec144"] = jnp.roll(state["fRec144"], 1) 
		state["fRec143"] = jnp.roll(state["fRec143"], 1) 
		state["fRec142"] = jnp.roll(state["fRec142"], 1) 
		state["fRec141"] = jnp.roll(state["fRec141"], 1) 
		state["fRec150"] = jnp.roll(state["fRec150"], 1) 
		state["fRec149"] = jnp.roll(state["fRec149"], 1) 
		state["fRec148"] = jnp.roll(state["fRec148"], 1) 
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
