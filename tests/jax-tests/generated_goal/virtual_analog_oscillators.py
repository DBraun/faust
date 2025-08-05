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
		ui_path.append("virtual_analog_oscillators") 
		ui_path.append("0x00") 
		ui_path.append("VIRTUAL ANALOG OSCILLATORS") 
		ui_path.append("Signal Levels") 
		self.add_vslider("fVslider1", ui_path, "Sawtooth", 1.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.append("Pulse Train") 
		self.add_button("fCheckbox2", ui_path, "Order 3", unnorm_funcs) 
		self.add_vslider("fVslider7", ui_path, "0x00", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider8", ui_path, "Duty Cycle", 0.5, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		self.add_vslider("fVslider5", ui_path, "Square", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider6", ui_path, "Triangle", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider9", ui_path, "Pink Noise", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_vslider("fVslider0", ui_path, "Ext Input", 0.0, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.append("Signal Parameters") 
		ui_path.append("0x00") 
		self.add_hslider("fHslider4", ui_path, "Mix Amplitude", -2e+01, -1.2e+02, 1e+01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider5", ui_path, "Frequency", 49.0, 1.0, 88.0, unnorm_funcs, "linear") 
		ui_path.pop()
		self.add_vslider("fVslider3", ui_path, "Detuning 1", -0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider4", ui_path, "Detuning 2", 0.1, -1e+01, 1e+01, unnorm_funcs, "linear") 
		self.add_vslider("fVslider2", ui_path, "Portamento", 0.1, 0.001, 1e+01, unnorm_funcs, "log") 
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
		self.add_hslider("fHslider3", ui_path, "Corner Frequency", 25.0, 1.0, 88.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider6", ui_path, "Corner Resonance", 0.9, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "VCF Output Level", 5.0, -6e+01, 2e+01, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		ui_path.append("0x00") 
		ui_path.append("CONSTANT-Q SPECTRUM ANALYZER (6E), 15 bands spanning LP, 9 octaves below 16000 Hz, HP") 
		self.add_vbargraph("fVbargraph14", ui_path, "vbargraph0", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph13", ui_path, "vbargraph1", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph12", ui_path, "vbargraph2", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph11", ui_path, "vbargraph3", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph10", ui_path, "vbargraph4", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph9", ui_path, "vbargraph5", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph8", ui_path, "vbargraph6", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph7", ui_path, "vbargraph7", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph6", ui_path, "vbargraph8", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph5", ui_path, "vbargraph9", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph4", ui_path, "vbargraph10", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph3", ui_path, "vbargraph11", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph2", ui_path, "vbargraph12", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph1", ui_path, "vbargraph13", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph0", ui_path, "vbargraph14", -5e+01, 1e+01, unnorm_funcs) 
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
		self._fConst16 = (np.float32(0.25) * self._fConst0) 
		self._fConst17 = (np.float32(0.041666668) * np.power(self._fConst0, np.float32(2.0))) 
		self._fConst18 = (np.float32(0.0052083335) * np.power(self._fConst0, np.float32(3.0))) 
		self._fConst19 = (np.float32(0.083333336) * self._fConst0) 
		self._fConst20 = (np.float32(0.5) * self._fConst0) 
		self._fConst21 = (np.float32(1.3333334) / self._fConst0) 
		self._fConst22 = (np.float32(0.013888889) * self._fConst0) 
		self._fConst23 = (np.float32(3.1415927) / self._fConst0) 
		self._fConst24 = (((self._fConst3 + np.float32(-3.1897273)) / self._fConst2) + np.float32(4.0767817)) 
		self._fConst25 = (np.float32(1.0) / self._fConst5) 
		self._fConst26 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst25)) 
		self._fConst27 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst12)) 
		self._fConst28 = (((self._fConst3 + np.float32(-0.74313045)) / self._fConst2) + np.float32(1.4500711)) 
		self._fConst29 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst25)) 
		self._fConst30 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst9)) 
		self._fConst31 = (((self._fConst3 + np.float32(-0.15748216)) / self._fConst2) + np.float32(0.9351402)) 
		self._fConst32 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst25)) 
		self._fConst33 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst6)) 
		self._fConst34 = np.tan((np.float32(31665.27) / self._fConst0)) 
		self._fConst35 = (np.float32(1.0) / self._fConst34) 
		self._fConst36 = (np.float32(1.0) / (((self._fConst35 + np.float32(0.15748216)) / self._fConst34) + np.float32(0.9351402))) 
		self._fConst37 = np.power(self._fConst34, np.float32(2.0)) 
		self._fConst38 = (np.float32(50.06381) / self._fConst37) 
		self._fConst39 = (self._fConst38 + np.float32(0.9351402)) 
		self._fConst40 = (np.float32(1.0) / (((self._fConst35 + np.float32(0.74313045)) / self._fConst34) + np.float32(1.4500711))) 
		self._fConst41 = (np.float32(11.0520525) / self._fConst37) 
		self._fConst42 = (self._fConst41 + np.float32(1.4500711)) 
		self._fConst43 = (np.float32(1.0) / (((self._fConst35 + np.float32(3.1897273)) / self._fConst34) + np.float32(4.0767817))) 
		self._fConst44 = (np.float32(0.0017661728) / self._fConst37) 
		self._fConst45 = (self._fConst44 + np.float32(0.0004076782)) 
		self._fConst46 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.16840488)) / self._fConst2) + np.float32(1.0693583))) 
		self._fConst47 = (self._fConst25 + np.float32(53.53615)) 
		self._fConst48 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.51247865)) / self._fConst2) + np.float32(0.6896214))) 
		self._fConst49 = (self._fConst25 + np.float32(7.6217313)) 
		self._fConst50 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.78241307)) / self._fConst2) + np.float32(0.2452915))) 
		self._fConst51 = (np.float32(0.0001) / self._fConst5) 
		self._fConst52 = (self._fConst51 + np.float32(0.0004332272)) 
		self._fConst53 = (((self._fConst3 + np.float32(-0.78241307)) / self._fConst2) + np.float32(0.2452915)) 
		self._fConst54 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst25)) 
		self._fConst55 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst51)) 
		self._fConst56 = (((self._fConst3 + np.float32(-0.51247865)) / self._fConst2) + np.float32(0.6896214)) 
		self._fConst57 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst25)) 
		self._fConst58 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst25)) 
		self._fConst59 = (((self._fConst3 + np.float32(-0.16840488)) / self._fConst2) + np.float32(1.0693583)) 
		self._fConst60 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst25)) 
		self._fConst61 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst25)) 
		self._fConst62 = (((self._fConst35 + np.float32(-3.1897273)) / self._fConst34) + np.float32(4.0767817)) 
		self._fConst63 = (np.float32(1.0) / self._fConst37) 
		self._fConst64 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst63)) 
		self._fConst65 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst44)) 
		self._fConst66 = (((self._fConst35 + np.float32(-0.74313045)) / self._fConst34) + np.float32(1.4500711)) 
		self._fConst67 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst63)) 
		self._fConst68 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst41)) 
		self._fConst69 = (((self._fConst35 + np.float32(-0.15748216)) / self._fConst34) + np.float32(0.9351402)) 
		self._fConst70 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst63)) 
		self._fConst71 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst38)) 
		self._fConst72 = np.tan((np.float32(19947.87) / self._fConst0)) 
		self._fConst73 = (np.float32(1.0) / self._fConst72) 
		self._fConst74 = (np.float32(1.0) / (((self._fConst73 + np.float32(0.15748216)) / self._fConst72) + np.float32(0.9351402))) 
		self._fConst75 = np.power(self._fConst72, np.float32(2.0)) 
		self._fConst76 = (np.float32(50.06381) / self._fConst75) 
		self._fConst77 = (self._fConst76 + np.float32(0.9351402)) 
		self._fConst78 = (np.float32(1.0) / (((self._fConst73 + np.float32(0.74313045)) / self._fConst72) + np.float32(1.4500711))) 
		self._fConst79 = (np.float32(11.0520525) / self._fConst75) 
		self._fConst80 = (self._fConst79 + np.float32(1.4500711)) 
		self._fConst81 = (np.float32(1.0) / (((self._fConst73 + np.float32(3.1897273)) / self._fConst72) + np.float32(4.0767817))) 
		self._fConst82 = (np.float32(0.0017661728) / self._fConst75) 
		self._fConst83 = (self._fConst82 + np.float32(0.0004076782)) 
		self._fConst84 = (np.float32(1.0) / (((self._fConst35 + np.float32(0.16840488)) / self._fConst34) + np.float32(1.0693583))) 
		self._fConst85 = (self._fConst63 + np.float32(53.53615)) 
		self._fConst86 = (np.float32(1.0) / (((self._fConst35 + np.float32(0.51247865)) / self._fConst34) + np.float32(0.6896214))) 
		self._fConst87 = (self._fConst63 + np.float32(7.6217313)) 
		self._fConst88 = (np.float32(1.0) / (((self._fConst35 + np.float32(0.78241307)) / self._fConst34) + np.float32(0.2452915))) 
		self._fConst89 = (np.float32(0.0001) / self._fConst37) 
		self._fConst90 = (self._fConst89 + np.float32(0.0004332272)) 
		self._fConst91 = (((self._fConst35 + np.float32(-0.78241307)) / self._fConst34) + np.float32(0.2452915)) 
		self._fConst92 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst63)) 
		self._fConst93 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst89)) 
		self._fConst94 = (((self._fConst35 + np.float32(-0.51247865)) / self._fConst34) + np.float32(0.6896214)) 
		self._fConst95 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst63)) 
		self._fConst96 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst63)) 
		self._fConst97 = (((self._fConst35 + np.float32(-0.16840488)) / self._fConst34) + np.float32(1.0693583)) 
		self._fConst98 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst63)) 
		self._fConst99 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst63)) 
		self._fConst100 = (((self._fConst73 + np.float32(-3.1897273)) / self._fConst72) + np.float32(4.0767817)) 
		self._fConst101 = (np.float32(1.0) / self._fConst75) 
		self._fConst102 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst101)) 
		self._fConst103 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst82)) 
		self._fConst104 = (((self._fConst73 + np.float32(-0.74313045)) / self._fConst72) + np.float32(1.4500711)) 
		self._fConst105 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst101)) 
		self._fConst106 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst79)) 
		self._fConst107 = (((self._fConst73 + np.float32(-0.15748216)) / self._fConst72) + np.float32(0.9351402)) 
		self._fConst108 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst101)) 
		self._fConst109 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst76)) 
		self._fConst110 = np.tan((np.float32(12566.371) / self._fConst0)) 
		self._fConst111 = (np.float32(1.0) / self._fConst110) 
		self._fConst112 = (np.float32(1.0) / (((self._fConst111 + np.float32(0.15748216)) / self._fConst110) + np.float32(0.9351402))) 
		self._fConst113 = np.power(self._fConst110, np.float32(2.0)) 
		self._fConst114 = (np.float32(50.06381) / self._fConst113) 
		self._fConst115 = (self._fConst114 + np.float32(0.9351402)) 
		self._fConst116 = (np.float32(1.0) / (((self._fConst111 + np.float32(0.74313045)) / self._fConst110) + np.float32(1.4500711))) 
		self._fConst117 = (np.float32(11.0520525) / self._fConst113) 
		self._fConst118 = (self._fConst117 + np.float32(1.4500711)) 
		self._fConst119 = (np.float32(1.0) / (((self._fConst111 + np.float32(3.1897273)) / self._fConst110) + np.float32(4.0767817))) 
		self._fConst120 = (np.float32(0.0017661728) / self._fConst113) 
		self._fConst121 = (self._fConst120 + np.float32(0.0004076782)) 
		self._fConst122 = (np.float32(1.0) / (((self._fConst73 + np.float32(0.16840488)) / self._fConst72) + np.float32(1.0693583))) 
		self._fConst123 = (self._fConst101 + np.float32(53.53615)) 
		self._fConst124 = (np.float32(1.0) / (((self._fConst73 + np.float32(0.51247865)) / self._fConst72) + np.float32(0.6896214))) 
		self._fConst125 = (self._fConst101 + np.float32(7.6217313)) 
		self._fConst126 = (np.float32(1.0) / (((self._fConst73 + np.float32(0.78241307)) / self._fConst72) + np.float32(0.2452915))) 
		self._fConst127 = (np.float32(0.0001) / self._fConst75) 
		self._fConst128 = (self._fConst127 + np.float32(0.0004332272)) 
		self._fConst129 = (((self._fConst73 + np.float32(-0.78241307)) / self._fConst72) + np.float32(0.2452915)) 
		self._fConst130 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst101)) 
		self._fConst131 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst127)) 
		self._fConst132 = (((self._fConst73 + np.float32(-0.51247865)) / self._fConst72) + np.float32(0.6896214)) 
		self._fConst133 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst101)) 
		self._fConst134 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst101)) 
		self._fConst135 = (((self._fConst73 + np.float32(-0.16840488)) / self._fConst72) + np.float32(1.0693583)) 
		self._fConst136 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst101)) 
		self._fConst137 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst101)) 
		self._fConst138 = (((self._fConst111 + np.float32(-3.1897273)) / self._fConst110) + np.float32(4.0767817)) 
		self._fConst139 = (np.float32(1.0) / self._fConst113) 
		self._fConst140 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst139)) 
		self._fConst141 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst120)) 
		self._fConst142 = (((self._fConst111 + np.float32(-0.74313045)) / self._fConst110) + np.float32(1.4500711)) 
		self._fConst143 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst139)) 
		self._fConst144 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst117)) 
		self._fConst145 = (((self._fConst111 + np.float32(-0.15748216)) / self._fConst110) + np.float32(0.9351402)) 
		self._fConst146 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst139)) 
		self._fConst147 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst114)) 
		self._fConst148 = np.tan((np.float32(7916.3174) / self._fConst0)) 
		self._fConst149 = (np.float32(1.0) / self._fConst148) 
		self._fConst150 = (np.float32(1.0) / (((self._fConst149 + np.float32(0.15748216)) / self._fConst148) + np.float32(0.9351402))) 
		self._fConst151 = np.power(self._fConst148, np.float32(2.0)) 
		self._fConst152 = (np.float32(50.06381) / self._fConst151) 
		self._fConst153 = (self._fConst152 + np.float32(0.9351402)) 
		self._fConst154 = (np.float32(1.0) / (((self._fConst149 + np.float32(0.74313045)) / self._fConst148) + np.float32(1.4500711))) 
		self._fConst155 = (np.float32(11.0520525) / self._fConst151) 
		self._fConst156 = (self._fConst155 + np.float32(1.4500711)) 
		self._fConst157 = (np.float32(1.0) / (((self._fConst149 + np.float32(3.1897273)) / self._fConst148) + np.float32(4.0767817))) 
		self._fConst158 = (np.float32(0.0017661728) / self._fConst151) 
		self._fConst159 = (self._fConst158 + np.float32(0.0004076782)) 
		self._fConst160 = (np.float32(1.0) / (((self._fConst111 + np.float32(0.16840488)) / self._fConst110) + np.float32(1.0693583))) 
		self._fConst161 = (self._fConst139 + np.float32(53.53615)) 
		self._fConst162 = (np.float32(1.0) / (((self._fConst111 + np.float32(0.51247865)) / self._fConst110) + np.float32(0.6896214))) 
		self._fConst163 = (self._fConst139 + np.float32(7.6217313)) 
		self._fConst164 = (np.float32(1.0) / (((self._fConst111 + np.float32(0.78241307)) / self._fConst110) + np.float32(0.2452915))) 
		self._fConst165 = (np.float32(0.0001) / self._fConst113) 
		self._fConst166 = (self._fConst165 + np.float32(0.0004332272)) 
		self._fConst167 = (((self._fConst111 + np.float32(-0.78241307)) / self._fConst110) + np.float32(0.2452915)) 
		self._fConst168 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst139)) 
		self._fConst169 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst165)) 
		self._fConst170 = (((self._fConst111 + np.float32(-0.51247865)) / self._fConst110) + np.float32(0.6896214)) 
		self._fConst171 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst139)) 
		self._fConst172 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst139)) 
		self._fConst173 = (((self._fConst111 + np.float32(-0.16840488)) / self._fConst110) + np.float32(1.0693583)) 
		self._fConst174 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst139)) 
		self._fConst175 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst139)) 
		self._fConst176 = (((self._fConst149 + np.float32(-3.1897273)) / self._fConst148) + np.float32(4.0767817)) 
		self._fConst177 = (np.float32(1.0) / self._fConst151) 
		self._fConst178 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst177)) 
		self._fConst179 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst158)) 
		self._fConst180 = (((self._fConst149 + np.float32(-0.74313045)) / self._fConst148) + np.float32(1.4500711)) 
		self._fConst181 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst177)) 
		self._fConst182 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst155)) 
		self._fConst183 = (((self._fConst149 + np.float32(-0.15748216)) / self._fConst148) + np.float32(0.9351402)) 
		self._fConst184 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst177)) 
		self._fConst185 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst152)) 
		self._fConst186 = np.tan((np.float32(4986.9673) / self._fConst0)) 
		self._fConst187 = (np.float32(1.0) / self._fConst186) 
		self._fConst188 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.15748216)) / self._fConst186) + np.float32(0.9351402))) 
		self._fConst189 = np.power(self._fConst186, np.float32(2.0)) 
		self._fConst190 = (np.float32(50.06381) / self._fConst189) 
		self._fConst191 = (self._fConst190 + np.float32(0.9351402)) 
		self._fConst192 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.74313045)) / self._fConst186) + np.float32(1.4500711))) 
		self._fConst193 = (np.float32(11.0520525) / self._fConst189) 
		self._fConst194 = (self._fConst193 + np.float32(1.4500711)) 
		self._fConst195 = (np.float32(1.0) / (((self._fConst187 + np.float32(3.1897273)) / self._fConst186) + np.float32(4.0767817))) 
		self._fConst196 = (np.float32(0.0017661728) / self._fConst189) 
		self._fConst197 = (self._fConst196 + np.float32(0.0004076782)) 
		self._fConst198 = (np.float32(1.0) / (((self._fConst149 + np.float32(0.16840488)) / self._fConst148) + np.float32(1.0693583))) 
		self._fConst199 = (self._fConst177 + np.float32(53.53615)) 
		self._fConst200 = (np.float32(1.0) / (((self._fConst149 + np.float32(0.51247865)) / self._fConst148) + np.float32(0.6896214))) 
		self._fConst201 = (self._fConst177 + np.float32(7.6217313)) 
		self._fConst202 = (np.float32(1.0) / (((self._fConst149 + np.float32(0.78241307)) / self._fConst148) + np.float32(0.2452915))) 
		self._fConst203 = (np.float32(0.0001) / self._fConst151) 
		self._fConst204 = (self._fConst203 + np.float32(0.0004332272)) 
		self._fConst205 = (((self._fConst149 + np.float32(-0.78241307)) / self._fConst148) + np.float32(0.2452915)) 
		self._fConst206 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst177)) 
		self._fConst207 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst203)) 
		self._fConst208 = (((self._fConst149 + np.float32(-0.51247865)) / self._fConst148) + np.float32(0.6896214)) 
		self._fConst209 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst177)) 
		self._fConst210 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst177)) 
		self._fConst211 = (((self._fConst149 + np.float32(-0.16840488)) / self._fConst148) + np.float32(1.0693583)) 
		self._fConst212 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst177)) 
		self._fConst213 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst177)) 
		self._fConst214 = (((self._fConst187 + np.float32(-3.1897273)) / self._fConst186) + np.float32(4.0767817)) 
		self._fConst215 = (np.float32(1.0) / self._fConst189) 
		self._fConst216 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst215)) 
		self._fConst217 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst196)) 
		self._fConst218 = (((self._fConst187 + np.float32(-0.74313045)) / self._fConst186) + np.float32(1.4500711)) 
		self._fConst219 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst215)) 
		self._fConst220 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst193)) 
		self._fConst221 = (((self._fConst187 + np.float32(-0.15748216)) / self._fConst186) + np.float32(0.9351402)) 
		self._fConst222 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst215)) 
		self._fConst223 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst190)) 
		self._fConst224 = np.tan((np.float32(3141.5928) / self._fConst0)) 
		self._fConst225 = (np.float32(1.0) / self._fConst224) 
		self._fConst226 = (np.float32(1.0) / (((self._fConst225 + np.float32(0.15748216)) / self._fConst224) + np.float32(0.9351402))) 
		self._fConst227 = np.power(self._fConst224, np.float32(2.0)) 
		self._fConst228 = (np.float32(50.06381) / self._fConst227) 
		self._fConst229 = (self._fConst228 + np.float32(0.9351402)) 
		self._fConst230 = (np.float32(1.0) / (((self._fConst225 + np.float32(0.74313045)) / self._fConst224) + np.float32(1.4500711))) 
		self._fConst231 = (np.float32(11.0520525) / self._fConst227) 
		self._fConst232 = (self._fConst231 + np.float32(1.4500711)) 
		self._fConst233 = (np.float32(1.0) / (((self._fConst225 + np.float32(3.1897273)) / self._fConst224) + np.float32(4.0767817))) 
		self._fConst234 = (np.float32(0.0017661728) / self._fConst227) 
		self._fConst235 = (self._fConst234 + np.float32(0.0004076782)) 
		self._fConst236 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.16840488)) / self._fConst186) + np.float32(1.0693583))) 
		self._fConst237 = (self._fConst215 + np.float32(53.53615)) 
		self._fConst238 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.51247865)) / self._fConst186) + np.float32(0.6896214))) 
		self._fConst239 = (self._fConst215 + np.float32(7.6217313)) 
		self._fConst240 = (np.float32(1.0) / (((self._fConst187 + np.float32(0.78241307)) / self._fConst186) + np.float32(0.2452915))) 
		self._fConst241 = (np.float32(0.0001) / self._fConst189) 
		self._fConst242 = (self._fConst241 + np.float32(0.0004332272)) 
		self._fConst243 = (((self._fConst187 + np.float32(-0.78241307)) / self._fConst186) + np.float32(0.2452915)) 
		self._fConst244 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst215)) 
		self._fConst245 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst241)) 
		self._fConst246 = (((self._fConst187 + np.float32(-0.51247865)) / self._fConst186) + np.float32(0.6896214)) 
		self._fConst247 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst215)) 
		self._fConst248 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst215)) 
		self._fConst249 = (((self._fConst187 + np.float32(-0.16840488)) / self._fConst186) + np.float32(1.0693583)) 
		self._fConst250 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst215)) 
		self._fConst251 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst215)) 
		self._fConst252 = (((self._fConst225 + np.float32(-3.1897273)) / self._fConst224) + np.float32(4.0767817)) 
		self._fConst253 = (np.float32(1.0) / self._fConst227) 
		self._fConst254 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst253)) 
		self._fConst255 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst234)) 
		self._fConst256 = (((self._fConst225 + np.float32(-0.74313045)) / self._fConst224) + np.float32(1.4500711)) 
		self._fConst257 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst253)) 
		self._fConst258 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst231)) 
		self._fConst259 = (((self._fConst225 + np.float32(-0.15748216)) / self._fConst224) + np.float32(0.9351402)) 
		self._fConst260 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst253)) 
		self._fConst261 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst228)) 
		self._fConst262 = np.tan((np.float32(1979.0793) / self._fConst0)) 
		self._fConst263 = (np.float32(1.0) / self._fConst262) 
		self._fConst264 = (np.float32(1.0) / (((self._fConst263 + np.float32(0.15748216)) / self._fConst262) + np.float32(0.9351402))) 
		self._fConst265 = np.power(self._fConst262, np.float32(2.0)) 
		self._fConst266 = (np.float32(50.06381) / self._fConst265) 
		self._fConst267 = (self._fConst266 + np.float32(0.9351402)) 
		self._fConst268 = (np.float32(1.0) / (((self._fConst263 + np.float32(0.74313045)) / self._fConst262) + np.float32(1.4500711))) 
		self._fConst269 = (np.float32(11.0520525) / self._fConst265) 
		self._fConst270 = (self._fConst269 + np.float32(1.4500711)) 
		self._fConst271 = (np.float32(1.0) / (((self._fConst263 + np.float32(3.1897273)) / self._fConst262) + np.float32(4.0767817))) 
		self._fConst272 = (np.float32(0.0017661728) / self._fConst265) 
		self._fConst273 = (self._fConst272 + np.float32(0.0004076782)) 
		self._fConst274 = (np.float32(1.0) / (((self._fConst225 + np.float32(0.16840488)) / self._fConst224) + np.float32(1.0693583))) 
		self._fConst275 = (self._fConst253 + np.float32(53.53615)) 
		self._fConst276 = (np.float32(1.0) / (((self._fConst225 + np.float32(0.51247865)) / self._fConst224) + np.float32(0.6896214))) 
		self._fConst277 = (self._fConst253 + np.float32(7.6217313)) 
		self._fConst278 = (np.float32(1.0) / (((self._fConst225 + np.float32(0.78241307)) / self._fConst224) + np.float32(0.2452915))) 
		self._fConst279 = (np.float32(0.0001) / self._fConst227) 
		self._fConst280 = (self._fConst279 + np.float32(0.0004332272)) 
		self._fConst281 = (((self._fConst225 + np.float32(-0.78241307)) / self._fConst224) + np.float32(0.2452915)) 
		self._fConst282 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst253)) 
		self._fConst283 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst279)) 
		self._fConst284 = (((self._fConst225 + np.float32(-0.51247865)) / self._fConst224) + np.float32(0.6896214)) 
		self._fConst285 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst253)) 
		self._fConst286 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst253)) 
		self._fConst287 = (((self._fConst225 + np.float32(-0.16840488)) / self._fConst224) + np.float32(1.0693583)) 
		self._fConst288 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst253)) 
		self._fConst289 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst253)) 
		self._fConst290 = (((self._fConst263 + np.float32(-3.1897273)) / self._fConst262) + np.float32(4.0767817)) 
		self._fConst291 = (np.float32(1.0) / self._fConst265) 
		self._fConst292 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst291)) 
		self._fConst293 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst272)) 
		self._fConst294 = (((self._fConst263 + np.float32(-0.74313045)) / self._fConst262) + np.float32(1.4500711)) 
		self._fConst295 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst291)) 
		self._fConst296 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst269)) 
		self._fConst297 = (((self._fConst263 + np.float32(-0.15748216)) / self._fConst262) + np.float32(0.9351402)) 
		self._fConst298 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst291)) 
		self._fConst299 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst266)) 
		self._fConst300 = np.tan((np.float32(1246.7418) / self._fConst0)) 
		self._fConst301 = (np.float32(1.0) / self._fConst300) 
		self._fConst302 = (np.float32(1.0) / (((self._fConst301 + np.float32(0.15748216)) / self._fConst300) + np.float32(0.9351402))) 
		self._fConst303 = np.power(self._fConst300, np.float32(2.0)) 
		self._fConst304 = (np.float32(50.06381) / self._fConst303) 
		self._fConst305 = (self._fConst304 + np.float32(0.9351402)) 
		self._fConst306 = (np.float32(1.0) / (((self._fConst301 + np.float32(0.74313045)) / self._fConst300) + np.float32(1.4500711))) 
		self._fConst307 = (np.float32(11.0520525) / self._fConst303) 
		self._fConst308 = (self._fConst307 + np.float32(1.4500711)) 
		self._fConst309 = (np.float32(1.0) / (((self._fConst301 + np.float32(3.1897273)) / self._fConst300) + np.float32(4.0767817))) 
		self._fConst310 = (np.float32(0.0017661728) / self._fConst303) 
		self._fConst311 = (self._fConst310 + np.float32(0.0004076782)) 
		self._fConst312 = (np.float32(1.0) / (((self._fConst263 + np.float32(0.16840488)) / self._fConst262) + np.float32(1.0693583))) 
		self._fConst313 = (self._fConst291 + np.float32(53.53615)) 
		self._fConst314 = (np.float32(1.0) / (((self._fConst263 + np.float32(0.51247865)) / self._fConst262) + np.float32(0.6896214))) 
		self._fConst315 = (self._fConst291 + np.float32(7.6217313)) 
		self._fConst316 = (np.float32(1.0) / (((self._fConst263 + np.float32(0.78241307)) / self._fConst262) + np.float32(0.2452915))) 
		self._fConst317 = (np.float32(0.0001) / self._fConst265) 
		self._fConst318 = (self._fConst317 + np.float32(0.0004332272)) 
		self._fConst319 = (((self._fConst263 + np.float32(-0.78241307)) / self._fConst262) + np.float32(0.2452915)) 
		self._fConst320 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst291)) 
		self._fConst321 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst317)) 
		self._fConst322 = (((self._fConst263 + np.float32(-0.51247865)) / self._fConst262) + np.float32(0.6896214)) 
		self._fConst323 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst291)) 
		self._fConst324 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst291)) 
		self._fConst325 = (((self._fConst263 + np.float32(-0.16840488)) / self._fConst262) + np.float32(1.0693583)) 
		self._fConst326 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst291)) 
		self._fConst327 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst291)) 
		self._fConst328 = (((self._fConst301 + np.float32(-3.1897273)) / self._fConst300) + np.float32(4.0767817)) 
		self._fConst329 = (np.float32(1.0) / self._fConst303) 
		self._fConst330 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst329)) 
		self._fConst331 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst310)) 
		self._fConst332 = (((self._fConst301 + np.float32(-0.74313045)) / self._fConst300) + np.float32(1.4500711)) 
		self._fConst333 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst329)) 
		self._fConst334 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst307)) 
		self._fConst335 = (((self._fConst301 + np.float32(-0.15748216)) / self._fConst300) + np.float32(0.9351402)) 
		self._fConst336 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst329)) 
		self._fConst337 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst304)) 
		self._fConst338 = np.tan((np.float32(785.3982) / self._fConst0)) 
		self._fConst339 = (np.float32(1.0) / self._fConst338) 
		self._fConst340 = (np.float32(1.0) / (((self._fConst339 + np.float32(0.15748216)) / self._fConst338) + np.float32(0.9351402))) 
		self._fConst341 = np.power(self._fConst338, np.float32(2.0)) 
		self._fConst342 = (np.float32(50.06381) / self._fConst341) 
		self._fConst343 = (self._fConst342 + np.float32(0.9351402)) 
		self._fConst344 = (np.float32(1.0) / (((self._fConst339 + np.float32(0.74313045)) / self._fConst338) + np.float32(1.4500711))) 
		self._fConst345 = (np.float32(11.0520525) / self._fConst341) 
		self._fConst346 = (self._fConst345 + np.float32(1.4500711)) 
		self._fConst347 = (np.float32(1.0) / (((self._fConst339 + np.float32(3.1897273)) / self._fConst338) + np.float32(4.0767817))) 
		self._fConst348 = (np.float32(0.0017661728) / self._fConst341) 
		self._fConst349 = (self._fConst348 + np.float32(0.0004076782)) 
		self._fConst350 = (np.float32(1.0) / (((self._fConst301 + np.float32(0.16840488)) / self._fConst300) + np.float32(1.0693583))) 
		self._fConst351 = (self._fConst329 + np.float32(53.53615)) 
		self._fConst352 = (np.float32(1.0) / (((self._fConst301 + np.float32(0.51247865)) / self._fConst300) + np.float32(0.6896214))) 
		self._fConst353 = (self._fConst329 + np.float32(7.6217313)) 
		self._fConst354 = (np.float32(1.0) / (((self._fConst301 + np.float32(0.78241307)) / self._fConst300) + np.float32(0.2452915))) 
		self._fConst355 = (np.float32(0.0001) / self._fConst303) 
		self._fConst356 = (self._fConst355 + np.float32(0.0004332272)) 
		self._fConst357 = (((self._fConst301 + np.float32(-0.78241307)) / self._fConst300) + np.float32(0.2452915)) 
		self._fConst358 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst329)) 
		self._fConst359 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst355)) 
		self._fConst360 = (((self._fConst301 + np.float32(-0.51247865)) / self._fConst300) + np.float32(0.6896214)) 
		self._fConst361 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst329)) 
		self._fConst362 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst329)) 
		self._fConst363 = (((self._fConst301 + np.float32(-0.16840488)) / self._fConst300) + np.float32(1.0693583)) 
		self._fConst364 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst329)) 
		self._fConst365 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst329)) 
		self._fConst366 = (((self._fConst339 + np.float32(-3.1897273)) / self._fConst338) + np.float32(4.0767817)) 
		self._fConst367 = (np.float32(1.0) / self._fConst341) 
		self._fConst368 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst367)) 
		self._fConst369 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst348)) 
		self._fConst370 = (((self._fConst339 + np.float32(-0.74313045)) / self._fConst338) + np.float32(1.4500711)) 
		self._fConst371 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst367)) 
		self._fConst372 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst345)) 
		self._fConst373 = (((self._fConst339 + np.float32(-0.15748216)) / self._fConst338) + np.float32(0.9351402)) 
		self._fConst374 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst367)) 
		self._fConst375 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst342)) 
		self._fConst376 = np.tan((np.float32(494.76984) / self._fConst0)) 
		self._fConst377 = (np.float32(1.0) / self._fConst376) 
		self._fConst378 = (np.float32(1.0) / (((self._fConst377 + np.float32(0.15748216)) / self._fConst376) + np.float32(0.9351402))) 
		self._fConst379 = np.power(self._fConst376, np.float32(2.0)) 
		self._fConst380 = (np.float32(50.06381) / self._fConst379) 
		self._fConst381 = (self._fConst380 + np.float32(0.9351402)) 
		self._fConst382 = (np.float32(1.0) / (((self._fConst377 + np.float32(0.74313045)) / self._fConst376) + np.float32(1.4500711))) 
		self._fConst383 = (np.float32(11.0520525) / self._fConst379) 
		self._fConst384 = (self._fConst383 + np.float32(1.4500711)) 
		self._fConst385 = (np.float32(1.0) / (((self._fConst377 + np.float32(3.1897273)) / self._fConst376) + np.float32(4.0767817))) 
		self._fConst386 = (np.float32(0.0017661728) / self._fConst379) 
		self._fConst387 = (self._fConst386 + np.float32(0.0004076782)) 
		self._fConst388 = (np.float32(1.0) / (((self._fConst339 + np.float32(0.16840488)) / self._fConst338) + np.float32(1.0693583))) 
		self._fConst389 = (self._fConst367 + np.float32(53.53615)) 
		self._fConst390 = (np.float32(1.0) / (((self._fConst339 + np.float32(0.51247865)) / self._fConst338) + np.float32(0.6896214))) 
		self._fConst391 = (self._fConst367 + np.float32(7.6217313)) 
		self._fConst392 = (np.float32(1.0) / (((self._fConst339 + np.float32(0.78241307)) / self._fConst338) + np.float32(0.2452915))) 
		self._fConst393 = (np.float32(0.0001) / self._fConst341) 
		self._fConst394 = (self._fConst393 + np.float32(0.0004332272)) 
		self._fConst395 = (((self._fConst339 + np.float32(-0.78241307)) / self._fConst338) + np.float32(0.2452915)) 
		self._fConst396 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst367)) 
		self._fConst397 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst393)) 
		self._fConst398 = (((self._fConst339 + np.float32(-0.51247865)) / self._fConst338) + np.float32(0.6896214)) 
		self._fConst399 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst367)) 
		self._fConst400 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst367)) 
		self._fConst401 = (((self._fConst339 + np.float32(-0.16840488)) / self._fConst338) + np.float32(1.0693583)) 
		self._fConst402 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst367)) 
		self._fConst403 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst367)) 
		self._fConst404 = (((self._fConst377 + np.float32(-3.1897273)) / self._fConst376) + np.float32(4.0767817)) 
		self._fConst405 = (np.float32(1.0) / self._fConst379) 
		self._fConst406 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst405)) 
		self._fConst407 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst386)) 
		self._fConst408 = (((self._fConst377 + np.float32(-0.74313045)) / self._fConst376) + np.float32(1.4500711)) 
		self._fConst409 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst405)) 
		self._fConst410 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst383)) 
		self._fConst411 = (((self._fConst377 + np.float32(-0.15748216)) / self._fConst376) + np.float32(0.9351402)) 
		self._fConst412 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst405)) 
		self._fConst413 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst380)) 
		self._fConst414 = np.tan((np.float32(311.68546) / self._fConst0)) 
		self._fConst415 = (np.float32(1.0) / self._fConst414) 
		self._fConst416 = (np.float32(1.0) / (((self._fConst415 + np.float32(0.15748216)) / self._fConst414) + np.float32(0.9351402))) 
		self._fConst417 = np.power(self._fConst414, np.float32(2.0)) 
		self._fConst418 = (np.float32(50.06381) / self._fConst417) 
		self._fConst419 = (self._fConst418 + np.float32(0.9351402)) 
		self._fConst420 = (np.float32(1.0) / (((self._fConst415 + np.float32(0.74313045)) / self._fConst414) + np.float32(1.4500711))) 
		self._fConst421 = (np.float32(11.0520525) / self._fConst417) 
		self._fConst422 = (self._fConst421 + np.float32(1.4500711)) 
		self._fConst423 = (np.float32(1.0) / (((self._fConst415 + np.float32(3.1897273)) / self._fConst414) + np.float32(4.0767817))) 
		self._fConst424 = (np.float32(0.0017661728) / self._fConst417) 
		self._fConst425 = (self._fConst424 + np.float32(0.0004076782)) 
		self._fConst426 = (np.float32(1.0) / (((self._fConst377 + np.float32(0.16840488)) / self._fConst376) + np.float32(1.0693583))) 
		self._fConst427 = (self._fConst405 + np.float32(53.53615)) 
		self._fConst428 = (np.float32(1.0) / (((self._fConst377 + np.float32(0.51247865)) / self._fConst376) + np.float32(0.6896214))) 
		self._fConst429 = (self._fConst405 + np.float32(7.6217313)) 
		self._fConst430 = (np.float32(1.0) / (((self._fConst377 + np.float32(0.78241307)) / self._fConst376) + np.float32(0.2452915))) 
		self._fConst431 = (np.float32(0.0001) / self._fConst379) 
		self._fConst432 = (self._fConst431 + np.float32(0.0004332272)) 
		self._fConst433 = (((self._fConst377 + np.float32(-0.78241307)) / self._fConst376) + np.float32(0.2452915)) 
		self._fConst434 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst405)) 
		self._fConst435 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst431)) 
		self._fConst436 = (((self._fConst377 + np.float32(-0.51247865)) / self._fConst376) + np.float32(0.6896214)) 
		self._fConst437 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst405)) 
		self._fConst438 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst405)) 
		self._fConst439 = (((self._fConst377 + np.float32(-0.16840488)) / self._fConst376) + np.float32(1.0693583)) 
		self._fConst440 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst405)) 
		self._fConst441 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst405)) 
		self._fConst442 = (((self._fConst415 + np.float32(-3.1897273)) / self._fConst414) + np.float32(4.0767817)) 
		self._fConst443 = (np.float32(1.0) / self._fConst417) 
		self._fConst444 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst443)) 
		self._fConst445 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst424)) 
		self._fConst446 = (((self._fConst415 + np.float32(-0.74313045)) / self._fConst414) + np.float32(1.4500711)) 
		self._fConst447 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst443)) 
		self._fConst448 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst421)) 
		self._fConst449 = (((self._fConst415 + np.float32(-0.15748216)) / self._fConst414) + np.float32(0.9351402)) 
		self._fConst450 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst443)) 
		self._fConst451 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst418)) 
		self._fConst452 = np.tan((np.float32(196.34955) / self._fConst0)) 
		self._fConst453 = (np.float32(1.0) / self._fConst452) 
		self._fConst454 = (np.float32(1.0) / (((self._fConst453 + np.float32(0.15748216)) / self._fConst452) + np.float32(0.9351402))) 
		self._fConst455 = np.power(self._fConst452, np.float32(2.0)) 
		self._fConst456 = (np.float32(50.06381) / self._fConst455) 
		self._fConst457 = (self._fConst456 + np.float32(0.9351402)) 
		self._fConst458 = (np.float32(1.0) / (((self._fConst453 + np.float32(0.74313045)) / self._fConst452) + np.float32(1.4500711))) 
		self._fConst459 = (np.float32(11.0520525) / self._fConst455) 
		self._fConst460 = (self._fConst459 + np.float32(1.4500711)) 
		self._fConst461 = (np.float32(1.0) / (((self._fConst453 + np.float32(3.1897273)) / self._fConst452) + np.float32(4.0767817))) 
		self._fConst462 = (np.float32(0.0017661728) / self._fConst455) 
		self._fConst463 = (self._fConst462 + np.float32(0.0004076782)) 
		self._fConst464 = (np.float32(1.0) / (((self._fConst415 + np.float32(0.16840488)) / self._fConst414) + np.float32(1.0693583))) 
		self._fConst465 = (self._fConst443 + np.float32(53.53615)) 
		self._fConst466 = (np.float32(1.0) / (((self._fConst415 + np.float32(0.51247865)) / self._fConst414) + np.float32(0.6896214))) 
		self._fConst467 = (self._fConst443 + np.float32(7.6217313)) 
		self._fConst468 = (np.float32(1.0) / (((self._fConst415 + np.float32(0.78241307)) / self._fConst414) + np.float32(0.2452915))) 
		self._fConst469 = (np.float32(0.0001) / self._fConst417) 
		self._fConst470 = (self._fConst469 + np.float32(0.0004332272)) 
		self._fConst471 = (((self._fConst415 + np.float32(-0.78241307)) / self._fConst414) + np.float32(0.2452915)) 
		self._fConst472 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst443)) 
		self._fConst473 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst469)) 
		self._fConst474 = (((self._fConst415 + np.float32(-0.51247865)) / self._fConst414) + np.float32(0.6896214)) 
		self._fConst475 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst443)) 
		self._fConst476 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst443)) 
		self._fConst477 = (((self._fConst415 + np.float32(-0.16840488)) / self._fConst414) + np.float32(1.0693583)) 
		self._fConst478 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst443)) 
		self._fConst479 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst443)) 
		self._fConst480 = (((self._fConst453 + np.float32(-3.1897273)) / self._fConst452) + np.float32(4.0767817)) 
		self._fConst481 = (np.float32(1.0) / self._fConst455) 
		self._fConst482 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst481)) 
		self._fConst483 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst462)) 
		self._fConst484 = (((self._fConst453 + np.float32(-0.74313045)) / self._fConst452) + np.float32(1.4500711)) 
		self._fConst485 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst481)) 
		self._fConst486 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst459)) 
		self._fConst487 = (((self._fConst453 + np.float32(-0.15748216)) / self._fConst452) + np.float32(0.9351402)) 
		self._fConst488 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst481)) 
		self._fConst489 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst456)) 
		self._fConst490 = np.tan((np.float32(123.69246) / self._fConst0)) 
		self._fConst491 = (np.float32(1.0) / self._fConst490) 
		self._fConst492 = (np.float32(1.0) / (((self._fConst491 + np.float32(0.15748216)) / self._fConst490) + np.float32(0.9351402))) 
		self._fConst493 = np.power(self._fConst490, np.float32(2.0)) 
		self._fConst494 = (np.float32(50.06381) / self._fConst493) 
		self._fConst495 = (self._fConst494 + np.float32(0.9351402)) 
		self._fConst496 = (np.float32(1.0) / (((self._fConst491 + np.float32(0.74313045)) / self._fConst490) + np.float32(1.4500711))) 
		self._fConst497 = (np.float32(11.0520525) / self._fConst493) 
		self._fConst498 = (self._fConst497 + np.float32(1.4500711)) 
		self._fConst499 = (np.float32(1.0) / (((self._fConst491 + np.float32(3.1897273)) / self._fConst490) + np.float32(4.0767817))) 
		self._fConst500 = (np.float32(0.0017661728) / self._fConst493) 
		self._fConst501 = (self._fConst500 + np.float32(0.0004076782)) 
		self._fConst502 = (np.float32(1.0) / (((self._fConst453 + np.float32(0.16840488)) / self._fConst452) + np.float32(1.0693583))) 
		self._fConst503 = (self._fConst481 + np.float32(53.53615)) 
		self._fConst504 = (np.float32(1.0) / (((self._fConst453 + np.float32(0.51247865)) / self._fConst452) + np.float32(0.6896214))) 
		self._fConst505 = (self._fConst481 + np.float32(7.6217313)) 
		self._fConst506 = (np.float32(1.0) / (((self._fConst453 + np.float32(0.78241307)) / self._fConst452) + np.float32(0.2452915))) 
		self._fConst507 = (np.float32(0.0001) / self._fConst455) 
		self._fConst508 = (self._fConst507 + np.float32(0.0004332272)) 
		self._fConst509 = (((self._fConst453 + np.float32(-0.78241307)) / self._fConst452) + np.float32(0.2452915)) 
		self._fConst510 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst481)) 
		self._fConst511 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst507)) 
		self._fConst512 = (((self._fConst453 + np.float32(-0.51247865)) / self._fConst452) + np.float32(0.6896214)) 
		self._fConst513 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst481)) 
		self._fConst514 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst481)) 
		self._fConst515 = (((self._fConst453 + np.float32(-0.16840488)) / self._fConst452) + np.float32(1.0693583)) 
		self._fConst516 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst481)) 
		self._fConst517 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst481)) 
		self._fConst518 = (((self._fConst491 + np.float32(-3.1897273)) / self._fConst490) + np.float32(4.0767817)) 
		self._fConst519 = (np.float32(1.0) / self._fConst493) 
		self._fConst520 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst519)) 
		self._fConst521 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst500)) 
		self._fConst522 = (((self._fConst491 + np.float32(-0.74313045)) / self._fConst490) + np.float32(1.4500711)) 
		self._fConst523 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst519)) 
		self._fConst524 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst497)) 
		self._fConst525 = (((self._fConst491 + np.float32(-0.15748216)) / self._fConst490) + np.float32(0.9351402)) 
		self._fConst526 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst519)) 
		self._fConst527 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst494)) 
		self._fConst528 = (np.float32(1.0) / (((self._fConst491 + np.float32(0.16840488)) / self._fConst490) + np.float32(1.0693583))) 
		self._fConst529 = (self._fConst519 + np.float32(53.53615)) 
		self._fConst530 = (np.float32(1.0) / (((self._fConst491 + np.float32(0.51247865)) / self._fConst490) + np.float32(0.6896214))) 
		self._fConst531 = (self._fConst519 + np.float32(7.6217313)) 
		self._fConst532 = (np.float32(1.0) / (((self._fConst491 + np.float32(0.78241307)) / self._fConst490) + np.float32(0.2452915))) 
		self._fConst533 = (np.float32(0.0001) / self._fConst493) 
		self._fConst534 = (self._fConst533 + np.float32(0.0004332272)) 
		self._fConst535 = (((self._fConst491 + np.float32(-0.78241307)) / self._fConst490) + np.float32(0.2452915)) 
		self._fConst536 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst519)) 
		self._fConst537 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst533)) 
		self._fConst538 = (((self._fConst491 + np.float32(-0.51247865)) / self._fConst490) + np.float32(0.6896214)) 
		self._fConst539 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst519)) 
		self._fConst540 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst519)) 
		self._fConst541 = (((self._fConst491 + np.float32(-0.16840488)) / self._fConst490) + np.float32(1.0693583)) 
		self._fConst542 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst519)) 
		self._fConst543 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst519)) 
		
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
		state["fRec103"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec110"] = np.float32(0)
		state["fRec117"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec124"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec16"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec20"] = np.float32(0)
		state["fRec21"] = np.float32(0)
		state["fRec22"] = np.float32(0)
		state["fRec27"] = np.float32(0)
		state["fRec29"] = np.float32(0)
		state["fRec30"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec33"] = np.float32(0)
		state["fRec4"] = np.float32(0)
		state["fRec40"] = np.float32(0)
		state["fRec47"] = np.float32(0)
		state["fRec5"] = np.float32(0)
		state["fRec54"] = np.float32(0)
		state["fRec6"] = np.float32(0)
		state["fRec61"] = np.float32(0)
		state["fRec68"] = np.float32(0)
		state["fRec7"] = np.float32(0)
		state["fRec75"] = np.float32(0)
		state["fRec8"] = np.float32(0)
		state["fRec82"] = np.float32(0)
		state["fRec89"] = np.float32(0)
		state["fRec9"] = np.float32(0)
		state["fRec96"] = np.float32(0)
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
		state["fVec21"] = np.float32(0)
		state["fVec23"] = np.float32(0)
		state["fVec25"] = np.float32(0)
		state["fVec26"] = np.float32(0)
		state["fVec28"] = np.float32(0)
		state["fVec29"] = np.float32(0)
		state["fVec3"] = np.float32(0)
		state["fVec31"] = np.float32(0)
		state["fVec32"] = np.float32(0)
		state["fVec4"] = np.float32(0)
		state["fVec5"] = np.float32(0)
		state["fVec6"] = np.float32(0)
		state["fVec7"] = np.float32(0)
		state["fVec8"] = np.float32(0)
		state["fVec9"] = np.float32(0)
		state["iRec24"] = np.int32(0)
		# Initialize array delays
		state["iVec0"] = np.zeros((4,), dtype=np.int32)
		state["fVec20"] = np.zeros((4096,), dtype=np.float32)
		state["fVec22"] = np.zeros((4096,), dtype=np.float32)
		state["fVec24"] = np.zeros((4096,), dtype=np.float32)
		state["fVec27"] = np.zeros((4096,), dtype=np.float32)
		state["fVec30"] = np.zeros((4096,), dtype=np.float32)
		state["fVec33"] = np.zeros((4096,), dtype=np.float32)
		state["fRec23"] = np.zeros((4,), dtype=np.float32)
		state["fRec26"] = np.zeros((3,), dtype=np.float32)
		state["fRec25"] = np.zeros((3,), dtype=np.float32)
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec39"] = np.zeros((3,), dtype=np.float32)
		state["fRec38"] = np.zeros((3,), dtype=np.float32)
		state["fRec37"] = np.zeros((3,), dtype=np.float32)
		state["fRec36"] = np.zeros((3,), dtype=np.float32)
		state["fRec35"] = np.zeros((3,), dtype=np.float32)
		state["fRec34"] = np.zeros((3,), dtype=np.float32)
		state["fRec46"] = np.zeros((3,), dtype=np.float32)
		state["fRec45"] = np.zeros((3,), dtype=np.float32)
		state["fRec44"] = np.zeros((3,), dtype=np.float32)
		state["fRec43"] = np.zeros((3,), dtype=np.float32)
		state["fRec42"] = np.zeros((3,), dtype=np.float32)
		state["fRec41"] = np.zeros((3,), dtype=np.float32)
		state["fRec53"] = np.zeros((3,), dtype=np.float32)
		state["fRec52"] = np.zeros((3,), dtype=np.float32)
		state["fRec51"] = np.zeros((3,), dtype=np.float32)
		state["fRec50"] = np.zeros((3,), dtype=np.float32)
		state["fRec49"] = np.zeros((3,), dtype=np.float32)
		state["fRec48"] = np.zeros((3,), dtype=np.float32)
		state["fRec60"] = np.zeros((3,), dtype=np.float32)
		state["fRec59"] = np.zeros((3,), dtype=np.float32)
		state["fRec58"] = np.zeros((3,), dtype=np.float32)
		state["fRec57"] = np.zeros((3,), dtype=np.float32)
		state["fRec56"] = np.zeros((3,), dtype=np.float32)
		state["fRec55"] = np.zeros((3,), dtype=np.float32)
		state["fRec67"] = np.zeros((3,), dtype=np.float32)
		state["fRec66"] = np.zeros((3,), dtype=np.float32)
		state["fRec65"] = np.zeros((3,), dtype=np.float32)
		state["fRec64"] = np.zeros((3,), dtype=np.float32)
		state["fRec63"] = np.zeros((3,), dtype=np.float32)
		state["fRec62"] = np.zeros((3,), dtype=np.float32)
		state["fRec74"] = np.zeros((3,), dtype=np.float32)
		state["fRec73"] = np.zeros((3,), dtype=np.float32)
		state["fRec72"] = np.zeros((3,), dtype=np.float32)
		state["fRec71"] = np.zeros((3,), dtype=np.float32)
		state["fRec70"] = np.zeros((3,), dtype=np.float32)
		state["fRec69"] = np.zeros((3,), dtype=np.float32)
		state["fRec81"] = np.zeros((3,), dtype=np.float32)
		state["fRec80"] = np.zeros((3,), dtype=np.float32)
		state["fRec79"] = np.zeros((3,), dtype=np.float32)
		state["fRec78"] = np.zeros((3,), dtype=np.float32)
		state["fRec77"] = np.zeros((3,), dtype=np.float32)
		state["fRec76"] = np.zeros((3,), dtype=np.float32)
		state["fRec88"] = np.zeros((3,), dtype=np.float32)
		state["fRec87"] = np.zeros((3,), dtype=np.float32)
		state["fRec86"] = np.zeros((3,), dtype=np.float32)
		state["fRec85"] = np.zeros((3,), dtype=np.float32)
		state["fRec84"] = np.zeros((3,), dtype=np.float32)
		state["fRec83"] = np.zeros((3,), dtype=np.float32)
		state["fRec95"] = np.zeros((3,), dtype=np.float32)
		state["fRec94"] = np.zeros((3,), dtype=np.float32)
		state["fRec93"] = np.zeros((3,), dtype=np.float32)
		state["fRec92"] = np.zeros((3,), dtype=np.float32)
		state["fRec91"] = np.zeros((3,), dtype=np.float32)
		state["fRec90"] = np.zeros((3,), dtype=np.float32)
		state["fRec102"] = np.zeros((3,), dtype=np.float32)
		state["fRec101"] = np.zeros((3,), dtype=np.float32)
		state["fRec100"] = np.zeros((3,), dtype=np.float32)
		state["fRec99"] = np.zeros((3,), dtype=np.float32)
		state["fRec98"] = np.zeros((3,), dtype=np.float32)
		state["fRec97"] = np.zeros((3,), dtype=np.float32)
		state["fRec109"] = np.zeros((3,), dtype=np.float32)
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec116"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec114"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec112"] = np.zeros((3,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fRec123"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec120"] = np.zeros((3,), dtype=np.float32)
		state["fRec119"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec127"] = np.zeros((3,), dtype=np.float32)
		state["fRec126"] = np.zeros((3,), dtype=np.float32)
		state["fRec125"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = params["fHslider0"] 
		fSlow1 = params["fHslider1"] 
		fSlow2 = jnp.where((((jnp.float32(0.001) * fSlow1) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst1 / fSlow1))), jnp.float32(0.0)) 
		iSlow3 = jnp.int32(params["fCheckbox0"]) 
		fSlow4 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider2"]))) 
		iSlow5 = jnp.int32(params["fCheckbox1"]) 
		fSlow6 = (jnp.float32(0.44) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fHslider3"] + jnp.float32(-49.0))))) 
		fSlow7 = params["fVslider0"] 
		fSlow8 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fHslider4"]))) 
		fSlow9 = (jnp.float32(0.33333334) * params["fVslider1"]) 
		iSlow10 = jnp.int32((params["fEntry0"] + jnp.float32(-1.0))) 
		iSlow11 = (iSlow10 >= jnp.int32(2)).astype(jnp.int32) 
		iSlow12 = (iSlow10 >= jnp.int32(1)).astype(jnp.int32) 
		fSlow13 = params["fVslider2"] 
		fSlow14 = jnp.where(((fSlow13 > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst15 / fSlow13))), jnp.float32(0.0)) 
		fSlow15 = ((jnp.float32(4.4e+02) * jnp.power(jnp.float32(2.0), (jnp.float32(0.083333336) * (params["fHslider5"] + jnp.float32(-49.0))))) * (jnp.float32(1.0) - fSlow14)) 
		iSlow16 = (iSlow10 >= jnp.int32(3)).astype(jnp.int32) 
		fSlow17 = (jnp.float32(1.0) - (jnp.float32(0.01) * params["fVslider3"])) 
		fSlow18 = ((jnp.float32(0.01) * params["fVslider4"]) + jnp.float32(1.0)) 
		fSlow19 = (self._fConst19 * params["fVslider5"]) 
		fSlow20 = (self._fConst21 * params["fVslider6"]) 
		fSlow21 = (self._fConst0 * params["fVslider7"]) 
		fSlow22 = params["fCheckbox2"] 
		fSlow23 = (self._fConst22 * fSlow22) 
		fSlow24 = (jnp.float32(0.01) * params["fVslider8"]) 
		fSlow25 = (jnp.float32(0.083333336) * (jnp.float32(1.0) - fSlow22)) 
		fSlow26 = params["fVslider9"] 
		fSlow27 = params["fHslider6"] 
		fSlow28 = (jnp.float32(4.0) * jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.power(fSlow27, jnp.float32(4.0)), jnp.float32(0.999999)))) 
		iSlow29 = jnp.int32(params["fCheckbox3"]) 
		fSlow30 = jnp.minimum(jnp.float32(1.4127994), (jnp.float32(1.4142135) * fSlow27)) 
		fSlow31 = (jnp.float32(1.4142135) * fSlow30) 
		fSlow32 = jnp.power(fSlow30, jnp.float32(2.0)) 
		fSlow33 = (fSlow31 + fSlow32) 
		fSlow34 = (fSlow31 + jnp.float32(2.0)) 
		fSlow35 = (jnp.float32(2.0) - fSlow31) 
		fSlow36 = jnp.power((jnp.float32(1.4127994) * fSlow27), jnp.float32(2.0)) 
		fSlow37 = (jnp.float32(1.998) * fSlow27) 
		fSlow38 = (fSlow36 + fSlow37) 
		fSlow39 = (fSlow37 + jnp.float32(2.0)) 
		fSlow40 = (jnp.float32(2.0) - fSlow37) 
		fSlow41 = (jnp.float32(1.0) - fSlow2) 
		fRec4_temp = state["fRec4"] 
		fRec10_temp = state["fRec10"] 
		fRec11_temp = state["fRec11"] 
		fRec13_temp = state["fRec13"] 
		fRec12_temp = state["fRec12"] 
		fVec1_temp = state["fVec1"] 
		fVec2_temp = state["fVec2"] 
		fVec3_temp = state["fVec3"] 
		fVec4_temp = state["fVec4"] 
		fVec5_temp = state["fVec5"] 
		fVec6_temp = state["fVec6"] 
		fRec14_temp = state["fRec14"] 
		fVec7_temp = state["fVec7"] 
		fVec8_temp = state["fVec8"] 
		fVec9_temp = state["fVec9"] 
		fVec10_temp = state["fVec10"] 
		fVec11_temp = state["fVec11"] 
		fVec12_temp = state["fVec12"] 
		fRec15_temp = state["fRec15"] 
		fVec13_temp = state["fVec13"] 
		fVec14_temp = state["fVec14"] 
		fVec15_temp = state["fVec15"] 
		fVec16_temp = state["fVec16"] 
		fVec17_temp = state["fVec17"] 
		fVec18_temp = state["fVec18"] 
		fRec16_temp = state["fRec16"] 
		fVec19_temp = state["fVec19"] 
		fRec17_temp = state["fRec17"] 
		fVec21_temp = state["fVec21"] 
		fRec18_temp = state["fRec18"] 
		fVec23_temp = state["fVec23"] 
		fRec19_temp = state["fRec19"] 
		fRec20_temp = state["fRec20"] 
		fRec21_temp = state["fRec21"] 
		fVec25_temp = state["fVec25"] 
		fVec26_temp = state["fVec26"] 
		fRec22_temp = state["fRec22"] 
		fVec28_temp = state["fVec28"] 
		fVec29_temp = state["fVec29"] 
		fVec31_temp = state["fVec31"] 
		fVec32_temp = state["fVec32"] 
		iRec24_temp = state["iRec24"] 
		fRec9_temp = state["fRec9"] 
		fRec8_temp = state["fRec8"] 
		fRec7_temp = state["fRec7"] 
		fRec6_temp = state["fRec6"] 
		fRec5_temp = state["fRec5"] 
		fRec29_temp = state["fRec29"] 
		fRec27_temp = state["fRec27"] 
		fRec32_temp = state["fRec32"] 
		fRec30_temp = state["fRec30"] 
		fRec0_temp = state["fRec0"] 
		fRec33_temp = state["fRec33"] 
		fRec40_temp = state["fRec40"] 
		fRec47_temp = state["fRec47"] 
		fRec54_temp = state["fRec54"] 
		fRec61_temp = state["fRec61"] 
		fRec68_temp = state["fRec68"] 
		fRec75_temp = state["fRec75"] 
		fRec82_temp = state["fRec82"] 
		fRec89_temp = state["fRec89"] 
		fRec96_temp = state["fRec96"] 
		fRec103_temp = state["fRec103"] 
		fRec110_temp = state["fRec110"] 
		fRec117_temp = state["fRec117"] 
		fRec124_temp = state["fRec124"] 
		state["fRec4"] = (fSlow4 + (jnp.float32(0.999) * fRec4_temp)) 
		state["fRec10"] = (fSlow6 + (jnp.float32(0.999) * fRec10_temp)) 
		fTemp0 = (self._fConst14 * state["fRec10"]) 
		fTemp1 = (jnp.float32(1.0) - fTemp0) 
		state["fRec11"] = (fSlow8 + (jnp.float32(0.999) * fRec11_temp)) 
		state["iVec0"] = state["iVec0"].at[0].set(jnp.int32(1)) 
		state["fRec13"] = ((fRec13_temp * fSlow14) + fSlow15) 
		fTemp2 = jnp.maximum(jnp.float32(2e+01), jnp.abs(state["fRec13"])) 
		fTemp3 = (fRec12_temp + (self._fConst15 * fTemp2)) 
		state["fRec12"] = (fTemp3 - jnp.floor(fTemp3)) 
		fTemp4 = (jnp.float32(2.0) * state["fRec12"]) 
		fTemp5 = (fTemp4 + jnp.float32(-1.0)) 
		fTemp6 = (state["iVec0"][1]) 
		fTemp7 = jnp.power(fTemp5, jnp.float32(2.0)) 
		state["fVec1"] = jnp.float32(fTemp7) 
		fTemp8 = (state["iVec0"][2]) 
		fTemp9 = jnp.power(fTemp5, jnp.float32(3.0)) 
		state["fVec2"] = (fTemp9 + (jnp.float32(1.0) - fTemp4)) 
		fTemp10 = ((fTemp9 + (jnp.float32(1.0) - (fTemp4 + fVec2_temp))) / fTemp2) 
		state["fVec3"] = jnp.float32(fTemp10) 
		fTemp11 = (state["iVec0"][3]) 
		fTemp12 = (fTemp7 * (fTemp7 + jnp.float32(-2.0))) 
		state["fVec4"] = jnp.float32(fTemp12) 
		fTemp13 = ((fTemp12 - fVec4_temp) / fTemp2) 
		state["fVec5"] = jnp.float32(fTemp13) 
		fTemp14 = ((fTemp13 - fVec5_temp) / fTemp2) 
		state["fVec6"] = jnp.float32(fTemp14) 
		fTemp15 = (fSlow17 * state["fRec13"]) 
		fTemp16 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp15)) 
		fTemp17 = (fRec14_temp + (self._fConst15 * fTemp16)) 
		state["fRec14"] = (fTemp17 - jnp.floor(fTemp17)) 
		fTemp18 = (jnp.float32(2.0) * state["fRec14"]) 
		fTemp19 = (fTemp18 + jnp.float32(-1.0)) 
		fTemp20 = jnp.power(fTemp19, jnp.float32(2.0)) 
		state["fVec7"] = jnp.float32(fTemp20) 
		fTemp21 = jnp.power(fTemp19, jnp.float32(3.0)) 
		state["fVec8"] = (fTemp21 + (jnp.float32(1.0) - fTemp18)) 
		fTemp22 = ((fTemp21 + (jnp.float32(1.0) - (fTemp18 + fVec8_temp))) / fTemp16) 
		state["fVec9"] = jnp.float32(fTemp22) 
		fTemp23 = (fTemp20 * (fTemp20 + jnp.float32(-2.0))) 
		state["fVec10"] = jnp.float32(fTemp23) 
		fTemp24 = ((fTemp23 - fVec10_temp) / fTemp16) 
		state["fVec11"] = jnp.float32(fTemp24) 
		fTemp25 = ((fTemp24 - fVec11_temp) / fTemp16) 
		state["fVec12"] = jnp.float32(fTemp25) 
		fTemp26 = (fSlow18 * state["fRec13"]) 
		fTemp27 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp26)) 
		fTemp28 = (fRec15_temp + (self._fConst15 * fTemp27)) 
		state["fRec15"] = (fTemp28 - jnp.floor(fTemp28)) 
		fTemp29 = (jnp.float32(2.0) * state["fRec15"]) 
		fTemp30 = (fTemp29 + jnp.float32(-1.0)) 
		fTemp31 = jnp.power(fTemp30, jnp.float32(2.0)) 
		state["fVec13"] = jnp.float32(fTemp31) 
		fTemp32 = jnp.power(fTemp30, jnp.float32(3.0)) 
		state["fVec14"] = (fTemp32 + (jnp.float32(1.0) - fTemp29)) 
		fTemp33 = ((fTemp32 + (jnp.float32(1.0) - (fTemp29 + fVec14_temp))) / fTemp27) 
		state["fVec15"] = jnp.float32(fTemp33) 
		fTemp34 = (fTemp31 * (fTemp31 + jnp.float32(-2.0))) 
		state["fVec16"] = jnp.float32(fTemp34) 
		fTemp35 = ((fTemp34 - fVec16_temp) / fTemp27) 
		state["fVec17"] = jnp.float32(fTemp35) 
		fTemp36 = ((fTemp35 - fVec17_temp) / fTemp27) 
		state["fVec18"] = jnp.float32(fTemp36) 
		fTemp37 = jnp.maximum(state["fRec13"], jnp.float32(23.44895)) 
		fTemp38 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp37)) 
		fTemp39 = (fRec16_temp + (self._fConst15 * fTemp38)) 
		state["fRec16"] = (fTemp39 - jnp.floor(fTemp39)) 
		fTemp40 = (jnp.float32(2.0) * state["fRec16"]) 
		fTemp41 = (fTemp40 + jnp.float32(-1.0)) 
		fTemp42 = jnp.power(fTemp41, jnp.float32(2.0)) 
		state["fVec19"] = jnp.float32(fTemp42) 
		fTemp43 = ((fTemp6 * (fTemp42 - fVec19_temp)) / fTemp38) 
		state["fVec20"] = state["fVec20"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp43) 
		fTemp44 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst20 / fTemp37))) 
		iTemp45 = jnp.int32(fTemp44) 
		fTemp46 = (iTemp45) 
		fTemp47 = ((fTemp43 - (state["fVec20"][((state["IOTA0"] - iTemp45) & 4095).astype(jnp.int32)] * (fTemp46 + (jnp.float32(1.0) - fTemp44)))) - ((fTemp44 - fTemp46) * state["fVec20"][((state["IOTA0"] - (iTemp45 + 1)) & 4095).astype(jnp.int32)])) 
		fTemp48 = jnp.maximum(fTemp15, jnp.float32(23.44895)) 
		fTemp49 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp48)) 
		fTemp50 = (fRec17_temp + (self._fConst15 * fTemp49)) 
		state["fRec17"] = (fTemp50 - jnp.floor(fTemp50)) 
		fTemp51 = (jnp.float32(2.0) * state["fRec17"]) 
		fTemp52 = (fTemp51 + jnp.float32(-1.0)) 
		fTemp53 = jnp.power(fTemp52, jnp.float32(2.0)) 
		state["fVec21"] = jnp.float32(fTemp53) 
		fTemp54 = ((fTemp6 * (fTemp53 - fVec21_temp)) / fTemp49) 
		state["fVec22"] = state["fVec22"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp54) 
		fTemp55 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst20 / fTemp48))) 
		iTemp56 = jnp.int32(fTemp55) 
		fTemp57 = (iTemp56) 
		fTemp58 = ((fTemp54 - (state["fVec22"][((state["IOTA0"] - iTemp56) & 4095).astype(jnp.int32)] * (fTemp57 + (jnp.float32(1.0) - fTemp55)))) - ((fTemp55 - fTemp57) * state["fVec22"][((state["IOTA0"] - (iTemp56 + 1)) & 4095).astype(jnp.int32)])) 
		fTemp59 = jnp.maximum(fTemp26, jnp.float32(23.44895)) 
		fTemp60 = jnp.maximum(jnp.float32(2e+01), jnp.abs(fTemp59)) 
		fTemp61 = (fRec18_temp + (self._fConst15 * fTemp60)) 
		state["fRec18"] = (fTemp61 - jnp.floor(fTemp61)) 
		fTemp62 = (jnp.float32(2.0) * state["fRec18"]) 
		fTemp63 = (fTemp62 + jnp.float32(-1.0)) 
		fTemp64 = jnp.power(fTemp63, jnp.float32(2.0)) 
		state["fVec23"] = jnp.float32(fTemp64) 
		fTemp65 = ((fTemp6 * (fTemp64 - fVec23_temp)) / fTemp60) 
		state["fVec24"] = state["fVec24"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp65) 
		fTemp66 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst20 / fTemp59))) 
		iTemp67 = jnp.int32(fTemp66) 
		fTemp68 = (iTemp67) 
		fTemp69 = ((fTemp65 - (state["fVec24"][((state["IOTA0"] - iTemp67) & 4095).astype(jnp.int32)] * (fTemp68 + (jnp.float32(1.0) - fTemp66)))) - ((fTemp66 - fTemp68) * state["fVec24"][((state["IOTA0"] - (iTemp67 + 1)) & 4095).astype(jnp.int32)])) 
		state["fRec19"] = ((self._fConst16 * fTemp47) + (jnp.float32(0.999) * fRec19_temp)) 
		state["fRec20"] = ((self._fConst16 * fTemp58) + (jnp.float32(0.999) * fRec20_temp)) 
		state["fRec21"] = ((self._fConst16 * fTemp69) + (jnp.float32(0.999) * fRec21_temp)) 
		fTemp70 = jnp.power(fTemp41, jnp.float32(3.0)) 
		state["fVec25"] = (fTemp70 + (jnp.float32(1.0) - fTemp40)) 
		fTemp71 = ((fTemp70 + (jnp.float32(1.0) - (fTemp40 + fVec25_temp))) / fTemp38) 
		state["fVec26"] = jnp.float32(fTemp71) 
		fTemp72 = ((fTemp8 * (fTemp71 - fVec26_temp)) / fTemp38) 
		state["fVec27"] = state["fVec27"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp72) 
		state["fRec22"] = (fSlow24 + (jnp.float32(0.99) * fRec22_temp)) 
		fTemp73 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst0 * (state["fRec22"] / fTemp37)))) 
		iTemp74 = jnp.int32(fTemp73) 
		fTemp75 = (iTemp74) 
		fTemp76 = (fTemp75 + (jnp.float32(1.0) - fTemp73)) 
		fTemp77 = (fTemp73 - fTemp75) 
		iTemp78 = (iTemp74 + jnp.int32(1)) 
		fTemp79 = jnp.power(fTemp52, jnp.float32(3.0)) 
		state["fVec28"] = (fTemp79 + (jnp.float32(1.0) - fTemp51)) 
		fTemp80 = ((fTemp79 + (jnp.float32(1.0) - (fTemp51 + fVec28_temp))) / fTemp49) 
		state["fVec29"] = jnp.float32(fTemp80) 
		fTemp81 = ((fTemp8 * (fTemp80 - fVec29_temp)) / fTemp49) 
		state["fVec30"] = state["fVec30"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp81) 
		fTemp82 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst0 * (state["fRec22"] / fTemp48)))) 
		iTemp83 = jnp.int32(fTemp82) 
		fTemp84 = (iTemp83) 
		fTemp85 = (fTemp84 + (jnp.float32(1.0) - fTemp82)) 
		fTemp86 = (fTemp82 - fTemp84) 
		iTemp87 = (iTemp83 + jnp.int32(1)) 
		fTemp88 = jnp.power(fTemp63, jnp.float32(3.0)) 
		state["fVec31"] = (fTemp88 + (jnp.float32(1.0) - fTemp62)) 
		fTemp89 = ((fTemp88 + (jnp.float32(1.0) - (fTemp62 + fVec31_temp))) / fTemp60) 
		state["fVec32"] = jnp.float32(fTemp89) 
		fTemp90 = ((fTemp8 * (fTemp89 - fVec32_temp)) / fTemp60) 
		state["fVec33"] = state["fVec33"].at[(state["IOTA0"] & 4095).astype(jnp.int32)].set(fTemp90) 
		fTemp91 = jnp.maximum(jnp.float32(0.0), jnp.minimum(jnp.float32(2047.0), (self._fConst0 * (state["fRec22"] / fTemp59)))) 
		iTemp92 = jnp.int32(fTemp91) 
		fTemp93 = (iTemp92) 
		fTemp94 = (fTemp93 + (jnp.float32(1.0) - fTemp91)) 
		fTemp95 = (fTemp91 - fTemp93) 
		iTemp96 = (iTemp92 + jnp.int32(1)) 
		state["iRec24"] = ((jnp.int32(1103515245) * iRec24_temp) + jnp.int32(12345)) 
		state["fRec23"] = state["fRec23"].at[0].set((((jnp.float32(0.5221894) * state["fRec23"][3]) + ((jnp.float32(4.656613e-10) * (state["iRec24"])) + (jnp.float32(2.494956) * state["fRec23"][1]))) - (jnp.float32(2.0172658) * state["fRec23"][2]))) 
		fTemp97 = ((fSlow7 * inputs[0]) + (state["fRec11"] * (((((fSlow9 * ((jnp.where((iSlow11 != 0), jnp.where((iSlow16 != 0), (self._fConst18 * ((fTemp11 * (fTemp14 - fVec6_temp)) / fTemp2)), (self._fConst17 * ((fTemp8 * (fTemp10 - fVec3_temp)) / fTemp2))), jnp.where((iSlow12 != 0), (self._fConst16 * ((fTemp6 * (fTemp7 - fVec1_temp)) / fTemp2)), fTemp5)) + jnp.where((iSlow11 != 0), jnp.where((iSlow16 != 0), (self._fConst18 * ((fTemp11 * (fTemp25 - fVec12_temp)) / fTemp16)), (self._fConst17 * ((fTemp8 * (fTemp22 - fVec9_temp)) / fTemp16))), jnp.where((iSlow12 != 0), (self._fConst16 * ((fTemp6 * (fTemp20 - fVec7_temp)) / fTemp16)), fTemp19))) + jnp.where((iSlow11 != 0), jnp.where((iSlow16 != 0), (self._fConst18 * ((fTemp11 * (fTemp36 - fVec18_temp)) / fTemp27)), (self._fConst17 * ((fTemp8 * (fTemp33 - fVec15_temp)) / fTemp27))), jnp.where((iSlow12 != 0), (self._fConst16 * ((fTemp6 * (fTemp31 - fVec13_temp)) / fTemp27)), fTemp30)))) + (fSlow19 * ((fTemp47 + fTemp58) + fTemp69))) + (fSlow20 * (state["fRec13"] * ((state["fRec19"] + (fSlow17 * state["fRec20"])) + (fSlow18 * state["fRec21"]))))) + (fSlow21 * ((fSlow23 * ((((fTemp72 - (state["fVec27"][((state["IOTA0"] - iTemp74) & 4095).astype(jnp.int32)] * fTemp76)) - (fTemp77 * state["fVec27"][((state["IOTA0"] - iTemp78) & 4095).astype(jnp.int32)])) + ((fTemp81 - (state["fVec30"][((state["IOTA0"] - iTemp83) & 4095).astype(jnp.int32)] * fTemp85)) - (fTemp86 * state["fVec30"][((state["IOTA0"] - iTemp87) & 4095).astype(jnp.int32)]))) + ((fTemp90 - (state["fVec33"][((state["IOTA0"] - iTemp92) & 4095).astype(jnp.int32)] * fTemp94)) - (fTemp95 * state["fVec33"][((state["IOTA0"] - iTemp96) & 4095).astype(jnp.int32)])))) + (fSlow25 * ((((fTemp43 - (fTemp76 * state["fVec20"][((state["IOTA0"] - iTemp74) & 4095).astype(jnp.int32)])) - (fTemp77 * state["fVec20"][((state["IOTA0"] - iTemp78) & 4095).astype(jnp.int32)])) + ((fTemp54 - (fTemp85 * state["fVec22"][((state["IOTA0"] - iTemp83) & 4095).astype(jnp.int32)])) - (fTemp86 * state["fVec22"][((state["IOTA0"] - iTemp87) & 4095).astype(jnp.int32)]))) + ((fTemp65 - (fTemp94 * state["fVec24"][((state["IOTA0"] - iTemp92) & 4095).astype(jnp.int32)])) - (fTemp95 * state["fVec24"][((state["IOTA0"] - iTemp96) & 4095).astype(jnp.int32)]))))))) + (fSlow26 * (((jnp.float32(0.049922034) * state["fRec23"][0]) + (jnp.float32(0.0506127) * state["fRec23"][2])) - ((jnp.float32(0.095993534) * state["fRec23"][1]) + (jnp.float32(0.004408786) * state["fRec23"][3]))))))) 
		fTemp98 = jnp.where((iSlow3 != 0), jnp.float32(0.0), fTemp97) 
		state["fRec9"] = (((fTemp1 * fRec9_temp) + fTemp98) - (fSlow28 * fRec5_temp)) 
		state["fRec8"] = (state["fRec9"] + (fTemp1 * fRec8_temp)) 
		state["fRec7"] = (state["fRec8"] + (fTemp1 * fRec7_temp)) 
		state["fRec6"] = (state["fRec7"] + (fRec6_temp * fTemp1)) 
		state["fRec5"] = (state["fRec6"] * jnp.power(fTemp0, jnp.float32(4.0))) 
		fTemp99 = jnp.tan((self._fConst23 * jnp.maximum(jnp.float32(2e+01), jnp.minimum(jnp.float32(1e+04), state["fRec10"])))) 
		fTemp100 = (jnp.float32(1.0) / fTemp99) 
		fTemp101 = (jnp.float32(1.0) - (jnp.float32(1.0) / jnp.power(fTemp99, jnp.float32(2.0)))) 
		fTemp102 = (fSlow33 + (((fSlow34 + fTemp100) / fTemp99) + jnp.float32(1.0))) 
		state["fRec26"] = state["fRec26"].at[0].set((fTemp98 - (((state["fRec26"][2] * (fSlow33 + (((fTemp100 - fSlow34) / fTemp99) + jnp.float32(1.0)))) + (jnp.float32(2.0) * (state["fRec26"][1] * (fSlow33 + fTemp101)))) / fTemp102))) 
		fTemp103 = (fSlow32 + ((((fSlow35 + fTemp100) / fTemp99) + jnp.float32(1.0)) - fSlow31)) 
		state["fRec25"] = state["fRec25"].at[0].set((((state["fRec26"][2] + (state["fRec26"][0] + (jnp.float32(2.0) * state["fRec26"][1]))) / fTemp102) - (((state["fRec25"][2] * (fSlow32 + ((((fTemp100 - fSlow35) / fTemp99) + jnp.float32(1.0)) - fSlow31))) + (jnp.float32(2.0) * (state["fRec25"][1] * (fSlow32 + (fTemp101 - fSlow31))))) / fTemp103))) 
		fTemp104 = jnp.tan((self._fConst23 * jnp.maximum(state["fRec10"], jnp.float32(2e+01)))) 
		fTemp105 = (jnp.float32(1.0) / fTemp104) 
		fTemp106 = (fSlow38 + (((fSlow39 + fTemp105) / fTemp104) + jnp.float32(1.0))) 
		fTemp107 = ((fSlow38 + (jnp.float32(1.0) - ((fSlow39 - fTemp105) / fTemp104))) / fTemp106) 
		fTemp108 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), fTemp107)) 
		fTemp109 = (jnp.float32(1.0) - jnp.power(fTemp108, jnp.float32(2.0))) 
		fTemp110 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp109)) 
		fTemp111 = ((fTemp98 * fTemp110) - (fTemp108 * fRec27_temp)) 
		fTemp112 = (jnp.float32(1.0) - (jnp.float32(1.0) / jnp.power(fTemp104, jnp.float32(2.0)))) 
		fTemp113 = (fSlow38 + fTemp112) 
		fTemp114 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), (jnp.float32(2.0) * (fTemp113 / (fTemp106 * (fTemp107 + jnp.float32(1.0))))))) 
		fTemp115 = (jnp.float32(1.0) - jnp.power(fTemp114, jnp.float32(2.0))) 
		fTemp116 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp115)) 
		state["fRec29"] = ((fTemp111 * fTemp116) - (fTemp114 * fRec29_temp)) 
		state["fRec27"] = ((fTemp111 * fTemp114) + (fRec29_temp * fTemp116)) 
		fRec28 = state["fRec29"] 
		fTemp117 = (jnp.float32(1.0) - (fTemp113 / fTemp106)) 
		fTemp118 = jnp.sqrt(fTemp109) 
		fTemp119 = ((((fTemp98 * fTemp108) + (fRec27_temp * fTemp110)) + (jnp.float32(2.0) * ((state["fRec27"] * fTemp117) / fTemp118))) + ((fRec28 * ((jnp.float32(1.0) - fTemp107) - (jnp.float32(2.0) * (fTemp114 * fTemp117)))) / (fTemp118 * jnp.sqrt(fTemp115)))) 
		fTemp120 = (fSlow36 + ((((fSlow40 + fTemp105) / fTemp104) + jnp.float32(1.0)) - fSlow37)) 
		fTemp121 = ((fSlow36 + ((((fTemp105 - fSlow40) / fTemp104) + jnp.float32(1.0)) - fSlow37)) / fTemp120) 
		fTemp122 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), fTemp121)) 
		fTemp123 = (jnp.float32(1.0) - jnp.power(fTemp122, jnp.float32(2.0))) 
		fTemp124 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp123)) 
		fTemp125 = (((fTemp119 * fTemp124) / fTemp106) - (fTemp122 * fRec30_temp)) 
		fTemp126 = (fSlow36 + (fTemp112 - fSlow37)) 
		fTemp127 = jnp.maximum(jnp.float32(-0.9999), jnp.minimum(jnp.float32(0.9999), (jnp.float32(2.0) * (fTemp126 / (fTemp120 * (fTemp121 + jnp.float32(1.0))))))) 
		fTemp128 = (jnp.float32(1.0) - jnp.power(fTemp127, jnp.float32(2.0))) 
		fTemp129 = jnp.sqrt(jnp.maximum(jnp.float32(0.0), fTemp128)) 
		state["fRec32"] = ((fTemp125 * fTemp129) - (fTemp127 * fRec32_temp)) 
		state["fRec30"] = ((fTemp125 * fTemp127) + (fRec32_temp * fTemp129)) 
		fRec31 = state["fRec32"] 
		fTemp130 = (jnp.float32(1.0) - (fTemp126 / fTemp120)) 
		fTemp131 = jnp.sqrt(fTemp123) 
		fTemp132 = jnp.where((iSlow3 != 0), fTemp97, (state["fRec4"] * jnp.where((iSlow5 != 0), jnp.where((iSlow29 != 0), ((((((fTemp119 * fTemp122) / fTemp106) + (fRec30_temp * fTemp124)) + (jnp.float32(2.0) * ((state["fRec30"] * fTemp130) / fTemp131))) + ((fRec31 * ((jnp.float32(1.0) - fTemp121) - (jnp.float32(2.0) * (fTemp127 * fTemp130)))) / (fTemp131 * jnp.sqrt(fTemp128)))) / fTemp120), ((state["fRec25"][2] + (state["fRec25"][0] + (jnp.float32(2.0) * state["fRec25"][1]))) / fTemp103)), state["fRec5"]))) 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp132 - (self._fConst11 * ((self._fConst24 * state["fRec3"][2]) + (self._fConst26 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst11 * (((self._fConst13 * state["fRec3"][0]) + (self._fConst27 * state["fRec3"][1])) + (self._fConst13 * state["fRec3"][2]))) - (self._fConst8 * ((self._fConst28 * state["fRec2"][2]) + (self._fConst29 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst8 * (((self._fConst10 * state["fRec2"][0]) + (self._fConst30 * state["fRec2"][1])) + (self._fConst10 * state["fRec2"][2]))) - (self._fConst4 * ((self._fConst31 * state["fRec1"][2]) + (self._fConst32 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fRec0_temp * fSlow2) + (jnp.abs((self._fConst4 * (((self._fConst7 * state["fRec1"][0]) + (self._fConst33 * state["fRec1"][1])) + (self._fConst7 * state["fRec1"][2])))) * fSlow41)) 
		fVbargraph0 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		# self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec39"] = state["fRec39"].at[0].set((fTemp132 - (self._fConst50 * ((self._fConst53 * state["fRec39"][2]) + (self._fConst54 * state["fRec39"][1]))))) 
		state["fRec38"] = state["fRec38"].at[0].set(((self._fConst50 * (((self._fConst52 * state["fRec39"][0]) + (self._fConst55 * state["fRec39"][1])) + (self._fConst52 * state["fRec39"][2]))) - (self._fConst48 * ((self._fConst56 * state["fRec38"][2]) + (self._fConst57 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst48 * (((self._fConst49 * state["fRec38"][0]) + (self._fConst58 * state["fRec38"][1])) + (self._fConst49 * state["fRec38"][2]))) - (self._fConst46 * ((self._fConst59 * state["fRec37"][2]) + (self._fConst60 * state["fRec37"][1]))))) 
		fTemp133 = (self._fConst46 * (((self._fConst47 * state["fRec37"][0]) + (self._fConst61 * state["fRec37"][1])) + (self._fConst47 * state["fRec37"][2]))) 
		state["fRec36"] = state["fRec36"].at[0].set((fTemp133 - (self._fConst43 * ((self._fConst62 * state["fRec36"][2]) + (self._fConst64 * state["fRec36"][1]))))) 
		state["fRec35"] = state["fRec35"].at[0].set(((self._fConst43 * (((self._fConst45 * state["fRec36"][0]) + (self._fConst65 * state["fRec36"][1])) + (self._fConst45 * state["fRec36"][2]))) - (self._fConst40 * ((self._fConst66 * state["fRec35"][2]) + (self._fConst67 * state["fRec35"][1]))))) 
		state["fRec34"] = state["fRec34"].at[0].set(((self._fConst40 * (((self._fConst42 * state["fRec35"][0]) + (self._fConst68 * state["fRec35"][1])) + (self._fConst42 * state["fRec35"][2]))) - (self._fConst36 * ((self._fConst69 * state["fRec34"][2]) + (self._fConst70 * state["fRec34"][1]))))) 
		state["fRec33"] = ((fSlow2 * fRec33_temp) + (fSlow41 * jnp.abs((self._fConst36 * (((self._fConst39 * state["fRec34"][0]) + (self._fConst71 * state["fRec34"][1])) + (self._fConst39 * state["fRec34"][2])))))) 
		fVbargraph1 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec33"])))
		# self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec46"] = state["fRec46"].at[0].set((fTemp133 - (self._fConst88 * ((self._fConst91 * state["fRec46"][2]) + (self._fConst92 * state["fRec46"][1]))))) 
		state["fRec45"] = state["fRec45"].at[0].set(((self._fConst88 * (((self._fConst90 * state["fRec46"][0]) + (self._fConst93 * state["fRec46"][1])) + (self._fConst90 * state["fRec46"][2]))) - (self._fConst86 * ((self._fConst94 * state["fRec45"][2]) + (self._fConst95 * state["fRec45"][1]))))) 
		state["fRec44"] = state["fRec44"].at[0].set(((self._fConst86 * (((self._fConst87 * state["fRec45"][0]) + (self._fConst96 * state["fRec45"][1])) + (self._fConst87 * state["fRec45"][2]))) - (self._fConst84 * ((self._fConst97 * state["fRec44"][2]) + (self._fConst98 * state["fRec44"][1]))))) 
		fTemp134 = (self._fConst84 * (((self._fConst85 * state["fRec44"][0]) + (self._fConst99 * state["fRec44"][1])) + (self._fConst85 * state["fRec44"][2]))) 
		state["fRec43"] = state["fRec43"].at[0].set((fTemp134 - (self._fConst81 * ((self._fConst100 * state["fRec43"][2]) + (self._fConst102 * state["fRec43"][1]))))) 
		state["fRec42"] = state["fRec42"].at[0].set(((self._fConst81 * (((self._fConst83 * state["fRec43"][0]) + (self._fConst103 * state["fRec43"][1])) + (self._fConst83 * state["fRec43"][2]))) - (self._fConst78 * ((self._fConst104 * state["fRec42"][2]) + (self._fConst105 * state["fRec42"][1]))))) 
		state["fRec41"] = state["fRec41"].at[0].set(((self._fConst78 * (((self._fConst80 * state["fRec42"][0]) + (self._fConst106 * state["fRec42"][1])) + (self._fConst80 * state["fRec42"][2]))) - (self._fConst74 * ((self._fConst107 * state["fRec41"][2]) + (self._fConst108 * state["fRec41"][1]))))) 
		state["fRec40"] = ((fSlow2 * fRec40_temp) + (fSlow41 * jnp.abs((self._fConst74 * (((self._fConst77 * state["fRec41"][0]) + (self._fConst109 * state["fRec41"][1])) + (self._fConst77 * state["fRec41"][2])))))) 
		fVbargraph2 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec40"])))
		# self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec53"] = state["fRec53"].at[0].set((fTemp134 - (self._fConst126 * ((self._fConst129 * state["fRec53"][2]) + (self._fConst130 * state["fRec53"][1]))))) 
		state["fRec52"] = state["fRec52"].at[0].set(((self._fConst126 * (((self._fConst128 * state["fRec53"][0]) + (self._fConst131 * state["fRec53"][1])) + (self._fConst128 * state["fRec53"][2]))) - (self._fConst124 * ((self._fConst132 * state["fRec52"][2]) + (self._fConst133 * state["fRec52"][1]))))) 
		state["fRec51"] = state["fRec51"].at[0].set(((self._fConst124 * (((self._fConst125 * state["fRec52"][0]) + (self._fConst134 * state["fRec52"][1])) + (self._fConst125 * state["fRec52"][2]))) - (self._fConst122 * ((self._fConst135 * state["fRec51"][2]) + (self._fConst136 * state["fRec51"][1]))))) 
		fTemp135 = (self._fConst122 * (((self._fConst123 * state["fRec51"][0]) + (self._fConst137 * state["fRec51"][1])) + (self._fConst123 * state["fRec51"][2]))) 
		state["fRec50"] = state["fRec50"].at[0].set((fTemp135 - (self._fConst119 * ((self._fConst138 * state["fRec50"][2]) + (self._fConst140 * state["fRec50"][1]))))) 
		state["fRec49"] = state["fRec49"].at[0].set(((self._fConst119 * (((self._fConst121 * state["fRec50"][0]) + (self._fConst141 * state["fRec50"][1])) + (self._fConst121 * state["fRec50"][2]))) - (self._fConst116 * ((self._fConst142 * state["fRec49"][2]) + (self._fConst143 * state["fRec49"][1]))))) 
		state["fRec48"] = state["fRec48"].at[0].set(((self._fConst116 * (((self._fConst118 * state["fRec49"][0]) + (self._fConst144 * state["fRec49"][1])) + (self._fConst118 * state["fRec49"][2]))) - (self._fConst112 * ((self._fConst145 * state["fRec48"][2]) + (self._fConst146 * state["fRec48"][1]))))) 
		state["fRec47"] = ((fSlow2 * fRec47_temp) + (fSlow41 * jnp.abs((self._fConst112 * (((self._fConst115 * state["fRec48"][0]) + (self._fConst147 * state["fRec48"][1])) + (self._fConst115 * state["fRec48"][2])))))) 
		fVbargraph3 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec47"])))
		# self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec60"] = state["fRec60"].at[0].set((fTemp135 - (self._fConst164 * ((self._fConst167 * state["fRec60"][2]) + (self._fConst168 * state["fRec60"][1]))))) 
		state["fRec59"] = state["fRec59"].at[0].set(((self._fConst164 * (((self._fConst166 * state["fRec60"][0]) + (self._fConst169 * state["fRec60"][1])) + (self._fConst166 * state["fRec60"][2]))) - (self._fConst162 * ((self._fConst170 * state["fRec59"][2]) + (self._fConst171 * state["fRec59"][1]))))) 
		state["fRec58"] = state["fRec58"].at[0].set(((self._fConst162 * (((self._fConst163 * state["fRec59"][0]) + (self._fConst172 * state["fRec59"][1])) + (self._fConst163 * state["fRec59"][2]))) - (self._fConst160 * ((self._fConst173 * state["fRec58"][2]) + (self._fConst174 * state["fRec58"][1]))))) 
		fTemp136 = (self._fConst160 * (((self._fConst161 * state["fRec58"][0]) + (self._fConst175 * state["fRec58"][1])) + (self._fConst161 * state["fRec58"][2]))) 
		state["fRec57"] = state["fRec57"].at[0].set((fTemp136 - (self._fConst157 * ((self._fConst176 * state["fRec57"][2]) + (self._fConst178 * state["fRec57"][1]))))) 
		state["fRec56"] = state["fRec56"].at[0].set(((self._fConst157 * (((self._fConst159 * state["fRec57"][0]) + (self._fConst179 * state["fRec57"][1])) + (self._fConst159 * state["fRec57"][2]))) - (self._fConst154 * ((self._fConst180 * state["fRec56"][2]) + (self._fConst181 * state["fRec56"][1]))))) 
		state["fRec55"] = state["fRec55"].at[0].set(((self._fConst154 * (((self._fConst156 * state["fRec56"][0]) + (self._fConst182 * state["fRec56"][1])) + (self._fConst156 * state["fRec56"][2]))) - (self._fConst150 * ((self._fConst183 * state["fRec55"][2]) + (self._fConst184 * state["fRec55"][1]))))) 
		state["fRec54"] = ((fSlow2 * fRec54_temp) + (fSlow41 * jnp.abs((self._fConst150 * (((self._fConst153 * state["fRec55"][0]) + (self._fConst185 * state["fRec55"][1])) + (self._fConst153 * state["fRec55"][2])))))) 
		fVbargraph4 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec54"])))
		# self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec67"] = state["fRec67"].at[0].set((fTemp136 - (self._fConst202 * ((self._fConst205 * state["fRec67"][2]) + (self._fConst206 * state["fRec67"][1]))))) 
		state["fRec66"] = state["fRec66"].at[0].set(((self._fConst202 * (((self._fConst204 * state["fRec67"][0]) + (self._fConst207 * state["fRec67"][1])) + (self._fConst204 * state["fRec67"][2]))) - (self._fConst200 * ((self._fConst208 * state["fRec66"][2]) + (self._fConst209 * state["fRec66"][1]))))) 
		state["fRec65"] = state["fRec65"].at[0].set(((self._fConst200 * (((self._fConst201 * state["fRec66"][0]) + (self._fConst210 * state["fRec66"][1])) + (self._fConst201 * state["fRec66"][2]))) - (self._fConst198 * ((self._fConst211 * state["fRec65"][2]) + (self._fConst212 * state["fRec65"][1]))))) 
		fTemp137 = (self._fConst198 * (((self._fConst199 * state["fRec65"][0]) + (self._fConst213 * state["fRec65"][1])) + (self._fConst199 * state["fRec65"][2]))) 
		state["fRec64"] = state["fRec64"].at[0].set((fTemp137 - (self._fConst195 * ((self._fConst214 * state["fRec64"][2]) + (self._fConst216 * state["fRec64"][1]))))) 
		state["fRec63"] = state["fRec63"].at[0].set(((self._fConst195 * (((self._fConst197 * state["fRec64"][0]) + (self._fConst217 * state["fRec64"][1])) + (self._fConst197 * state["fRec64"][2]))) - (self._fConst192 * ((self._fConst218 * state["fRec63"][2]) + (self._fConst219 * state["fRec63"][1]))))) 
		state["fRec62"] = state["fRec62"].at[0].set(((self._fConst192 * (((self._fConst194 * state["fRec63"][0]) + (self._fConst220 * state["fRec63"][1])) + (self._fConst194 * state["fRec63"][2]))) - (self._fConst188 * ((self._fConst221 * state["fRec62"][2]) + (self._fConst222 * state["fRec62"][1]))))) 
		state["fRec61"] = ((fSlow2 * fRec61_temp) + (fSlow41 * jnp.abs((self._fConst188 * (((self._fConst191 * state["fRec62"][0]) + (self._fConst223 * state["fRec62"][1])) + (self._fConst191 * state["fRec62"][2])))))) 
		fVbargraph5 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec61"])))
		# self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec74"] = state["fRec74"].at[0].set((fTemp137 - (self._fConst240 * ((self._fConst243 * state["fRec74"][2]) + (self._fConst244 * state["fRec74"][1]))))) 
		state["fRec73"] = state["fRec73"].at[0].set(((self._fConst240 * (((self._fConst242 * state["fRec74"][0]) + (self._fConst245 * state["fRec74"][1])) + (self._fConst242 * state["fRec74"][2]))) - (self._fConst238 * ((self._fConst246 * state["fRec73"][2]) + (self._fConst247 * state["fRec73"][1]))))) 
		state["fRec72"] = state["fRec72"].at[0].set(((self._fConst238 * (((self._fConst239 * state["fRec73"][0]) + (self._fConst248 * state["fRec73"][1])) + (self._fConst239 * state["fRec73"][2]))) - (self._fConst236 * ((self._fConst249 * state["fRec72"][2]) + (self._fConst250 * state["fRec72"][1]))))) 
		fTemp138 = (self._fConst236 * (((self._fConst237 * state["fRec72"][0]) + (self._fConst251 * state["fRec72"][1])) + (self._fConst237 * state["fRec72"][2]))) 
		state["fRec71"] = state["fRec71"].at[0].set((fTemp138 - (self._fConst233 * ((self._fConst252 * state["fRec71"][2]) + (self._fConst254 * state["fRec71"][1]))))) 
		state["fRec70"] = state["fRec70"].at[0].set(((self._fConst233 * (((self._fConst235 * state["fRec71"][0]) + (self._fConst255 * state["fRec71"][1])) + (self._fConst235 * state["fRec71"][2]))) - (self._fConst230 * ((self._fConst256 * state["fRec70"][2]) + (self._fConst257 * state["fRec70"][1]))))) 
		state["fRec69"] = state["fRec69"].at[0].set(((self._fConst230 * (((self._fConst232 * state["fRec70"][0]) + (self._fConst258 * state["fRec70"][1])) + (self._fConst232 * state["fRec70"][2]))) - (self._fConst226 * ((self._fConst259 * state["fRec69"][2]) + (self._fConst260 * state["fRec69"][1]))))) 
		state["fRec68"] = ((fSlow2 * fRec68_temp) + (fSlow41 * jnp.abs((self._fConst226 * (((self._fConst229 * state["fRec69"][0]) + (self._fConst261 * state["fRec69"][1])) + (self._fConst229 * state["fRec69"][2])))))) 
		fVbargraph6 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec68"])))
		# self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec81"] = state["fRec81"].at[0].set((fTemp138 - (self._fConst278 * ((self._fConst281 * state["fRec81"][2]) + (self._fConst282 * state["fRec81"][1]))))) 
		state["fRec80"] = state["fRec80"].at[0].set(((self._fConst278 * (((self._fConst280 * state["fRec81"][0]) + (self._fConst283 * state["fRec81"][1])) + (self._fConst280 * state["fRec81"][2]))) - (self._fConst276 * ((self._fConst284 * state["fRec80"][2]) + (self._fConst285 * state["fRec80"][1]))))) 
		state["fRec79"] = state["fRec79"].at[0].set(((self._fConst276 * (((self._fConst277 * state["fRec80"][0]) + (self._fConst286 * state["fRec80"][1])) + (self._fConst277 * state["fRec80"][2]))) - (self._fConst274 * ((self._fConst287 * state["fRec79"][2]) + (self._fConst288 * state["fRec79"][1]))))) 
		fTemp139 = (self._fConst274 * (((self._fConst275 * state["fRec79"][0]) + (self._fConst289 * state["fRec79"][1])) + (self._fConst275 * state["fRec79"][2]))) 
		state["fRec78"] = state["fRec78"].at[0].set((fTemp139 - (self._fConst271 * ((self._fConst290 * state["fRec78"][2]) + (self._fConst292 * state["fRec78"][1]))))) 
		state["fRec77"] = state["fRec77"].at[0].set(((self._fConst271 * (((self._fConst273 * state["fRec78"][0]) + (self._fConst293 * state["fRec78"][1])) + (self._fConst273 * state["fRec78"][2]))) - (self._fConst268 * ((self._fConst294 * state["fRec77"][2]) + (self._fConst295 * state["fRec77"][1]))))) 
		state["fRec76"] = state["fRec76"].at[0].set(((self._fConst268 * (((self._fConst270 * state["fRec77"][0]) + (self._fConst296 * state["fRec77"][1])) + (self._fConst270 * state["fRec77"][2]))) - (self._fConst264 * ((self._fConst297 * state["fRec76"][2]) + (self._fConst298 * state["fRec76"][1]))))) 
		state["fRec75"] = ((fSlow2 * fRec75_temp) + (fSlow41 * jnp.abs((self._fConst264 * (((self._fConst267 * state["fRec76"][0]) + (self._fConst299 * state["fRec76"][1])) + (self._fConst267 * state["fRec76"][2])))))) 
		fVbargraph7 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec75"])))
		# self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec88"] = state["fRec88"].at[0].set((fTemp139 - (self._fConst316 * ((self._fConst319 * state["fRec88"][2]) + (self._fConst320 * state["fRec88"][1]))))) 
		state["fRec87"] = state["fRec87"].at[0].set(((self._fConst316 * (((self._fConst318 * state["fRec88"][0]) + (self._fConst321 * state["fRec88"][1])) + (self._fConst318 * state["fRec88"][2]))) - (self._fConst314 * ((self._fConst322 * state["fRec87"][2]) + (self._fConst323 * state["fRec87"][1]))))) 
		state["fRec86"] = state["fRec86"].at[0].set(((self._fConst314 * (((self._fConst315 * state["fRec87"][0]) + (self._fConst324 * state["fRec87"][1])) + (self._fConst315 * state["fRec87"][2]))) - (self._fConst312 * ((self._fConst325 * state["fRec86"][2]) + (self._fConst326 * state["fRec86"][1]))))) 
		fTemp140 = (self._fConst312 * (((self._fConst313 * state["fRec86"][0]) + (self._fConst327 * state["fRec86"][1])) + (self._fConst313 * state["fRec86"][2]))) 
		state["fRec85"] = state["fRec85"].at[0].set((fTemp140 - (self._fConst309 * ((self._fConst328 * state["fRec85"][2]) + (self._fConst330 * state["fRec85"][1]))))) 
		state["fRec84"] = state["fRec84"].at[0].set(((self._fConst309 * (((self._fConst311 * state["fRec85"][0]) + (self._fConst331 * state["fRec85"][1])) + (self._fConst311 * state["fRec85"][2]))) - (self._fConst306 * ((self._fConst332 * state["fRec84"][2]) + (self._fConst333 * state["fRec84"][1]))))) 
		state["fRec83"] = state["fRec83"].at[0].set(((self._fConst306 * (((self._fConst308 * state["fRec84"][0]) + (self._fConst334 * state["fRec84"][1])) + (self._fConst308 * state["fRec84"][2]))) - (self._fConst302 * ((self._fConst335 * state["fRec83"][2]) + (self._fConst336 * state["fRec83"][1]))))) 
		state["fRec82"] = ((fSlow2 * fRec82_temp) + (fSlow41 * jnp.abs((self._fConst302 * (((self._fConst305 * state["fRec83"][0]) + (self._fConst337 * state["fRec83"][1])) + (self._fConst305 * state["fRec83"][2])))))) 
		fVbargraph8 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec82"])))
		# self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec95"] = state["fRec95"].at[0].set((fTemp140 - (self._fConst354 * ((self._fConst357 * state["fRec95"][2]) + (self._fConst358 * state["fRec95"][1]))))) 
		state["fRec94"] = state["fRec94"].at[0].set(((self._fConst354 * (((self._fConst356 * state["fRec95"][0]) + (self._fConst359 * state["fRec95"][1])) + (self._fConst356 * state["fRec95"][2]))) - (self._fConst352 * ((self._fConst360 * state["fRec94"][2]) + (self._fConst361 * state["fRec94"][1]))))) 
		state["fRec93"] = state["fRec93"].at[0].set(((self._fConst352 * (((self._fConst353 * state["fRec94"][0]) + (self._fConst362 * state["fRec94"][1])) + (self._fConst353 * state["fRec94"][2]))) - (self._fConst350 * ((self._fConst363 * state["fRec93"][2]) + (self._fConst364 * state["fRec93"][1]))))) 
		fTemp141 = (self._fConst350 * (((self._fConst351 * state["fRec93"][0]) + (self._fConst365 * state["fRec93"][1])) + (self._fConst351 * state["fRec93"][2]))) 
		state["fRec92"] = state["fRec92"].at[0].set((fTemp141 - (self._fConst347 * ((self._fConst366 * state["fRec92"][2]) + (self._fConst368 * state["fRec92"][1]))))) 
		state["fRec91"] = state["fRec91"].at[0].set(((self._fConst347 * (((self._fConst349 * state["fRec92"][0]) + (self._fConst369 * state["fRec92"][1])) + (self._fConst349 * state["fRec92"][2]))) - (self._fConst344 * ((self._fConst370 * state["fRec91"][2]) + (self._fConst371 * state["fRec91"][1]))))) 
		state["fRec90"] = state["fRec90"].at[0].set(((self._fConst344 * (((self._fConst346 * state["fRec91"][0]) + (self._fConst372 * state["fRec91"][1])) + (self._fConst346 * state["fRec91"][2]))) - (self._fConst340 * ((self._fConst373 * state["fRec90"][2]) + (self._fConst374 * state["fRec90"][1]))))) 
		state["fRec89"] = ((fSlow2 * fRec89_temp) + (fSlow41 * jnp.abs((self._fConst340 * (((self._fConst343 * state["fRec90"][0]) + (self._fConst375 * state["fRec90"][1])) + (self._fConst343 * state["fRec90"][2])))))) 
		fVbargraph9 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec89"])))
		# self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec102"] = state["fRec102"].at[0].set((fTemp141 - (self._fConst392 * ((self._fConst395 * state["fRec102"][2]) + (self._fConst396 * state["fRec102"][1]))))) 
		state["fRec101"] = state["fRec101"].at[0].set(((self._fConst392 * (((self._fConst394 * state["fRec102"][0]) + (self._fConst397 * state["fRec102"][1])) + (self._fConst394 * state["fRec102"][2]))) - (self._fConst390 * ((self._fConst398 * state["fRec101"][2]) + (self._fConst399 * state["fRec101"][1]))))) 
		state["fRec100"] = state["fRec100"].at[0].set(((self._fConst390 * (((self._fConst391 * state["fRec101"][0]) + (self._fConst400 * state["fRec101"][1])) + (self._fConst391 * state["fRec101"][2]))) - (self._fConst388 * ((self._fConst401 * state["fRec100"][2]) + (self._fConst402 * state["fRec100"][1]))))) 
		fTemp142 = (self._fConst388 * (((self._fConst389 * state["fRec100"][0]) + (self._fConst403 * state["fRec100"][1])) + (self._fConst389 * state["fRec100"][2]))) 
		state["fRec99"] = state["fRec99"].at[0].set((fTemp142 - (self._fConst385 * ((self._fConst404 * state["fRec99"][2]) + (self._fConst406 * state["fRec99"][1]))))) 
		state["fRec98"] = state["fRec98"].at[0].set(((self._fConst385 * (((self._fConst387 * state["fRec99"][0]) + (self._fConst407 * state["fRec99"][1])) + (self._fConst387 * state["fRec99"][2]))) - (self._fConst382 * ((self._fConst408 * state["fRec98"][2]) + (self._fConst409 * state["fRec98"][1]))))) 
		state["fRec97"] = state["fRec97"].at[0].set(((self._fConst382 * (((self._fConst384 * state["fRec98"][0]) + (self._fConst410 * state["fRec98"][1])) + (self._fConst384 * state["fRec98"][2]))) - (self._fConst378 * ((self._fConst411 * state["fRec97"][2]) + (self._fConst412 * state["fRec97"][1]))))) 
		state["fRec96"] = ((fSlow2 * fRec96_temp) + (fSlow41 * jnp.abs((self._fConst378 * (((self._fConst381 * state["fRec97"][0]) + (self._fConst413 * state["fRec97"][1])) + (self._fConst381 * state["fRec97"][2])))))) 
		fVbargraph10 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec96"])))
		# self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec109"] = state["fRec109"].at[0].set((fTemp142 - (self._fConst430 * ((self._fConst433 * state["fRec109"][2]) + (self._fConst434 * state["fRec109"][1]))))) 
		state["fRec108"] = state["fRec108"].at[0].set(((self._fConst430 * (((self._fConst432 * state["fRec109"][0]) + (self._fConst435 * state["fRec109"][1])) + (self._fConst432 * state["fRec109"][2]))) - (self._fConst428 * ((self._fConst436 * state["fRec108"][2]) + (self._fConst437 * state["fRec108"][1]))))) 
		state["fRec107"] = state["fRec107"].at[0].set(((self._fConst428 * (((self._fConst429 * state["fRec108"][0]) + (self._fConst438 * state["fRec108"][1])) + (self._fConst429 * state["fRec108"][2]))) - (self._fConst426 * ((self._fConst439 * state["fRec107"][2]) + (self._fConst440 * state["fRec107"][1]))))) 
		fTemp143 = (self._fConst426 * (((self._fConst427 * state["fRec107"][0]) + (self._fConst441 * state["fRec107"][1])) + (self._fConst427 * state["fRec107"][2]))) 
		state["fRec106"] = state["fRec106"].at[0].set((fTemp143 - (self._fConst423 * ((self._fConst442 * state["fRec106"][2]) + (self._fConst444 * state["fRec106"][1]))))) 
		state["fRec105"] = state["fRec105"].at[0].set(((self._fConst423 * (((self._fConst425 * state["fRec106"][0]) + (self._fConst445 * state["fRec106"][1])) + (self._fConst425 * state["fRec106"][2]))) - (self._fConst420 * ((self._fConst446 * state["fRec105"][2]) + (self._fConst447 * state["fRec105"][1]))))) 
		state["fRec104"] = state["fRec104"].at[0].set(((self._fConst420 * (((self._fConst422 * state["fRec105"][0]) + (self._fConst448 * state["fRec105"][1])) + (self._fConst422 * state["fRec105"][2]))) - (self._fConst416 * ((self._fConst449 * state["fRec104"][2]) + (self._fConst450 * state["fRec104"][1]))))) 
		state["fRec103"] = ((fSlow2 * fRec103_temp) + (fSlow41 * jnp.abs((self._fConst416 * (((self._fConst419 * state["fRec104"][0]) + (self._fConst451 * state["fRec104"][1])) + (self._fConst419 * state["fRec104"][2])))))) 
		fVbargraph11 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec103"])))
		# self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec116"] = state["fRec116"].at[0].set((fTemp143 - (self._fConst468 * ((self._fConst471 * state["fRec116"][2]) + (self._fConst472 * state["fRec116"][1]))))) 
		state["fRec115"] = state["fRec115"].at[0].set(((self._fConst468 * (((self._fConst470 * state["fRec116"][0]) + (self._fConst473 * state["fRec116"][1])) + (self._fConst470 * state["fRec116"][2]))) - (self._fConst466 * ((self._fConst474 * state["fRec115"][2]) + (self._fConst475 * state["fRec115"][1]))))) 
		state["fRec114"] = state["fRec114"].at[0].set(((self._fConst466 * (((self._fConst467 * state["fRec115"][0]) + (self._fConst476 * state["fRec115"][1])) + (self._fConst467 * state["fRec115"][2]))) - (self._fConst464 * ((self._fConst477 * state["fRec114"][2]) + (self._fConst478 * state["fRec114"][1]))))) 
		fTemp144 = (self._fConst464 * (((self._fConst465 * state["fRec114"][0]) + (self._fConst479 * state["fRec114"][1])) + (self._fConst465 * state["fRec114"][2]))) 
		state["fRec113"] = state["fRec113"].at[0].set((fTemp144 - (self._fConst461 * ((self._fConst480 * state["fRec113"][2]) + (self._fConst482 * state["fRec113"][1]))))) 
		state["fRec112"] = state["fRec112"].at[0].set(((self._fConst461 * (((self._fConst463 * state["fRec113"][0]) + (self._fConst483 * state["fRec113"][1])) + (self._fConst463 * state["fRec113"][2]))) - (self._fConst458 * ((self._fConst484 * state["fRec112"][2]) + (self._fConst485 * state["fRec112"][1]))))) 
		state["fRec111"] = state["fRec111"].at[0].set(((self._fConst458 * (((self._fConst460 * state["fRec112"][0]) + (self._fConst486 * state["fRec112"][1])) + (self._fConst460 * state["fRec112"][2]))) - (self._fConst454 * ((self._fConst487 * state["fRec111"][2]) + (self._fConst488 * state["fRec111"][1]))))) 
		state["fRec110"] = ((fSlow2 * fRec110_temp) + (fSlow41 * jnp.abs((self._fConst454 * (((self._fConst457 * state["fRec111"][0]) + (self._fConst489 * state["fRec111"][1])) + (self._fConst457 * state["fRec111"][2])))))) 
		fVbargraph12 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec110"])))
		# self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec123"] = state["fRec123"].at[0].set((fTemp144 - (self._fConst506 * ((self._fConst509 * state["fRec123"][2]) + (self._fConst510 * state["fRec123"][1]))))) 
		state["fRec122"] = state["fRec122"].at[0].set(((self._fConst506 * (((self._fConst508 * state["fRec123"][0]) + (self._fConst511 * state["fRec123"][1])) + (self._fConst508 * state["fRec123"][2]))) - (self._fConst504 * ((self._fConst512 * state["fRec122"][2]) + (self._fConst513 * state["fRec122"][1]))))) 
		state["fRec121"] = state["fRec121"].at[0].set(((self._fConst504 * (((self._fConst505 * state["fRec122"][0]) + (self._fConst514 * state["fRec122"][1])) + (self._fConst505 * state["fRec122"][2]))) - (self._fConst502 * ((self._fConst515 * state["fRec121"][2]) + (self._fConst516 * state["fRec121"][1]))))) 
		fTemp145 = (self._fConst502 * (((self._fConst503 * state["fRec121"][0]) + (self._fConst517 * state["fRec121"][1])) + (self._fConst503 * state["fRec121"][2]))) 
		state["fRec120"] = state["fRec120"].at[0].set((fTemp145 - (self._fConst499 * ((self._fConst518 * state["fRec120"][2]) + (self._fConst520 * state["fRec120"][1]))))) 
		state["fRec119"] = state["fRec119"].at[0].set(((self._fConst499 * (((self._fConst501 * state["fRec120"][0]) + (self._fConst521 * state["fRec120"][1])) + (self._fConst501 * state["fRec120"][2]))) - (self._fConst496 * ((self._fConst522 * state["fRec119"][2]) + (self._fConst523 * state["fRec119"][1]))))) 
		state["fRec118"] = state["fRec118"].at[0].set(((self._fConst496 * (((self._fConst498 * state["fRec119"][0]) + (self._fConst524 * state["fRec119"][1])) + (self._fConst498 * state["fRec119"][2]))) - (self._fConst492 * ((self._fConst525 * state["fRec118"][2]) + (self._fConst526 * state["fRec118"][1]))))) 
		state["fRec117"] = ((fSlow2 * fRec117_temp) + (fSlow41 * jnp.abs((self._fConst492 * (((self._fConst495 * state["fRec118"][0]) + (self._fConst527 * state["fRec118"][1])) + (self._fConst495 * state["fRec118"][2])))))) 
		fVbargraph13 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec117"])))
		# self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec127"] = state["fRec127"].at[0].set((fTemp145 - (self._fConst532 * ((self._fConst535 * state["fRec127"][2]) + (self._fConst536 * state["fRec127"][1]))))) 
		state["fRec126"] = state["fRec126"].at[0].set(((self._fConst532 * (((self._fConst534 * state["fRec127"][0]) + (self._fConst537 * state["fRec127"][1])) + (self._fConst534 * state["fRec127"][2]))) - (self._fConst530 * ((self._fConst538 * state["fRec126"][2]) + (self._fConst539 * state["fRec126"][1]))))) 
		state["fRec125"] = state["fRec125"].at[0].set(((self._fConst530 * (((self._fConst531 * state["fRec126"][0]) + (self._fConst540 * state["fRec126"][1])) + (self._fConst531 * state["fRec126"][2]))) - (self._fConst528 * ((self._fConst541 * state["fRec125"][2]) + (self._fConst542 * state["fRec125"][1]))))) 
		state["fRec124"] = ((fSlow2 * fRec124_temp) + (fSlow41 * jnp.abs((self._fConst528 * (((self._fConst529 * state["fRec125"][0]) + (self._fConst543 * state["fRec125"][1])) + (self._fConst529 * state["fRec125"][2])))))) 
		fVbargraph14 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec124"])))
		# self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		fTemp146 = fTemp132 
		_result0 = fTemp146 
		_result1 = fTemp146 
		state["iVec0"] = jnp.roll(state["iVec0"], 1) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec23"] = jnp.roll(state["fRec23"], 1) 
		state["fRec26"] = jnp.roll(state["fRec26"], 1) 
		state["fRec25"] = jnp.roll(state["fRec25"], 1) 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec39"] = jnp.roll(state["fRec39"], 1) 
		state["fRec38"] = jnp.roll(state["fRec38"], 1) 
		state["fRec37"] = jnp.roll(state["fRec37"], 1) 
		state["fRec36"] = jnp.roll(state["fRec36"], 1) 
		state["fRec35"] = jnp.roll(state["fRec35"], 1) 
		state["fRec34"] = jnp.roll(state["fRec34"], 1) 
		state["fRec46"] = jnp.roll(state["fRec46"], 1) 
		state["fRec45"] = jnp.roll(state["fRec45"], 1) 
		state["fRec44"] = jnp.roll(state["fRec44"], 1) 
		state["fRec43"] = jnp.roll(state["fRec43"], 1) 
		state["fRec42"] = jnp.roll(state["fRec42"], 1) 
		state["fRec41"] = jnp.roll(state["fRec41"], 1) 
		state["fRec53"] = jnp.roll(state["fRec53"], 1) 
		state["fRec52"] = jnp.roll(state["fRec52"], 1) 
		state["fRec51"] = jnp.roll(state["fRec51"], 1) 
		state["fRec50"] = jnp.roll(state["fRec50"], 1) 
		state["fRec49"] = jnp.roll(state["fRec49"], 1) 
		state["fRec48"] = jnp.roll(state["fRec48"], 1) 
		state["fRec60"] = jnp.roll(state["fRec60"], 1) 
		state["fRec59"] = jnp.roll(state["fRec59"], 1) 
		state["fRec58"] = jnp.roll(state["fRec58"], 1) 
		state["fRec57"] = jnp.roll(state["fRec57"], 1) 
		state["fRec56"] = jnp.roll(state["fRec56"], 1) 
		state["fRec55"] = jnp.roll(state["fRec55"], 1) 
		state["fRec67"] = jnp.roll(state["fRec67"], 1) 
		state["fRec66"] = jnp.roll(state["fRec66"], 1) 
		state["fRec65"] = jnp.roll(state["fRec65"], 1) 
		state["fRec64"] = jnp.roll(state["fRec64"], 1) 
		state["fRec63"] = jnp.roll(state["fRec63"], 1) 
		state["fRec62"] = jnp.roll(state["fRec62"], 1) 
		state["fRec74"] = jnp.roll(state["fRec74"], 1) 
		state["fRec73"] = jnp.roll(state["fRec73"], 1) 
		state["fRec72"] = jnp.roll(state["fRec72"], 1) 
		state["fRec71"] = jnp.roll(state["fRec71"], 1) 
		state["fRec70"] = jnp.roll(state["fRec70"], 1) 
		state["fRec69"] = jnp.roll(state["fRec69"], 1) 
		state["fRec81"] = jnp.roll(state["fRec81"], 1) 
		state["fRec80"] = jnp.roll(state["fRec80"], 1) 
		state["fRec79"] = jnp.roll(state["fRec79"], 1) 
		state["fRec78"] = jnp.roll(state["fRec78"], 1) 
		state["fRec77"] = jnp.roll(state["fRec77"], 1) 
		state["fRec76"] = jnp.roll(state["fRec76"], 1) 
		state["fRec88"] = jnp.roll(state["fRec88"], 1) 
		state["fRec87"] = jnp.roll(state["fRec87"], 1) 
		state["fRec86"] = jnp.roll(state["fRec86"], 1) 
		state["fRec85"] = jnp.roll(state["fRec85"], 1) 
		state["fRec84"] = jnp.roll(state["fRec84"], 1) 
		state["fRec83"] = jnp.roll(state["fRec83"], 1) 
		state["fRec95"] = jnp.roll(state["fRec95"], 1) 
		state["fRec94"] = jnp.roll(state["fRec94"], 1) 
		state["fRec93"] = jnp.roll(state["fRec93"], 1) 
		state["fRec92"] = jnp.roll(state["fRec92"], 1) 
		state["fRec91"] = jnp.roll(state["fRec91"], 1) 
		state["fRec90"] = jnp.roll(state["fRec90"], 1) 
		state["fRec102"] = jnp.roll(state["fRec102"], 1) 
		state["fRec101"] = jnp.roll(state["fRec101"], 1) 
		state["fRec100"] = jnp.roll(state["fRec100"], 1) 
		state["fRec99"] = jnp.roll(state["fRec99"], 1) 
		state["fRec98"] = jnp.roll(state["fRec98"], 1) 
		state["fRec97"] = jnp.roll(state["fRec97"], 1) 
		state["fRec109"] = jnp.roll(state["fRec109"], 1) 
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec116"] = jnp.roll(state["fRec116"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec114"] = jnp.roll(state["fRec114"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec112"] = jnp.roll(state["fRec112"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec123"] = jnp.roll(state["fRec123"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec120"] = jnp.roll(state["fRec120"], 1) 
		state["fRec119"] = jnp.roll(state["fRec119"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec127"] = jnp.roll(state["fRec127"], 1) 
		state["fRec126"] = jnp.roll(state["fRec126"], 1) 
		state["fRec125"] = jnp.roll(state["fRec125"], 1) 
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
