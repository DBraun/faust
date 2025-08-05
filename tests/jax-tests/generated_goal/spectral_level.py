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
		ui_path.append("spectral_level") 
		ui_path.append("CONSTANT-Q SPECTRUM ANALYZER (6E), 30 bands spanning LP, 9 octaves below 16000 Hz, HP") 
		self.add_vbargraph("fVbargraph29", ui_path, "vbargraph0", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph28", ui_path, "vbargraph1", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph27", ui_path, "vbargraph2", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph26", ui_path, "vbargraph3", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph25", ui_path, "vbargraph4", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph24", ui_path, "vbargraph5", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph23", ui_path, "vbargraph6", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph22", ui_path, "vbargraph7", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph21", ui_path, "vbargraph8", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph20", ui_path, "vbargraph9", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph19", ui_path, "vbargraph10", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph18", ui_path, "vbargraph11", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph17", ui_path, "vbargraph12", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph16", ui_path, "vbargraph13", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph15", ui_path, "vbargraph14", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph14", ui_path, "vbargraph15", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph13", ui_path, "vbargraph16", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph12", ui_path, "vbargraph17", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph11", ui_path, "vbargraph18", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph10", ui_path, "vbargraph19", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph9", ui_path, "vbargraph20", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph8", ui_path, "vbargraph21", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph7", ui_path, "vbargraph22", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph6", ui_path, "vbargraph23", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph5", ui_path, "vbargraph24", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph4", ui_path, "vbargraph25", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph3", ui_path, "vbargraph26", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph2", ui_path, "vbargraph27", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph1", ui_path, "vbargraph28", -5e+01, 1e+01, unnorm_funcs) 
		self.add_vbargraph("fVbargraph0", ui_path, "vbargraph29", -5e+01, 1e+01, unnorm_funcs) 
		ui_path.pop()
		ui_path.append("SPECTRUM ANALYZER CONTROLS") 
		self.add_hslider("fHslider1", ui_path, "Level Averaging Time", 1e+02, 1.0, 1e+04, unnorm_funcs, "log") 
		self.add_hslider("fHslider0", ui_path, "Level dB Offset", 5e+01, 0.0, 1e+02, unnorm_funcs, "linear") 
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
		self._fConst14 = (((self._fConst3 + np.float32(-3.1897273)) / self._fConst2) + np.float32(4.0767817)) 
		self._fConst15 = (np.float32(1.0) / self._fConst5) 
		self._fConst16 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst15)) 
		self._fConst17 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst12)) 
		self._fConst18 = (((self._fConst3 + np.float32(-0.74313045)) / self._fConst2) + np.float32(1.4500711)) 
		self._fConst19 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst15)) 
		self._fConst20 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst9)) 
		self._fConst21 = (((self._fConst3 + np.float32(-0.15748216)) / self._fConst2) + np.float32(0.9351402)) 
		self._fConst22 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst15)) 
		self._fConst23 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst6)) 
		self._fConst24 = np.tan((np.float32(39895.74) / self._fConst0)) 
		self._fConst25 = (np.float32(1.0) / self._fConst24) 
		self._fConst26 = (np.float32(1.0) / (((self._fConst25 + np.float32(0.15748216)) / self._fConst24) + np.float32(0.9351402))) 
		self._fConst27 = np.power(self._fConst24, np.float32(2.0)) 
		self._fConst28 = (np.float32(50.06381) / self._fConst27) 
		self._fConst29 = (self._fConst28 + np.float32(0.9351402)) 
		self._fConst30 = (np.float32(1.0) / (((self._fConst25 + np.float32(0.74313045)) / self._fConst24) + np.float32(1.4500711))) 
		self._fConst31 = (np.float32(11.0520525) / self._fConst27) 
		self._fConst32 = (self._fConst31 + np.float32(1.4500711)) 
		self._fConst33 = (np.float32(1.0) / (((self._fConst25 + np.float32(3.1897273)) / self._fConst24) + np.float32(4.0767817))) 
		self._fConst34 = (np.float32(0.0017661728) / self._fConst27) 
		self._fConst35 = (self._fConst34 + np.float32(0.0004076782)) 
		self._fConst36 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.16840488)) / self._fConst2) + np.float32(1.0693583))) 
		self._fConst37 = (self._fConst15 + np.float32(53.53615)) 
		self._fConst38 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.51247865)) / self._fConst2) + np.float32(0.6896214))) 
		self._fConst39 = (self._fConst15 + np.float32(7.6217313)) 
		self._fConst40 = (np.float32(1.0) / (((self._fConst3 + np.float32(0.78241307)) / self._fConst2) + np.float32(0.2452915))) 
		self._fConst41 = (np.float32(0.0001) / self._fConst5) 
		self._fConst42 = (self._fConst41 + np.float32(0.0004332272)) 
		self._fConst43 = (((self._fConst3 + np.float32(-0.78241307)) / self._fConst2) + np.float32(0.2452915)) 
		self._fConst44 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst15)) 
		self._fConst45 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst41)) 
		self._fConst46 = (((self._fConst3 + np.float32(-0.51247865)) / self._fConst2) + np.float32(0.6896214)) 
		self._fConst47 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst15)) 
		self._fConst48 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst15)) 
		self._fConst49 = (((self._fConst3 + np.float32(-0.16840488)) / self._fConst2) + np.float32(1.0693583)) 
		self._fConst50 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst15)) 
		self._fConst51 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst15)) 
		self._fConst52 = (((self._fConst25 + np.float32(-3.1897273)) / self._fConst24) + np.float32(4.0767817)) 
		self._fConst53 = (np.float32(1.0) / self._fConst27) 
		self._fConst54 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst53)) 
		self._fConst55 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst34)) 
		self._fConst56 = (((self._fConst25 + np.float32(-0.74313045)) / self._fConst24) + np.float32(1.4500711)) 
		self._fConst57 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst53)) 
		self._fConst58 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst31)) 
		self._fConst59 = (((self._fConst25 + np.float32(-0.15748216)) / self._fConst24) + np.float32(0.9351402)) 
		self._fConst60 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst53)) 
		self._fConst61 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst28)) 
		self._fConst62 = np.tan((np.float32(31665.27) / self._fConst0)) 
		self._fConst63 = (np.float32(1.0) / self._fConst62) 
		self._fConst64 = (np.float32(1.0) / (((self._fConst63 + np.float32(0.15748216)) / self._fConst62) + np.float32(0.9351402))) 
		self._fConst65 = np.power(self._fConst62, np.float32(2.0)) 
		self._fConst66 = (np.float32(50.06381) / self._fConst65) 
		self._fConst67 = (self._fConst66 + np.float32(0.9351402)) 
		self._fConst68 = (np.float32(1.0) / (((self._fConst63 + np.float32(0.74313045)) / self._fConst62) + np.float32(1.4500711))) 
		self._fConst69 = (np.float32(11.0520525) / self._fConst65) 
		self._fConst70 = (self._fConst69 + np.float32(1.4500711)) 
		self._fConst71 = (np.float32(1.0) / (((self._fConst63 + np.float32(3.1897273)) / self._fConst62) + np.float32(4.0767817))) 
		self._fConst72 = (np.float32(0.0017661728) / self._fConst65) 
		self._fConst73 = (self._fConst72 + np.float32(0.0004076782)) 
		self._fConst74 = (np.float32(1.0) / (((self._fConst25 + np.float32(0.16840488)) / self._fConst24) + np.float32(1.0693583))) 
		self._fConst75 = (self._fConst53 + np.float32(53.53615)) 
		self._fConst76 = (np.float32(1.0) / (((self._fConst25 + np.float32(0.51247865)) / self._fConst24) + np.float32(0.6896214))) 
		self._fConst77 = (self._fConst53 + np.float32(7.6217313)) 
		self._fConst78 = (np.float32(1.0) / (((self._fConst25 + np.float32(0.78241307)) / self._fConst24) + np.float32(0.2452915))) 
		self._fConst79 = (np.float32(0.0001) / self._fConst27) 
		self._fConst80 = (self._fConst79 + np.float32(0.0004332272)) 
		self._fConst81 = (((self._fConst25 + np.float32(-0.78241307)) / self._fConst24) + np.float32(0.2452915)) 
		self._fConst82 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst53)) 
		self._fConst83 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst79)) 
		self._fConst84 = (((self._fConst25 + np.float32(-0.51247865)) / self._fConst24) + np.float32(0.6896214)) 
		self._fConst85 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst53)) 
		self._fConst86 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst53)) 
		self._fConst87 = (((self._fConst25 + np.float32(-0.16840488)) / self._fConst24) + np.float32(1.0693583)) 
		self._fConst88 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst53)) 
		self._fConst89 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst53)) 
		self._fConst90 = (((self._fConst63 + np.float32(-3.1897273)) / self._fConst62) + np.float32(4.0767817)) 
		self._fConst91 = (np.float32(1.0) / self._fConst65) 
		self._fConst92 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst91)) 
		self._fConst93 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst72)) 
		self._fConst94 = (((self._fConst63 + np.float32(-0.74313045)) / self._fConst62) + np.float32(1.4500711)) 
		self._fConst95 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst91)) 
		self._fConst96 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst69)) 
		self._fConst97 = (((self._fConst63 + np.float32(-0.15748216)) / self._fConst62) + np.float32(0.9351402)) 
		self._fConst98 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst91)) 
		self._fConst99 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst66)) 
		self._fConst100 = np.tan((np.float32(25132.742) / self._fConst0)) 
		self._fConst101 = (np.float32(1.0) / self._fConst100) 
		self._fConst102 = (np.float32(1.0) / (((self._fConst101 + np.float32(0.15748216)) / self._fConst100) + np.float32(0.9351402))) 
		self._fConst103 = np.power(self._fConst100, np.float32(2.0)) 
		self._fConst104 = (np.float32(50.06381) / self._fConst103) 
		self._fConst105 = (self._fConst104 + np.float32(0.9351402)) 
		self._fConst106 = (np.float32(1.0) / (((self._fConst101 + np.float32(0.74313045)) / self._fConst100) + np.float32(1.4500711))) 
		self._fConst107 = (np.float32(11.0520525) / self._fConst103) 
		self._fConst108 = (self._fConst107 + np.float32(1.4500711)) 
		self._fConst109 = (np.float32(1.0) / (((self._fConst101 + np.float32(3.1897273)) / self._fConst100) + np.float32(4.0767817))) 
		self._fConst110 = (np.float32(0.0017661728) / self._fConst103) 
		self._fConst111 = (self._fConst110 + np.float32(0.0004076782)) 
		self._fConst112 = (np.float32(1.0) / (((self._fConst63 + np.float32(0.16840488)) / self._fConst62) + np.float32(1.0693583))) 
		self._fConst113 = (self._fConst91 + np.float32(53.53615)) 
		self._fConst114 = (np.float32(1.0) / (((self._fConst63 + np.float32(0.51247865)) / self._fConst62) + np.float32(0.6896214))) 
		self._fConst115 = (self._fConst91 + np.float32(7.6217313)) 
		self._fConst116 = (np.float32(1.0) / (((self._fConst63 + np.float32(0.78241307)) / self._fConst62) + np.float32(0.2452915))) 
		self._fConst117 = (np.float32(0.0001) / self._fConst65) 
		self._fConst118 = (self._fConst117 + np.float32(0.0004332272)) 
		self._fConst119 = (((self._fConst63 + np.float32(-0.78241307)) / self._fConst62) + np.float32(0.2452915)) 
		self._fConst120 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst91)) 
		self._fConst121 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst117)) 
		self._fConst122 = (((self._fConst63 + np.float32(-0.51247865)) / self._fConst62) + np.float32(0.6896214)) 
		self._fConst123 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst91)) 
		self._fConst124 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst91)) 
		self._fConst125 = (((self._fConst63 + np.float32(-0.16840488)) / self._fConst62) + np.float32(1.0693583)) 
		self._fConst126 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst91)) 
		self._fConst127 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst91)) 
		self._fConst128 = (((self._fConst101 + np.float32(-3.1897273)) / self._fConst100) + np.float32(4.0767817)) 
		self._fConst129 = (np.float32(1.0) / self._fConst103) 
		self._fConst130 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst129)) 
		self._fConst131 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst110)) 
		self._fConst132 = (((self._fConst101 + np.float32(-0.74313045)) / self._fConst100) + np.float32(1.4500711)) 
		self._fConst133 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst129)) 
		self._fConst134 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst107)) 
		self._fConst135 = (((self._fConst101 + np.float32(-0.15748216)) / self._fConst100) + np.float32(0.9351402)) 
		self._fConst136 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst129)) 
		self._fConst137 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst104)) 
		self._fConst138 = np.tan((np.float32(19947.87) / self._fConst0)) 
		self._fConst139 = (np.float32(1.0) / self._fConst138) 
		self._fConst140 = (np.float32(1.0) / (((self._fConst139 + np.float32(0.15748216)) / self._fConst138) + np.float32(0.9351402))) 
		self._fConst141 = np.power(self._fConst138, np.float32(2.0)) 
		self._fConst142 = (np.float32(50.06381) / self._fConst141) 
		self._fConst143 = (self._fConst142 + np.float32(0.9351402)) 
		self._fConst144 = (np.float32(1.0) / (((self._fConst139 + np.float32(0.74313045)) / self._fConst138) + np.float32(1.4500711))) 
		self._fConst145 = (np.float32(11.0520525) / self._fConst141) 
		self._fConst146 = (self._fConst145 + np.float32(1.4500711)) 
		self._fConst147 = (np.float32(1.0) / (((self._fConst139 + np.float32(3.1897273)) / self._fConst138) + np.float32(4.0767817))) 
		self._fConst148 = (np.float32(0.0017661728) / self._fConst141) 
		self._fConst149 = (self._fConst148 + np.float32(0.0004076782)) 
		self._fConst150 = (np.float32(1.0) / (((self._fConst101 + np.float32(0.16840488)) / self._fConst100) + np.float32(1.0693583))) 
		self._fConst151 = (self._fConst129 + np.float32(53.53615)) 
		self._fConst152 = (np.float32(1.0) / (((self._fConst101 + np.float32(0.51247865)) / self._fConst100) + np.float32(0.6896214))) 
		self._fConst153 = (self._fConst129 + np.float32(7.6217313)) 
		self._fConst154 = (np.float32(1.0) / (((self._fConst101 + np.float32(0.78241307)) / self._fConst100) + np.float32(0.2452915))) 
		self._fConst155 = (np.float32(0.0001) / self._fConst103) 
		self._fConst156 = (self._fConst155 + np.float32(0.0004332272)) 
		self._fConst157 = (((self._fConst101 + np.float32(-0.78241307)) / self._fConst100) + np.float32(0.2452915)) 
		self._fConst158 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst129)) 
		self._fConst159 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst155)) 
		self._fConst160 = (((self._fConst101 + np.float32(-0.51247865)) / self._fConst100) + np.float32(0.6896214)) 
		self._fConst161 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst129)) 
		self._fConst162 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst129)) 
		self._fConst163 = (((self._fConst101 + np.float32(-0.16840488)) / self._fConst100) + np.float32(1.0693583)) 
		self._fConst164 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst129)) 
		self._fConst165 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst129)) 
		self._fConst166 = (((self._fConst139 + np.float32(-3.1897273)) / self._fConst138) + np.float32(4.0767817)) 
		self._fConst167 = (np.float32(1.0) / self._fConst141) 
		self._fConst168 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst167)) 
		self._fConst169 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst148)) 
		self._fConst170 = (((self._fConst139 + np.float32(-0.74313045)) / self._fConst138) + np.float32(1.4500711)) 
		self._fConst171 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst167)) 
		self._fConst172 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst145)) 
		self._fConst173 = (((self._fConst139 + np.float32(-0.15748216)) / self._fConst138) + np.float32(0.9351402)) 
		self._fConst174 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst167)) 
		self._fConst175 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst142)) 
		self._fConst176 = np.tan((np.float32(15832.635) / self._fConst0)) 
		self._fConst177 = (np.float32(1.0) / self._fConst176) 
		self._fConst178 = (np.float32(1.0) / (((self._fConst177 + np.float32(0.15748216)) / self._fConst176) + np.float32(0.9351402))) 
		self._fConst179 = np.power(self._fConst176, np.float32(2.0)) 
		self._fConst180 = (np.float32(50.06381) / self._fConst179) 
		self._fConst181 = (self._fConst180 + np.float32(0.9351402)) 
		self._fConst182 = (np.float32(1.0) / (((self._fConst177 + np.float32(0.74313045)) / self._fConst176) + np.float32(1.4500711))) 
		self._fConst183 = (np.float32(11.0520525) / self._fConst179) 
		self._fConst184 = (self._fConst183 + np.float32(1.4500711)) 
		self._fConst185 = (np.float32(1.0) / (((self._fConst177 + np.float32(3.1897273)) / self._fConst176) + np.float32(4.0767817))) 
		self._fConst186 = (np.float32(0.0017661728) / self._fConst179) 
		self._fConst187 = (self._fConst186 + np.float32(0.0004076782)) 
		self._fConst188 = (np.float32(1.0) / (((self._fConst139 + np.float32(0.16840488)) / self._fConst138) + np.float32(1.0693583))) 
		self._fConst189 = (self._fConst167 + np.float32(53.53615)) 
		self._fConst190 = (np.float32(1.0) / (((self._fConst139 + np.float32(0.51247865)) / self._fConst138) + np.float32(0.6896214))) 
		self._fConst191 = (self._fConst167 + np.float32(7.6217313)) 
		self._fConst192 = (np.float32(1.0) / (((self._fConst139 + np.float32(0.78241307)) / self._fConst138) + np.float32(0.2452915))) 
		self._fConst193 = (np.float32(0.0001) / self._fConst141) 
		self._fConst194 = (self._fConst193 + np.float32(0.0004332272)) 
		self._fConst195 = (((self._fConst139 + np.float32(-0.78241307)) / self._fConst138) + np.float32(0.2452915)) 
		self._fConst196 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst167)) 
		self._fConst197 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst193)) 
		self._fConst198 = (((self._fConst139 + np.float32(-0.51247865)) / self._fConst138) + np.float32(0.6896214)) 
		self._fConst199 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst167)) 
		self._fConst200 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst167)) 
		self._fConst201 = (((self._fConst139 + np.float32(-0.16840488)) / self._fConst138) + np.float32(1.0693583)) 
		self._fConst202 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst167)) 
		self._fConst203 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst167)) 
		self._fConst204 = (((self._fConst177 + np.float32(-3.1897273)) / self._fConst176) + np.float32(4.0767817)) 
		self._fConst205 = (np.float32(1.0) / self._fConst179) 
		self._fConst206 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst205)) 
		self._fConst207 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst186)) 
		self._fConst208 = (((self._fConst177 + np.float32(-0.74313045)) / self._fConst176) + np.float32(1.4500711)) 
		self._fConst209 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst205)) 
		self._fConst210 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst183)) 
		self._fConst211 = (((self._fConst177 + np.float32(-0.15748216)) / self._fConst176) + np.float32(0.9351402)) 
		self._fConst212 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst205)) 
		self._fConst213 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst180)) 
		self._fConst214 = np.tan((np.float32(12566.371) / self._fConst0)) 
		self._fConst215 = (np.float32(1.0) / self._fConst214) 
		self._fConst216 = (np.float32(1.0) / (((self._fConst215 + np.float32(0.15748216)) / self._fConst214) + np.float32(0.9351402))) 
		self._fConst217 = np.power(self._fConst214, np.float32(2.0)) 
		self._fConst218 = (np.float32(50.06381) / self._fConst217) 
		self._fConst219 = (self._fConst218 + np.float32(0.9351402)) 
		self._fConst220 = (np.float32(1.0) / (((self._fConst215 + np.float32(0.74313045)) / self._fConst214) + np.float32(1.4500711))) 
		self._fConst221 = (np.float32(11.0520525) / self._fConst217) 
		self._fConst222 = (self._fConst221 + np.float32(1.4500711)) 
		self._fConst223 = (np.float32(1.0) / (((self._fConst215 + np.float32(3.1897273)) / self._fConst214) + np.float32(4.0767817))) 
		self._fConst224 = (np.float32(0.0017661728) / self._fConst217) 
		self._fConst225 = (self._fConst224 + np.float32(0.0004076782)) 
		self._fConst226 = (np.float32(1.0) / (((self._fConst177 + np.float32(0.16840488)) / self._fConst176) + np.float32(1.0693583))) 
		self._fConst227 = (self._fConst205 + np.float32(53.53615)) 
		self._fConst228 = (np.float32(1.0) / (((self._fConst177 + np.float32(0.51247865)) / self._fConst176) + np.float32(0.6896214))) 
		self._fConst229 = (self._fConst205 + np.float32(7.6217313)) 
		self._fConst230 = (np.float32(1.0) / (((self._fConst177 + np.float32(0.78241307)) / self._fConst176) + np.float32(0.2452915))) 
		self._fConst231 = (np.float32(0.0001) / self._fConst179) 
		self._fConst232 = (self._fConst231 + np.float32(0.0004332272)) 
		self._fConst233 = (((self._fConst177 + np.float32(-0.78241307)) / self._fConst176) + np.float32(0.2452915)) 
		self._fConst234 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst205)) 
		self._fConst235 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst231)) 
		self._fConst236 = (((self._fConst177 + np.float32(-0.51247865)) / self._fConst176) + np.float32(0.6896214)) 
		self._fConst237 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst205)) 
		self._fConst238 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst205)) 
		self._fConst239 = (((self._fConst177 + np.float32(-0.16840488)) / self._fConst176) + np.float32(1.0693583)) 
		self._fConst240 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst205)) 
		self._fConst241 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst205)) 
		self._fConst242 = (((self._fConst215 + np.float32(-3.1897273)) / self._fConst214) + np.float32(4.0767817)) 
		self._fConst243 = (np.float32(1.0) / self._fConst217) 
		self._fConst244 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst243)) 
		self._fConst245 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst224)) 
		self._fConst246 = (((self._fConst215 + np.float32(-0.74313045)) / self._fConst214) + np.float32(1.4500711)) 
		self._fConst247 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst243)) 
		self._fConst248 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst221)) 
		self._fConst249 = (((self._fConst215 + np.float32(-0.15748216)) / self._fConst214) + np.float32(0.9351402)) 
		self._fConst250 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst243)) 
		self._fConst251 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst218)) 
		self._fConst252 = np.tan((np.float32(9973.935) / self._fConst0)) 
		self._fConst253 = (np.float32(1.0) / self._fConst252) 
		self._fConst254 = (np.float32(1.0) / (((self._fConst253 + np.float32(0.15748216)) / self._fConst252) + np.float32(0.9351402))) 
		self._fConst255 = np.power(self._fConst252, np.float32(2.0)) 
		self._fConst256 = (np.float32(50.06381) / self._fConst255) 
		self._fConst257 = (self._fConst256 + np.float32(0.9351402)) 
		self._fConst258 = (np.float32(1.0) / (((self._fConst253 + np.float32(0.74313045)) / self._fConst252) + np.float32(1.4500711))) 
		self._fConst259 = (np.float32(11.0520525) / self._fConst255) 
		self._fConst260 = (self._fConst259 + np.float32(1.4500711)) 
		self._fConst261 = (np.float32(1.0) / (((self._fConst253 + np.float32(3.1897273)) / self._fConst252) + np.float32(4.0767817))) 
		self._fConst262 = (np.float32(0.0017661728) / self._fConst255) 
		self._fConst263 = (self._fConst262 + np.float32(0.0004076782)) 
		self._fConst264 = (np.float32(1.0) / (((self._fConst215 + np.float32(0.16840488)) / self._fConst214) + np.float32(1.0693583))) 
		self._fConst265 = (self._fConst243 + np.float32(53.53615)) 
		self._fConst266 = (np.float32(1.0) / (((self._fConst215 + np.float32(0.51247865)) / self._fConst214) + np.float32(0.6896214))) 
		self._fConst267 = (self._fConst243 + np.float32(7.6217313)) 
		self._fConst268 = (np.float32(1.0) / (((self._fConst215 + np.float32(0.78241307)) / self._fConst214) + np.float32(0.2452915))) 
		self._fConst269 = (np.float32(0.0001) / self._fConst217) 
		self._fConst270 = (self._fConst269 + np.float32(0.0004332272)) 
		self._fConst271 = (((self._fConst215 + np.float32(-0.78241307)) / self._fConst214) + np.float32(0.2452915)) 
		self._fConst272 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst243)) 
		self._fConst273 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst269)) 
		self._fConst274 = (((self._fConst215 + np.float32(-0.51247865)) / self._fConst214) + np.float32(0.6896214)) 
		self._fConst275 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst243)) 
		self._fConst276 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst243)) 
		self._fConst277 = (((self._fConst215 + np.float32(-0.16840488)) / self._fConst214) + np.float32(1.0693583)) 
		self._fConst278 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst243)) 
		self._fConst279 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst243)) 
		self._fConst280 = (((self._fConst253 + np.float32(-3.1897273)) / self._fConst252) + np.float32(4.0767817)) 
		self._fConst281 = (np.float32(1.0) / self._fConst255) 
		self._fConst282 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst281)) 
		self._fConst283 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst262)) 
		self._fConst284 = (((self._fConst253 + np.float32(-0.74313045)) / self._fConst252) + np.float32(1.4500711)) 
		self._fConst285 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst281)) 
		self._fConst286 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst259)) 
		self._fConst287 = (((self._fConst253 + np.float32(-0.15748216)) / self._fConst252) + np.float32(0.9351402)) 
		self._fConst288 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst281)) 
		self._fConst289 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst256)) 
		self._fConst290 = np.tan((np.float32(7916.3174) / self._fConst0)) 
		self._fConst291 = (np.float32(1.0) / self._fConst290) 
		self._fConst292 = (np.float32(1.0) / (((self._fConst291 + np.float32(0.15748216)) / self._fConst290) + np.float32(0.9351402))) 
		self._fConst293 = np.power(self._fConst290, np.float32(2.0)) 
		self._fConst294 = (np.float32(50.06381) / self._fConst293) 
		self._fConst295 = (self._fConst294 + np.float32(0.9351402)) 
		self._fConst296 = (np.float32(1.0) / (((self._fConst291 + np.float32(0.74313045)) / self._fConst290) + np.float32(1.4500711))) 
		self._fConst297 = (np.float32(11.0520525) / self._fConst293) 
		self._fConst298 = (self._fConst297 + np.float32(1.4500711)) 
		self._fConst299 = (np.float32(1.0) / (((self._fConst291 + np.float32(3.1897273)) / self._fConst290) + np.float32(4.0767817))) 
		self._fConst300 = (np.float32(0.0017661728) / self._fConst293) 
		self._fConst301 = (self._fConst300 + np.float32(0.0004076782)) 
		self._fConst302 = (np.float32(1.0) / (((self._fConst253 + np.float32(0.16840488)) / self._fConst252) + np.float32(1.0693583))) 
		self._fConst303 = (self._fConst281 + np.float32(53.53615)) 
		self._fConst304 = (np.float32(1.0) / (((self._fConst253 + np.float32(0.51247865)) / self._fConst252) + np.float32(0.6896214))) 
		self._fConst305 = (self._fConst281 + np.float32(7.6217313)) 
		self._fConst306 = (np.float32(1.0) / (((self._fConst253 + np.float32(0.78241307)) / self._fConst252) + np.float32(0.2452915))) 
		self._fConst307 = (np.float32(0.0001) / self._fConst255) 
		self._fConst308 = (self._fConst307 + np.float32(0.0004332272)) 
		self._fConst309 = (((self._fConst253 + np.float32(-0.78241307)) / self._fConst252) + np.float32(0.2452915)) 
		self._fConst310 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst281)) 
		self._fConst311 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst307)) 
		self._fConst312 = (((self._fConst253 + np.float32(-0.51247865)) / self._fConst252) + np.float32(0.6896214)) 
		self._fConst313 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst281)) 
		self._fConst314 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst281)) 
		self._fConst315 = (((self._fConst253 + np.float32(-0.16840488)) / self._fConst252) + np.float32(1.0693583)) 
		self._fConst316 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst281)) 
		self._fConst317 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst281)) 
		self._fConst318 = (((self._fConst291 + np.float32(-3.1897273)) / self._fConst290) + np.float32(4.0767817)) 
		self._fConst319 = (np.float32(1.0) / self._fConst293) 
		self._fConst320 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst319)) 
		self._fConst321 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst300)) 
		self._fConst322 = (((self._fConst291 + np.float32(-0.74313045)) / self._fConst290) + np.float32(1.4500711)) 
		self._fConst323 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst319)) 
		self._fConst324 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst297)) 
		self._fConst325 = (((self._fConst291 + np.float32(-0.15748216)) / self._fConst290) + np.float32(0.9351402)) 
		self._fConst326 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst319)) 
		self._fConst327 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst294)) 
		self._fConst328 = np.tan((np.float32(6283.1855) / self._fConst0)) 
		self._fConst329 = (np.float32(1.0) / self._fConst328) 
		self._fConst330 = (np.float32(1.0) / (((self._fConst329 + np.float32(0.15748216)) / self._fConst328) + np.float32(0.9351402))) 
		self._fConst331 = np.power(self._fConst328, np.float32(2.0)) 
		self._fConst332 = (np.float32(50.06381) / self._fConst331) 
		self._fConst333 = (self._fConst332 + np.float32(0.9351402)) 
		self._fConst334 = (np.float32(1.0) / (((self._fConst329 + np.float32(0.74313045)) / self._fConst328) + np.float32(1.4500711))) 
		self._fConst335 = (np.float32(11.0520525) / self._fConst331) 
		self._fConst336 = (self._fConst335 + np.float32(1.4500711)) 
		self._fConst337 = (np.float32(1.0) / (((self._fConst329 + np.float32(3.1897273)) / self._fConst328) + np.float32(4.0767817))) 
		self._fConst338 = (np.float32(0.0017661728) / self._fConst331) 
		self._fConst339 = (self._fConst338 + np.float32(0.0004076782)) 
		self._fConst340 = (np.float32(1.0) / (((self._fConst291 + np.float32(0.16840488)) / self._fConst290) + np.float32(1.0693583))) 
		self._fConst341 = (self._fConst319 + np.float32(53.53615)) 
		self._fConst342 = (np.float32(1.0) / (((self._fConst291 + np.float32(0.51247865)) / self._fConst290) + np.float32(0.6896214))) 
		self._fConst343 = (self._fConst319 + np.float32(7.6217313)) 
		self._fConst344 = (np.float32(1.0) / (((self._fConst291 + np.float32(0.78241307)) / self._fConst290) + np.float32(0.2452915))) 
		self._fConst345 = (np.float32(0.0001) / self._fConst293) 
		self._fConst346 = (self._fConst345 + np.float32(0.0004332272)) 
		self._fConst347 = (((self._fConst291 + np.float32(-0.78241307)) / self._fConst290) + np.float32(0.2452915)) 
		self._fConst348 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst319)) 
		self._fConst349 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst345)) 
		self._fConst350 = (((self._fConst291 + np.float32(-0.51247865)) / self._fConst290) + np.float32(0.6896214)) 
		self._fConst351 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst319)) 
		self._fConst352 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst319)) 
		self._fConst353 = (((self._fConst291 + np.float32(-0.16840488)) / self._fConst290) + np.float32(1.0693583)) 
		self._fConst354 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst319)) 
		self._fConst355 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst319)) 
		self._fConst356 = (((self._fConst329 + np.float32(-3.1897273)) / self._fConst328) + np.float32(4.0767817)) 
		self._fConst357 = (np.float32(1.0) / self._fConst331) 
		self._fConst358 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst357)) 
		self._fConst359 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst338)) 
		self._fConst360 = (((self._fConst329 + np.float32(-0.74313045)) / self._fConst328) + np.float32(1.4500711)) 
		self._fConst361 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst357)) 
		self._fConst362 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst335)) 
		self._fConst363 = (((self._fConst329 + np.float32(-0.15748216)) / self._fConst328) + np.float32(0.9351402)) 
		self._fConst364 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst357)) 
		self._fConst365 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst332)) 
		self._fConst366 = np.tan((np.float32(4986.9673) / self._fConst0)) 
		self._fConst367 = (np.float32(1.0) / self._fConst366) 
		self._fConst368 = (np.float32(1.0) / (((self._fConst367 + np.float32(0.15748216)) / self._fConst366) + np.float32(0.9351402))) 
		self._fConst369 = np.power(self._fConst366, np.float32(2.0)) 
		self._fConst370 = (np.float32(50.06381) / self._fConst369) 
		self._fConst371 = (self._fConst370 + np.float32(0.9351402)) 
		self._fConst372 = (np.float32(1.0) / (((self._fConst367 + np.float32(0.74313045)) / self._fConst366) + np.float32(1.4500711))) 
		self._fConst373 = (np.float32(11.0520525) / self._fConst369) 
		self._fConst374 = (self._fConst373 + np.float32(1.4500711)) 
		self._fConst375 = (np.float32(1.0) / (((self._fConst367 + np.float32(3.1897273)) / self._fConst366) + np.float32(4.0767817))) 
		self._fConst376 = (np.float32(0.0017661728) / self._fConst369) 
		self._fConst377 = (self._fConst376 + np.float32(0.0004076782)) 
		self._fConst378 = (np.float32(1.0) / (((self._fConst329 + np.float32(0.16840488)) / self._fConst328) + np.float32(1.0693583))) 
		self._fConst379 = (self._fConst357 + np.float32(53.53615)) 
		self._fConst380 = (np.float32(1.0) / (((self._fConst329 + np.float32(0.51247865)) / self._fConst328) + np.float32(0.6896214))) 
		self._fConst381 = (self._fConst357 + np.float32(7.6217313)) 
		self._fConst382 = (np.float32(1.0) / (((self._fConst329 + np.float32(0.78241307)) / self._fConst328) + np.float32(0.2452915))) 
		self._fConst383 = (np.float32(0.0001) / self._fConst331) 
		self._fConst384 = (self._fConst383 + np.float32(0.0004332272)) 
		self._fConst385 = (((self._fConst329 + np.float32(-0.78241307)) / self._fConst328) + np.float32(0.2452915)) 
		self._fConst386 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst357)) 
		self._fConst387 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst383)) 
		self._fConst388 = (((self._fConst329 + np.float32(-0.51247865)) / self._fConst328) + np.float32(0.6896214)) 
		self._fConst389 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst357)) 
		self._fConst390 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst357)) 
		self._fConst391 = (((self._fConst329 + np.float32(-0.16840488)) / self._fConst328) + np.float32(1.0693583)) 
		self._fConst392 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst357)) 
		self._fConst393 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst357)) 
		self._fConst394 = (((self._fConst367 + np.float32(-3.1897273)) / self._fConst366) + np.float32(4.0767817)) 
		self._fConst395 = (np.float32(1.0) / self._fConst369) 
		self._fConst396 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst395)) 
		self._fConst397 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst376)) 
		self._fConst398 = (((self._fConst367 + np.float32(-0.74313045)) / self._fConst366) + np.float32(1.4500711)) 
		self._fConst399 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst395)) 
		self._fConst400 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst373)) 
		self._fConst401 = (((self._fConst367 + np.float32(-0.15748216)) / self._fConst366) + np.float32(0.9351402)) 
		self._fConst402 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst395)) 
		self._fConst403 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst370)) 
		self._fConst404 = np.tan((np.float32(3958.1587) / self._fConst0)) 
		self._fConst405 = (np.float32(1.0) / self._fConst404) 
		self._fConst406 = (np.float32(1.0) / (((self._fConst405 + np.float32(0.15748216)) / self._fConst404) + np.float32(0.9351402))) 
		self._fConst407 = np.power(self._fConst404, np.float32(2.0)) 
		self._fConst408 = (np.float32(50.06381) / self._fConst407) 
		self._fConst409 = (self._fConst408 + np.float32(0.9351402)) 
		self._fConst410 = (np.float32(1.0) / (((self._fConst405 + np.float32(0.74313045)) / self._fConst404) + np.float32(1.4500711))) 
		self._fConst411 = (np.float32(11.0520525) / self._fConst407) 
		self._fConst412 = (self._fConst411 + np.float32(1.4500711)) 
		self._fConst413 = (np.float32(1.0) / (((self._fConst405 + np.float32(3.1897273)) / self._fConst404) + np.float32(4.0767817))) 
		self._fConst414 = (np.float32(0.0017661728) / self._fConst407) 
		self._fConst415 = (self._fConst414 + np.float32(0.0004076782)) 
		self._fConst416 = (np.float32(1.0) / (((self._fConst367 + np.float32(0.16840488)) / self._fConst366) + np.float32(1.0693583))) 
		self._fConst417 = (self._fConst395 + np.float32(53.53615)) 
		self._fConst418 = (np.float32(1.0) / (((self._fConst367 + np.float32(0.51247865)) / self._fConst366) + np.float32(0.6896214))) 
		self._fConst419 = (self._fConst395 + np.float32(7.6217313)) 
		self._fConst420 = (np.float32(1.0) / (((self._fConst367 + np.float32(0.78241307)) / self._fConst366) + np.float32(0.2452915))) 
		self._fConst421 = (np.float32(0.0001) / self._fConst369) 
		self._fConst422 = (self._fConst421 + np.float32(0.0004332272)) 
		self._fConst423 = (((self._fConst367 + np.float32(-0.78241307)) / self._fConst366) + np.float32(0.2452915)) 
		self._fConst424 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst395)) 
		self._fConst425 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst421)) 
		self._fConst426 = (((self._fConst367 + np.float32(-0.51247865)) / self._fConst366) + np.float32(0.6896214)) 
		self._fConst427 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst395)) 
		self._fConst428 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst395)) 
		self._fConst429 = (((self._fConst367 + np.float32(-0.16840488)) / self._fConst366) + np.float32(1.0693583)) 
		self._fConst430 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst395)) 
		self._fConst431 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst395)) 
		self._fConst432 = (((self._fConst405 + np.float32(-3.1897273)) / self._fConst404) + np.float32(4.0767817)) 
		self._fConst433 = (np.float32(1.0) / self._fConst407) 
		self._fConst434 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst433)) 
		self._fConst435 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst414)) 
		self._fConst436 = (((self._fConst405 + np.float32(-0.74313045)) / self._fConst404) + np.float32(1.4500711)) 
		self._fConst437 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst433)) 
		self._fConst438 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst411)) 
		self._fConst439 = (((self._fConst405 + np.float32(-0.15748216)) / self._fConst404) + np.float32(0.9351402)) 
		self._fConst440 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst433)) 
		self._fConst441 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst408)) 
		self._fConst442 = np.tan((np.float32(3141.5928) / self._fConst0)) 
		self._fConst443 = (np.float32(1.0) / self._fConst442) 
		self._fConst444 = (np.float32(1.0) / (((self._fConst443 + np.float32(0.15748216)) / self._fConst442) + np.float32(0.9351402))) 
		self._fConst445 = np.power(self._fConst442, np.float32(2.0)) 
		self._fConst446 = (np.float32(50.06381) / self._fConst445) 
		self._fConst447 = (self._fConst446 + np.float32(0.9351402)) 
		self._fConst448 = (np.float32(1.0) / (((self._fConst443 + np.float32(0.74313045)) / self._fConst442) + np.float32(1.4500711))) 
		self._fConst449 = (np.float32(11.0520525) / self._fConst445) 
		self._fConst450 = (self._fConst449 + np.float32(1.4500711)) 
		self._fConst451 = (np.float32(1.0) / (((self._fConst443 + np.float32(3.1897273)) / self._fConst442) + np.float32(4.0767817))) 
		self._fConst452 = (np.float32(0.0017661728) / self._fConst445) 
		self._fConst453 = (self._fConst452 + np.float32(0.0004076782)) 
		self._fConst454 = (np.float32(1.0) / (((self._fConst405 + np.float32(0.16840488)) / self._fConst404) + np.float32(1.0693583))) 
		self._fConst455 = (self._fConst433 + np.float32(53.53615)) 
		self._fConst456 = (np.float32(1.0) / (((self._fConst405 + np.float32(0.51247865)) / self._fConst404) + np.float32(0.6896214))) 
		self._fConst457 = (self._fConst433 + np.float32(7.6217313)) 
		self._fConst458 = (np.float32(1.0) / (((self._fConst405 + np.float32(0.78241307)) / self._fConst404) + np.float32(0.2452915))) 
		self._fConst459 = (np.float32(0.0001) / self._fConst407) 
		self._fConst460 = (self._fConst459 + np.float32(0.0004332272)) 
		self._fConst461 = (((self._fConst405 + np.float32(-0.78241307)) / self._fConst404) + np.float32(0.2452915)) 
		self._fConst462 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst433)) 
		self._fConst463 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst459)) 
		self._fConst464 = (((self._fConst405 + np.float32(-0.51247865)) / self._fConst404) + np.float32(0.6896214)) 
		self._fConst465 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst433)) 
		self._fConst466 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst433)) 
		self._fConst467 = (((self._fConst405 + np.float32(-0.16840488)) / self._fConst404) + np.float32(1.0693583)) 
		self._fConst468 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst433)) 
		self._fConst469 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst433)) 
		self._fConst470 = (((self._fConst443 + np.float32(-3.1897273)) / self._fConst442) + np.float32(4.0767817)) 
		self._fConst471 = (np.float32(1.0) / self._fConst445) 
		self._fConst472 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst471)) 
		self._fConst473 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst452)) 
		self._fConst474 = (((self._fConst443 + np.float32(-0.74313045)) / self._fConst442) + np.float32(1.4500711)) 
		self._fConst475 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst471)) 
		self._fConst476 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst449)) 
		self._fConst477 = (((self._fConst443 + np.float32(-0.15748216)) / self._fConst442) + np.float32(0.9351402)) 
		self._fConst478 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst471)) 
		self._fConst479 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst446)) 
		self._fConst480 = np.tan((np.float32(2493.4836) / self._fConst0)) 
		self._fConst481 = (np.float32(1.0) / self._fConst480) 
		self._fConst482 = (np.float32(1.0) / (((self._fConst481 + np.float32(0.15748216)) / self._fConst480) + np.float32(0.9351402))) 
		self._fConst483 = np.power(self._fConst480, np.float32(2.0)) 
		self._fConst484 = (np.float32(50.06381) / self._fConst483) 
		self._fConst485 = (self._fConst484 + np.float32(0.9351402)) 
		self._fConst486 = (np.float32(1.0) / (((self._fConst481 + np.float32(0.74313045)) / self._fConst480) + np.float32(1.4500711))) 
		self._fConst487 = (np.float32(11.0520525) / self._fConst483) 
		self._fConst488 = (self._fConst487 + np.float32(1.4500711)) 
		self._fConst489 = (np.float32(1.0) / (((self._fConst481 + np.float32(3.1897273)) / self._fConst480) + np.float32(4.0767817))) 
		self._fConst490 = (np.float32(0.0017661728) / self._fConst483) 
		self._fConst491 = (self._fConst490 + np.float32(0.0004076782)) 
		self._fConst492 = (np.float32(1.0) / (((self._fConst443 + np.float32(0.16840488)) / self._fConst442) + np.float32(1.0693583))) 
		self._fConst493 = (self._fConst471 + np.float32(53.53615)) 
		self._fConst494 = (np.float32(1.0) / (((self._fConst443 + np.float32(0.51247865)) / self._fConst442) + np.float32(0.6896214))) 
		self._fConst495 = (self._fConst471 + np.float32(7.6217313)) 
		self._fConst496 = (np.float32(1.0) / (((self._fConst443 + np.float32(0.78241307)) / self._fConst442) + np.float32(0.2452915))) 
		self._fConst497 = (np.float32(0.0001) / self._fConst445) 
		self._fConst498 = (self._fConst497 + np.float32(0.0004332272)) 
		self._fConst499 = (((self._fConst443 + np.float32(-0.78241307)) / self._fConst442) + np.float32(0.2452915)) 
		self._fConst500 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst471)) 
		self._fConst501 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst497)) 
		self._fConst502 = (((self._fConst443 + np.float32(-0.51247865)) / self._fConst442) + np.float32(0.6896214)) 
		self._fConst503 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst471)) 
		self._fConst504 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst471)) 
		self._fConst505 = (((self._fConst443 + np.float32(-0.16840488)) / self._fConst442) + np.float32(1.0693583)) 
		self._fConst506 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst471)) 
		self._fConst507 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst471)) 
		self._fConst508 = (((self._fConst481 + np.float32(-3.1897273)) / self._fConst480) + np.float32(4.0767817)) 
		self._fConst509 = (np.float32(1.0) / self._fConst483) 
		self._fConst510 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst509)) 
		self._fConst511 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst490)) 
		self._fConst512 = (((self._fConst481 + np.float32(-0.74313045)) / self._fConst480) + np.float32(1.4500711)) 
		self._fConst513 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst509)) 
		self._fConst514 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst487)) 
		self._fConst515 = (((self._fConst481 + np.float32(-0.15748216)) / self._fConst480) + np.float32(0.9351402)) 
		self._fConst516 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst509)) 
		self._fConst517 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst484)) 
		self._fConst518 = np.tan((np.float32(1979.0793) / self._fConst0)) 
		self._fConst519 = (np.float32(1.0) / self._fConst518) 
		self._fConst520 = (np.float32(1.0) / (((self._fConst519 + np.float32(0.15748216)) / self._fConst518) + np.float32(0.9351402))) 
		self._fConst521 = np.power(self._fConst518, np.float32(2.0)) 
		self._fConst522 = (np.float32(50.06381) / self._fConst521) 
		self._fConst523 = (self._fConst522 + np.float32(0.9351402)) 
		self._fConst524 = (np.float32(1.0) / (((self._fConst519 + np.float32(0.74313045)) / self._fConst518) + np.float32(1.4500711))) 
		self._fConst525 = (np.float32(11.0520525) / self._fConst521) 
		self._fConst526 = (self._fConst525 + np.float32(1.4500711)) 
		self._fConst527 = (np.float32(1.0) / (((self._fConst519 + np.float32(3.1897273)) / self._fConst518) + np.float32(4.0767817))) 
		self._fConst528 = (np.float32(0.0017661728) / self._fConst521) 
		self._fConst529 = (self._fConst528 + np.float32(0.0004076782)) 
		self._fConst530 = (np.float32(1.0) / (((self._fConst481 + np.float32(0.16840488)) / self._fConst480) + np.float32(1.0693583))) 
		self._fConst531 = (self._fConst509 + np.float32(53.53615)) 
		self._fConst532 = (np.float32(1.0) / (((self._fConst481 + np.float32(0.51247865)) / self._fConst480) + np.float32(0.6896214))) 
		self._fConst533 = (self._fConst509 + np.float32(7.6217313)) 
		self._fConst534 = (np.float32(1.0) / (((self._fConst481 + np.float32(0.78241307)) / self._fConst480) + np.float32(0.2452915))) 
		self._fConst535 = (np.float32(0.0001) / self._fConst483) 
		self._fConst536 = (self._fConst535 + np.float32(0.0004332272)) 
		self._fConst537 = (((self._fConst481 + np.float32(-0.78241307)) / self._fConst480) + np.float32(0.2452915)) 
		self._fConst538 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst509)) 
		self._fConst539 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst535)) 
		self._fConst540 = (((self._fConst481 + np.float32(-0.51247865)) / self._fConst480) + np.float32(0.6896214)) 
		self._fConst541 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst509)) 
		self._fConst542 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst509)) 
		self._fConst543 = (((self._fConst481 + np.float32(-0.16840488)) / self._fConst480) + np.float32(1.0693583)) 
		self._fConst544 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst509)) 
		self._fConst545 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst509)) 
		self._fConst546 = (((self._fConst519 + np.float32(-3.1897273)) / self._fConst518) + np.float32(4.0767817)) 
		self._fConst547 = (np.float32(1.0) / self._fConst521) 
		self._fConst548 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst547)) 
		self._fConst549 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst528)) 
		self._fConst550 = (((self._fConst519 + np.float32(-0.74313045)) / self._fConst518) + np.float32(1.4500711)) 
		self._fConst551 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst547)) 
		self._fConst552 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst525)) 
		self._fConst553 = (((self._fConst519 + np.float32(-0.15748216)) / self._fConst518) + np.float32(0.9351402)) 
		self._fConst554 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst547)) 
		self._fConst555 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst522)) 
		self._fConst556 = np.tan((np.float32(1570.7964) / self._fConst0)) 
		self._fConst557 = (np.float32(1.0) / self._fConst556) 
		self._fConst558 = (np.float32(1.0) / (((self._fConst557 + np.float32(0.15748216)) / self._fConst556) + np.float32(0.9351402))) 
		self._fConst559 = np.power(self._fConst556, np.float32(2.0)) 
		self._fConst560 = (np.float32(50.06381) / self._fConst559) 
		self._fConst561 = (self._fConst560 + np.float32(0.9351402)) 
		self._fConst562 = (np.float32(1.0) / (((self._fConst557 + np.float32(0.74313045)) / self._fConst556) + np.float32(1.4500711))) 
		self._fConst563 = (np.float32(11.0520525) / self._fConst559) 
		self._fConst564 = (self._fConst563 + np.float32(1.4500711)) 
		self._fConst565 = (np.float32(1.0) / (((self._fConst557 + np.float32(3.1897273)) / self._fConst556) + np.float32(4.0767817))) 
		self._fConst566 = (np.float32(0.0017661728) / self._fConst559) 
		self._fConst567 = (self._fConst566 + np.float32(0.0004076782)) 
		self._fConst568 = (np.float32(1.0) / (((self._fConst519 + np.float32(0.16840488)) / self._fConst518) + np.float32(1.0693583))) 
		self._fConst569 = (self._fConst547 + np.float32(53.53615)) 
		self._fConst570 = (np.float32(1.0) / (((self._fConst519 + np.float32(0.51247865)) / self._fConst518) + np.float32(0.6896214))) 
		self._fConst571 = (self._fConst547 + np.float32(7.6217313)) 
		self._fConst572 = (np.float32(1.0) / (((self._fConst519 + np.float32(0.78241307)) / self._fConst518) + np.float32(0.2452915))) 
		self._fConst573 = (np.float32(0.0001) / self._fConst521) 
		self._fConst574 = (self._fConst573 + np.float32(0.0004332272)) 
		self._fConst575 = (((self._fConst519 + np.float32(-0.78241307)) / self._fConst518) + np.float32(0.2452915)) 
		self._fConst576 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst547)) 
		self._fConst577 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst573)) 
		self._fConst578 = (((self._fConst519 + np.float32(-0.51247865)) / self._fConst518) + np.float32(0.6896214)) 
		self._fConst579 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst547)) 
		self._fConst580 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst547)) 
		self._fConst581 = (((self._fConst519 + np.float32(-0.16840488)) / self._fConst518) + np.float32(1.0693583)) 
		self._fConst582 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst547)) 
		self._fConst583 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst547)) 
		self._fConst584 = (((self._fConst557 + np.float32(-3.1897273)) / self._fConst556) + np.float32(4.0767817)) 
		self._fConst585 = (np.float32(1.0) / self._fConst559) 
		self._fConst586 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst585)) 
		self._fConst587 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst566)) 
		self._fConst588 = (((self._fConst557 + np.float32(-0.74313045)) / self._fConst556) + np.float32(1.4500711)) 
		self._fConst589 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst585)) 
		self._fConst590 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst563)) 
		self._fConst591 = (((self._fConst557 + np.float32(-0.15748216)) / self._fConst556) + np.float32(0.9351402)) 
		self._fConst592 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst585)) 
		self._fConst593 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst560)) 
		self._fConst594 = np.tan((np.float32(1246.7418) / self._fConst0)) 
		self._fConst595 = (np.float32(1.0) / self._fConst594) 
		self._fConst596 = (np.float32(1.0) / (((self._fConst595 + np.float32(0.15748216)) / self._fConst594) + np.float32(0.9351402))) 
		self._fConst597 = np.power(self._fConst594, np.float32(2.0)) 
		self._fConst598 = (np.float32(50.06381) / self._fConst597) 
		self._fConst599 = (self._fConst598 + np.float32(0.9351402)) 
		self._fConst600 = (np.float32(1.0) / (((self._fConst595 + np.float32(0.74313045)) / self._fConst594) + np.float32(1.4500711))) 
		self._fConst601 = (np.float32(11.0520525) / self._fConst597) 
		self._fConst602 = (self._fConst601 + np.float32(1.4500711)) 
		self._fConst603 = (np.float32(1.0) / (((self._fConst595 + np.float32(3.1897273)) / self._fConst594) + np.float32(4.0767817))) 
		self._fConst604 = (np.float32(0.0017661728) / self._fConst597) 
		self._fConst605 = (self._fConst604 + np.float32(0.0004076782)) 
		self._fConst606 = (np.float32(1.0) / (((self._fConst557 + np.float32(0.16840488)) / self._fConst556) + np.float32(1.0693583))) 
		self._fConst607 = (self._fConst585 + np.float32(53.53615)) 
		self._fConst608 = (np.float32(1.0) / (((self._fConst557 + np.float32(0.51247865)) / self._fConst556) + np.float32(0.6896214))) 
		self._fConst609 = (self._fConst585 + np.float32(7.6217313)) 
		self._fConst610 = (np.float32(1.0) / (((self._fConst557 + np.float32(0.78241307)) / self._fConst556) + np.float32(0.2452915))) 
		self._fConst611 = (np.float32(0.0001) / self._fConst559) 
		self._fConst612 = (self._fConst611 + np.float32(0.0004332272)) 
		self._fConst613 = (((self._fConst557 + np.float32(-0.78241307)) / self._fConst556) + np.float32(0.2452915)) 
		self._fConst614 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst585)) 
		self._fConst615 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst611)) 
		self._fConst616 = (((self._fConst557 + np.float32(-0.51247865)) / self._fConst556) + np.float32(0.6896214)) 
		self._fConst617 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst585)) 
		self._fConst618 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst585)) 
		self._fConst619 = (((self._fConst557 + np.float32(-0.16840488)) / self._fConst556) + np.float32(1.0693583)) 
		self._fConst620 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst585)) 
		self._fConst621 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst585)) 
		self._fConst622 = (((self._fConst595 + np.float32(-3.1897273)) / self._fConst594) + np.float32(4.0767817)) 
		self._fConst623 = (np.float32(1.0) / self._fConst597) 
		self._fConst624 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst623)) 
		self._fConst625 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst604)) 
		self._fConst626 = (((self._fConst595 + np.float32(-0.74313045)) / self._fConst594) + np.float32(1.4500711)) 
		self._fConst627 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst623)) 
		self._fConst628 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst601)) 
		self._fConst629 = (((self._fConst595 + np.float32(-0.15748216)) / self._fConst594) + np.float32(0.9351402)) 
		self._fConst630 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst623)) 
		self._fConst631 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst598)) 
		self._fConst632 = np.tan((np.float32(989.5397) / self._fConst0)) 
		self._fConst633 = (np.float32(1.0) / self._fConst632) 
		self._fConst634 = (np.float32(1.0) / (((self._fConst633 + np.float32(0.15748216)) / self._fConst632) + np.float32(0.9351402))) 
		self._fConst635 = np.power(self._fConst632, np.float32(2.0)) 
		self._fConst636 = (np.float32(50.06381) / self._fConst635) 
		self._fConst637 = (self._fConst636 + np.float32(0.9351402)) 
		self._fConst638 = (np.float32(1.0) / (((self._fConst633 + np.float32(0.74313045)) / self._fConst632) + np.float32(1.4500711))) 
		self._fConst639 = (np.float32(11.0520525) / self._fConst635) 
		self._fConst640 = (self._fConst639 + np.float32(1.4500711)) 
		self._fConst641 = (np.float32(1.0) / (((self._fConst633 + np.float32(3.1897273)) / self._fConst632) + np.float32(4.0767817))) 
		self._fConst642 = (np.float32(0.0017661728) / self._fConst635) 
		self._fConst643 = (self._fConst642 + np.float32(0.0004076782)) 
		self._fConst644 = (np.float32(1.0) / (((self._fConst595 + np.float32(0.16840488)) / self._fConst594) + np.float32(1.0693583))) 
		self._fConst645 = (self._fConst623 + np.float32(53.53615)) 
		self._fConst646 = (np.float32(1.0) / (((self._fConst595 + np.float32(0.51247865)) / self._fConst594) + np.float32(0.6896214))) 
		self._fConst647 = (self._fConst623 + np.float32(7.6217313)) 
		self._fConst648 = (np.float32(1.0) / (((self._fConst595 + np.float32(0.78241307)) / self._fConst594) + np.float32(0.2452915))) 
		self._fConst649 = (np.float32(0.0001) / self._fConst597) 
		self._fConst650 = (self._fConst649 + np.float32(0.0004332272)) 
		self._fConst651 = (((self._fConst595 + np.float32(-0.78241307)) / self._fConst594) + np.float32(0.2452915)) 
		self._fConst652 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst623)) 
		self._fConst653 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst649)) 
		self._fConst654 = (((self._fConst595 + np.float32(-0.51247865)) / self._fConst594) + np.float32(0.6896214)) 
		self._fConst655 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst623)) 
		self._fConst656 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst623)) 
		self._fConst657 = (((self._fConst595 + np.float32(-0.16840488)) / self._fConst594) + np.float32(1.0693583)) 
		self._fConst658 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst623)) 
		self._fConst659 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst623)) 
		self._fConst660 = (((self._fConst633 + np.float32(-3.1897273)) / self._fConst632) + np.float32(4.0767817)) 
		self._fConst661 = (np.float32(1.0) / self._fConst635) 
		self._fConst662 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst661)) 
		self._fConst663 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst642)) 
		self._fConst664 = (((self._fConst633 + np.float32(-0.74313045)) / self._fConst632) + np.float32(1.4500711)) 
		self._fConst665 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst661)) 
		self._fConst666 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst639)) 
		self._fConst667 = (((self._fConst633 + np.float32(-0.15748216)) / self._fConst632) + np.float32(0.9351402)) 
		self._fConst668 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst661)) 
		self._fConst669 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst636)) 
		self._fConst670 = np.tan((np.float32(785.3982) / self._fConst0)) 
		self._fConst671 = (np.float32(1.0) / self._fConst670) 
		self._fConst672 = (np.float32(1.0) / (((self._fConst671 + np.float32(0.15748216)) / self._fConst670) + np.float32(0.9351402))) 
		self._fConst673 = np.power(self._fConst670, np.float32(2.0)) 
		self._fConst674 = (np.float32(50.06381) / self._fConst673) 
		self._fConst675 = (self._fConst674 + np.float32(0.9351402)) 
		self._fConst676 = (np.float32(1.0) / (((self._fConst671 + np.float32(0.74313045)) / self._fConst670) + np.float32(1.4500711))) 
		self._fConst677 = (np.float32(11.0520525) / self._fConst673) 
		self._fConst678 = (self._fConst677 + np.float32(1.4500711)) 
		self._fConst679 = (np.float32(1.0) / (((self._fConst671 + np.float32(3.1897273)) / self._fConst670) + np.float32(4.0767817))) 
		self._fConst680 = (np.float32(0.0017661728) / self._fConst673) 
		self._fConst681 = (self._fConst680 + np.float32(0.0004076782)) 
		self._fConst682 = (np.float32(1.0) / (((self._fConst633 + np.float32(0.16840488)) / self._fConst632) + np.float32(1.0693583))) 
		self._fConst683 = (self._fConst661 + np.float32(53.53615)) 
		self._fConst684 = (np.float32(1.0) / (((self._fConst633 + np.float32(0.51247865)) / self._fConst632) + np.float32(0.6896214))) 
		self._fConst685 = (self._fConst661 + np.float32(7.6217313)) 
		self._fConst686 = (np.float32(1.0) / (((self._fConst633 + np.float32(0.78241307)) / self._fConst632) + np.float32(0.2452915))) 
		self._fConst687 = (np.float32(0.0001) / self._fConst635) 
		self._fConst688 = (self._fConst687 + np.float32(0.0004332272)) 
		self._fConst689 = (((self._fConst633 + np.float32(-0.78241307)) / self._fConst632) + np.float32(0.2452915)) 
		self._fConst690 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst661)) 
		self._fConst691 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst687)) 
		self._fConst692 = (((self._fConst633 + np.float32(-0.51247865)) / self._fConst632) + np.float32(0.6896214)) 
		self._fConst693 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst661)) 
		self._fConst694 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst661)) 
		self._fConst695 = (((self._fConst633 + np.float32(-0.16840488)) / self._fConst632) + np.float32(1.0693583)) 
		self._fConst696 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst661)) 
		self._fConst697 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst661)) 
		self._fConst698 = (((self._fConst671 + np.float32(-3.1897273)) / self._fConst670) + np.float32(4.0767817)) 
		self._fConst699 = (np.float32(1.0) / self._fConst673) 
		self._fConst700 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst699)) 
		self._fConst701 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst680)) 
		self._fConst702 = (((self._fConst671 + np.float32(-0.74313045)) / self._fConst670) + np.float32(1.4500711)) 
		self._fConst703 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst699)) 
		self._fConst704 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst677)) 
		self._fConst705 = (((self._fConst671 + np.float32(-0.15748216)) / self._fConst670) + np.float32(0.9351402)) 
		self._fConst706 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst699)) 
		self._fConst707 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst674)) 
		self._fConst708 = np.tan((np.float32(623.3709) / self._fConst0)) 
		self._fConst709 = (np.float32(1.0) / self._fConst708) 
		self._fConst710 = (np.float32(1.0) / (((self._fConst709 + np.float32(0.15748216)) / self._fConst708) + np.float32(0.9351402))) 
		self._fConst711 = np.power(self._fConst708, np.float32(2.0)) 
		self._fConst712 = (np.float32(50.06381) / self._fConst711) 
		self._fConst713 = (self._fConst712 + np.float32(0.9351402)) 
		self._fConst714 = (np.float32(1.0) / (((self._fConst709 + np.float32(0.74313045)) / self._fConst708) + np.float32(1.4500711))) 
		self._fConst715 = (np.float32(11.0520525) / self._fConst711) 
		self._fConst716 = (self._fConst715 + np.float32(1.4500711)) 
		self._fConst717 = (np.float32(1.0) / (((self._fConst709 + np.float32(3.1897273)) / self._fConst708) + np.float32(4.0767817))) 
		self._fConst718 = (np.float32(0.0017661728) / self._fConst711) 
		self._fConst719 = (self._fConst718 + np.float32(0.0004076782)) 
		self._fConst720 = (np.float32(1.0) / (((self._fConst671 + np.float32(0.16840488)) / self._fConst670) + np.float32(1.0693583))) 
		self._fConst721 = (self._fConst699 + np.float32(53.53615)) 
		self._fConst722 = (np.float32(1.0) / (((self._fConst671 + np.float32(0.51247865)) / self._fConst670) + np.float32(0.6896214))) 
		self._fConst723 = (self._fConst699 + np.float32(7.6217313)) 
		self._fConst724 = (np.float32(1.0) / (((self._fConst671 + np.float32(0.78241307)) / self._fConst670) + np.float32(0.2452915))) 
		self._fConst725 = (np.float32(0.0001) / self._fConst673) 
		self._fConst726 = (self._fConst725 + np.float32(0.0004332272)) 
		self._fConst727 = (((self._fConst671 + np.float32(-0.78241307)) / self._fConst670) + np.float32(0.2452915)) 
		self._fConst728 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst699)) 
		self._fConst729 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst725)) 
		self._fConst730 = (((self._fConst671 + np.float32(-0.51247865)) / self._fConst670) + np.float32(0.6896214)) 
		self._fConst731 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst699)) 
		self._fConst732 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst699)) 
		self._fConst733 = (((self._fConst671 + np.float32(-0.16840488)) / self._fConst670) + np.float32(1.0693583)) 
		self._fConst734 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst699)) 
		self._fConst735 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst699)) 
		self._fConst736 = (((self._fConst709 + np.float32(-3.1897273)) / self._fConst708) + np.float32(4.0767817)) 
		self._fConst737 = (np.float32(1.0) / self._fConst711) 
		self._fConst738 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst737)) 
		self._fConst739 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst718)) 
		self._fConst740 = (((self._fConst709 + np.float32(-0.74313045)) / self._fConst708) + np.float32(1.4500711)) 
		self._fConst741 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst737)) 
		self._fConst742 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst715)) 
		self._fConst743 = (((self._fConst709 + np.float32(-0.15748216)) / self._fConst708) + np.float32(0.9351402)) 
		self._fConst744 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst737)) 
		self._fConst745 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst712)) 
		self._fConst746 = np.tan((np.float32(494.76984) / self._fConst0)) 
		self._fConst747 = (np.float32(1.0) / self._fConst746) 
		self._fConst748 = (np.float32(1.0) / (((self._fConst747 + np.float32(0.15748216)) / self._fConst746) + np.float32(0.9351402))) 
		self._fConst749 = np.power(self._fConst746, np.float32(2.0)) 
		self._fConst750 = (np.float32(50.06381) / self._fConst749) 
		self._fConst751 = (self._fConst750 + np.float32(0.9351402)) 
		self._fConst752 = (np.float32(1.0) / (((self._fConst747 + np.float32(0.74313045)) / self._fConst746) + np.float32(1.4500711))) 
		self._fConst753 = (np.float32(11.0520525) / self._fConst749) 
		self._fConst754 = (self._fConst753 + np.float32(1.4500711)) 
		self._fConst755 = (np.float32(1.0) / (((self._fConst747 + np.float32(3.1897273)) / self._fConst746) + np.float32(4.0767817))) 
		self._fConst756 = (np.float32(0.0017661728) / self._fConst749) 
		self._fConst757 = (self._fConst756 + np.float32(0.0004076782)) 
		self._fConst758 = (np.float32(1.0) / (((self._fConst709 + np.float32(0.16840488)) / self._fConst708) + np.float32(1.0693583))) 
		self._fConst759 = (self._fConst737 + np.float32(53.53615)) 
		self._fConst760 = (np.float32(1.0) / (((self._fConst709 + np.float32(0.51247865)) / self._fConst708) + np.float32(0.6896214))) 
		self._fConst761 = (self._fConst737 + np.float32(7.6217313)) 
		self._fConst762 = (np.float32(1.0) / (((self._fConst709 + np.float32(0.78241307)) / self._fConst708) + np.float32(0.2452915))) 
		self._fConst763 = (np.float32(0.0001) / self._fConst711) 
		self._fConst764 = (self._fConst763 + np.float32(0.0004332272)) 
		self._fConst765 = (((self._fConst709 + np.float32(-0.78241307)) / self._fConst708) + np.float32(0.2452915)) 
		self._fConst766 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst737)) 
		self._fConst767 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst763)) 
		self._fConst768 = (((self._fConst709 + np.float32(-0.51247865)) / self._fConst708) + np.float32(0.6896214)) 
		self._fConst769 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst737)) 
		self._fConst770 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst737)) 
		self._fConst771 = (((self._fConst709 + np.float32(-0.16840488)) / self._fConst708) + np.float32(1.0693583)) 
		self._fConst772 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst737)) 
		self._fConst773 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst737)) 
		self._fConst774 = (((self._fConst747 + np.float32(-3.1897273)) / self._fConst746) + np.float32(4.0767817)) 
		self._fConst775 = (np.float32(1.0) / self._fConst749) 
		self._fConst776 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst775)) 
		self._fConst777 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst756)) 
		self._fConst778 = (((self._fConst747 + np.float32(-0.74313045)) / self._fConst746) + np.float32(1.4500711)) 
		self._fConst779 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst775)) 
		self._fConst780 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst753)) 
		self._fConst781 = (((self._fConst747 + np.float32(-0.15748216)) / self._fConst746) + np.float32(0.9351402)) 
		self._fConst782 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst775)) 
		self._fConst783 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst750)) 
		self._fConst784 = np.tan((np.float32(392.6991) / self._fConst0)) 
		self._fConst785 = (np.float32(1.0) / self._fConst784) 
		self._fConst786 = (np.float32(1.0) / (((self._fConst785 + np.float32(0.15748216)) / self._fConst784) + np.float32(0.9351402))) 
		self._fConst787 = np.power(self._fConst784, np.float32(2.0)) 
		self._fConst788 = (np.float32(50.06381) / self._fConst787) 
		self._fConst789 = (self._fConst788 + np.float32(0.9351402)) 
		self._fConst790 = (np.float32(1.0) / (((self._fConst785 + np.float32(0.74313045)) / self._fConst784) + np.float32(1.4500711))) 
		self._fConst791 = (np.float32(11.0520525) / self._fConst787) 
		self._fConst792 = (self._fConst791 + np.float32(1.4500711)) 
		self._fConst793 = (np.float32(1.0) / (((self._fConst785 + np.float32(3.1897273)) / self._fConst784) + np.float32(4.0767817))) 
		self._fConst794 = (np.float32(0.0017661728) / self._fConst787) 
		self._fConst795 = (self._fConst794 + np.float32(0.0004076782)) 
		self._fConst796 = (np.float32(1.0) / (((self._fConst747 + np.float32(0.16840488)) / self._fConst746) + np.float32(1.0693583))) 
		self._fConst797 = (self._fConst775 + np.float32(53.53615)) 
		self._fConst798 = (np.float32(1.0) / (((self._fConst747 + np.float32(0.51247865)) / self._fConst746) + np.float32(0.6896214))) 
		self._fConst799 = (self._fConst775 + np.float32(7.6217313)) 
		self._fConst800 = (np.float32(1.0) / (((self._fConst747 + np.float32(0.78241307)) / self._fConst746) + np.float32(0.2452915))) 
		self._fConst801 = (np.float32(0.0001) / self._fConst749) 
		self._fConst802 = (self._fConst801 + np.float32(0.0004332272)) 
		self._fConst803 = (((self._fConst747 + np.float32(-0.78241307)) / self._fConst746) + np.float32(0.2452915)) 
		self._fConst804 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst775)) 
		self._fConst805 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst801)) 
		self._fConst806 = (((self._fConst747 + np.float32(-0.51247865)) / self._fConst746) + np.float32(0.6896214)) 
		self._fConst807 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst775)) 
		self._fConst808 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst775)) 
		self._fConst809 = (((self._fConst747 + np.float32(-0.16840488)) / self._fConst746) + np.float32(1.0693583)) 
		self._fConst810 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst775)) 
		self._fConst811 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst775)) 
		self._fConst812 = (((self._fConst785 + np.float32(-3.1897273)) / self._fConst784) + np.float32(4.0767817)) 
		self._fConst813 = (np.float32(1.0) / self._fConst787) 
		self._fConst814 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst813)) 
		self._fConst815 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst794)) 
		self._fConst816 = (((self._fConst785 + np.float32(-0.74313045)) / self._fConst784) + np.float32(1.4500711)) 
		self._fConst817 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst813)) 
		self._fConst818 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst791)) 
		self._fConst819 = (((self._fConst785 + np.float32(-0.15748216)) / self._fConst784) + np.float32(0.9351402)) 
		self._fConst820 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst813)) 
		self._fConst821 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst788)) 
		self._fConst822 = np.tan((np.float32(311.68546) / self._fConst0)) 
		self._fConst823 = (np.float32(1.0) / self._fConst822) 
		self._fConst824 = (np.float32(1.0) / (((self._fConst823 + np.float32(0.15748216)) / self._fConst822) + np.float32(0.9351402))) 
		self._fConst825 = np.power(self._fConst822, np.float32(2.0)) 
		self._fConst826 = (np.float32(50.06381) / self._fConst825) 
		self._fConst827 = (self._fConst826 + np.float32(0.9351402)) 
		self._fConst828 = (np.float32(1.0) / (((self._fConst823 + np.float32(0.74313045)) / self._fConst822) + np.float32(1.4500711))) 
		self._fConst829 = (np.float32(11.0520525) / self._fConst825) 
		self._fConst830 = (self._fConst829 + np.float32(1.4500711)) 
		self._fConst831 = (np.float32(1.0) / (((self._fConst823 + np.float32(3.1897273)) / self._fConst822) + np.float32(4.0767817))) 
		self._fConst832 = (np.float32(0.0017661728) / self._fConst825) 
		self._fConst833 = (self._fConst832 + np.float32(0.0004076782)) 
		self._fConst834 = (np.float32(1.0) / (((self._fConst785 + np.float32(0.16840488)) / self._fConst784) + np.float32(1.0693583))) 
		self._fConst835 = (self._fConst813 + np.float32(53.53615)) 
		self._fConst836 = (np.float32(1.0) / (((self._fConst785 + np.float32(0.51247865)) / self._fConst784) + np.float32(0.6896214))) 
		self._fConst837 = (self._fConst813 + np.float32(7.6217313)) 
		self._fConst838 = (np.float32(1.0) / (((self._fConst785 + np.float32(0.78241307)) / self._fConst784) + np.float32(0.2452915))) 
		self._fConst839 = (np.float32(0.0001) / self._fConst787) 
		self._fConst840 = (self._fConst839 + np.float32(0.0004332272)) 
		self._fConst841 = (((self._fConst785 + np.float32(-0.78241307)) / self._fConst784) + np.float32(0.2452915)) 
		self._fConst842 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst813)) 
		self._fConst843 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst839)) 
		self._fConst844 = (((self._fConst785 + np.float32(-0.51247865)) / self._fConst784) + np.float32(0.6896214)) 
		self._fConst845 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst813)) 
		self._fConst846 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst813)) 
		self._fConst847 = (((self._fConst785 + np.float32(-0.16840488)) / self._fConst784) + np.float32(1.0693583)) 
		self._fConst848 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst813)) 
		self._fConst849 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst813)) 
		self._fConst850 = (((self._fConst823 + np.float32(-3.1897273)) / self._fConst822) + np.float32(4.0767817)) 
		self._fConst851 = (np.float32(1.0) / self._fConst825) 
		self._fConst852 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst851)) 
		self._fConst853 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst832)) 
		self._fConst854 = (((self._fConst823 + np.float32(-0.74313045)) / self._fConst822) + np.float32(1.4500711)) 
		self._fConst855 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst851)) 
		self._fConst856 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst829)) 
		self._fConst857 = (((self._fConst823 + np.float32(-0.15748216)) / self._fConst822) + np.float32(0.9351402)) 
		self._fConst858 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst851)) 
		self._fConst859 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst826)) 
		self._fConst860 = np.tan((np.float32(247.38492) / self._fConst0)) 
		self._fConst861 = (np.float32(1.0) / self._fConst860) 
		self._fConst862 = (np.float32(1.0) / (((self._fConst861 + np.float32(0.15748216)) / self._fConst860) + np.float32(0.9351402))) 
		self._fConst863 = np.power(self._fConst860, np.float32(2.0)) 
		self._fConst864 = (np.float32(50.06381) / self._fConst863) 
		self._fConst865 = (self._fConst864 + np.float32(0.9351402)) 
		self._fConst866 = (np.float32(1.0) / (((self._fConst861 + np.float32(0.74313045)) / self._fConst860) + np.float32(1.4500711))) 
		self._fConst867 = (np.float32(11.0520525) / self._fConst863) 
		self._fConst868 = (self._fConst867 + np.float32(1.4500711)) 
		self._fConst869 = (np.float32(1.0) / (((self._fConst861 + np.float32(3.1897273)) / self._fConst860) + np.float32(4.0767817))) 
		self._fConst870 = (np.float32(0.0017661728) / self._fConst863) 
		self._fConst871 = (self._fConst870 + np.float32(0.0004076782)) 
		self._fConst872 = (np.float32(1.0) / (((self._fConst823 + np.float32(0.16840488)) / self._fConst822) + np.float32(1.0693583))) 
		self._fConst873 = (self._fConst851 + np.float32(53.53615)) 
		self._fConst874 = (np.float32(1.0) / (((self._fConst823 + np.float32(0.51247865)) / self._fConst822) + np.float32(0.6896214))) 
		self._fConst875 = (self._fConst851 + np.float32(7.6217313)) 
		self._fConst876 = (np.float32(1.0) / (((self._fConst823 + np.float32(0.78241307)) / self._fConst822) + np.float32(0.2452915))) 
		self._fConst877 = (np.float32(0.0001) / self._fConst825) 
		self._fConst878 = (self._fConst877 + np.float32(0.0004332272)) 
		self._fConst879 = (((self._fConst823 + np.float32(-0.78241307)) / self._fConst822) + np.float32(0.2452915)) 
		self._fConst880 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst851)) 
		self._fConst881 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst877)) 
		self._fConst882 = (((self._fConst823 + np.float32(-0.51247865)) / self._fConst822) + np.float32(0.6896214)) 
		self._fConst883 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst851)) 
		self._fConst884 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst851)) 
		self._fConst885 = (((self._fConst823 + np.float32(-0.16840488)) / self._fConst822) + np.float32(1.0693583)) 
		self._fConst886 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst851)) 
		self._fConst887 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst851)) 
		self._fConst888 = (((self._fConst861 + np.float32(-3.1897273)) / self._fConst860) + np.float32(4.0767817)) 
		self._fConst889 = (np.float32(1.0) / self._fConst863) 
		self._fConst890 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst889)) 
		self._fConst891 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst870)) 
		self._fConst892 = (((self._fConst861 + np.float32(-0.74313045)) / self._fConst860) + np.float32(1.4500711)) 
		self._fConst893 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst889)) 
		self._fConst894 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst867)) 
		self._fConst895 = (((self._fConst861 + np.float32(-0.15748216)) / self._fConst860) + np.float32(0.9351402)) 
		self._fConst896 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst889)) 
		self._fConst897 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst864)) 
		self._fConst898 = np.tan((np.float32(196.34955) / self._fConst0)) 
		self._fConst899 = (np.float32(1.0) / self._fConst898) 
		self._fConst900 = (np.float32(1.0) / (((self._fConst899 + np.float32(0.15748216)) / self._fConst898) + np.float32(0.9351402))) 
		self._fConst901 = np.power(self._fConst898, np.float32(2.0)) 
		self._fConst902 = (np.float32(50.06381) / self._fConst901) 
		self._fConst903 = (self._fConst902 + np.float32(0.9351402)) 
		self._fConst904 = (np.float32(1.0) / (((self._fConst899 + np.float32(0.74313045)) / self._fConst898) + np.float32(1.4500711))) 
		self._fConst905 = (np.float32(11.0520525) / self._fConst901) 
		self._fConst906 = (self._fConst905 + np.float32(1.4500711)) 
		self._fConst907 = (np.float32(1.0) / (((self._fConst899 + np.float32(3.1897273)) / self._fConst898) + np.float32(4.0767817))) 
		self._fConst908 = (np.float32(0.0017661728) / self._fConst901) 
		self._fConst909 = (self._fConst908 + np.float32(0.0004076782)) 
		self._fConst910 = (np.float32(1.0) / (((self._fConst861 + np.float32(0.16840488)) / self._fConst860) + np.float32(1.0693583))) 
		self._fConst911 = (self._fConst889 + np.float32(53.53615)) 
		self._fConst912 = (np.float32(1.0) / (((self._fConst861 + np.float32(0.51247865)) / self._fConst860) + np.float32(0.6896214))) 
		self._fConst913 = (self._fConst889 + np.float32(7.6217313)) 
		self._fConst914 = (np.float32(1.0) / (((self._fConst861 + np.float32(0.78241307)) / self._fConst860) + np.float32(0.2452915))) 
		self._fConst915 = (np.float32(0.0001) / self._fConst863) 
		self._fConst916 = (self._fConst915 + np.float32(0.0004332272)) 
		self._fConst917 = (((self._fConst861 + np.float32(-0.78241307)) / self._fConst860) + np.float32(0.2452915)) 
		self._fConst918 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst889)) 
		self._fConst919 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst915)) 
		self._fConst920 = (((self._fConst861 + np.float32(-0.51247865)) / self._fConst860) + np.float32(0.6896214)) 
		self._fConst921 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst889)) 
		self._fConst922 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst889)) 
		self._fConst923 = (((self._fConst861 + np.float32(-0.16840488)) / self._fConst860) + np.float32(1.0693583)) 
		self._fConst924 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst889)) 
		self._fConst925 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst889)) 
		self._fConst926 = (((self._fConst899 + np.float32(-3.1897273)) / self._fConst898) + np.float32(4.0767817)) 
		self._fConst927 = (np.float32(1.0) / self._fConst901) 
		self._fConst928 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst927)) 
		self._fConst929 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst908)) 
		self._fConst930 = (((self._fConst899 + np.float32(-0.74313045)) / self._fConst898) + np.float32(1.4500711)) 
		self._fConst931 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst927)) 
		self._fConst932 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst905)) 
		self._fConst933 = (((self._fConst899 + np.float32(-0.15748216)) / self._fConst898) + np.float32(0.9351402)) 
		self._fConst934 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst927)) 
		self._fConst935 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst902)) 
		self._fConst936 = np.tan((np.float32(155.84273) / self._fConst0)) 
		self._fConst937 = (np.float32(1.0) / self._fConst936) 
		self._fConst938 = (np.float32(1.0) / (((self._fConst937 + np.float32(0.15748216)) / self._fConst936) + np.float32(0.9351402))) 
		self._fConst939 = np.power(self._fConst936, np.float32(2.0)) 
		self._fConst940 = (np.float32(50.06381) / self._fConst939) 
		self._fConst941 = (self._fConst940 + np.float32(0.9351402)) 
		self._fConst942 = (np.float32(1.0) / (((self._fConst937 + np.float32(0.74313045)) / self._fConst936) + np.float32(1.4500711))) 
		self._fConst943 = (np.float32(11.0520525) / self._fConst939) 
		self._fConst944 = (self._fConst943 + np.float32(1.4500711)) 
		self._fConst945 = (np.float32(1.0) / (((self._fConst937 + np.float32(3.1897273)) / self._fConst936) + np.float32(4.0767817))) 
		self._fConst946 = (np.float32(0.0017661728) / self._fConst939) 
		self._fConst947 = (self._fConst946 + np.float32(0.0004076782)) 
		self._fConst948 = (np.float32(1.0) / (((self._fConst899 + np.float32(0.16840488)) / self._fConst898) + np.float32(1.0693583))) 
		self._fConst949 = (self._fConst927 + np.float32(53.53615)) 
		self._fConst950 = (np.float32(1.0) / (((self._fConst899 + np.float32(0.51247865)) / self._fConst898) + np.float32(0.6896214))) 
		self._fConst951 = (self._fConst927 + np.float32(7.6217313)) 
		self._fConst952 = (np.float32(1.0) / (((self._fConst899 + np.float32(0.78241307)) / self._fConst898) + np.float32(0.2452915))) 
		self._fConst953 = (np.float32(0.0001) / self._fConst901) 
		self._fConst954 = (self._fConst953 + np.float32(0.0004332272)) 
		self._fConst955 = (((self._fConst899 + np.float32(-0.78241307)) / self._fConst898) + np.float32(0.2452915)) 
		self._fConst956 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst927)) 
		self._fConst957 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst953)) 
		self._fConst958 = (((self._fConst899 + np.float32(-0.51247865)) / self._fConst898) + np.float32(0.6896214)) 
		self._fConst959 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst927)) 
		self._fConst960 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst927)) 
		self._fConst961 = (((self._fConst899 + np.float32(-0.16840488)) / self._fConst898) + np.float32(1.0693583)) 
		self._fConst962 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst927)) 
		self._fConst963 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst927)) 
		self._fConst964 = (((self._fConst937 + np.float32(-3.1897273)) / self._fConst936) + np.float32(4.0767817)) 
		self._fConst965 = (np.float32(1.0) / self._fConst939) 
		self._fConst966 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst965)) 
		self._fConst967 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst946)) 
		self._fConst968 = (((self._fConst937 + np.float32(-0.74313045)) / self._fConst936) + np.float32(1.4500711)) 
		self._fConst969 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst965)) 
		self._fConst970 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst943)) 
		self._fConst971 = (((self._fConst937 + np.float32(-0.15748216)) / self._fConst936) + np.float32(0.9351402)) 
		self._fConst972 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst965)) 
		self._fConst973 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst940)) 
		self._fConst974 = np.tan((np.float32(123.69246) / self._fConst0)) 
		self._fConst975 = (np.float32(1.0) / self._fConst974) 
		self._fConst976 = (np.float32(1.0) / (((self._fConst975 + np.float32(0.15748216)) / self._fConst974) + np.float32(0.9351402))) 
		self._fConst977 = np.power(self._fConst974, np.float32(2.0)) 
		self._fConst978 = (np.float32(50.06381) / self._fConst977) 
		self._fConst979 = (self._fConst978 + np.float32(0.9351402)) 
		self._fConst980 = (np.float32(1.0) / (((self._fConst975 + np.float32(0.74313045)) / self._fConst974) + np.float32(1.4500711))) 
		self._fConst981 = (np.float32(11.0520525) / self._fConst977) 
		self._fConst982 = (self._fConst981 + np.float32(1.4500711)) 
		self._fConst983 = (np.float32(1.0) / (((self._fConst975 + np.float32(3.1897273)) / self._fConst974) + np.float32(4.0767817))) 
		self._fConst984 = (np.float32(0.0017661728) / self._fConst977) 
		self._fConst985 = (self._fConst984 + np.float32(0.0004076782)) 
		self._fConst986 = (np.float32(1.0) / (((self._fConst937 + np.float32(0.16840488)) / self._fConst936) + np.float32(1.0693583))) 
		self._fConst987 = (self._fConst965 + np.float32(53.53615)) 
		self._fConst988 = (np.float32(1.0) / (((self._fConst937 + np.float32(0.51247865)) / self._fConst936) + np.float32(0.6896214))) 
		self._fConst989 = (self._fConst965 + np.float32(7.6217313)) 
		self._fConst990 = (np.float32(1.0) / (((self._fConst937 + np.float32(0.78241307)) / self._fConst936) + np.float32(0.2452915))) 
		self._fConst991 = (np.float32(0.0001) / self._fConst939) 
		self._fConst992 = (self._fConst991 + np.float32(0.0004332272)) 
		self._fConst993 = (((self._fConst937 + np.float32(-0.78241307)) / self._fConst936) + np.float32(0.2452915)) 
		self._fConst994 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst965)) 
		self._fConst995 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst991)) 
		self._fConst996 = (((self._fConst937 + np.float32(-0.51247865)) / self._fConst936) + np.float32(0.6896214)) 
		self._fConst997 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst965)) 
		self._fConst998 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst965)) 
		self._fConst999 = (((self._fConst937 + np.float32(-0.16840488)) / self._fConst936) + np.float32(1.0693583)) 
		self._fConst1000 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst965)) 
		self._fConst1001 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst965)) 
		self._fConst1002 = (((self._fConst975 + np.float32(-3.1897273)) / self._fConst974) + np.float32(4.0767817)) 
		self._fConst1003 = (np.float32(1.0) / self._fConst977) 
		self._fConst1004 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst1003)) 
		self._fConst1005 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst984)) 
		self._fConst1006 = (((self._fConst975 + np.float32(-0.74313045)) / self._fConst974) + np.float32(1.4500711)) 
		self._fConst1007 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst1003)) 
		self._fConst1008 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst981)) 
		self._fConst1009 = (((self._fConst975 + np.float32(-0.15748216)) / self._fConst974) + np.float32(0.9351402)) 
		self._fConst1010 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst1003)) 
		self._fConst1011 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst978)) 
		self._fConst1012 = np.tan((np.float32(98.174774) / self._fConst0)) 
		self._fConst1013 = (np.float32(1.0) / self._fConst1012) 
		self._fConst1014 = (np.float32(1.0) / (((self._fConst1013 + np.float32(0.15748216)) / self._fConst1012) + np.float32(0.9351402))) 
		self._fConst1015 = np.power(self._fConst1012, np.float32(2.0)) 
		self._fConst1016 = (np.float32(50.06381) / self._fConst1015) 
		self._fConst1017 = (self._fConst1016 + np.float32(0.9351402)) 
		self._fConst1018 = (np.float32(1.0) / (((self._fConst1013 + np.float32(0.74313045)) / self._fConst1012) + np.float32(1.4500711))) 
		self._fConst1019 = (np.float32(11.0520525) / self._fConst1015) 
		self._fConst1020 = (self._fConst1019 + np.float32(1.4500711)) 
		self._fConst1021 = (np.float32(1.0) / (((self._fConst1013 + np.float32(3.1897273)) / self._fConst1012) + np.float32(4.0767817))) 
		self._fConst1022 = (np.float32(0.0017661728) / self._fConst1015) 
		self._fConst1023 = (self._fConst1022 + np.float32(0.0004076782)) 
		self._fConst1024 = (np.float32(1.0) / (((self._fConst975 + np.float32(0.16840488)) / self._fConst974) + np.float32(1.0693583))) 
		self._fConst1025 = (self._fConst1003 + np.float32(53.53615)) 
		self._fConst1026 = (np.float32(1.0) / (((self._fConst975 + np.float32(0.51247865)) / self._fConst974) + np.float32(0.6896214))) 
		self._fConst1027 = (self._fConst1003 + np.float32(7.6217313)) 
		self._fConst1028 = (np.float32(1.0) / (((self._fConst975 + np.float32(0.78241307)) / self._fConst974) + np.float32(0.2452915))) 
		self._fConst1029 = (np.float32(0.0001) / self._fConst977) 
		self._fConst1030 = (self._fConst1029 + np.float32(0.0004332272)) 
		self._fConst1031 = (((self._fConst975 + np.float32(-0.78241307)) / self._fConst974) + np.float32(0.2452915)) 
		self._fConst1032 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst1003)) 
		self._fConst1033 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst1029)) 
		self._fConst1034 = (((self._fConst975 + np.float32(-0.51247865)) / self._fConst974) + np.float32(0.6896214)) 
		self._fConst1035 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst1003)) 
		self._fConst1036 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst1003)) 
		self._fConst1037 = (((self._fConst975 + np.float32(-0.16840488)) / self._fConst974) + np.float32(1.0693583)) 
		self._fConst1038 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst1003)) 
		self._fConst1039 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst1003)) 
		self._fConst1040 = (((self._fConst1013 + np.float32(-3.1897273)) / self._fConst1012) + np.float32(4.0767817)) 
		self._fConst1041 = (np.float32(1.0) / self._fConst1015) 
		self._fConst1042 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst1041)) 
		self._fConst1043 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst1022)) 
		self._fConst1044 = (((self._fConst1013 + np.float32(-0.74313045)) / self._fConst1012) + np.float32(1.4500711)) 
		self._fConst1045 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst1041)) 
		self._fConst1046 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst1019)) 
		self._fConst1047 = (((self._fConst1013 + np.float32(-0.15748216)) / self._fConst1012) + np.float32(0.9351402)) 
		self._fConst1048 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst1041)) 
		self._fConst1049 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst1016)) 
		self._fConst1050 = np.tan((np.float32(77.921364) / self._fConst0)) 
		self._fConst1051 = (np.float32(1.0) / self._fConst1050) 
		self._fConst1052 = (np.float32(1.0) / (((self._fConst1051 + np.float32(0.15748216)) / self._fConst1050) + np.float32(0.9351402))) 
		self._fConst1053 = np.power(self._fConst1050, np.float32(2.0)) 
		self._fConst1054 = (np.float32(50.06381) / self._fConst1053) 
		self._fConst1055 = (self._fConst1054 + np.float32(0.9351402)) 
		self._fConst1056 = (np.float32(1.0) / (((self._fConst1051 + np.float32(0.74313045)) / self._fConst1050) + np.float32(1.4500711))) 
		self._fConst1057 = (np.float32(11.0520525) / self._fConst1053) 
		self._fConst1058 = (self._fConst1057 + np.float32(1.4500711)) 
		self._fConst1059 = (np.float32(1.0) / (((self._fConst1051 + np.float32(3.1897273)) / self._fConst1050) + np.float32(4.0767817))) 
		self._fConst1060 = (np.float32(0.0017661728) / self._fConst1053) 
		self._fConst1061 = (self._fConst1060 + np.float32(0.0004076782)) 
		self._fConst1062 = (np.float32(1.0) / (((self._fConst1013 + np.float32(0.16840488)) / self._fConst1012) + np.float32(1.0693583))) 
		self._fConst1063 = (self._fConst1041 + np.float32(53.53615)) 
		self._fConst1064 = (np.float32(1.0) / (((self._fConst1013 + np.float32(0.51247865)) / self._fConst1012) + np.float32(0.6896214))) 
		self._fConst1065 = (self._fConst1041 + np.float32(7.6217313)) 
		self._fConst1066 = (np.float32(1.0) / (((self._fConst1013 + np.float32(0.78241307)) / self._fConst1012) + np.float32(0.2452915))) 
		self._fConst1067 = (np.float32(0.0001) / self._fConst1015) 
		self._fConst1068 = (self._fConst1067 + np.float32(0.0004332272)) 
		self._fConst1069 = (((self._fConst1013 + np.float32(-0.78241307)) / self._fConst1012) + np.float32(0.2452915)) 
		self._fConst1070 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst1041)) 
		self._fConst1071 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst1067)) 
		self._fConst1072 = (((self._fConst1013 + np.float32(-0.51247865)) / self._fConst1012) + np.float32(0.6896214)) 
		self._fConst1073 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst1041)) 
		self._fConst1074 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst1041)) 
		self._fConst1075 = (((self._fConst1013 + np.float32(-0.16840488)) / self._fConst1012) + np.float32(1.0693583)) 
		self._fConst1076 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst1041)) 
		self._fConst1077 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst1041)) 
		self._fConst1078 = (((self._fConst1051 + np.float32(-3.1897273)) / self._fConst1050) + np.float32(4.0767817)) 
		self._fConst1079 = (np.float32(1.0) / self._fConst1053) 
		self._fConst1080 = (np.float32(2.0) * (np.float32(4.0767817) - self._fConst1079)) 
		self._fConst1081 = (np.float32(2.0) * (np.float32(0.0004076782) - self._fConst1060)) 
		self._fConst1082 = (((self._fConst1051 + np.float32(-0.74313045)) / self._fConst1050) + np.float32(1.4500711)) 
		self._fConst1083 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst1079)) 
		self._fConst1084 = (np.float32(2.0) * (np.float32(1.4500711) - self._fConst1057)) 
		self._fConst1085 = (((self._fConst1051 + np.float32(-0.15748216)) / self._fConst1050) + np.float32(0.9351402)) 
		self._fConst1086 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst1079)) 
		self._fConst1087 = (np.float32(2.0) * (np.float32(0.9351402) - self._fConst1054)) 
		self._fConst1088 = (np.float32(1.0) / (((self._fConst1051 + np.float32(0.16840488)) / self._fConst1050) + np.float32(1.0693583))) 
		self._fConst1089 = (self._fConst1079 + np.float32(53.53615)) 
		self._fConst1090 = (np.float32(1.0) / (((self._fConst1051 + np.float32(0.51247865)) / self._fConst1050) + np.float32(0.6896214))) 
		self._fConst1091 = (self._fConst1079 + np.float32(7.6217313)) 
		self._fConst1092 = (np.float32(1.0) / (((self._fConst1051 + np.float32(0.78241307)) / self._fConst1050) + np.float32(0.2452915))) 
		self._fConst1093 = (np.float32(0.0001) / self._fConst1053) 
		self._fConst1094 = (self._fConst1093 + np.float32(0.0004332272)) 
		self._fConst1095 = (((self._fConst1051 + np.float32(-0.78241307)) / self._fConst1050) + np.float32(0.2452915)) 
		self._fConst1096 = (np.float32(2.0) * (np.float32(0.2452915) - self._fConst1079)) 
		self._fConst1097 = (np.float32(2.0) * (np.float32(0.0004332272) - self._fConst1093)) 
		self._fConst1098 = (((self._fConst1051 + np.float32(-0.51247865)) / self._fConst1050) + np.float32(0.6896214)) 
		self._fConst1099 = (np.float32(2.0) * (np.float32(0.6896214) - self._fConst1079)) 
		self._fConst1100 = (np.float32(2.0) * (np.float32(7.6217313) - self._fConst1079)) 
		self._fConst1101 = (((self._fConst1051 + np.float32(-0.16840488)) / self._fConst1050) + np.float32(1.0693583)) 
		self._fConst1102 = (np.float32(2.0) * (np.float32(1.0693583) - self._fConst1079)) 
		self._fConst1103 = (np.float32(2.0) * (np.float32(53.53615) - self._fConst1079)) 
		
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
		state["fRec102"] = np.float32(0)
		state["fRec109"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec116"] = np.float32(0)
		state["fRec123"] = np.float32(0)
		state["fRec130"] = np.float32(0)
		state["fRec137"] = np.float32(0)
		state["fRec144"] = np.float32(0)
		state["fRec151"] = np.float32(0)
		state["fRec158"] = np.float32(0)
		state["fRec165"] = np.float32(0)
		state["fRec172"] = np.float32(0)
		state["fRec179"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec186"] = np.float32(0)
		state["fRec193"] = np.float32(0)
		state["fRec200"] = np.float32(0)
		state["fRec25"] = np.float32(0)
		state["fRec32"] = np.float32(0)
		state["fRec39"] = np.float32(0)
		state["fRec4"] = np.float32(0)
		state["fRec46"] = np.float32(0)
		state["fRec53"] = np.float32(0)
		state["fRec60"] = np.float32(0)
		state["fRec67"] = np.float32(0)
		state["fRec74"] = np.float32(0)
		state["fRec81"] = np.float32(0)
		state["fRec88"] = np.float32(0)
		state["fRec95"] = np.float32(0)
		# Initialize array delays
		state["fRec3"] = np.zeros((3,), dtype=np.float32)
		state["fRec2"] = np.zeros((3,), dtype=np.float32)
		state["fRec1"] = np.zeros((3,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fRec9"] = np.zeros((3,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fRec7"] = np.zeros((3,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fRec5"] = np.zeros((3,), dtype=np.float32)
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
		state["fRec108"] = np.zeros((3,), dtype=np.float32)
		state["fRec107"] = np.zeros((3,), dtype=np.float32)
		state["fRec106"] = np.zeros((3,), dtype=np.float32)
		state["fRec105"] = np.zeros((3,), dtype=np.float32)
		state["fRec104"] = np.zeros((3,), dtype=np.float32)
		state["fRec103"] = np.zeros((3,), dtype=np.float32)
		state["fRec115"] = np.zeros((3,), dtype=np.float32)
		state["fRec114"] = np.zeros((3,), dtype=np.float32)
		state["fRec113"] = np.zeros((3,), dtype=np.float32)
		state["fRec112"] = np.zeros((3,), dtype=np.float32)
		state["fRec111"] = np.zeros((3,), dtype=np.float32)
		state["fRec110"] = np.zeros((3,), dtype=np.float32)
		state["fRec122"] = np.zeros((3,), dtype=np.float32)
		state["fRec121"] = np.zeros((3,), dtype=np.float32)
		state["fRec120"] = np.zeros((3,), dtype=np.float32)
		state["fRec119"] = np.zeros((3,), dtype=np.float32)
		state["fRec118"] = np.zeros((3,), dtype=np.float32)
		state["fRec117"] = np.zeros((3,), dtype=np.float32)
		state["fRec129"] = np.zeros((3,), dtype=np.float32)
		state["fRec128"] = np.zeros((3,), dtype=np.float32)
		state["fRec127"] = np.zeros((3,), dtype=np.float32)
		state["fRec126"] = np.zeros((3,), dtype=np.float32)
		state["fRec125"] = np.zeros((3,), dtype=np.float32)
		state["fRec124"] = np.zeros((3,), dtype=np.float32)
		state["fRec136"] = np.zeros((3,), dtype=np.float32)
		state["fRec135"] = np.zeros((3,), dtype=np.float32)
		state["fRec134"] = np.zeros((3,), dtype=np.float32)
		state["fRec133"] = np.zeros((3,), dtype=np.float32)
		state["fRec132"] = np.zeros((3,), dtype=np.float32)
		state["fRec131"] = np.zeros((3,), dtype=np.float32)
		state["fRec143"] = np.zeros((3,), dtype=np.float32)
		state["fRec142"] = np.zeros((3,), dtype=np.float32)
		state["fRec141"] = np.zeros((3,), dtype=np.float32)
		state["fRec140"] = np.zeros((3,), dtype=np.float32)
		state["fRec139"] = np.zeros((3,), dtype=np.float32)
		state["fRec138"] = np.zeros((3,), dtype=np.float32)
		state["fRec150"] = np.zeros((3,), dtype=np.float32)
		state["fRec149"] = np.zeros((3,), dtype=np.float32)
		state["fRec148"] = np.zeros((3,), dtype=np.float32)
		state["fRec147"] = np.zeros((3,), dtype=np.float32)
		state["fRec146"] = np.zeros((3,), dtype=np.float32)
		state["fRec145"] = np.zeros((3,), dtype=np.float32)
		state["fRec157"] = np.zeros((3,), dtype=np.float32)
		state["fRec156"] = np.zeros((3,), dtype=np.float32)
		state["fRec155"] = np.zeros((3,), dtype=np.float32)
		state["fRec154"] = np.zeros((3,), dtype=np.float32)
		state["fRec153"] = np.zeros((3,), dtype=np.float32)
		state["fRec152"] = np.zeros((3,), dtype=np.float32)
		state["fRec164"] = np.zeros((3,), dtype=np.float32)
		state["fRec163"] = np.zeros((3,), dtype=np.float32)
		state["fRec162"] = np.zeros((3,), dtype=np.float32)
		state["fRec161"] = np.zeros((3,), dtype=np.float32)
		state["fRec160"] = np.zeros((3,), dtype=np.float32)
		state["fRec159"] = np.zeros((3,), dtype=np.float32)
		state["fRec171"] = np.zeros((3,), dtype=np.float32)
		state["fRec170"] = np.zeros((3,), dtype=np.float32)
		state["fRec169"] = np.zeros((3,), dtype=np.float32)
		state["fRec168"] = np.zeros((3,), dtype=np.float32)
		state["fRec167"] = np.zeros((3,), dtype=np.float32)
		state["fRec166"] = np.zeros((3,), dtype=np.float32)
		state["fRec178"] = np.zeros((3,), dtype=np.float32)
		state["fRec177"] = np.zeros((3,), dtype=np.float32)
		state["fRec176"] = np.zeros((3,), dtype=np.float32)
		state["fRec175"] = np.zeros((3,), dtype=np.float32)
		state["fRec174"] = np.zeros((3,), dtype=np.float32)
		state["fRec173"] = np.zeros((3,), dtype=np.float32)
		state["fRec185"] = np.zeros((3,), dtype=np.float32)
		state["fRec184"] = np.zeros((3,), dtype=np.float32)
		state["fRec183"] = np.zeros((3,), dtype=np.float32)
		state["fRec182"] = np.zeros((3,), dtype=np.float32)
		state["fRec181"] = np.zeros((3,), dtype=np.float32)
		state["fRec180"] = np.zeros((3,), dtype=np.float32)
		state["fRec192"] = np.zeros((3,), dtype=np.float32)
		state["fRec191"] = np.zeros((3,), dtype=np.float32)
		state["fRec190"] = np.zeros((3,), dtype=np.float32)
		state["fRec189"] = np.zeros((3,), dtype=np.float32)
		state["fRec188"] = np.zeros((3,), dtype=np.float32)
		state["fRec187"] = np.zeros((3,), dtype=np.float32)
		state["fRec199"] = np.zeros((3,), dtype=np.float32)
		state["fRec198"] = np.zeros((3,), dtype=np.float32)
		state["fRec197"] = np.zeros((3,), dtype=np.float32)
		state["fRec196"] = np.zeros((3,), dtype=np.float32)
		state["fRec195"] = np.zeros((3,), dtype=np.float32)
		state["fRec194"] = np.zeros((3,), dtype=np.float32)
		state["fRec203"] = np.zeros((3,), dtype=np.float32)
		state["fRec202"] = np.zeros((3,), dtype=np.float32)
		state["fRec201"] = np.zeros((3,), dtype=np.float32)
		# Initialize waveform arrays for read-write tables
		return state

	def tick(self, params: dict, state: dict, inputs: jnp.ndarray, rng: jax.Array = None) -> Tuple[dict, jnp.ndarray]:
		
		rngs = nnx.Rngs(rng) if rng is not None else None
		
		fSlow0 = params["fHslider0"] 
		fSlow1 = params["fHslider1"] 
		fSlow2 = jnp.where((((jnp.float32(0.001) * fSlow1) > jnp.float32(0.0)).astype(jnp.int32) != 0), jnp.exp(-((self._fConst1 / fSlow1))), jnp.float32(0.0)) 
		fSlow3 = (jnp.float32(1.0) - fSlow2) 
		fRec0_temp = state["fRec0"] 
		fRec4_temp = state["fRec4"] 
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
		fRec109_temp = state["fRec109"] 
		fRec116_temp = state["fRec116"] 
		fRec123_temp = state["fRec123"] 
		fRec130_temp = state["fRec130"] 
		fRec137_temp = state["fRec137"] 
		fRec144_temp = state["fRec144"] 
		fRec151_temp = state["fRec151"] 
		fRec158_temp = state["fRec158"] 
		fRec165_temp = state["fRec165"] 
		fRec172_temp = state["fRec172"] 
		fRec179_temp = state["fRec179"] 
		fRec186_temp = state["fRec186"] 
		fRec193_temp = state["fRec193"] 
		fRec200_temp = state["fRec200"] 
		fTemp0 = inputs[0] 
		state["fRec3"] = state["fRec3"].at[0].set((fTemp0 - (self._fConst11 * ((self._fConst14 * state["fRec3"][2]) + (self._fConst16 * state["fRec3"][1]))))) 
		state["fRec2"] = state["fRec2"].at[0].set(((self._fConst11 * (((self._fConst13 * state["fRec3"][0]) + (self._fConst17 * state["fRec3"][1])) + (self._fConst13 * state["fRec3"][2]))) - (self._fConst8 * ((self._fConst18 * state["fRec2"][2]) + (self._fConst19 * state["fRec2"][1]))))) 
		state["fRec1"] = state["fRec1"].at[0].set(((self._fConst8 * (((self._fConst10 * state["fRec2"][0]) + (self._fConst20 * state["fRec2"][1])) + (self._fConst10 * state["fRec2"][2]))) - (self._fConst4 * ((self._fConst21 * state["fRec1"][2]) + (self._fConst22 * state["fRec1"][1]))))) 
		state["fRec0"] = ((fRec0_temp * fSlow2) + (jnp.abs((self._fConst4 * (((self._fConst7 * state["fRec1"][0]) + (self._fConst23 * state["fRec1"][1])) + (self._fConst7 * state["fRec1"][2])))) * fSlow3)) 
		fVbargraph0 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec0"])))
		# self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		state["fRec10"] = state["fRec10"].at[0].set((fTemp0 - (self._fConst40 * ((self._fConst43 * state["fRec10"][2]) + (self._fConst44 * state["fRec10"][1]))))) 
		state["fRec9"] = state["fRec9"].at[0].set(((self._fConst40 * (((self._fConst42 * state["fRec10"][0]) + (self._fConst45 * state["fRec10"][1])) + (self._fConst42 * state["fRec10"][2]))) - (self._fConst38 * ((self._fConst46 * state["fRec9"][2]) + (self._fConst47 * state["fRec9"][1]))))) 
		state["fRec8"] = state["fRec8"].at[0].set(((self._fConst38 * (((self._fConst39 * state["fRec9"][0]) + (self._fConst48 * state["fRec9"][1])) + (self._fConst39 * state["fRec9"][2]))) - (self._fConst36 * ((self._fConst49 * state["fRec8"][2]) + (self._fConst50 * state["fRec8"][1]))))) 
		fTemp1 = (self._fConst36 * (((self._fConst37 * state["fRec8"][0]) + (self._fConst51 * state["fRec8"][1])) + (self._fConst37 * state["fRec8"][2]))) 
		state["fRec7"] = state["fRec7"].at[0].set((fTemp1 - (self._fConst33 * ((self._fConst52 * state["fRec7"][2]) + (self._fConst54 * state["fRec7"][1]))))) 
		state["fRec6"] = state["fRec6"].at[0].set(((self._fConst33 * (((self._fConst35 * state["fRec7"][0]) + (self._fConst55 * state["fRec7"][1])) + (self._fConst35 * state["fRec7"][2]))) - (self._fConst30 * ((self._fConst56 * state["fRec6"][2]) + (self._fConst57 * state["fRec6"][1]))))) 
		state["fRec5"] = state["fRec5"].at[0].set(((self._fConst30 * (((self._fConst32 * state["fRec6"][0]) + (self._fConst58 * state["fRec6"][1])) + (self._fConst32 * state["fRec6"][2]))) - (self._fConst26 * ((self._fConst59 * state["fRec5"][2]) + (self._fConst60 * state["fRec5"][1]))))) 
		state["fRec4"] = ((fSlow2 * fRec4_temp) + (fSlow3 * jnp.abs((self._fConst26 * (((self._fConst29 * state["fRec5"][0]) + (self._fConst61 * state["fRec5"][1])) + (self._fConst29 * state["fRec5"][2])))))) 
		fVbargraph1 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec4"])))
		# self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		state["fRec17"] = state["fRec17"].at[0].set((fTemp1 - (self._fConst78 * ((self._fConst81 * state["fRec17"][2]) + (self._fConst82 * state["fRec17"][1]))))) 
		state["fRec16"] = state["fRec16"].at[0].set(((self._fConst78 * (((self._fConst80 * state["fRec17"][0]) + (self._fConst83 * state["fRec17"][1])) + (self._fConst80 * state["fRec17"][2]))) - (self._fConst76 * ((self._fConst84 * state["fRec16"][2]) + (self._fConst85 * state["fRec16"][1]))))) 
		state["fRec15"] = state["fRec15"].at[0].set(((self._fConst76 * (((self._fConst77 * state["fRec16"][0]) + (self._fConst86 * state["fRec16"][1])) + (self._fConst77 * state["fRec16"][2]))) - (self._fConst74 * ((self._fConst87 * state["fRec15"][2]) + (self._fConst88 * state["fRec15"][1]))))) 
		fTemp2 = (self._fConst74 * (((self._fConst75 * state["fRec15"][0]) + (self._fConst89 * state["fRec15"][1])) + (self._fConst75 * state["fRec15"][2]))) 
		state["fRec14"] = state["fRec14"].at[0].set((fTemp2 - (self._fConst71 * ((self._fConst90 * state["fRec14"][2]) + (self._fConst92 * state["fRec14"][1]))))) 
		state["fRec13"] = state["fRec13"].at[0].set(((self._fConst71 * (((self._fConst73 * state["fRec14"][0]) + (self._fConst93 * state["fRec14"][1])) + (self._fConst73 * state["fRec14"][2]))) - (self._fConst68 * ((self._fConst94 * state["fRec13"][2]) + (self._fConst95 * state["fRec13"][1]))))) 
		state["fRec12"] = state["fRec12"].at[0].set(((self._fConst68 * (((self._fConst70 * state["fRec13"][0]) + (self._fConst96 * state["fRec13"][1])) + (self._fConst70 * state["fRec13"][2]))) - (self._fConst64 * ((self._fConst97 * state["fRec12"][2]) + (self._fConst98 * state["fRec12"][1]))))) 
		state["fRec11"] = ((fSlow2 * fRec11_temp) + (fSlow3 * jnp.abs((self._fConst64 * (((self._fConst67 * state["fRec12"][0]) + (self._fConst99 * state["fRec12"][1])) + (self._fConst67 * state["fRec12"][2])))))) 
		fVbargraph2 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec11"])))
		# self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		state["fRec24"] = state["fRec24"].at[0].set((fTemp2 - (self._fConst116 * ((self._fConst119 * state["fRec24"][2]) + (self._fConst120 * state["fRec24"][1]))))) 
		state["fRec23"] = state["fRec23"].at[0].set(((self._fConst116 * (((self._fConst118 * state["fRec24"][0]) + (self._fConst121 * state["fRec24"][1])) + (self._fConst118 * state["fRec24"][2]))) - (self._fConst114 * ((self._fConst122 * state["fRec23"][2]) + (self._fConst123 * state["fRec23"][1]))))) 
		state["fRec22"] = state["fRec22"].at[0].set(((self._fConst114 * (((self._fConst115 * state["fRec23"][0]) + (self._fConst124 * state["fRec23"][1])) + (self._fConst115 * state["fRec23"][2]))) - (self._fConst112 * ((self._fConst125 * state["fRec22"][2]) + (self._fConst126 * state["fRec22"][1]))))) 
		fTemp3 = (self._fConst112 * (((self._fConst113 * state["fRec22"][0]) + (self._fConst127 * state["fRec22"][1])) + (self._fConst113 * state["fRec22"][2]))) 
		state["fRec21"] = state["fRec21"].at[0].set((fTemp3 - (self._fConst109 * ((self._fConst128 * state["fRec21"][2]) + (self._fConst130 * state["fRec21"][1]))))) 
		state["fRec20"] = state["fRec20"].at[0].set(((self._fConst109 * (((self._fConst111 * state["fRec21"][0]) + (self._fConst131 * state["fRec21"][1])) + (self._fConst111 * state["fRec21"][2]))) - (self._fConst106 * ((self._fConst132 * state["fRec20"][2]) + (self._fConst133 * state["fRec20"][1]))))) 
		state["fRec19"] = state["fRec19"].at[0].set(((self._fConst106 * (((self._fConst108 * state["fRec20"][0]) + (self._fConst134 * state["fRec20"][1])) + (self._fConst108 * state["fRec20"][2]))) - (self._fConst102 * ((self._fConst135 * state["fRec19"][2]) + (self._fConst136 * state["fRec19"][1]))))) 
		state["fRec18"] = ((fSlow2 * fRec18_temp) + (fSlow3 * jnp.abs((self._fConst102 * (((self._fConst105 * state["fRec19"][0]) + (self._fConst137 * state["fRec19"][1])) + (self._fConst105 * state["fRec19"][2])))))) 
		fVbargraph3 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec18"])))
		# self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		state["fRec31"] = state["fRec31"].at[0].set((fTemp3 - (self._fConst154 * ((self._fConst157 * state["fRec31"][2]) + (self._fConst158 * state["fRec31"][1]))))) 
		state["fRec30"] = state["fRec30"].at[0].set(((self._fConst154 * (((self._fConst156 * state["fRec31"][0]) + (self._fConst159 * state["fRec31"][1])) + (self._fConst156 * state["fRec31"][2]))) - (self._fConst152 * ((self._fConst160 * state["fRec30"][2]) + (self._fConst161 * state["fRec30"][1]))))) 
		state["fRec29"] = state["fRec29"].at[0].set(((self._fConst152 * (((self._fConst153 * state["fRec30"][0]) + (self._fConst162 * state["fRec30"][1])) + (self._fConst153 * state["fRec30"][2]))) - (self._fConst150 * ((self._fConst163 * state["fRec29"][2]) + (self._fConst164 * state["fRec29"][1]))))) 
		fTemp4 = (self._fConst150 * (((self._fConst151 * state["fRec29"][0]) + (self._fConst165 * state["fRec29"][1])) + (self._fConst151 * state["fRec29"][2]))) 
		state["fRec28"] = state["fRec28"].at[0].set((fTemp4 - (self._fConst147 * ((self._fConst166 * state["fRec28"][2]) + (self._fConst168 * state["fRec28"][1]))))) 
		state["fRec27"] = state["fRec27"].at[0].set(((self._fConst147 * (((self._fConst149 * state["fRec28"][0]) + (self._fConst169 * state["fRec28"][1])) + (self._fConst149 * state["fRec28"][2]))) - (self._fConst144 * ((self._fConst170 * state["fRec27"][2]) + (self._fConst171 * state["fRec27"][1]))))) 
		state["fRec26"] = state["fRec26"].at[0].set(((self._fConst144 * (((self._fConst146 * state["fRec27"][0]) + (self._fConst172 * state["fRec27"][1])) + (self._fConst146 * state["fRec27"][2]))) - (self._fConst140 * ((self._fConst173 * state["fRec26"][2]) + (self._fConst174 * state["fRec26"][1]))))) 
		state["fRec25"] = ((fSlow2 * fRec25_temp) + (fSlow3 * jnp.abs((self._fConst140 * (((self._fConst143 * state["fRec26"][0]) + (self._fConst175 * state["fRec26"][1])) + (self._fConst143 * state["fRec26"][2])))))) 
		fVbargraph4 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec25"])))
		# self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		state["fRec38"] = state["fRec38"].at[0].set((fTemp4 - (self._fConst192 * ((self._fConst195 * state["fRec38"][2]) + (self._fConst196 * state["fRec38"][1]))))) 
		state["fRec37"] = state["fRec37"].at[0].set(((self._fConst192 * (((self._fConst194 * state["fRec38"][0]) + (self._fConst197 * state["fRec38"][1])) + (self._fConst194 * state["fRec38"][2]))) - (self._fConst190 * ((self._fConst198 * state["fRec37"][2]) + (self._fConst199 * state["fRec37"][1]))))) 
		state["fRec36"] = state["fRec36"].at[0].set(((self._fConst190 * (((self._fConst191 * state["fRec37"][0]) + (self._fConst200 * state["fRec37"][1])) + (self._fConst191 * state["fRec37"][2]))) - (self._fConst188 * ((self._fConst201 * state["fRec36"][2]) + (self._fConst202 * state["fRec36"][1]))))) 
		fTemp5 = (self._fConst188 * (((self._fConst189 * state["fRec36"][0]) + (self._fConst203 * state["fRec36"][1])) + (self._fConst189 * state["fRec36"][2]))) 
		state["fRec35"] = state["fRec35"].at[0].set((fTemp5 - (self._fConst185 * ((self._fConst204 * state["fRec35"][2]) + (self._fConst206 * state["fRec35"][1]))))) 
		state["fRec34"] = state["fRec34"].at[0].set(((self._fConst185 * (((self._fConst187 * state["fRec35"][0]) + (self._fConst207 * state["fRec35"][1])) + (self._fConst187 * state["fRec35"][2]))) - (self._fConst182 * ((self._fConst208 * state["fRec34"][2]) + (self._fConst209 * state["fRec34"][1]))))) 
		state["fRec33"] = state["fRec33"].at[0].set(((self._fConst182 * (((self._fConst184 * state["fRec34"][0]) + (self._fConst210 * state["fRec34"][1])) + (self._fConst184 * state["fRec34"][2]))) - (self._fConst178 * ((self._fConst211 * state["fRec33"][2]) + (self._fConst212 * state["fRec33"][1]))))) 
		state["fRec32"] = ((fSlow2 * fRec32_temp) + (fSlow3 * jnp.abs((self._fConst178 * (((self._fConst181 * state["fRec33"][0]) + (self._fConst213 * state["fRec33"][1])) + (self._fConst181 * state["fRec33"][2])))))) 
		fVbargraph5 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec32"])))
		# self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		state["fRec45"] = state["fRec45"].at[0].set((fTemp5 - (self._fConst230 * ((self._fConst233 * state["fRec45"][2]) + (self._fConst234 * state["fRec45"][1]))))) 
		state["fRec44"] = state["fRec44"].at[0].set(((self._fConst230 * (((self._fConst232 * state["fRec45"][0]) + (self._fConst235 * state["fRec45"][1])) + (self._fConst232 * state["fRec45"][2]))) - (self._fConst228 * ((self._fConst236 * state["fRec44"][2]) + (self._fConst237 * state["fRec44"][1]))))) 
		state["fRec43"] = state["fRec43"].at[0].set(((self._fConst228 * (((self._fConst229 * state["fRec44"][0]) + (self._fConst238 * state["fRec44"][1])) + (self._fConst229 * state["fRec44"][2]))) - (self._fConst226 * ((self._fConst239 * state["fRec43"][2]) + (self._fConst240 * state["fRec43"][1]))))) 
		fTemp6 = (self._fConst226 * (((self._fConst227 * state["fRec43"][0]) + (self._fConst241 * state["fRec43"][1])) + (self._fConst227 * state["fRec43"][2]))) 
		state["fRec42"] = state["fRec42"].at[0].set((fTemp6 - (self._fConst223 * ((self._fConst242 * state["fRec42"][2]) + (self._fConst244 * state["fRec42"][1]))))) 
		state["fRec41"] = state["fRec41"].at[0].set(((self._fConst223 * (((self._fConst225 * state["fRec42"][0]) + (self._fConst245 * state["fRec42"][1])) + (self._fConst225 * state["fRec42"][2]))) - (self._fConst220 * ((self._fConst246 * state["fRec41"][2]) + (self._fConst247 * state["fRec41"][1]))))) 
		state["fRec40"] = state["fRec40"].at[0].set(((self._fConst220 * (((self._fConst222 * state["fRec41"][0]) + (self._fConst248 * state["fRec41"][1])) + (self._fConst222 * state["fRec41"][2]))) - (self._fConst216 * ((self._fConst249 * state["fRec40"][2]) + (self._fConst250 * state["fRec40"][1]))))) 
		state["fRec39"] = ((fSlow2 * fRec39_temp) + (fSlow3 * jnp.abs((self._fConst216 * (((self._fConst219 * state["fRec40"][0]) + (self._fConst251 * state["fRec40"][1])) + (self._fConst219 * state["fRec40"][2])))))) 
		fVbargraph6 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec39"])))
		# self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		state["fRec52"] = state["fRec52"].at[0].set((fTemp6 - (self._fConst268 * ((self._fConst271 * state["fRec52"][2]) + (self._fConst272 * state["fRec52"][1]))))) 
		state["fRec51"] = state["fRec51"].at[0].set(((self._fConst268 * (((self._fConst270 * state["fRec52"][0]) + (self._fConst273 * state["fRec52"][1])) + (self._fConst270 * state["fRec52"][2]))) - (self._fConst266 * ((self._fConst274 * state["fRec51"][2]) + (self._fConst275 * state["fRec51"][1]))))) 
		state["fRec50"] = state["fRec50"].at[0].set(((self._fConst266 * (((self._fConst267 * state["fRec51"][0]) + (self._fConst276 * state["fRec51"][1])) + (self._fConst267 * state["fRec51"][2]))) - (self._fConst264 * ((self._fConst277 * state["fRec50"][2]) + (self._fConst278 * state["fRec50"][1]))))) 
		fTemp7 = (self._fConst264 * (((self._fConst265 * state["fRec50"][0]) + (self._fConst279 * state["fRec50"][1])) + (self._fConst265 * state["fRec50"][2]))) 
		state["fRec49"] = state["fRec49"].at[0].set((fTemp7 - (self._fConst261 * ((self._fConst280 * state["fRec49"][2]) + (self._fConst282 * state["fRec49"][1]))))) 
		state["fRec48"] = state["fRec48"].at[0].set(((self._fConst261 * (((self._fConst263 * state["fRec49"][0]) + (self._fConst283 * state["fRec49"][1])) + (self._fConst263 * state["fRec49"][2]))) - (self._fConst258 * ((self._fConst284 * state["fRec48"][2]) + (self._fConst285 * state["fRec48"][1]))))) 
		state["fRec47"] = state["fRec47"].at[0].set(((self._fConst258 * (((self._fConst260 * state["fRec48"][0]) + (self._fConst286 * state["fRec48"][1])) + (self._fConst260 * state["fRec48"][2]))) - (self._fConst254 * ((self._fConst287 * state["fRec47"][2]) + (self._fConst288 * state["fRec47"][1]))))) 
		state["fRec46"] = ((fSlow2 * fRec46_temp) + (fSlow3 * jnp.abs((self._fConst254 * (((self._fConst257 * state["fRec47"][0]) + (self._fConst289 * state["fRec47"][1])) + (self._fConst257 * state["fRec47"][2])))))) 
		fVbargraph7 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec46"])))
		# self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		state["fRec59"] = state["fRec59"].at[0].set((fTemp7 - (self._fConst306 * ((self._fConst309 * state["fRec59"][2]) + (self._fConst310 * state["fRec59"][1]))))) 
		state["fRec58"] = state["fRec58"].at[0].set(((self._fConst306 * (((self._fConst308 * state["fRec59"][0]) + (self._fConst311 * state["fRec59"][1])) + (self._fConst308 * state["fRec59"][2]))) - (self._fConst304 * ((self._fConst312 * state["fRec58"][2]) + (self._fConst313 * state["fRec58"][1]))))) 
		state["fRec57"] = state["fRec57"].at[0].set(((self._fConst304 * (((self._fConst305 * state["fRec58"][0]) + (self._fConst314 * state["fRec58"][1])) + (self._fConst305 * state["fRec58"][2]))) - (self._fConst302 * ((self._fConst315 * state["fRec57"][2]) + (self._fConst316 * state["fRec57"][1]))))) 
		fTemp8 = (self._fConst302 * (((self._fConst303 * state["fRec57"][0]) + (self._fConst317 * state["fRec57"][1])) + (self._fConst303 * state["fRec57"][2]))) 
		state["fRec56"] = state["fRec56"].at[0].set((fTemp8 - (self._fConst299 * ((self._fConst318 * state["fRec56"][2]) + (self._fConst320 * state["fRec56"][1]))))) 
		state["fRec55"] = state["fRec55"].at[0].set(((self._fConst299 * (((self._fConst301 * state["fRec56"][0]) + (self._fConst321 * state["fRec56"][1])) + (self._fConst301 * state["fRec56"][2]))) - (self._fConst296 * ((self._fConst322 * state["fRec55"][2]) + (self._fConst323 * state["fRec55"][1]))))) 
		state["fRec54"] = state["fRec54"].at[0].set(((self._fConst296 * (((self._fConst298 * state["fRec55"][0]) + (self._fConst324 * state["fRec55"][1])) + (self._fConst298 * state["fRec55"][2]))) - (self._fConst292 * ((self._fConst325 * state["fRec54"][2]) + (self._fConst326 * state["fRec54"][1]))))) 
		state["fRec53"] = ((fSlow2 * fRec53_temp) + (fSlow3 * jnp.abs((self._fConst292 * (((self._fConst295 * state["fRec54"][0]) + (self._fConst327 * state["fRec54"][1])) + (self._fConst295 * state["fRec54"][2])))))) 
		fVbargraph8 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec53"])))
		# self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		state["fRec66"] = state["fRec66"].at[0].set((fTemp8 - (self._fConst344 * ((self._fConst347 * state["fRec66"][2]) + (self._fConst348 * state["fRec66"][1]))))) 
		state["fRec65"] = state["fRec65"].at[0].set(((self._fConst344 * (((self._fConst346 * state["fRec66"][0]) + (self._fConst349 * state["fRec66"][1])) + (self._fConst346 * state["fRec66"][2]))) - (self._fConst342 * ((self._fConst350 * state["fRec65"][2]) + (self._fConst351 * state["fRec65"][1]))))) 
		state["fRec64"] = state["fRec64"].at[0].set(((self._fConst342 * (((self._fConst343 * state["fRec65"][0]) + (self._fConst352 * state["fRec65"][1])) + (self._fConst343 * state["fRec65"][2]))) - (self._fConst340 * ((self._fConst353 * state["fRec64"][2]) + (self._fConst354 * state["fRec64"][1]))))) 
		fTemp9 = (self._fConst340 * (((self._fConst341 * state["fRec64"][0]) + (self._fConst355 * state["fRec64"][1])) + (self._fConst341 * state["fRec64"][2]))) 
		state["fRec63"] = state["fRec63"].at[0].set((fTemp9 - (self._fConst337 * ((self._fConst356 * state["fRec63"][2]) + (self._fConst358 * state["fRec63"][1]))))) 
		state["fRec62"] = state["fRec62"].at[0].set(((self._fConst337 * (((self._fConst339 * state["fRec63"][0]) + (self._fConst359 * state["fRec63"][1])) + (self._fConst339 * state["fRec63"][2]))) - (self._fConst334 * ((self._fConst360 * state["fRec62"][2]) + (self._fConst361 * state["fRec62"][1]))))) 
		state["fRec61"] = state["fRec61"].at[0].set(((self._fConst334 * (((self._fConst336 * state["fRec62"][0]) + (self._fConst362 * state["fRec62"][1])) + (self._fConst336 * state["fRec62"][2]))) - (self._fConst330 * ((self._fConst363 * state["fRec61"][2]) + (self._fConst364 * state["fRec61"][1]))))) 
		state["fRec60"] = ((fSlow2 * fRec60_temp) + (fSlow3 * jnp.abs((self._fConst330 * (((self._fConst333 * state["fRec61"][0]) + (self._fConst365 * state["fRec61"][1])) + (self._fConst333 * state["fRec61"][2])))))) 
		fVbargraph9 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec60"])))
		# self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		state["fRec73"] = state["fRec73"].at[0].set((fTemp9 - (self._fConst382 * ((self._fConst385 * state["fRec73"][2]) + (self._fConst386 * state["fRec73"][1]))))) 
		state["fRec72"] = state["fRec72"].at[0].set(((self._fConst382 * (((self._fConst384 * state["fRec73"][0]) + (self._fConst387 * state["fRec73"][1])) + (self._fConst384 * state["fRec73"][2]))) - (self._fConst380 * ((self._fConst388 * state["fRec72"][2]) + (self._fConst389 * state["fRec72"][1]))))) 
		state["fRec71"] = state["fRec71"].at[0].set(((self._fConst380 * (((self._fConst381 * state["fRec72"][0]) + (self._fConst390 * state["fRec72"][1])) + (self._fConst381 * state["fRec72"][2]))) - (self._fConst378 * ((self._fConst391 * state["fRec71"][2]) + (self._fConst392 * state["fRec71"][1]))))) 
		fTemp10 = (self._fConst378 * (((self._fConst379 * state["fRec71"][0]) + (self._fConst393 * state["fRec71"][1])) + (self._fConst379 * state["fRec71"][2]))) 
		state["fRec70"] = state["fRec70"].at[0].set((fTemp10 - (self._fConst375 * ((self._fConst394 * state["fRec70"][2]) + (self._fConst396 * state["fRec70"][1]))))) 
		state["fRec69"] = state["fRec69"].at[0].set(((self._fConst375 * (((self._fConst377 * state["fRec70"][0]) + (self._fConst397 * state["fRec70"][1])) + (self._fConst377 * state["fRec70"][2]))) - (self._fConst372 * ((self._fConst398 * state["fRec69"][2]) + (self._fConst399 * state["fRec69"][1]))))) 
		state["fRec68"] = state["fRec68"].at[0].set(((self._fConst372 * (((self._fConst374 * state["fRec69"][0]) + (self._fConst400 * state["fRec69"][1])) + (self._fConst374 * state["fRec69"][2]))) - (self._fConst368 * ((self._fConst401 * state["fRec68"][2]) + (self._fConst402 * state["fRec68"][1]))))) 
		state["fRec67"] = ((fSlow2 * fRec67_temp) + (fSlow3 * jnp.abs((self._fConst368 * (((self._fConst371 * state["fRec68"][0]) + (self._fConst403 * state["fRec68"][1])) + (self._fConst371 * state["fRec68"][2])))))) 
		fVbargraph10 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec67"])))
		# self.sow("intermediates", "fVbargraph10", fVbargraph10) 
		state["fRec80"] = state["fRec80"].at[0].set((fTemp10 - (self._fConst420 * ((self._fConst423 * state["fRec80"][2]) + (self._fConst424 * state["fRec80"][1]))))) 
		state["fRec79"] = state["fRec79"].at[0].set(((self._fConst420 * (((self._fConst422 * state["fRec80"][0]) + (self._fConst425 * state["fRec80"][1])) + (self._fConst422 * state["fRec80"][2]))) - (self._fConst418 * ((self._fConst426 * state["fRec79"][2]) + (self._fConst427 * state["fRec79"][1]))))) 
		state["fRec78"] = state["fRec78"].at[0].set(((self._fConst418 * (((self._fConst419 * state["fRec79"][0]) + (self._fConst428 * state["fRec79"][1])) + (self._fConst419 * state["fRec79"][2]))) - (self._fConst416 * ((self._fConst429 * state["fRec78"][2]) + (self._fConst430 * state["fRec78"][1]))))) 
		fTemp11 = (self._fConst416 * (((self._fConst417 * state["fRec78"][0]) + (self._fConst431 * state["fRec78"][1])) + (self._fConst417 * state["fRec78"][2]))) 
		state["fRec77"] = state["fRec77"].at[0].set((fTemp11 - (self._fConst413 * ((self._fConst432 * state["fRec77"][2]) + (self._fConst434 * state["fRec77"][1]))))) 
		state["fRec76"] = state["fRec76"].at[0].set(((self._fConst413 * (((self._fConst415 * state["fRec77"][0]) + (self._fConst435 * state["fRec77"][1])) + (self._fConst415 * state["fRec77"][2]))) - (self._fConst410 * ((self._fConst436 * state["fRec76"][2]) + (self._fConst437 * state["fRec76"][1]))))) 
		state["fRec75"] = state["fRec75"].at[0].set(((self._fConst410 * (((self._fConst412 * state["fRec76"][0]) + (self._fConst438 * state["fRec76"][1])) + (self._fConst412 * state["fRec76"][2]))) - (self._fConst406 * ((self._fConst439 * state["fRec75"][2]) + (self._fConst440 * state["fRec75"][1]))))) 
		state["fRec74"] = ((fSlow2 * fRec74_temp) + (fSlow3 * jnp.abs((self._fConst406 * (((self._fConst409 * state["fRec75"][0]) + (self._fConst441 * state["fRec75"][1])) + (self._fConst409 * state["fRec75"][2])))))) 
		fVbargraph11 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec74"])))
		# self.sow("intermediates", "fVbargraph11", fVbargraph11) 
		state["fRec87"] = state["fRec87"].at[0].set((fTemp11 - (self._fConst458 * ((self._fConst461 * state["fRec87"][2]) + (self._fConst462 * state["fRec87"][1]))))) 
		state["fRec86"] = state["fRec86"].at[0].set(((self._fConst458 * (((self._fConst460 * state["fRec87"][0]) + (self._fConst463 * state["fRec87"][1])) + (self._fConst460 * state["fRec87"][2]))) - (self._fConst456 * ((self._fConst464 * state["fRec86"][2]) + (self._fConst465 * state["fRec86"][1]))))) 
		state["fRec85"] = state["fRec85"].at[0].set(((self._fConst456 * (((self._fConst457 * state["fRec86"][0]) + (self._fConst466 * state["fRec86"][1])) + (self._fConst457 * state["fRec86"][2]))) - (self._fConst454 * ((self._fConst467 * state["fRec85"][2]) + (self._fConst468 * state["fRec85"][1]))))) 
		fTemp12 = (self._fConst454 * (((self._fConst455 * state["fRec85"][0]) + (self._fConst469 * state["fRec85"][1])) + (self._fConst455 * state["fRec85"][2]))) 
		state["fRec84"] = state["fRec84"].at[0].set((fTemp12 - (self._fConst451 * ((self._fConst470 * state["fRec84"][2]) + (self._fConst472 * state["fRec84"][1]))))) 
		state["fRec83"] = state["fRec83"].at[0].set(((self._fConst451 * (((self._fConst453 * state["fRec84"][0]) + (self._fConst473 * state["fRec84"][1])) + (self._fConst453 * state["fRec84"][2]))) - (self._fConst448 * ((self._fConst474 * state["fRec83"][2]) + (self._fConst475 * state["fRec83"][1]))))) 
		state["fRec82"] = state["fRec82"].at[0].set(((self._fConst448 * (((self._fConst450 * state["fRec83"][0]) + (self._fConst476 * state["fRec83"][1])) + (self._fConst450 * state["fRec83"][2]))) - (self._fConst444 * ((self._fConst477 * state["fRec82"][2]) + (self._fConst478 * state["fRec82"][1]))))) 
		state["fRec81"] = ((fSlow2 * fRec81_temp) + (fSlow3 * jnp.abs((self._fConst444 * (((self._fConst447 * state["fRec82"][0]) + (self._fConst479 * state["fRec82"][1])) + (self._fConst447 * state["fRec82"][2])))))) 
		fVbargraph12 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec81"])))
		# self.sow("intermediates", "fVbargraph12", fVbargraph12) 
		state["fRec94"] = state["fRec94"].at[0].set((fTemp12 - (self._fConst496 * ((self._fConst499 * state["fRec94"][2]) + (self._fConst500 * state["fRec94"][1]))))) 
		state["fRec93"] = state["fRec93"].at[0].set(((self._fConst496 * (((self._fConst498 * state["fRec94"][0]) + (self._fConst501 * state["fRec94"][1])) + (self._fConst498 * state["fRec94"][2]))) - (self._fConst494 * ((self._fConst502 * state["fRec93"][2]) + (self._fConst503 * state["fRec93"][1]))))) 
		state["fRec92"] = state["fRec92"].at[0].set(((self._fConst494 * (((self._fConst495 * state["fRec93"][0]) + (self._fConst504 * state["fRec93"][1])) + (self._fConst495 * state["fRec93"][2]))) - (self._fConst492 * ((self._fConst505 * state["fRec92"][2]) + (self._fConst506 * state["fRec92"][1]))))) 
		fTemp13 = (self._fConst492 * (((self._fConst493 * state["fRec92"][0]) + (self._fConst507 * state["fRec92"][1])) + (self._fConst493 * state["fRec92"][2]))) 
		state["fRec91"] = state["fRec91"].at[0].set((fTemp13 - (self._fConst489 * ((self._fConst508 * state["fRec91"][2]) + (self._fConst510 * state["fRec91"][1]))))) 
		state["fRec90"] = state["fRec90"].at[0].set(((self._fConst489 * (((self._fConst491 * state["fRec91"][0]) + (self._fConst511 * state["fRec91"][1])) + (self._fConst491 * state["fRec91"][2]))) - (self._fConst486 * ((self._fConst512 * state["fRec90"][2]) + (self._fConst513 * state["fRec90"][1]))))) 
		state["fRec89"] = state["fRec89"].at[0].set(((self._fConst486 * (((self._fConst488 * state["fRec90"][0]) + (self._fConst514 * state["fRec90"][1])) + (self._fConst488 * state["fRec90"][2]))) - (self._fConst482 * ((self._fConst515 * state["fRec89"][2]) + (self._fConst516 * state["fRec89"][1]))))) 
		state["fRec88"] = ((fSlow2 * fRec88_temp) + (fSlow3 * jnp.abs((self._fConst482 * (((self._fConst485 * state["fRec89"][0]) + (self._fConst517 * state["fRec89"][1])) + (self._fConst485 * state["fRec89"][2])))))) 
		fVbargraph13 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec88"])))
		# self.sow("intermediates", "fVbargraph13", fVbargraph13) 
		state["fRec101"] = state["fRec101"].at[0].set((fTemp13 - (self._fConst534 * ((self._fConst537 * state["fRec101"][2]) + (self._fConst538 * state["fRec101"][1]))))) 
		state["fRec100"] = state["fRec100"].at[0].set(((self._fConst534 * (((self._fConst536 * state["fRec101"][0]) + (self._fConst539 * state["fRec101"][1])) + (self._fConst536 * state["fRec101"][2]))) - (self._fConst532 * ((self._fConst540 * state["fRec100"][2]) + (self._fConst541 * state["fRec100"][1]))))) 
		state["fRec99"] = state["fRec99"].at[0].set(((self._fConst532 * (((self._fConst533 * state["fRec100"][0]) + (self._fConst542 * state["fRec100"][1])) + (self._fConst533 * state["fRec100"][2]))) - (self._fConst530 * ((self._fConst543 * state["fRec99"][2]) + (self._fConst544 * state["fRec99"][1]))))) 
		fTemp14 = (self._fConst530 * (((self._fConst531 * state["fRec99"][0]) + (self._fConst545 * state["fRec99"][1])) + (self._fConst531 * state["fRec99"][2]))) 
		state["fRec98"] = state["fRec98"].at[0].set((fTemp14 - (self._fConst527 * ((self._fConst546 * state["fRec98"][2]) + (self._fConst548 * state["fRec98"][1]))))) 
		state["fRec97"] = state["fRec97"].at[0].set(((self._fConst527 * (((self._fConst529 * state["fRec98"][0]) + (self._fConst549 * state["fRec98"][1])) + (self._fConst529 * state["fRec98"][2]))) - (self._fConst524 * ((self._fConst550 * state["fRec97"][2]) + (self._fConst551 * state["fRec97"][1]))))) 
		state["fRec96"] = state["fRec96"].at[0].set(((self._fConst524 * (((self._fConst526 * state["fRec97"][0]) + (self._fConst552 * state["fRec97"][1])) + (self._fConst526 * state["fRec97"][2]))) - (self._fConst520 * ((self._fConst553 * state["fRec96"][2]) + (self._fConst554 * state["fRec96"][1]))))) 
		state["fRec95"] = ((fSlow2 * fRec95_temp) + (fSlow3 * jnp.abs((self._fConst520 * (((self._fConst523 * state["fRec96"][0]) + (self._fConst555 * state["fRec96"][1])) + (self._fConst523 * state["fRec96"][2])))))) 
		fVbargraph14 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec95"])))
		# self.sow("intermediates", "fVbargraph14", fVbargraph14) 
		state["fRec108"] = state["fRec108"].at[0].set((fTemp14 - (self._fConst572 * ((self._fConst575 * state["fRec108"][2]) + (self._fConst576 * state["fRec108"][1]))))) 
		state["fRec107"] = state["fRec107"].at[0].set(((self._fConst572 * (((self._fConst574 * state["fRec108"][0]) + (self._fConst577 * state["fRec108"][1])) + (self._fConst574 * state["fRec108"][2]))) - (self._fConst570 * ((self._fConst578 * state["fRec107"][2]) + (self._fConst579 * state["fRec107"][1]))))) 
		state["fRec106"] = state["fRec106"].at[0].set(((self._fConst570 * (((self._fConst571 * state["fRec107"][0]) + (self._fConst580 * state["fRec107"][1])) + (self._fConst571 * state["fRec107"][2]))) - (self._fConst568 * ((self._fConst581 * state["fRec106"][2]) + (self._fConst582 * state["fRec106"][1]))))) 
		fTemp15 = (self._fConst568 * (((self._fConst569 * state["fRec106"][0]) + (self._fConst583 * state["fRec106"][1])) + (self._fConst569 * state["fRec106"][2]))) 
		state["fRec105"] = state["fRec105"].at[0].set((fTemp15 - (self._fConst565 * ((self._fConst584 * state["fRec105"][2]) + (self._fConst586 * state["fRec105"][1]))))) 
		state["fRec104"] = state["fRec104"].at[0].set(((self._fConst565 * (((self._fConst567 * state["fRec105"][0]) + (self._fConst587 * state["fRec105"][1])) + (self._fConst567 * state["fRec105"][2]))) - (self._fConst562 * ((self._fConst588 * state["fRec104"][2]) + (self._fConst589 * state["fRec104"][1]))))) 
		state["fRec103"] = state["fRec103"].at[0].set(((self._fConst562 * (((self._fConst564 * state["fRec104"][0]) + (self._fConst590 * state["fRec104"][1])) + (self._fConst564 * state["fRec104"][2]))) - (self._fConst558 * ((self._fConst591 * state["fRec103"][2]) + (self._fConst592 * state["fRec103"][1]))))) 
		state["fRec102"] = ((fSlow2 * fRec102_temp) + (fSlow3 * jnp.abs((self._fConst558 * (((self._fConst561 * state["fRec103"][0]) + (self._fConst593 * state["fRec103"][1])) + (self._fConst561 * state["fRec103"][2])))))) 
		fVbargraph15 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec102"])))
		# self.sow("intermediates", "fVbargraph15", fVbargraph15) 
		state["fRec115"] = state["fRec115"].at[0].set((fTemp15 - (self._fConst610 * ((self._fConst613 * state["fRec115"][2]) + (self._fConst614 * state["fRec115"][1]))))) 
		state["fRec114"] = state["fRec114"].at[0].set(((self._fConst610 * (((self._fConst612 * state["fRec115"][0]) + (self._fConst615 * state["fRec115"][1])) + (self._fConst612 * state["fRec115"][2]))) - (self._fConst608 * ((self._fConst616 * state["fRec114"][2]) + (self._fConst617 * state["fRec114"][1]))))) 
		state["fRec113"] = state["fRec113"].at[0].set(((self._fConst608 * (((self._fConst609 * state["fRec114"][0]) + (self._fConst618 * state["fRec114"][1])) + (self._fConst609 * state["fRec114"][2]))) - (self._fConst606 * ((self._fConst619 * state["fRec113"][2]) + (self._fConst620 * state["fRec113"][1]))))) 
		fTemp16 = (self._fConst606 * (((self._fConst607 * state["fRec113"][0]) + (self._fConst621 * state["fRec113"][1])) + (self._fConst607 * state["fRec113"][2]))) 
		state["fRec112"] = state["fRec112"].at[0].set((fTemp16 - (self._fConst603 * ((self._fConst622 * state["fRec112"][2]) + (self._fConst624 * state["fRec112"][1]))))) 
		state["fRec111"] = state["fRec111"].at[0].set(((self._fConst603 * (((self._fConst605 * state["fRec112"][0]) + (self._fConst625 * state["fRec112"][1])) + (self._fConst605 * state["fRec112"][2]))) - (self._fConst600 * ((self._fConst626 * state["fRec111"][2]) + (self._fConst627 * state["fRec111"][1]))))) 
		state["fRec110"] = state["fRec110"].at[0].set(((self._fConst600 * (((self._fConst602 * state["fRec111"][0]) + (self._fConst628 * state["fRec111"][1])) + (self._fConst602 * state["fRec111"][2]))) - (self._fConst596 * ((self._fConst629 * state["fRec110"][2]) + (self._fConst630 * state["fRec110"][1]))))) 
		state["fRec109"] = ((fSlow2 * fRec109_temp) + (fSlow3 * jnp.abs((self._fConst596 * (((self._fConst599 * state["fRec110"][0]) + (self._fConst631 * state["fRec110"][1])) + (self._fConst599 * state["fRec110"][2])))))) 
		fVbargraph16 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec109"])))
		# self.sow("intermediates", "fVbargraph16", fVbargraph16) 
		state["fRec122"] = state["fRec122"].at[0].set((fTemp16 - (self._fConst648 * ((self._fConst651 * state["fRec122"][2]) + (self._fConst652 * state["fRec122"][1]))))) 
		state["fRec121"] = state["fRec121"].at[0].set(((self._fConst648 * (((self._fConst650 * state["fRec122"][0]) + (self._fConst653 * state["fRec122"][1])) + (self._fConst650 * state["fRec122"][2]))) - (self._fConst646 * ((self._fConst654 * state["fRec121"][2]) + (self._fConst655 * state["fRec121"][1]))))) 
		state["fRec120"] = state["fRec120"].at[0].set(((self._fConst646 * (((self._fConst647 * state["fRec121"][0]) + (self._fConst656 * state["fRec121"][1])) + (self._fConst647 * state["fRec121"][2]))) - (self._fConst644 * ((self._fConst657 * state["fRec120"][2]) + (self._fConst658 * state["fRec120"][1]))))) 
		fTemp17 = (self._fConst644 * (((self._fConst645 * state["fRec120"][0]) + (self._fConst659 * state["fRec120"][1])) + (self._fConst645 * state["fRec120"][2]))) 
		state["fRec119"] = state["fRec119"].at[0].set((fTemp17 - (self._fConst641 * ((self._fConst660 * state["fRec119"][2]) + (self._fConst662 * state["fRec119"][1]))))) 
		state["fRec118"] = state["fRec118"].at[0].set(((self._fConst641 * (((self._fConst643 * state["fRec119"][0]) + (self._fConst663 * state["fRec119"][1])) + (self._fConst643 * state["fRec119"][2]))) - (self._fConst638 * ((self._fConst664 * state["fRec118"][2]) + (self._fConst665 * state["fRec118"][1]))))) 
		state["fRec117"] = state["fRec117"].at[0].set(((self._fConst638 * (((self._fConst640 * state["fRec118"][0]) + (self._fConst666 * state["fRec118"][1])) + (self._fConst640 * state["fRec118"][2]))) - (self._fConst634 * ((self._fConst667 * state["fRec117"][2]) + (self._fConst668 * state["fRec117"][1]))))) 
		state["fRec116"] = ((fSlow2 * fRec116_temp) + (fSlow3 * jnp.abs((self._fConst634 * (((self._fConst637 * state["fRec117"][0]) + (self._fConst669 * state["fRec117"][1])) + (self._fConst637 * state["fRec117"][2])))))) 
		fVbargraph17 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec116"])))
		# self.sow("intermediates", "fVbargraph17", fVbargraph17) 
		state["fRec129"] = state["fRec129"].at[0].set((fTemp17 - (self._fConst686 * ((self._fConst689 * state["fRec129"][2]) + (self._fConst690 * state["fRec129"][1]))))) 
		state["fRec128"] = state["fRec128"].at[0].set(((self._fConst686 * (((self._fConst688 * state["fRec129"][0]) + (self._fConst691 * state["fRec129"][1])) + (self._fConst688 * state["fRec129"][2]))) - (self._fConst684 * ((self._fConst692 * state["fRec128"][2]) + (self._fConst693 * state["fRec128"][1]))))) 
		state["fRec127"] = state["fRec127"].at[0].set(((self._fConst684 * (((self._fConst685 * state["fRec128"][0]) + (self._fConst694 * state["fRec128"][1])) + (self._fConst685 * state["fRec128"][2]))) - (self._fConst682 * ((self._fConst695 * state["fRec127"][2]) + (self._fConst696 * state["fRec127"][1]))))) 
		fTemp18 = (self._fConst682 * (((self._fConst683 * state["fRec127"][0]) + (self._fConst697 * state["fRec127"][1])) + (self._fConst683 * state["fRec127"][2]))) 
		state["fRec126"] = state["fRec126"].at[0].set((fTemp18 - (self._fConst679 * ((self._fConst698 * state["fRec126"][2]) + (self._fConst700 * state["fRec126"][1]))))) 
		state["fRec125"] = state["fRec125"].at[0].set(((self._fConst679 * (((self._fConst681 * state["fRec126"][0]) + (self._fConst701 * state["fRec126"][1])) + (self._fConst681 * state["fRec126"][2]))) - (self._fConst676 * ((self._fConst702 * state["fRec125"][2]) + (self._fConst703 * state["fRec125"][1]))))) 
		state["fRec124"] = state["fRec124"].at[0].set(((self._fConst676 * (((self._fConst678 * state["fRec125"][0]) + (self._fConst704 * state["fRec125"][1])) + (self._fConst678 * state["fRec125"][2]))) - (self._fConst672 * ((self._fConst705 * state["fRec124"][2]) + (self._fConst706 * state["fRec124"][1]))))) 
		state["fRec123"] = ((fSlow2 * fRec123_temp) + (fSlow3 * jnp.abs((self._fConst672 * (((self._fConst675 * state["fRec124"][0]) + (self._fConst707 * state["fRec124"][1])) + (self._fConst675 * state["fRec124"][2])))))) 
		fVbargraph18 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec123"])))
		# self.sow("intermediates", "fVbargraph18", fVbargraph18) 
		state["fRec136"] = state["fRec136"].at[0].set((fTemp18 - (self._fConst724 * ((self._fConst727 * state["fRec136"][2]) + (self._fConst728 * state["fRec136"][1]))))) 
		state["fRec135"] = state["fRec135"].at[0].set(((self._fConst724 * (((self._fConst726 * state["fRec136"][0]) + (self._fConst729 * state["fRec136"][1])) + (self._fConst726 * state["fRec136"][2]))) - (self._fConst722 * ((self._fConst730 * state["fRec135"][2]) + (self._fConst731 * state["fRec135"][1]))))) 
		state["fRec134"] = state["fRec134"].at[0].set(((self._fConst722 * (((self._fConst723 * state["fRec135"][0]) + (self._fConst732 * state["fRec135"][1])) + (self._fConst723 * state["fRec135"][2]))) - (self._fConst720 * ((self._fConst733 * state["fRec134"][2]) + (self._fConst734 * state["fRec134"][1]))))) 
		fTemp19 = (self._fConst720 * (((self._fConst721 * state["fRec134"][0]) + (self._fConst735 * state["fRec134"][1])) + (self._fConst721 * state["fRec134"][2]))) 
		state["fRec133"] = state["fRec133"].at[0].set((fTemp19 - (self._fConst717 * ((self._fConst736 * state["fRec133"][2]) + (self._fConst738 * state["fRec133"][1]))))) 
		state["fRec132"] = state["fRec132"].at[0].set(((self._fConst717 * (((self._fConst719 * state["fRec133"][0]) + (self._fConst739 * state["fRec133"][1])) + (self._fConst719 * state["fRec133"][2]))) - (self._fConst714 * ((self._fConst740 * state["fRec132"][2]) + (self._fConst741 * state["fRec132"][1]))))) 
		state["fRec131"] = state["fRec131"].at[0].set(((self._fConst714 * (((self._fConst716 * state["fRec132"][0]) + (self._fConst742 * state["fRec132"][1])) + (self._fConst716 * state["fRec132"][2]))) - (self._fConst710 * ((self._fConst743 * state["fRec131"][2]) + (self._fConst744 * state["fRec131"][1]))))) 
		state["fRec130"] = ((fSlow2 * fRec130_temp) + (fSlow3 * jnp.abs((self._fConst710 * (((self._fConst713 * state["fRec131"][0]) + (self._fConst745 * state["fRec131"][1])) + (self._fConst713 * state["fRec131"][2])))))) 
		fVbargraph19 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec130"])))
		# self.sow("intermediates", "fVbargraph19", fVbargraph19) 
		state["fRec143"] = state["fRec143"].at[0].set((fTemp19 - (self._fConst762 * ((self._fConst765 * state["fRec143"][2]) + (self._fConst766 * state["fRec143"][1]))))) 
		state["fRec142"] = state["fRec142"].at[0].set(((self._fConst762 * (((self._fConst764 * state["fRec143"][0]) + (self._fConst767 * state["fRec143"][1])) + (self._fConst764 * state["fRec143"][2]))) - (self._fConst760 * ((self._fConst768 * state["fRec142"][2]) + (self._fConst769 * state["fRec142"][1]))))) 
		state["fRec141"] = state["fRec141"].at[0].set(((self._fConst760 * (((self._fConst761 * state["fRec142"][0]) + (self._fConst770 * state["fRec142"][1])) + (self._fConst761 * state["fRec142"][2]))) - (self._fConst758 * ((self._fConst771 * state["fRec141"][2]) + (self._fConst772 * state["fRec141"][1]))))) 
		fTemp20 = (self._fConst758 * (((self._fConst759 * state["fRec141"][0]) + (self._fConst773 * state["fRec141"][1])) + (self._fConst759 * state["fRec141"][2]))) 
		state["fRec140"] = state["fRec140"].at[0].set((fTemp20 - (self._fConst755 * ((self._fConst774 * state["fRec140"][2]) + (self._fConst776 * state["fRec140"][1]))))) 
		state["fRec139"] = state["fRec139"].at[0].set(((self._fConst755 * (((self._fConst757 * state["fRec140"][0]) + (self._fConst777 * state["fRec140"][1])) + (self._fConst757 * state["fRec140"][2]))) - (self._fConst752 * ((self._fConst778 * state["fRec139"][2]) + (self._fConst779 * state["fRec139"][1]))))) 
		state["fRec138"] = state["fRec138"].at[0].set(((self._fConst752 * (((self._fConst754 * state["fRec139"][0]) + (self._fConst780 * state["fRec139"][1])) + (self._fConst754 * state["fRec139"][2]))) - (self._fConst748 * ((self._fConst781 * state["fRec138"][2]) + (self._fConst782 * state["fRec138"][1]))))) 
		state["fRec137"] = ((fSlow2 * fRec137_temp) + (fSlow3 * jnp.abs((self._fConst748 * (((self._fConst751 * state["fRec138"][0]) + (self._fConst783 * state["fRec138"][1])) + (self._fConst751 * state["fRec138"][2])))))) 
		fVbargraph20 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec137"])))
		# self.sow("intermediates", "fVbargraph20", fVbargraph20) 
		state["fRec150"] = state["fRec150"].at[0].set((fTemp20 - (self._fConst800 * ((self._fConst803 * state["fRec150"][2]) + (self._fConst804 * state["fRec150"][1]))))) 
		state["fRec149"] = state["fRec149"].at[0].set(((self._fConst800 * (((self._fConst802 * state["fRec150"][0]) + (self._fConst805 * state["fRec150"][1])) + (self._fConst802 * state["fRec150"][2]))) - (self._fConst798 * ((self._fConst806 * state["fRec149"][2]) + (self._fConst807 * state["fRec149"][1]))))) 
		state["fRec148"] = state["fRec148"].at[0].set(((self._fConst798 * (((self._fConst799 * state["fRec149"][0]) + (self._fConst808 * state["fRec149"][1])) + (self._fConst799 * state["fRec149"][2]))) - (self._fConst796 * ((self._fConst809 * state["fRec148"][2]) + (self._fConst810 * state["fRec148"][1]))))) 
		fTemp21 = (self._fConst796 * (((self._fConst797 * state["fRec148"][0]) + (self._fConst811 * state["fRec148"][1])) + (self._fConst797 * state["fRec148"][2]))) 
		state["fRec147"] = state["fRec147"].at[0].set((fTemp21 - (self._fConst793 * ((self._fConst812 * state["fRec147"][2]) + (self._fConst814 * state["fRec147"][1]))))) 
		state["fRec146"] = state["fRec146"].at[0].set(((self._fConst793 * (((self._fConst795 * state["fRec147"][0]) + (self._fConst815 * state["fRec147"][1])) + (self._fConst795 * state["fRec147"][2]))) - (self._fConst790 * ((self._fConst816 * state["fRec146"][2]) + (self._fConst817 * state["fRec146"][1]))))) 
		state["fRec145"] = state["fRec145"].at[0].set(((self._fConst790 * (((self._fConst792 * state["fRec146"][0]) + (self._fConst818 * state["fRec146"][1])) + (self._fConst792 * state["fRec146"][2]))) - (self._fConst786 * ((self._fConst819 * state["fRec145"][2]) + (self._fConst820 * state["fRec145"][1]))))) 
		state["fRec144"] = ((fSlow2 * fRec144_temp) + (fSlow3 * jnp.abs((self._fConst786 * (((self._fConst789 * state["fRec145"][0]) + (self._fConst821 * state["fRec145"][1])) + (self._fConst789 * state["fRec145"][2])))))) 
		fVbargraph21 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec144"])))
		# self.sow("intermediates", "fVbargraph21", fVbargraph21) 
		state["fRec157"] = state["fRec157"].at[0].set((fTemp21 - (self._fConst838 * ((self._fConst841 * state["fRec157"][2]) + (self._fConst842 * state["fRec157"][1]))))) 
		state["fRec156"] = state["fRec156"].at[0].set(((self._fConst838 * (((self._fConst840 * state["fRec157"][0]) + (self._fConst843 * state["fRec157"][1])) + (self._fConst840 * state["fRec157"][2]))) - (self._fConst836 * ((self._fConst844 * state["fRec156"][2]) + (self._fConst845 * state["fRec156"][1]))))) 
		state["fRec155"] = state["fRec155"].at[0].set(((self._fConst836 * (((self._fConst837 * state["fRec156"][0]) + (self._fConst846 * state["fRec156"][1])) + (self._fConst837 * state["fRec156"][2]))) - (self._fConst834 * ((self._fConst847 * state["fRec155"][2]) + (self._fConst848 * state["fRec155"][1]))))) 
		fTemp22 = (self._fConst834 * (((self._fConst835 * state["fRec155"][0]) + (self._fConst849 * state["fRec155"][1])) + (self._fConst835 * state["fRec155"][2]))) 
		state["fRec154"] = state["fRec154"].at[0].set((fTemp22 - (self._fConst831 * ((self._fConst850 * state["fRec154"][2]) + (self._fConst852 * state["fRec154"][1]))))) 
		state["fRec153"] = state["fRec153"].at[0].set(((self._fConst831 * (((self._fConst833 * state["fRec154"][0]) + (self._fConst853 * state["fRec154"][1])) + (self._fConst833 * state["fRec154"][2]))) - (self._fConst828 * ((self._fConst854 * state["fRec153"][2]) + (self._fConst855 * state["fRec153"][1]))))) 
		state["fRec152"] = state["fRec152"].at[0].set(((self._fConst828 * (((self._fConst830 * state["fRec153"][0]) + (self._fConst856 * state["fRec153"][1])) + (self._fConst830 * state["fRec153"][2]))) - (self._fConst824 * ((self._fConst857 * state["fRec152"][2]) + (self._fConst858 * state["fRec152"][1]))))) 
		state["fRec151"] = ((fSlow2 * fRec151_temp) + (fSlow3 * jnp.abs((self._fConst824 * (((self._fConst827 * state["fRec152"][0]) + (self._fConst859 * state["fRec152"][1])) + (self._fConst827 * state["fRec152"][2])))))) 
		fVbargraph22 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec151"])))
		# self.sow("intermediates", "fVbargraph22", fVbargraph22) 
		state["fRec164"] = state["fRec164"].at[0].set((fTemp22 - (self._fConst876 * ((self._fConst879 * state["fRec164"][2]) + (self._fConst880 * state["fRec164"][1]))))) 
		state["fRec163"] = state["fRec163"].at[0].set(((self._fConst876 * (((self._fConst878 * state["fRec164"][0]) + (self._fConst881 * state["fRec164"][1])) + (self._fConst878 * state["fRec164"][2]))) - (self._fConst874 * ((self._fConst882 * state["fRec163"][2]) + (self._fConst883 * state["fRec163"][1]))))) 
		state["fRec162"] = state["fRec162"].at[0].set(((self._fConst874 * (((self._fConst875 * state["fRec163"][0]) + (self._fConst884 * state["fRec163"][1])) + (self._fConst875 * state["fRec163"][2]))) - (self._fConst872 * ((self._fConst885 * state["fRec162"][2]) + (self._fConst886 * state["fRec162"][1]))))) 
		fTemp23 = (self._fConst872 * (((self._fConst873 * state["fRec162"][0]) + (self._fConst887 * state["fRec162"][1])) + (self._fConst873 * state["fRec162"][2]))) 
		state["fRec161"] = state["fRec161"].at[0].set((fTemp23 - (self._fConst869 * ((self._fConst888 * state["fRec161"][2]) + (self._fConst890 * state["fRec161"][1]))))) 
		state["fRec160"] = state["fRec160"].at[0].set(((self._fConst869 * (((self._fConst871 * state["fRec161"][0]) + (self._fConst891 * state["fRec161"][1])) + (self._fConst871 * state["fRec161"][2]))) - (self._fConst866 * ((self._fConst892 * state["fRec160"][2]) + (self._fConst893 * state["fRec160"][1]))))) 
		state["fRec159"] = state["fRec159"].at[0].set(((self._fConst866 * (((self._fConst868 * state["fRec160"][0]) + (self._fConst894 * state["fRec160"][1])) + (self._fConst868 * state["fRec160"][2]))) - (self._fConst862 * ((self._fConst895 * state["fRec159"][2]) + (self._fConst896 * state["fRec159"][1]))))) 
		state["fRec158"] = ((fSlow2 * fRec158_temp) + (fSlow3 * jnp.abs((self._fConst862 * (((self._fConst865 * state["fRec159"][0]) + (self._fConst897 * state["fRec159"][1])) + (self._fConst865 * state["fRec159"][2])))))) 
		fVbargraph23 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec158"])))
		# self.sow("intermediates", "fVbargraph23", fVbargraph23) 
		state["fRec171"] = state["fRec171"].at[0].set((fTemp23 - (self._fConst914 * ((self._fConst917 * state["fRec171"][2]) + (self._fConst918 * state["fRec171"][1]))))) 
		state["fRec170"] = state["fRec170"].at[0].set(((self._fConst914 * (((self._fConst916 * state["fRec171"][0]) + (self._fConst919 * state["fRec171"][1])) + (self._fConst916 * state["fRec171"][2]))) - (self._fConst912 * ((self._fConst920 * state["fRec170"][2]) + (self._fConst921 * state["fRec170"][1]))))) 
		state["fRec169"] = state["fRec169"].at[0].set(((self._fConst912 * (((self._fConst913 * state["fRec170"][0]) + (self._fConst922 * state["fRec170"][1])) + (self._fConst913 * state["fRec170"][2]))) - (self._fConst910 * ((self._fConst923 * state["fRec169"][2]) + (self._fConst924 * state["fRec169"][1]))))) 
		fTemp24 = (self._fConst910 * (((self._fConst911 * state["fRec169"][0]) + (self._fConst925 * state["fRec169"][1])) + (self._fConst911 * state["fRec169"][2]))) 
		state["fRec168"] = state["fRec168"].at[0].set((fTemp24 - (self._fConst907 * ((self._fConst926 * state["fRec168"][2]) + (self._fConst928 * state["fRec168"][1]))))) 
		state["fRec167"] = state["fRec167"].at[0].set(((self._fConst907 * (((self._fConst909 * state["fRec168"][0]) + (self._fConst929 * state["fRec168"][1])) + (self._fConst909 * state["fRec168"][2]))) - (self._fConst904 * ((self._fConst930 * state["fRec167"][2]) + (self._fConst931 * state["fRec167"][1]))))) 
		state["fRec166"] = state["fRec166"].at[0].set(((self._fConst904 * (((self._fConst906 * state["fRec167"][0]) + (self._fConst932 * state["fRec167"][1])) + (self._fConst906 * state["fRec167"][2]))) - (self._fConst900 * ((self._fConst933 * state["fRec166"][2]) + (self._fConst934 * state["fRec166"][1]))))) 
		state["fRec165"] = ((fSlow2 * fRec165_temp) + (fSlow3 * jnp.abs((self._fConst900 * (((self._fConst903 * state["fRec166"][0]) + (self._fConst935 * state["fRec166"][1])) + (self._fConst903 * state["fRec166"][2])))))) 
		fVbargraph24 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec165"])))
		# self.sow("intermediates", "fVbargraph24", fVbargraph24) 
		state["fRec178"] = state["fRec178"].at[0].set((fTemp24 - (self._fConst952 * ((self._fConst955 * state["fRec178"][2]) + (self._fConst956 * state["fRec178"][1]))))) 
		state["fRec177"] = state["fRec177"].at[0].set(((self._fConst952 * (((self._fConst954 * state["fRec178"][0]) + (self._fConst957 * state["fRec178"][1])) + (self._fConst954 * state["fRec178"][2]))) - (self._fConst950 * ((self._fConst958 * state["fRec177"][2]) + (self._fConst959 * state["fRec177"][1]))))) 
		state["fRec176"] = state["fRec176"].at[0].set(((self._fConst950 * (((self._fConst951 * state["fRec177"][0]) + (self._fConst960 * state["fRec177"][1])) + (self._fConst951 * state["fRec177"][2]))) - (self._fConst948 * ((self._fConst961 * state["fRec176"][2]) + (self._fConst962 * state["fRec176"][1]))))) 
		fTemp25 = (self._fConst948 * (((self._fConst949 * state["fRec176"][0]) + (self._fConst963 * state["fRec176"][1])) + (self._fConst949 * state["fRec176"][2]))) 
		state["fRec175"] = state["fRec175"].at[0].set((fTemp25 - (self._fConst945 * ((self._fConst964 * state["fRec175"][2]) + (self._fConst966 * state["fRec175"][1]))))) 
		state["fRec174"] = state["fRec174"].at[0].set(((self._fConst945 * (((self._fConst947 * state["fRec175"][0]) + (self._fConst967 * state["fRec175"][1])) + (self._fConst947 * state["fRec175"][2]))) - (self._fConst942 * ((self._fConst968 * state["fRec174"][2]) + (self._fConst969 * state["fRec174"][1]))))) 
		state["fRec173"] = state["fRec173"].at[0].set(((self._fConst942 * (((self._fConst944 * state["fRec174"][0]) + (self._fConst970 * state["fRec174"][1])) + (self._fConst944 * state["fRec174"][2]))) - (self._fConst938 * ((self._fConst971 * state["fRec173"][2]) + (self._fConst972 * state["fRec173"][1]))))) 
		state["fRec172"] = ((fSlow2 * fRec172_temp) + (fSlow3 * jnp.abs((self._fConst938 * (((self._fConst941 * state["fRec173"][0]) + (self._fConst973 * state["fRec173"][1])) + (self._fConst941 * state["fRec173"][2])))))) 
		fVbargraph25 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec172"])))
		# self.sow("intermediates", "fVbargraph25", fVbargraph25) 
		state["fRec185"] = state["fRec185"].at[0].set((fTemp25 - (self._fConst990 * ((self._fConst993 * state["fRec185"][2]) + (self._fConst994 * state["fRec185"][1]))))) 
		state["fRec184"] = state["fRec184"].at[0].set(((self._fConst990 * (((self._fConst992 * state["fRec185"][0]) + (self._fConst995 * state["fRec185"][1])) + (self._fConst992 * state["fRec185"][2]))) - (self._fConst988 * ((self._fConst996 * state["fRec184"][2]) + (self._fConst997 * state["fRec184"][1]))))) 
		state["fRec183"] = state["fRec183"].at[0].set(((self._fConst988 * (((self._fConst989 * state["fRec184"][0]) + (self._fConst998 * state["fRec184"][1])) + (self._fConst989 * state["fRec184"][2]))) - (self._fConst986 * ((self._fConst999 * state["fRec183"][2]) + (self._fConst1000 * state["fRec183"][1]))))) 
		fTemp26 = (self._fConst986 * (((self._fConst987 * state["fRec183"][0]) + (self._fConst1001 * state["fRec183"][1])) + (self._fConst987 * state["fRec183"][2]))) 
		state["fRec182"] = state["fRec182"].at[0].set((fTemp26 - (self._fConst983 * ((self._fConst1002 * state["fRec182"][2]) + (self._fConst1004 * state["fRec182"][1]))))) 
		state["fRec181"] = state["fRec181"].at[0].set(((self._fConst983 * (((self._fConst985 * state["fRec182"][0]) + (self._fConst1005 * state["fRec182"][1])) + (self._fConst985 * state["fRec182"][2]))) - (self._fConst980 * ((self._fConst1006 * state["fRec181"][2]) + (self._fConst1007 * state["fRec181"][1]))))) 
		state["fRec180"] = state["fRec180"].at[0].set(((self._fConst980 * (((self._fConst982 * state["fRec181"][0]) + (self._fConst1008 * state["fRec181"][1])) + (self._fConst982 * state["fRec181"][2]))) - (self._fConst976 * ((self._fConst1009 * state["fRec180"][2]) + (self._fConst1010 * state["fRec180"][1]))))) 
		state["fRec179"] = ((fSlow2 * fRec179_temp) + (fSlow3 * jnp.abs((self._fConst976 * (((self._fConst979 * state["fRec180"][0]) + (self._fConst1011 * state["fRec180"][1])) + (self._fConst979 * state["fRec180"][2])))))) 
		fVbargraph26 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec179"])))
		# self.sow("intermediates", "fVbargraph26", fVbargraph26) 
		state["fRec192"] = state["fRec192"].at[0].set((fTemp26 - (self._fConst1028 * ((self._fConst1031 * state["fRec192"][2]) + (self._fConst1032 * state["fRec192"][1]))))) 
		state["fRec191"] = state["fRec191"].at[0].set(((self._fConst1028 * (((self._fConst1030 * state["fRec192"][0]) + (self._fConst1033 * state["fRec192"][1])) + (self._fConst1030 * state["fRec192"][2]))) - (self._fConst1026 * ((self._fConst1034 * state["fRec191"][2]) + (self._fConst1035 * state["fRec191"][1]))))) 
		state["fRec190"] = state["fRec190"].at[0].set(((self._fConst1026 * (((self._fConst1027 * state["fRec191"][0]) + (self._fConst1036 * state["fRec191"][1])) + (self._fConst1027 * state["fRec191"][2]))) - (self._fConst1024 * ((self._fConst1037 * state["fRec190"][2]) + (self._fConst1038 * state["fRec190"][1]))))) 
		fTemp27 = (self._fConst1024 * (((self._fConst1025 * state["fRec190"][0]) + (self._fConst1039 * state["fRec190"][1])) + (self._fConst1025 * state["fRec190"][2]))) 
		state["fRec189"] = state["fRec189"].at[0].set((fTemp27 - (self._fConst1021 * ((self._fConst1040 * state["fRec189"][2]) + (self._fConst1042 * state["fRec189"][1]))))) 
		state["fRec188"] = state["fRec188"].at[0].set(((self._fConst1021 * (((self._fConst1023 * state["fRec189"][0]) + (self._fConst1043 * state["fRec189"][1])) + (self._fConst1023 * state["fRec189"][2]))) - (self._fConst1018 * ((self._fConst1044 * state["fRec188"][2]) + (self._fConst1045 * state["fRec188"][1]))))) 
		state["fRec187"] = state["fRec187"].at[0].set(((self._fConst1018 * (((self._fConst1020 * state["fRec188"][0]) + (self._fConst1046 * state["fRec188"][1])) + (self._fConst1020 * state["fRec188"][2]))) - (self._fConst1014 * ((self._fConst1047 * state["fRec187"][2]) + (self._fConst1048 * state["fRec187"][1]))))) 
		state["fRec186"] = ((fSlow2 * fRec186_temp) + (fSlow3 * jnp.abs((self._fConst1014 * (((self._fConst1017 * state["fRec187"][0]) + (self._fConst1049 * state["fRec187"][1])) + (self._fConst1017 * state["fRec187"][2])))))) 
		fVbargraph27 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec186"])))
		# self.sow("intermediates", "fVbargraph27", fVbargraph27) 
		state["fRec199"] = state["fRec199"].at[0].set((fTemp27 - (self._fConst1066 * ((self._fConst1069 * state["fRec199"][2]) + (self._fConst1070 * state["fRec199"][1]))))) 
		state["fRec198"] = state["fRec198"].at[0].set(((self._fConst1066 * (((self._fConst1068 * state["fRec199"][0]) + (self._fConst1071 * state["fRec199"][1])) + (self._fConst1068 * state["fRec199"][2]))) - (self._fConst1064 * ((self._fConst1072 * state["fRec198"][2]) + (self._fConst1073 * state["fRec198"][1]))))) 
		state["fRec197"] = state["fRec197"].at[0].set(((self._fConst1064 * (((self._fConst1065 * state["fRec198"][0]) + (self._fConst1074 * state["fRec198"][1])) + (self._fConst1065 * state["fRec198"][2]))) - (self._fConst1062 * ((self._fConst1075 * state["fRec197"][2]) + (self._fConst1076 * state["fRec197"][1]))))) 
		fTemp28 = (self._fConst1062 * (((self._fConst1063 * state["fRec197"][0]) + (self._fConst1077 * state["fRec197"][1])) + (self._fConst1063 * state["fRec197"][2]))) 
		state["fRec196"] = state["fRec196"].at[0].set((fTemp28 - (self._fConst1059 * ((self._fConst1078 * state["fRec196"][2]) + (self._fConst1080 * state["fRec196"][1]))))) 
		state["fRec195"] = state["fRec195"].at[0].set(((self._fConst1059 * (((self._fConst1061 * state["fRec196"][0]) + (self._fConst1081 * state["fRec196"][1])) + (self._fConst1061 * state["fRec196"][2]))) - (self._fConst1056 * ((self._fConst1082 * state["fRec195"][2]) + (self._fConst1083 * state["fRec195"][1]))))) 
		state["fRec194"] = state["fRec194"].at[0].set(((self._fConst1056 * (((self._fConst1058 * state["fRec195"][0]) + (self._fConst1084 * state["fRec195"][1])) + (self._fConst1058 * state["fRec195"][2]))) - (self._fConst1052 * ((self._fConst1085 * state["fRec194"][2]) + (self._fConst1086 * state["fRec194"][1]))))) 
		state["fRec193"] = ((fSlow2 * fRec193_temp) + (fSlow3 * jnp.abs((self._fConst1052 * (((self._fConst1055 * state["fRec194"][0]) + (self._fConst1087 * state["fRec194"][1])) + (self._fConst1055 * state["fRec194"][2])))))) 
		fVbargraph28 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec193"])))
		# self.sow("intermediates", "fVbargraph28", fVbargraph28) 
		state["fRec203"] = state["fRec203"].at[0].set((fTemp28 - (self._fConst1092 * ((self._fConst1095 * state["fRec203"][2]) + (self._fConst1096 * state["fRec203"][1]))))) 
		state["fRec202"] = state["fRec202"].at[0].set(((self._fConst1092 * (((self._fConst1094 * state["fRec203"][0]) + (self._fConst1097 * state["fRec203"][1])) + (self._fConst1094 * state["fRec203"][2]))) - (self._fConst1090 * ((self._fConst1098 * state["fRec202"][2]) + (self._fConst1099 * state["fRec202"][1]))))) 
		state["fRec201"] = state["fRec201"].at[0].set(((self._fConst1090 * (((self._fConst1091 * state["fRec202"][0]) + (self._fConst1100 * state["fRec202"][1])) + (self._fConst1091 * state["fRec202"][2]))) - (self._fConst1088 * ((self._fConst1101 * state["fRec201"][2]) + (self._fConst1102 * state["fRec201"][1]))))) 
		state["fRec200"] = ((fSlow2 * fRec200_temp) + (fSlow3 * jnp.abs((self._fConst1088 * (((self._fConst1089 * state["fRec201"][0]) + (self._fConst1103 * state["fRec201"][1])) + (self._fConst1089 * state["fRec201"][2])))))) 
		fVbargraph29 = (fSlow0 + (jnp.float32(2e+01) * jnp.log10(state["fRec200"])))
		# self.sow("intermediates", "fVbargraph29", fVbargraph29) 
		fTemp29 = fTemp0 
		_result0 = fTemp29 
		_result1 = fTemp29 
		state["fRec3"] = jnp.roll(state["fRec3"], 1) 
		state["fRec2"] = jnp.roll(state["fRec2"], 1) 
		state["fRec1"] = jnp.roll(state["fRec1"], 1) 
		state["fRec10"] = jnp.roll(state["fRec10"], 1) 
		state["fRec9"] = jnp.roll(state["fRec9"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec7"] = jnp.roll(state["fRec7"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec5"] = jnp.roll(state["fRec5"], 1) 
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
		state["fRec108"] = jnp.roll(state["fRec108"], 1) 
		state["fRec107"] = jnp.roll(state["fRec107"], 1) 
		state["fRec106"] = jnp.roll(state["fRec106"], 1) 
		state["fRec105"] = jnp.roll(state["fRec105"], 1) 
		state["fRec104"] = jnp.roll(state["fRec104"], 1) 
		state["fRec103"] = jnp.roll(state["fRec103"], 1) 
		state["fRec115"] = jnp.roll(state["fRec115"], 1) 
		state["fRec114"] = jnp.roll(state["fRec114"], 1) 
		state["fRec113"] = jnp.roll(state["fRec113"], 1) 
		state["fRec112"] = jnp.roll(state["fRec112"], 1) 
		state["fRec111"] = jnp.roll(state["fRec111"], 1) 
		state["fRec110"] = jnp.roll(state["fRec110"], 1) 
		state["fRec122"] = jnp.roll(state["fRec122"], 1) 
		state["fRec121"] = jnp.roll(state["fRec121"], 1) 
		state["fRec120"] = jnp.roll(state["fRec120"], 1) 
		state["fRec119"] = jnp.roll(state["fRec119"], 1) 
		state["fRec118"] = jnp.roll(state["fRec118"], 1) 
		state["fRec117"] = jnp.roll(state["fRec117"], 1) 
		state["fRec129"] = jnp.roll(state["fRec129"], 1) 
		state["fRec128"] = jnp.roll(state["fRec128"], 1) 
		state["fRec127"] = jnp.roll(state["fRec127"], 1) 
		state["fRec126"] = jnp.roll(state["fRec126"], 1) 
		state["fRec125"] = jnp.roll(state["fRec125"], 1) 
		state["fRec124"] = jnp.roll(state["fRec124"], 1) 
		state["fRec136"] = jnp.roll(state["fRec136"], 1) 
		state["fRec135"] = jnp.roll(state["fRec135"], 1) 
		state["fRec134"] = jnp.roll(state["fRec134"], 1) 
		state["fRec133"] = jnp.roll(state["fRec133"], 1) 
		state["fRec132"] = jnp.roll(state["fRec132"], 1) 
		state["fRec131"] = jnp.roll(state["fRec131"], 1) 
		state["fRec143"] = jnp.roll(state["fRec143"], 1) 
		state["fRec142"] = jnp.roll(state["fRec142"], 1) 
		state["fRec141"] = jnp.roll(state["fRec141"], 1) 
		state["fRec140"] = jnp.roll(state["fRec140"], 1) 
		state["fRec139"] = jnp.roll(state["fRec139"], 1) 
		state["fRec138"] = jnp.roll(state["fRec138"], 1) 
		state["fRec150"] = jnp.roll(state["fRec150"], 1) 
		state["fRec149"] = jnp.roll(state["fRec149"], 1) 
		state["fRec148"] = jnp.roll(state["fRec148"], 1) 
		state["fRec147"] = jnp.roll(state["fRec147"], 1) 
		state["fRec146"] = jnp.roll(state["fRec146"], 1) 
		state["fRec145"] = jnp.roll(state["fRec145"], 1) 
		state["fRec157"] = jnp.roll(state["fRec157"], 1) 
		state["fRec156"] = jnp.roll(state["fRec156"], 1) 
		state["fRec155"] = jnp.roll(state["fRec155"], 1) 
		state["fRec154"] = jnp.roll(state["fRec154"], 1) 
		state["fRec153"] = jnp.roll(state["fRec153"], 1) 
		state["fRec152"] = jnp.roll(state["fRec152"], 1) 
		state["fRec164"] = jnp.roll(state["fRec164"], 1) 
		state["fRec163"] = jnp.roll(state["fRec163"], 1) 
		state["fRec162"] = jnp.roll(state["fRec162"], 1) 
		state["fRec161"] = jnp.roll(state["fRec161"], 1) 
		state["fRec160"] = jnp.roll(state["fRec160"], 1) 
		state["fRec159"] = jnp.roll(state["fRec159"], 1) 
		state["fRec171"] = jnp.roll(state["fRec171"], 1) 
		state["fRec170"] = jnp.roll(state["fRec170"], 1) 
		state["fRec169"] = jnp.roll(state["fRec169"], 1) 
		state["fRec168"] = jnp.roll(state["fRec168"], 1) 
		state["fRec167"] = jnp.roll(state["fRec167"], 1) 
		state["fRec166"] = jnp.roll(state["fRec166"], 1) 
		state["fRec178"] = jnp.roll(state["fRec178"], 1) 
		state["fRec177"] = jnp.roll(state["fRec177"], 1) 
		state["fRec176"] = jnp.roll(state["fRec176"], 1) 
		state["fRec175"] = jnp.roll(state["fRec175"], 1) 
		state["fRec174"] = jnp.roll(state["fRec174"], 1) 
		state["fRec173"] = jnp.roll(state["fRec173"], 1) 
		state["fRec185"] = jnp.roll(state["fRec185"], 1) 
		state["fRec184"] = jnp.roll(state["fRec184"], 1) 
		state["fRec183"] = jnp.roll(state["fRec183"], 1) 
		state["fRec182"] = jnp.roll(state["fRec182"], 1) 
		state["fRec181"] = jnp.roll(state["fRec181"], 1) 
		state["fRec180"] = jnp.roll(state["fRec180"], 1) 
		state["fRec192"] = jnp.roll(state["fRec192"], 1) 
		state["fRec191"] = jnp.roll(state["fRec191"], 1) 
		state["fRec190"] = jnp.roll(state["fRec190"], 1) 
		state["fRec189"] = jnp.roll(state["fRec189"], 1) 
		state["fRec188"] = jnp.roll(state["fRec188"], 1) 
		state["fRec187"] = jnp.roll(state["fRec187"], 1) 
		state["fRec199"] = jnp.roll(state["fRec199"], 1) 
		state["fRec198"] = jnp.roll(state["fRec198"], 1) 
		state["fRec197"] = jnp.roll(state["fRec197"], 1) 
		state["fRec196"] = jnp.roll(state["fRec196"], 1) 
		state["fRec195"] = jnp.roll(state["fRec195"], 1) 
		state["fRec194"] = jnp.roll(state["fRec194"], 1) 
		state["fRec203"] = jnp.roll(state["fRec203"], 1) 
		state["fRec202"] = jnp.roll(state["fRec202"], 1) 
		state["fRec201"] = jnp.roll(state["fRec201"], 1) 
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
