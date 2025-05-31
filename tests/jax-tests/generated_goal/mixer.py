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
		return 8
	
	@property
	def num_outputs(self):
		return 2
	
	# fmt: off
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fRec0"] = np.float32(0)
		state["fRec1"] = np.float32(0)
		state["fRec10"] = np.float32(0)
		state["fRec11"] = np.float32(0)
		state["fRec12"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec14"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec16"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec18"] = np.float32(0)
		state["fRec2"] = np.float32(0)
		state["fRec3"] = np.float32(0)
		state["fRec4"] = np.float32(0)
		state["fRec5"] = np.float32(0)
		state["fRec6"] = np.float32(0)
		state["fRec7"] = np.float32(0)
		state["fRec8"] = np.float32(0)
		state["fRec9"] = np.float32(0)
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
		ui_path.append("mixer") 
		ui_path.append("Ch 0") 
		self.add_nentry("fEntry0", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider1", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph0", ui_path, "vbargraph0", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox0", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 1") 
		self.add_nentry("fEntry1", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider2", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph1", ui_path, "vbargraph1", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox1", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 2") 
		self.add_nentry("fEntry2", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider3", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph2", ui_path, "vbargraph2", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox2", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 3") 
		self.add_nentry("fEntry3", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider4", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph3", ui_path, "vbargraph3", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox3", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 4") 
		self.add_nentry("fEntry4", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider5", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph4", ui_path, "vbargraph4", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox4", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 5") 
		self.add_nentry("fEntry5", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider6", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph5", ui_path, "vbargraph5", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox5", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 6") 
		self.add_nentry("fEntry6", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider7", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph6", ui_path, "vbargraph6", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox6", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("Ch 7") 
		self.add_nentry("fEntry7", ui_path, "pan", 0.0, -9e+01, 9e+01, 1.0, unnorm_funcs, "linear") 
		ui_path.append("0x00") 
		self.add_vslider("fVslider8", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		self.add_vbargraph("fVbargraph7", ui_path, "vbargraph7", -7e+01, 5.0) 
		ui_path.pop()
		self.add_button("fCheckbox7", ui_path, "mute", unnorm_funcs) 
		ui_path.pop()
		ui_path.append("stereo out") 
		ui_path.append("L") 
		self.add_vbargraph("fVbargraph8", ui_path, "vbargraph8", -7e+01, 5.0) 
		ui_path.pop()
		ui_path.append("R") 
		self.add_vbargraph("fVbargraph9", ui_path, "vbargraph9", -7e+01, 5.0) 
		ui_path.pop()
		self.add_vslider("fVslider0", ui_path, "0x00", 0.0, -7e+01, 4.0, unnorm_funcs, "linear") 
		ui_path.pop()
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
		self._fConst0 = (np.float32(1.0) / np.minimum(np.float32(1.92e+05), np.maximum(np.float32(1.0), (self.sample_rate)))) 
		
	def tick(self, params: dict, state: dict, inputs: jnp.array) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider0"]))) 
		fSlow1 = (jnp.float32(0.0055555557) * (params["fEntry0"] + jnp.float32(-9e+01))) 
		fSlow2 = jnp.sqrt(-fSlow1) 
		fSlow3 = (jnp.float32(1.0) - params["fCheckbox0"]) 
		fSlow4 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider1"]))) 
		fSlow5 = (jnp.float32(0.0055555557) * (params["fEntry1"] + jnp.float32(-9e+01))) 
		fSlow6 = jnp.sqrt(-fSlow5) 
		fSlow7 = (jnp.float32(1.0) - params["fCheckbox1"]) 
		fSlow8 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider2"]))) 
		fSlow9 = (jnp.float32(0.0055555557) * (params["fEntry2"] + jnp.float32(-9e+01))) 
		fSlow10 = jnp.sqrt(-fSlow9) 
		fSlow11 = (jnp.float32(1.0) - params["fCheckbox2"]) 
		fSlow12 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider3"]))) 
		fSlow13 = (jnp.float32(0.0055555557) * (params["fEntry3"] + jnp.float32(-9e+01))) 
		fSlow14 = jnp.sqrt(-fSlow13) 
		fSlow15 = (jnp.float32(1.0) - params["fCheckbox3"]) 
		fSlow16 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider4"]))) 
		fSlow17 = (jnp.float32(0.0055555557) * (params["fEntry4"] + jnp.float32(-9e+01))) 
		fSlow18 = jnp.sqrt(-fSlow17) 
		fSlow19 = (jnp.float32(1.0) - params["fCheckbox4"]) 
		fSlow20 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider5"]))) 
		fSlow21 = (jnp.float32(0.0055555557) * (params["fEntry5"] + jnp.float32(-9e+01))) 
		fSlow22 = jnp.sqrt(-fSlow21) 
		fSlow23 = (jnp.float32(1.0) - params["fCheckbox5"]) 
		fSlow24 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider6"]))) 
		fSlow25 = (jnp.float32(0.0055555557) * (params["fEntry6"] + jnp.float32(-9e+01))) 
		fSlow26 = jnp.sqrt(-fSlow25) 
		fSlow27 = (jnp.float32(1.0) - params["fCheckbox6"]) 
		fSlow28 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider7"]))) 
		fSlow29 = (jnp.float32(0.0055555557) * (params["fEntry7"] + jnp.float32(-9e+01))) 
		fSlow30 = jnp.sqrt(-fSlow29) 
		fSlow31 = (jnp.float32(1.0) - params["fCheckbox7"]) 
		fSlow32 = (jnp.float32(0.001) * jnp.power(jnp.float32(1e+01), (jnp.float32(0.05) * params["fVslider8"]))) 
		fSlow33 = jnp.sqrt((fSlow1 + jnp.float32(1.0))) 
		fSlow34 = jnp.sqrt((fSlow5 + jnp.float32(1.0))) 
		fSlow35 = jnp.sqrt((fSlow9 + jnp.float32(1.0))) 
		fSlow36 = jnp.sqrt((fSlow13 + jnp.float32(1.0))) 
		fSlow37 = jnp.sqrt((fSlow17 + jnp.float32(1.0))) 
		fSlow38 = jnp.sqrt((fSlow21 + jnp.float32(1.0))) 
		fSlow39 = jnp.sqrt((fSlow25 + jnp.float32(1.0))) 
		fSlow40 = jnp.sqrt((fSlow29 + jnp.float32(1.0))) 
		fRec1_temp = state["fRec1"] 
		fRec3_temp = state["fRec3"] 
		fRec2_temp = state["fRec2"] 
		fRec5_temp = state["fRec5"] 
		fRec4_temp = state["fRec4"] 
		fRec7_temp = state["fRec7"] 
		fRec6_temp = state["fRec6"] 
		fRec9_temp = state["fRec9"] 
		fRec8_temp = state["fRec8"] 
		fRec11_temp = state["fRec11"] 
		fRec10_temp = state["fRec10"] 
		fRec13_temp = state["fRec13"] 
		fRec12_temp = state["fRec12"] 
		fRec15_temp = state["fRec15"] 
		fRec14_temp = state["fRec14"] 
		fRec17_temp = state["fRec17"] 
		fRec16_temp = state["fRec16"] 
		fRec0_temp = state["fRec0"] 
		fRec18_temp = state["fRec18"] 
		state["fRec1"] = (fSlow0 + (jnp.float32(0.999) * fRec1_temp)) 
		state["fRec3"] = (fSlow4 + (jnp.float32(0.999) * fRec3_temp)) 
		fTemp0 = (fSlow3 * (inputs[0] * state["fRec3"])) 
		state["fRec2"] = jnp.maximum((fRec2_temp - self._fConst0), jnp.abs(fTemp0)) 
		fVbargraph0 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec2"])))
		self.sow("intermediates", "fVbargraph0", fVbargraph0) 
		fTemp1 = fTemp0 
		state["fRec5"] = (fSlow8 + (jnp.float32(0.999) * fRec5_temp)) 
		fTemp2 = (fSlow7 * (inputs[1] * state["fRec5"])) 
		state["fRec4"] = jnp.maximum((fRec4_temp - self._fConst0), jnp.abs(fTemp2)) 
		fVbargraph1 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec4"])))
		self.sow("intermediates", "fVbargraph1", fVbargraph1) 
		fTemp3 = fTemp2 
		state["fRec7"] = (fSlow12 + (jnp.float32(0.999) * fRec7_temp)) 
		fTemp4 = (fSlow11 * (inputs[2] * state["fRec7"])) 
		state["fRec6"] = jnp.maximum((fRec6_temp - self._fConst0), jnp.abs(fTemp4)) 
		fVbargraph2 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec6"])))
		self.sow("intermediates", "fVbargraph2", fVbargraph2) 
		fTemp5 = fTemp4 
		state["fRec9"] = (fSlow16 + (jnp.float32(0.999) * fRec9_temp)) 
		fTemp6 = (fSlow15 * (inputs[3] * state["fRec9"])) 
		state["fRec8"] = jnp.maximum((fRec8_temp - self._fConst0), jnp.abs(fTemp6)) 
		fVbargraph3 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec8"])))
		self.sow("intermediates", "fVbargraph3", fVbargraph3) 
		fTemp7 = fTemp6 
		state["fRec11"] = (fSlow20 + (jnp.float32(0.999) * fRec11_temp)) 
		fTemp8 = (fSlow19 * (inputs[4] * state["fRec11"])) 
		state["fRec10"] = jnp.maximum((fRec10_temp - self._fConst0), jnp.abs(fTemp8)) 
		fVbargraph4 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec10"])))
		self.sow("intermediates", "fVbargraph4", fVbargraph4) 
		fTemp9 = fTemp8 
		state["fRec13"] = (fSlow24 + (jnp.float32(0.999) * fRec13_temp)) 
		fTemp10 = (fSlow23 * (inputs[5] * state["fRec13"])) 
		state["fRec12"] = jnp.maximum((fRec12_temp - self._fConst0), jnp.abs(fTemp10)) 
		fVbargraph5 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec12"])))
		self.sow("intermediates", "fVbargraph5", fVbargraph5) 
		fTemp11 = fTemp10 
		state["fRec15"] = (fSlow28 + (jnp.float32(0.999) * fRec15_temp)) 
		fTemp12 = (fSlow27 * (inputs[6] * state["fRec15"])) 
		state["fRec14"] = jnp.maximum((fRec14_temp - self._fConst0), jnp.abs(fTemp12)) 
		fVbargraph6 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec14"])))
		self.sow("intermediates", "fVbargraph6", fVbargraph6) 
		fTemp13 = fTemp12 
		state["fRec17"] = (fSlow32 + (jnp.float32(0.999) * fRec17_temp)) 
		fTemp14 = (fSlow31 * (inputs[7] * state["fRec17"])) 
		state["fRec16"] = jnp.maximum((fRec16_temp - self._fConst0), jnp.abs(fTemp14)) 
		fVbargraph7 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec16"])))
		self.sow("intermediates", "fVbargraph7", fVbargraph7) 
		fTemp15 = fTemp14 
		fTemp16 = (state["fRec1"] * ((((((((fSlow2 * fTemp1) + (fSlow6 * fTemp3)) + (fSlow10 * fTemp5)) + (fSlow14 * fTemp7)) + (fSlow18 * fTemp9)) + (fSlow22 * fTemp11)) + (fSlow26 * fTemp13)) + (fSlow30 * fTemp15))) 
		state["fRec0"] = jnp.maximum((fRec0_temp - self._fConst0), jnp.abs(fTemp16)) 
		fVbargraph8 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec0"])))
		self.sow("intermediates", "fVbargraph8", fVbargraph8) 
		_result0 = fTemp16 
		fTemp17 = (state["fRec1"] * ((((((((fSlow33 * fTemp1) + (fSlow34 * fTemp3)) + (fSlow35 * fTemp5)) + (fSlow36 * fTemp7)) + (fSlow37 * fTemp9)) + (fSlow38 * fTemp11)) + (fSlow39 * fTemp13)) + (fSlow40 * fTemp15))) 
		state["fRec18"] = jnp.maximum((fRec18_temp - self._fConst0), jnp.abs(fTemp17)) 
		fVbargraph9 = (jnp.float32(2e+01) * jnp.log10(jnp.maximum(jnp.float32(0.00031622776), state["fRec18"])))
		self.sow("intermediates", "fVbargraph9", fVbargraph9) 
		_result1 = fTemp17 
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
