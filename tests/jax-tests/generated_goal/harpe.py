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
		state["fRec11"] = np.float32(0)
		state["fRec13"] = np.float32(0)
		state["fRec15"] = np.float32(0)
		state["fRec17"] = np.float32(0)
		state["fRec19"] = np.float32(0)
		state["fRec2"] = np.float32(0)
		state["fRec21"] = np.float32(0)
		state["fRec23"] = np.float32(0)
		state["fRec3"] = np.float32(0)
		state["fRec5"] = np.float32(0)
		state["fRec7"] = np.float32(0)
		state["fRec9"] = np.float32(0)
		state["iRec1"] = np.int32(0)
		state["iVec0"] = np.int32(0)
		state["iVec10"] = np.int32(0)
		state["iVec12"] = np.int32(0)
		state["iVec14"] = np.int32(0)
		state["iVec16"] = np.int32(0)
		state["iVec18"] = np.int32(0)
		state["iVec2"] = np.int32(0)
		state["iVec20"] = np.int32(0)
		state["iVec4"] = np.int32(0)
		state["iVec6"] = np.int32(0)
		state["iVec8"] = np.int32(0)
		# Initialize array delays
		state["fVec1"] = np.zeros((128,), dtype=np.float32)
		state["fRec0"] = np.zeros((3,), dtype=np.float32)
		state["fVec3"] = np.zeros((128,), dtype=np.float32)
		state["fRec4"] = np.zeros((3,), dtype=np.float32)
		state["fVec5"] = np.zeros((128,), dtype=np.float32)
		state["fRec6"] = np.zeros((3,), dtype=np.float32)
		state["fVec7"] = np.zeros((128,), dtype=np.float32)
		state["fRec8"] = np.zeros((3,), dtype=np.float32)
		state["fVec9"] = np.zeros((64,), dtype=np.float32)
		state["fRec10"] = np.zeros((3,), dtype=np.float32)
		state["fVec11"] = np.zeros((64,), dtype=np.float32)
		state["fRec12"] = np.zeros((3,), dtype=np.float32)
		state["fVec13"] = np.zeros((64,), dtype=np.float32)
		state["fRec14"] = np.zeros((3,), dtype=np.float32)
		state["fVec15"] = np.zeros((64,), dtype=np.float32)
		state["fRec16"] = np.zeros((3,), dtype=np.float32)
		state["fVec17"] = np.zeros((64,), dtype=np.float32)
		state["fRec18"] = np.zeros((3,), dtype=np.float32)
		state["fVec19"] = np.zeros((32,), dtype=np.float32)
		state["fRec20"] = np.zeros((3,), dtype=np.float32)
		state["fVec21"] = np.zeros((32,), dtype=np.float32)
		state["fRec22"] = np.zeros((3,), dtype=np.float32)
		# Initialize IOTA variables
		state["IOTA0"] = np.int32(0)
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
		ui_path.append("Harpe") 
		self.add_hslider("fHslider0", ui_path, "attenuation", 0.0, 0.0, 0.01, unnorm_funcs, "linear") 
		self.add_hslider("fHslider2", ui_path, "hand", 0.43, 0.0, 1.0, unnorm_funcs, "linear") 
		self.add_hslider("fHslider1", ui_path, "level", 0.5, 0.0, 1.0, unnorm_funcs, "linear") 
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
	def tick(self, params: dict, state: dict, inputs: jnp.array) -> Tuple[dict, jnp.ndarray]:
		
		fSlow0 = (jnp.float32(0.5) * (jnp.float32(1.0) - params["fHslider0"])) 
		fSlow1 = (jnp.float32(4.656613e-10) * jnp.power(params["fHslider1"], jnp.float32(2.0))) 
		fSlow2 = (jnp.float32(0.1) * params["fHslider2"]) 
		iRec1_temp = state["iRec1"] 
		fRec3_temp = state["fRec3"] 
		iVec0_temp = state["iVec0"] 
		fRec2_temp = state["fRec2"] 
		iVec2_temp = state["iVec2"] 
		fRec5_temp = state["fRec5"] 
		iVec4_temp = state["iVec4"] 
		fRec7_temp = state["fRec7"] 
		iVec6_temp = state["iVec6"] 
		fRec9_temp = state["fRec9"] 
		iVec8_temp = state["iVec8"] 
		fRec11_temp = state["fRec11"] 
		iVec10_temp = state["iVec10"] 
		fRec13_temp = state["fRec13"] 
		iVec12_temp = state["iVec12"] 
		fRec15_temp = state["fRec15"] 
		iVec14_temp = state["iVec14"] 
		fRec17_temp = state["fRec17"] 
		iVec16_temp = state["iVec16"] 
		fRec19_temp = state["fRec19"] 
		iVec18_temp = state["iVec18"] 
		fRec21_temp = state["fRec21"] 
		iVec20_temp = state["iVec20"] 
		fRec23_temp = state["fRec23"] 
		state["iRec1"] = ((jnp.int32(1103515245) * iRec1_temp) + jnp.int32(12345)) 
		fTemp0 = (state["iRec1"]) 
		state["fRec3"] = (fSlow2 + (jnp.float32(0.9) * fRec3_temp)) 
		fTemp1 = jnp.minimum(state["fRec3"], fRec3_temp) 
		fTemp2 = jnp.maximum(state["fRec3"], fRec3_temp) 
		iTemp3 = ((fTemp1 < jnp.float32(0.045454547)).astype(jnp.int32) & (jnp.float32(0.045454547) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec0"] = iTemp3 
		state["fRec2"] = ((fRec2_temp + ((((iTemp3 - iVec0_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.009977324) * ((fRec2_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec1"] = state["fVec1"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow0 * (state["fRec0"][1] + state["fRec0"][2])) + (fSlow1 * (fTemp0 * ((state["fRec2"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec0"] = state["fRec0"].at[0].set(state["fVec1"][((state["IOTA0"] - 99) & 127).astype(jnp.int32)]) 
		iTemp4 = ((fTemp1 < jnp.float32(0.13636364)).astype(jnp.int32) & (jnp.float32(0.13636364) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec2"] = iTemp4 
		state["fRec5"] = ((fRec5_temp + ((((iTemp4 - iVec2_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.011460936) * ((fRec5_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec3"] = state["fVec3"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow0 * (state["fRec4"][1] + state["fRec4"][2])) + (fSlow1 * (fTemp0 * ((state["fRec5"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec4"] = state["fRec4"].at[0].set(state["fVec3"][((state["IOTA0"] - 86) & 127).astype(jnp.int32)]) 
		iTemp5 = ((fTemp1 < jnp.float32(0.22727273)).astype(jnp.int32) & (jnp.float32(0.22727273) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec4"] = iTemp5 
		state["fRec7"] = ((fRec7_temp + ((((iTemp5 - iVec4_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.013165158) * ((fRec7_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec5"] = state["fVec5"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow0 * (state["fRec6"][1] + state["fRec6"][2])) + (fSlow1 * (fTemp0 * ((state["fRec7"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec6"] = state["fRec6"].at[0].set(state["fVec5"][((state["IOTA0"] - 74) & 127).astype(jnp.int32)]) 
		iTemp6 = ((fTemp1 < jnp.float32(0.3181818)).astype(jnp.int32) & (jnp.float32(0.3181818) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec6"] = iTemp6 
		state["fRec9"] = ((fRec9_temp + ((((iTemp6 - iVec6_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.0151227955) * ((fRec9_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec7"] = state["fVec7"].at[(state["IOTA0"] & 127).astype(jnp.int32)].set(((fSlow0 * (state["fRec8"][1] + state["fRec8"][2])) + (fSlow1 * (fTemp0 * ((state["fRec9"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec8"] = state["fRec8"].at[0].set(state["fVec7"][((state["IOTA0"] - 65) & 127).astype(jnp.int32)]) 
		iTemp7 = ((fTemp1 < jnp.float32(0.4090909)).astype(jnp.int32) & (jnp.float32(0.4090909) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec8"] = iTemp7 
		state["fRec11"] = ((fRec11_temp + ((((iTemp7 - iVec8_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.01737153) * ((fRec11_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec9"] = state["fVec9"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow0 * (state["fRec10"][1] + state["fRec10"][2])) + (fSlow1 * (fTemp0 * ((state["fRec11"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec10"] = state["fRec10"].at[0].set(state["fVec9"][((state["IOTA0"] - 56) & 63).astype(jnp.int32)]) 
		iTemp8 = ((fTemp1 < jnp.float32(0.5)).astype(jnp.int32) & (jnp.float32(0.5) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec10"] = iTemp8 
		state["fRec13"] = ((fRec13_temp + ((((iTemp8 - iVec10_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.019954648) * ((fRec13_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec11"] = state["fVec11"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow0 * (state["fRec12"][1] + state["fRec12"][2])) + (fSlow1 * (fTemp0 * ((state["fRec13"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec12"] = state["fRec12"].at[0].set(state["fVec11"][((state["IOTA0"] - 49) & 63).astype(jnp.int32)]) 
		fTemp9 = (jnp.float32(0.70710677) * state["fRec12"][0]) 
		iTemp10 = ((fTemp1 < jnp.float32(0.59090906)).astype(jnp.int32) & (jnp.float32(0.59090906) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec12"] = iTemp10 
		state["fRec15"] = ((fRec15_temp + ((((iTemp10 - iVec12_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.022921871) * ((fRec15_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec13"] = state["fVec13"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow0 * (state["fRec14"][1] + state["fRec14"][2])) + (fSlow1 * (fTemp0 * ((state["fRec15"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec14"] = state["fRec14"].at[0].set(state["fVec13"][((state["IOTA0"] - 42) & 63).astype(jnp.int32)]) 
		iTemp11 = ((fTemp1 < jnp.float32(0.6818182)).astype(jnp.int32) & (jnp.float32(0.6818182) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec14"] = iTemp11 
		state["fRec17"] = ((fRec17_temp + ((((iTemp11 - iVec14_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.026330316) * ((fRec17_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec15"] = state["fVec15"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow0 * (state["fRec16"][1] + state["fRec16"][2])) + (fSlow1 * (fTemp0 * ((state["fRec17"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec16"] = state["fRec16"].at[0].set(state["fVec15"][((state["IOTA0"] - 36) & 63).astype(jnp.int32)]) 
		iTemp12 = ((fTemp1 < jnp.float32(0.77272725)).astype(jnp.int32) & (jnp.float32(0.77272725) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec16"] = iTemp12 
		state["fRec19"] = ((fRec19_temp + ((((iTemp12 - iVec16_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.030245591) * ((fRec19_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec17"] = state["fVec17"].at[(state["IOTA0"] & 63).astype(jnp.int32)].set(((fSlow0 * (state["fRec18"][1] + state["fRec18"][2])) + (fSlow1 * (fTemp0 * ((state["fRec19"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec18"] = state["fRec18"].at[0].set(state["fVec17"][((state["IOTA0"] - 32) & 63).astype(jnp.int32)]) 
		iTemp13 = ((fTemp1 < jnp.float32(0.8636364)).astype(jnp.int32) & (jnp.float32(0.8636364) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec18"] = iTemp13 
		state["fRec21"] = ((fRec21_temp + ((((iTemp13 - iVec18_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.03474306) * ((fRec21_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec19"] = state["fVec19"].at[(state["IOTA0"] & 31).astype(jnp.int32)].set(((fSlow0 * (state["fRec20"][1] + state["fRec20"][2])) + (fSlow1 * (fTemp0 * ((state["fRec21"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec20"] = state["fRec20"].at[0].set(state["fVec19"][((state["IOTA0"] - 27) & 31).astype(jnp.int32)]) 
		iTemp14 = ((fTemp1 < jnp.float32(0.95454544)).astype(jnp.int32) & (jnp.float32(0.95454544) < fTemp2).astype(jnp.int32)).astype(jnp.int32) 
		state["iVec20"] = iTemp14 
		state["fRec23"] = ((fRec23_temp + ((((iTemp14 - iVec20_temp)) > jnp.float32(0.0)).astype(jnp.int32))) - (jnp.float32(0.039909296) * ((fRec23_temp > jnp.float32(0.0)).astype(jnp.int32)))) 
		state["fVec21"] = state["fVec21"].at[(state["IOTA0"] & 31).astype(jnp.int32)].set(((fSlow0 * (state["fRec22"][1] + state["fRec22"][2])) + (fSlow1 * (fTemp0 * ((state["fRec23"] > jnp.float32(0.0)).astype(jnp.int32)))))) 
		state["fRec22"] = state["fRec22"].at[0].set(state["fVec21"][((state["IOTA0"] - 24) & 31).astype(jnp.int32)]) 
		_result0 = (((((((((((jnp.float32(0.9770084) * state["fRec0"][0]) + (jnp.float32(0.9293204) * state["fRec4"][0])) + (jnp.float32(0.87904906) * state["fRec6"][0])) + (jnp.float32(0.8257228) * state["fRec8"][0])) + (jnp.float32(0.76870614) * state["fRec10"][0])) + fTemp9) + (jnp.float32(0.6396021) * state["fRec14"][0])) + (jnp.float32(0.56407607) * state["fRec16"][0])) + (jnp.float32(0.4767313) * state["fRec18"][0])) + (jnp.float32(0.36927447) * state["fRec20"][0])) + (jnp.float32(0.21320072) * state["fRec22"][0])) 
		_result1 = ((((((fTemp9 + (((((jnp.float32(0.21320072) * state["fRec0"][0]) + (jnp.float32(0.36927447) * state["fRec4"][0])) + (jnp.float32(0.4767313) * state["fRec6"][0])) + (jnp.float32(0.56407607) * state["fRec8"][0])) + (jnp.float32(0.6396021) * state["fRec10"][0]))) + (jnp.float32(0.76870614) * state["fRec14"][0])) + (jnp.float32(0.8257228) * state["fRec16"][0])) + (jnp.float32(0.87904906) * state["fRec18"][0])) + (jnp.float32(0.9293204) * state["fRec20"][0])) + (jnp.float32(0.9770084) * state["fRec22"][0])) 
		state["IOTA0"] = (state["IOTA0"] + jnp.int32(1)) 
		state["fRec0"] = jnp.roll(state["fRec0"], 1) 
		state["fRec4"] = jnp.roll(state["fRec4"], 1) 
		state["fRec6"] = jnp.roll(state["fRec6"], 1) 
		state["fRec8"] = jnp.roll(state["fRec8"], 1) 
		state["fRec10"] = jnp.roll(state["fRec10"], 1) 
		state["fRec12"] = jnp.roll(state["fRec12"], 1) 
		state["fRec14"] = jnp.roll(state["fRec14"], 1) 
		state["fRec16"] = jnp.roll(state["fRec16"], 1) 
		state["fRec18"] = jnp.roll(state["fRec18"], 1) 
		state["fRec20"] = jnp.roll(state["fRec20"], 1) 
		state["fRec22"] = jnp.roll(state["fRec22"], 1) 
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
