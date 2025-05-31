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
		return 104
	
	@property
	def num_outputs(self):
		return 76
	
	# fmt: off
	def _initialize_carry(self, x: jnp.ndarray, length: int):
		state = {}
		
		# Initialize scalar delays
		state["fVec0"] = np.float32(0)
		state["iVec1"] = np.int32(0)
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
		ui_path.append("math_") 
		ui_path.pop()
		
		self._unnorm_funcs = unnorm_funcs
		# Initialize other constants
	def tick(self, params: dict, state: dict, inputs: jnp.array) -> Tuple[dict, jnp.ndarray]:
		
		fVec0_temp = state["fVec0"] 
		iVec1_temp = state["iVec1"] 
		state["fVec0"] = jnp.float32(2.0) 
		_result0 = jnp.power(fVec0_temp, jnp.float32(3e+01)) 
		state["iVec1"] = jnp.int32(2) 
		_result1 = jnp.int32(jnp.power((iVec1_temp), (jnp.int32(30)))) 
		_result2 = jnp.float32(1.0737418e+09) 
		_result3 = jnp.int32(1073741824) 
		_result4 = jnp.isnan(inputs[0]) 
		_result5 = jnp.isinf(inputs[1]) 
		_result6 = jnp.copysign(inputs[2], inputs[3]) 
		_result7 = (jnp.int32(inputs[4]) >> jnp.int32(inputs[5])) 
		_result8 = (jnp.int32(inputs[6]) << jnp.int32(inputs[7])) 
		_result9 = (jnp.int32((jnp.float32(10.5) * inputs[8])) % jnp.int32(3)) 
		_result10 = jnp.mod((jnp.float32(10.5) * inputs[9]), jnp.float32(3.0)) 
		_result11 = (jnp.int32(inputs[10]) & jnp.int32(inputs[11])).astype(jnp.int32) 
		_result12 = (jnp.int32((jnp.float32(3.5) * inputs[12])) & jnp.int32((jnp.float32(2.4) * inputs[13]))).astype(jnp.int32) 
		_result13 = (jnp.int32((jnp.float32(3.5) * inputs[14])) & jnp.int32((jnp.float32(2.4) * inputs[15]))).astype(jnp.int32) 
		_result14 = (jnp.float32(2.4) * ((jnp.int32((jnp.float32(3.5) * inputs[16])) & jnp.int32(inputs[17])).astype(jnp.int32))) 
		_result15 = (jnp.int32((jnp.float32(3.5) * inputs[18])) | jnp.int32((jnp.float32(2.4) * inputs[19]))).astype(jnp.int32) 
		_result16 = (jnp.int32((jnp.float32(3.5) * inputs[20])) | jnp.int32((jnp.float32(2.4) * inputs[21]))).astype(jnp.int32) 
		_result17 = (jnp.int32((jnp.float32(3.5) * inputs[22])) ^ jnp.int32((jnp.float32(2.4) * inputs[23]))) 
		_result18 = (jnp.int32((jnp.float32(3.5) * inputs[24])) ^ jnp.int32((jnp.float32(2.4) * inputs[25]))) 
		_result19 = jnp.int32(jnp.power((jnp.int32((jnp.float32(3.5) * inputs[26]))), (jnp.int32((jnp.float32(2.4) * inputs[27]))))) 
		_result20 = (jnp.float32(2.4) * (inputs[28] * jnp.power(jnp.float32(3.5), inputs[29]))) 
		_result21 = (jnp.int32((jnp.float32(3.5) * inputs[30])) > jnp.int32((jnp.float32(2.4) * inputs[31]))).astype(jnp.int32) 
		_result22 = ((jnp.float32(3.5) * inputs[32]) > (jnp.int32((jnp.float32(2.4) * inputs[33])))).astype(jnp.int32) 
		_result23 = ((jnp.float32(3.5) * inputs[34]) > (jnp.float32(2.4) * inputs[35])).astype(jnp.int32) 
		_result24 = (jnp.int32((jnp.float32(3.5) * inputs[36])) >= jnp.int32((jnp.float32(2.4) * inputs[37]))).astype(jnp.int32) 
		_result25 = ((jnp.float32(3.5) * inputs[38]) >= (jnp.float32(2.4) * inputs[39])).astype(jnp.int32) 
		_result26 = (jnp.int32((jnp.float32(3.5) * inputs[40])) < jnp.int32((jnp.float32(2.4) * inputs[41]))).astype(jnp.int32) 
		_result27 = ((jnp.float32(3.5) * inputs[42]) < (jnp.float32(2.4) * inputs[43])).astype(jnp.int32) 
		_result28 = (jnp.int32((jnp.float32(3.5) * inputs[44])) <= jnp.int32((jnp.float32(2.4) * inputs[45]))).astype(jnp.int32) 
		_result29 = ((jnp.float32(3.5) * inputs[46]) <= (jnp.float32(2.4) * inputs[47])).astype(jnp.int32) 
		_result30 = (jnp.int32((jnp.float32(3.5) * inputs[48])) == jnp.int32((jnp.float32(2.4) * inputs[49]))).astype(jnp.int32) 
		_result31 = ((jnp.float32(3.5) * inputs[50]) == (jnp.float32(2.4) * inputs[51])).astype(jnp.int32) 
		_result32 = (jnp.int32((jnp.float32(3.5) * inputs[52])) != jnp.int32((jnp.float32(2.4) * inputs[53]))).astype(jnp.int32) 
		_result33 = ((jnp.float32(3.5) * inputs[54]) != (jnp.float32(2.4) * inputs[55])).astype(jnp.int32) 
		_result34 = jnp.abs(jnp.int32((jnp.float32(4.4) * inputs[56]))) 
		_result35 = jnp.abs(jnp.int32(-((jnp.float32(4.4) * inputs[57])))) 
		_result36 = jnp.abs((jnp.float32(4.4) * inputs[58])) 
		_result37 = jnp.abs(-((jnp.float32(4.4) * inputs[59]))) 
		_result38 = jnp.arccos((jnp.float32(0.5) * inputs[60])) 
		_result39 = jnp.arccos((jnp.int32((jnp.float32(0.5) * inputs[61])))) 
		_result40 = jnp.arcsin((jnp.float32(0.5) * inputs[62])) 
		_result41 = jnp.arctan((jnp.float32(0.5) * inputs[63])) 
		_result42 = jnp.arctan2((jnp.float32(0.5) * inputs[64]), jnp.float32(4.0)) 
		_result43 = jnp.arctan2((jnp.int32((jnp.float32(0.5) * inputs[65]))), jnp.float32(4.0)) 
		_result44 = jnp.arctan2((jnp.int32((jnp.float32(0.5) * inputs[66]))), jnp.float32(4.0)) 
		_result45 = jnp.ceil((jnp.float32(1.3) * inputs[67])) 
		_result46 = jnp.cos((jnp.float32(0.3) * inputs[68])) 
		_result47 = jnp.exp((jnp.float32(0.5) * inputs[69])) 
		_result48 = jnp.floor((jnp.float32(6.5) * inputs[70])) 
		_result49 = jnp.mod((jnp.float32(9.2) * inputs[71]), jnp.float32(2.0)) 
		_result50 = jnp.log((jnp.float32(0.5) * (inputs[72] + jnp.float32(1.0)))) 
		_result51 = jnp.log10((jnp.float32(0.5) * (inputs[73] + jnp.float32(1.0)))) 
		_result52 = jnp.maximum((jnp.float32(0.5) * inputs[74]), (jnp.float32(0.4) * inputs[75])) 
		_result53 = jnp.minimum((jnp.float32(0.5) * inputs[76]), (jnp.float32(0.4) * inputs[77])) 
		_result54 = jnp.maximum((jnp.float32(0.5) * inputs[78]), (jnp.int32((jnp.float32(0.4) * inputs[79])))) 
		_result55 = jnp.minimum((jnp.float32(0.5) * inputs[80]), (jnp.int32((jnp.float32(0.4) * inputs[81])))) 
		_result56 = jnp.maximum(jnp.int32((jnp.float32(3.5) * inputs[82])), jnp.int32((jnp.float32(2.4) * inputs[83]))) 
		_result57 = jnp.minimum(jnp.int32((jnp.float32(3.5) * inputs[84])), jnp.int32((jnp.float32(2.4) * inputs[85]))) 
		_result58 = jnp.power((jnp.float32(0.5) * inputs[86]), jnp.float32(0.3)) 
		_result59 = jnp.power((jnp.int32((jnp.float32(0.5) * inputs[87]))), jnp.float32(0.3)) 
		_result60 = jnp.power((jnp.float32(0.5) * inputs[88]), jnp.float32(3.0)) 
		_result61 = jnp.int32(jnp.power((jnp.int32((jnp.float32(0.5) * inputs[89]))), (jnp.int32(3)))) 
		_result62 = jnp.int32(jnp.power((jnp.int32((jnp.float32(0.5) * inputs[90]))), (jnp.int32(3)))) 
		_result63 = jnp.power(jnp.float32(1e+01), (jnp.float32(3.0) * inputs[91])) 
		_result64 = remainder((jnp.float32(9.2) * inputs[92]), jnp.float32(2.0)) 
		_result65 = jnp.rint((jnp.float32(1.5) * inputs[93])) 
		_result66 = jnp.round((jnp.float32(1.5) * inputs[94])) 
		_result67 = jnp.sin((jnp.float32(0.3) * inputs[95])) 
		_result68 = jnp.sqrt((jnp.float32(0.3) * inputs[96])) 
		_result69 = jnp.tan((jnp.float32(0.3) * inputs[97])) 
		_result70 = jnp.arccosh(((jnp.float32(0.3) * inputs[98]) + jnp.float32(1e+01))) 
		_result71 = jnp.arcsinh(((jnp.float32(0.3) * inputs[99]) + jnp.float32(1e+01))) 
		_result72 = jnp.arctanh(((jnp.float32(0.3) * inputs[100]) + jnp.float32(0.5))) 
		_result73 = jnp.cosh(((jnp.float32(0.3) * inputs[101]) + jnp.float32(1e+01))) 
		_result74 = jnp.sinh(((jnp.float32(0.3) * inputs[102]) + jnp.float32(1e+01))) 
		_result75 = jnp.tanh(((jnp.float32(0.3) * inputs[103]) + jnp.float32(1e+01))) 
		return state, jnp.stack([_result0,_result1,_result2,_result3,_result4,_result5,_result6,_result7,_result8,_result9,_result10,_result11,_result12,_result13,_result14,_result15,_result16,_result17,_result18,_result19,_result20,_result21,_result22,_result23,_result24,_result25,_result26,_result27,_result28,_result29,_result30,_result31,_result32,_result33,_result34,_result35,_result36,_result37,_result38,_result39,_result40,_result41,_result42,_result43,_result44,_result45,_result46,_result47,_result48,_result49,_result50,_result51,_result52,_result53,_result54,_result55,_result56,_result57,_result58,_result59,_result60,_result61,_result62,_result63,_result64,_result65,_result66,_result67,_result68,_result69,_result70,_result71,_result72,_result73,_result74,_result75]) 
		
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
