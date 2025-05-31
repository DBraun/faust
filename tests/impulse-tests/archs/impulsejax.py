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

import sys
import os
from os import environ
environ["JAX_PLATFORM_NAME"] = "cpu"
environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
environ["JAX_PLATFORMS"] = "cpu"     # Force CPU only

import contextlib

import warnings
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_metal_message():
	with open(os.devnull, 'w') as fnull:
		# Suppress both stdout and stderr
		stdout_fileno = sys.stdout.fileno()
		stderr_fileno = sys.stderr.fileno()
		old_stdout = os.dup(stdout_fileno)
		old_stderr = os.dup(stderr_fileno)
		os.dup2(fnull.fileno(), stdout_fileno)
		os.dup2(fnull.fileno(), stderr_fileno)
		try:
			yield
		finally:
			os.dup2(old_stdout, stdout_fileno)
			os.dup2(old_stderr, stderr_fileno)
			os.close(old_stdout)
			os.close(old_stderr)

# Use it when importing or initializing JAX
with suppress_metal_message():
	import jax
	import math
	import json
	import dataclasses
	import re
	from typing import Any, Dict, List, Tuple
	from pathlib import Path
	import numpy as np
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
<<includeIntrinsic>>
<<includeclass>>

	def load_soundfile(self, filepath):
		audio = jnp.sin(jnp.linspace(0, 2*jnp.pi, num=4096, endpoint=False, dtype=FAUSTFLOAT))
		audio = jnp.stack([audio, audio])
		return audio, 44100
		
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
		# For deterministic impulse tests, use exact init values like C++ CheckControlUI
		label = "/".join(ui_path + [label])
		init = FAUSTFLOAT(init)
		
		# Create parameter with exact init value (no normalization for impulse tests)
		setattr(self, zone, self.param(label, nn.initializers.constant(init, dtype=FAUSTFLOAT), ()))
		
		# Create identity unnormalization function (parameter is already at correct value)
		unnorm_funcs[label] = (zone, lambda x: x)
	
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
		self.add_bargraph(zone, ui_path, label, a_min, a_max)
	
	def add_vbargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float):
		self.add_bargraph(zone, ui_path, label, a_min, a_max)

	def add_bargraph(self, zone: str, ui_path: list[str], label: str, a_min: float, a_max: float):
		setattr(self, zone, FAUSTFLOAT(0))

	def unnormalize(self, i: int) -> Dict[str, jnp.array]:
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
				if zone.startswith("fButton"):
					normalized_value = getattr(self, zone)
					params[zone] = jnp.where(i > 63, jnp.zeros_like(normalized_value), jnp.ones_like(normalized_value))
				else:
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
	
	def process_block(self, i, carry: Dict[str, jnp.array], inputs: jnp.array = None, length: int = None, unroll: int = 1) -> Tuple[jnp.array, Dict[str, jnp.array]]:
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
		params = self.unnormalize(i)
		
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
		params = self.unnormalize(0)
		
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


def test(args, N_SAMPLES, OFFSET, print_header=True):

	from jax import random
	from scipy.io import wavfile

	model = mydsp(sample_rate=args.sample_rate)

	# json_obj = model.getJSON()
	# print('json_obj: ', json_obj)

	key = random.PRNGKey(0)

	BLOCK_SIZE = 1

	N_CHANNELS = model.num_inputs

	if args.random:
		input_audio = -1.+2.*random.uniform(key, shape=(N_CHANNELS, BLOCK_SIZE), dtype=FAUSTFLOAT)
	else:
		input_audio = jnp.zeros((N_CHANNELS, BLOCK_SIZE), dtype=FAUSTFLOAT)
		input_audio = input_audio.at[:,0].set(1.)

	variables = model.init({"params": key, "rng_stream": key}, input_audio, length=BLOCK_SIZE)
	
	# Initialize carry state
	carry = model.apply(variables, method="initialize_carry")

	@jax.jit
	def process_block_jit(i, carry, inputs: jnp.ndarray, rng: jax.Array):
		return model.apply(variables, i, carry, inputs, length=BLOCK_SIZE, unroll=1, method="process_block", rngs={"rng_stream": rng})

	out_blocks = []
	did_silence_audio = False

	for i in range(math.ceil(N_SAMPLES/BLOCK_SIZE)):
		if not did_silence_audio and i != 0:
			did_silence_audio = True
			input_audio = jnp.zeros_like(input_audio)

		key, subkey = random.split(key)
		out_block, carry = process_block_jit(i, carry, input_audio, subkey)
		out_blocks.append(np.array(out_block))

	y = np.concatenate(out_blocks, axis=-1)
	y = y[:, :N_SAMPLES]

	assert y.ndim == 2
	assert y.shape[0] == model.num_outputs
	assert y.shape[1] == N_SAMPLES

	output_audio = y.T

	if args.output is not None:
		
		wavfile.write(args.output, args.sample_rate, output_audio)
	
	# print the header
	if print_header:
		print(f"number_of_inputs  : {model.num_inputs}")
		print(f"number_of_outputs : {model.num_outputs}")
		print(f"number_of_frames  : {N_SAMPLES*4}")

	# print the samples
	for i, frame in enumerate(output_audio):
		print(f"     {i+OFFSET} :  " + ' '.join(["{:.6f}".format(c) for c in frame]))

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Run a JAX/Flax model converted from Faust code')
	parser.add_argument('-sr', '--sample-rate', type=int, default=44100, help='Sample rate (such as 44100)')
	parser.add_argument('--random', type=bool, default=False, help="Whether the default audio is random. By default it's an impulse.")
	parser.add_argument('-o', '--output', type=str, default=None, help='Filepath for output audio WAV')

	args = parser.parse_args()

	test(args, 15000, 0)
	test(args, 15000, 15000, print_header=False)
