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
from flax import linen as nn

# Import librosa if available for soundfile loading
try:
	import librosa
	HAS_LIBROSA = True
except ImportError:
	HAS_LIBROSA = False
	print("Warning: librosa not available, soundfile loading will use dummy data")

# Generated code will be inserted here
<<includeIntrinsic>>
<<includeclass>>
	
	def load_soundfile(self, filepath: str):
		"""
		Load a soundfile from disk, searching in soundfile_dirs.
		
		Args:
			filepath: Path to the soundfile
			
		Returns:
			Tuple of (audio_data, sample_rate)
		"""
		if not HAS_LIBROSA:
			return np.zeros((1, 1024)), self.sample_rate
			
		# soundfile_dirs should always include at least current directory
		soundfile_dirs = [""] + list(self.soundfile_dirs)
		
		# Create list of potential paths to check
		potential_paths = ([Path(filepath)] if Path(filepath).is_absolute() 
						  else [Path(d) / filepath for d in soundfile_dirs])
		
		# Try to load from each path
		for full_path in potential_paths:
			try:
				audio, sr = librosa.load(str(full_path), mono=False, sr=None)
				if audio.ndim == 1:
					audio = np.expand_dims(audio, 0)
				return audio, sr
			except FileNotFoundError:
				continue
		
		# If not found, return silence
		return np.zeros((1, 1024)), self.sample_rate
	
	def add_soundfile(self, state, zone: str, ui_path: List[str], label: str, url: str, x):
		"""
		Add a soundfile UI element to the state.
		
		Args:
			state: Current state dictionary
			zone: Variable name in the DSP code (e.g., 'fSoundfile0')
			ui_path: Hierarchical path in the UI
			label: Display label
			url: Soundfile URL in format {'file1.wav';'file2.wav'}
			x: Input signal (unused but required by interface)
		"""
		# Parse multiple soundfile URLs from Faust format
		filepaths = url[2:-2].split("';'") if url.startswith("{") else [url]
		
		# Load all soundfiles
		fLength, fOffset, fSR, offset = [], [], [], 0
		audio_data = [self.load_soundfile(filepath) for filepath in filepaths]
		
		# Determine max channels and total length
		num_chans = max([y.shape[0] for y, _ in audio_data])
		total_length = sum([y.shape[1] for y, _ in audio_data])
		
		# Create buffer for all soundfiles
		fBuffers = jnp.zeros((num_chans, total_length))
		
		# Copy each soundfile into the buffer
		for y, sr in audio_data:
			fSR.append(sr)
			assert y.ndim == 2
			y = jnp.array(y)
			fLength.append(y.shape[1])
			fOffset.append(offset)
			fBuffers = fBuffers.at[:y.shape[0], offset:offset+y.shape[1]].set(y)
			offset += y.shape[1]
		
		# Handle parameter soundfiles (can be trained)
		if label.startswith('param:'):
			label = label[6:]  # remove 'param:' prefix
			full_label = "/".join(ui_path + [label])
			fBuffers = self.param("_" + full_label, (lambda key, shape: fBuffers), None)
		else:
			full_label = "/".join(ui_path + [label])
		
		# Store for introspection
		self.sow('intermediates', full_label, fBuffers)
		
		# Store in state with Faust's expected structure
		state[zone] = {
			'fLength': fLength,
			'fOffset': fOffset,
			'fBuffers': fBuffers,
			'fSR': fSR
		}
	
	def add_button(self, state, zone: str, ui_path: List[str], label: str):
		"""
		Add a button UI element to the state.
		
		Args:
			state: Current state dictionary
			zone: Variable name in the DSP code (e.g., 'fButton0')
			ui_path: Hierarchical path in the UI
			label: Display label
		"""
		full_label = "/".join(ui_path + [label])
		param = self.param("_" + full_label, nn.initializers.constant(0.), ())
		param = jnp.where(param > 0., 1., 0.)
		self.sow('intermediates', full_label, param)
		state[zone] = param
	
	def add_checkbox(self, state, zone: str, ui_path: List[str], label: str):
		"""
		Add a checkbox UI element to the state.
		Checkboxes are binary like buttons.
		"""
		self.add_button(state, zone, ui_path, label)

	def add_nentry(
		self, state, zone: str, ui_path: List[str], label: str,
		init: float, a_min: float, a_max: float, step_size: float,
		scale_mode: str = "linear",
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
		step_values = jnp.arange(num_steps, dtype=jnp.float32) * step_size + a_min

		# ---------- parameters ----------
		# (1) logits, initialised to favour the initial step
		def init_logits(key, shape):
			logits = jnp.zeros(shape)
			return logits.at[init_step].set(5.0)        # bias ≈ exp(5) ≈ 148
		logits = self.param("_" + label, init_logits, (num_steps,))

		# temperature (optional learnable scalar)
		# tau = self.param(f"_{label}_tau", nn.initializers.constant(1.0), ())
		tau = 1.0  # todo: user should be able to configure via UI Label metadata:
		# https://faustdoc.grame.fr/manual/syntax/#ui-label-metadata

		# At train-time pass rngs={"gumbel": key} to model.apply
		if self.has_rng("gumbel"):
			gumbel_noise = -jnp.log(-jnp.log(
				random.uniform(self.make_rng("gumbel"), shape=logits.shape) + 1e-10
			) + 1e-10)
			probs = nn.softmax((logits + gumbel_noise) / jnp.clip(tau, 1e-3))
		else:  # deterministic fallback (e.g. evaluation)
			probs = nn.softmax(logits / jnp.clip(tau, 1e-3))

		param_value = jnp.sum(probs * step_values)

		self.sow("intermediates", label + ":probs", probs)
		self.sow("intermediates", label, param_value)
		state[zone] = param_value
	
	def add_slider(self, state, zone: str, ui_path: List[str], label: str,
				   init: float, a_min: float, a_max: float, scale_mode='linear'):
		"""
		Add a slider UI element to the state.
		
		Sliders are continuous parameters that can use different scaling modes
		for better parameter optimization.
		
		Args:
			state: Current state dictionary
			zone: Variable name in the DSP code
			ui_path: Hierarchical path in the UI
			label: Display label
			init: Initial value
			a_min: Minimum value
			a_max: Maximum value
			scale_mode: 'linear', 'exp', or 'log' scaling
		"""
		full_label = "/".join(ui_path + [label])
		init, a_min, a_max = float(init), float(a_min), float(a_max)
		
		if scale_mode == 'linear':
			# Linear scaling: parameter in [-1, 1] maps linearly to [a_min, a_max]
			init_norm = jnp.interp(init, jnp.array([a_min, a_max]), jnp.array([-1., 1.]))
			param = self.param("_" + full_label, nn.initializers.constant(init_norm), ())
			param = jnp.clip(param, -1., 1.)
			param = jnp.interp(param, jnp.array([-1., 1.]), jnp.array([a_min, a_max]))
			
		elif scale_mode == 'exp':
			# Exponential scaling: parameter in [-1, 1] maps exponentially
			init_exp = jnp.interp(init, jnp.array([a_min, a_max]), jnp.array([1., jnp.e]))
			init_log = jnp.log(init_exp)
			init_norm = jnp.interp(init_log, jnp.array([0., 1.]), jnp.array([-1., 1.]))
			param = self.param("_" + full_label, nn.initializers.constant(init_norm), ())
			param = jnp.clip(param, -1., 1.)
			param_unit = jnp.interp(param, jnp.array([-1., 1.]), jnp.array([0., 1.]))
			param = jnp.interp(jnp.exp(param_unit), jnp.array([1., jnp.e]), jnp.array([a_min, a_max]))
			
		elif scale_mode == 'log':
			# Logarithmic scaling: parameter in [-1, 1] maps logarithmically
			init_log10 = jnp.interp(init, jnp.array([a_min, a_max]), jnp.array([-4., 0.]))
			init_linear = jnp.power(10., init_log10)
			init_norm = jnp.interp(init_linear, jnp.array([10.**-4., 1.]), jnp.array([-1., 1.]))
			param = self.param("_" + full_label, nn.initializers.constant(init_norm), ())
			param = jnp.clip(param, -1., 1.)
			param_linear = jnp.interp(param, jnp.array([-1., 1.]), jnp.array([10.**-4., 1.]))
			param = jnp.interp(jnp.log10(param_linear), jnp.array([-4., 0.]), jnp.array([a_min, a_max]))
			
		else:
			raise ValueError(f"Unknown scale mode '{scale_mode}'")
		
		self.sow('intermediates', full_label, param)
		state[zone] = param
	
	def add_hslider(self, state, zone: str, ui_path: List[str], label: str,
					init: float, a_min: float, a_max: float, step_size: float):
		"""Add a horizontal slider (linear scaling by default)."""
		self.add_slider(state, zone, ui_path, label, init, a_min, a_max, 'linear')
	
	def add_vslider(self, state, zone: str, ui_path: List[str], label: str,
					init: float, a_min: float, a_max: float, step_size: float):
		"""Add a vertical slider (linear scaling by default)."""
		self.add_slider(state, zone, ui_path, label, init, a_min, a_max, 'linear')
	
	def add_hbargraph(self, state, zone: str, ui_path: List[str], label: str,
					  a_min: float, a_max: float):
		"""
		Add a horizontal bargraph (output display) to the state.
		Bargraphs display values but don't create parameters.
		"""
		# Bargraphs are outputs, not inputs, so we just pass
		pass
	
	def add_vbargraph(self, state, zone: str, ui_path: List[str], label: str,
					  a_min: float, a_max: float):
		"""
		Add a vertical bargraph (output display) to the state.
		Bargraphs display values but don't create parameters.
		"""
		# Bargraphs are outputs, not inputs, so we just pass
		pass
	
	def initialize_carry(self, key: jax.random.PRNGKey, input_shape: Tuple[int]) -> Dict[str, jnp.array]:
		"""
		Initialize the carry state for real-time processing.
		
		Args:
			key: PRNG key for random initialization if needed
			input_shape: Shape of input without batch dimension (num_inputs,)
			
		Returns:
			Dictionary containing all stateful components (delays, filter states, etc.)
		"""
		# Create dummy input for initialization
		dummy_x = jnp.zeros((input_shape[0], 1))
		
		# Initialize the full state
		state = self.initialize(dummy_x, 1)
		state = self.build_interface(state, dummy_x, 1)
		
		# Convert to JAX arrays
		state = jax.tree.map(jnp.array, state)
		
		# Filter out non-stateful components if needed
		# For now, we keep everything as the tick function expects it
		return state
	
	def process_block(self, carry: Dict[str, jnp.array], inputs: jnp.array) -> Tuple[jnp.array, Dict[str, jnp.array]]:
		"""
		Process one block of audio and return updated state.
		
		Args:
			carry: State dictionary from previous block
			inputs: Input audio block of shape (num_inputs, block_size)
			
		Returns:
			Tuple of (output_block, new_carry) where:
			- output_block has shape (num_outputs, block_size)
			- new_carry is the updated state dictionary
		"""
		# Transpose for scan: (block_size, num_inputs)
		inputs_t = jnp.transpose(inputs, axes=(1, 0))
		
		scan_fn = nn.scan(tick, variable_broadcast="params", split_rngs={'rng_stream': True}, length=T)
		new_carry, outputs = scan_fn(self, carry, jnp.transpose(inputs_t, axes=(1, 0)))
		
		# Transpose back: (num_outputs, block_size)
		outputs_t = jnp.transpose(outputs, axes=(1, 0))

		return outputs_t, new_carry
	
	@nn.compact
	def __call__(self, x, T: int) -> jnp.array:
		"""
		Original interface for compatibility - internally uses process_block.
		
		Args:
			x: Input array of shape (num_inputs, num_samples) or None for generators
			T: Number of samples to process (used when x is None)
			
		Returns:
			Output array of shape (num_outputs, num_samples)
		"""
		# Handle generators (no input case)
		if x is None or (hasattr(x, 'shape') and x.shape[0] == 0):
			x = jnp.zeros((self.num_inputs, T))
		
		# Initialize carry state
		carry = self.initialize_carry(self.make_rng('params'), (self.num_inputs,))
		
		# Process as one block
		outputs, _ = self.process_block(carry, x)
		
		return outputs


# Example usage demonstrating real-time processing
def example_realtime_processing():
	from jax import random
	
	# Initialize model
	model = mydsp(sample_rate=48000)
	key = random.PRNGKey(0)
	
	# Initialize parameters with dummy input
	dummy_input = jnp.zeros((model.num_inputs, 1))
	variables = model.init({'params': key}, dummy_input, 1)
	
	# Initialize carry state for real-time processing
	carry = model.apply(variables, key, (model.num_inputs,), method=model.initialize_carry)
	
	# JIT compile the process_block method for efficiency
	@jax.jit
	def process_block_jit(carry, inputs):
		return model.apply(variables, carry, inputs, method=model.process_block)
	
	# Simulate real-time processing
	block_size = 512
	num_blocks = 10
	
	for block_idx in range(num_blocks):
		# Get input block (in real scenario, from audio interface)
		input_block = jnp.zeros((model.num_inputs, block_size))  # Replace with actual audio
		
		# Process block and get updated state
		output_block, carry = process_block_jit(carry, input_block)
		
		# Send output to audio interface (simulated here)
		print(f"Block {block_idx}: output shape = {output_block.shape}")
	
	print("Real-time processing example completed!")


def realtime_audio_example():
	"""
	Real-time audio streaming example using sounddevice.
	Demonstrates the real-time API with actual audio output.
	"""
	try:
		import sounddevice as sd
	except ImportError:
		print("sounddevice not installed. Install with: pip install sounddevice")
		print("Falling back to offline example.")
		example_realtime_processing()
		return
	
	import time
	from jax import random
	
	# Audio settings
	SAMPLE_RATE = 48000
	BLOCK_SIZE = 512
	
	# Initialize model
	model = mydsp(sample_rate=SAMPLE_RATE)
	key = random.PRNGKey(0)
	
	# Initialize parameters
	if model.num_inputs > 0:
		dummy_input = jnp.zeros((model.num_inputs, 1))
	else:
		dummy_input = jnp.zeros((0, 1))  # Empty input for generators
	
	variables = model.init({'params': key}, dummy_input, 1)
	
	# Initialize carry state
	carry = model.apply(variables, key, (model.num_inputs,), method=model.initialize_carry)
	
	# JIT compile the process method
	@jax.jit
	def process_block_jit(carry, inputs):
		return model.apply(variables, carry, inputs, method=model.process_block)
	
	# Create a generator for audio blocks
	def audio_generator():
		nonlocal carry
		while True:
			# For generators, create empty input
			if model.num_inputs == 0:
				inputs = jnp.zeros((0, BLOCK_SIZE))
			else:
				# For processors, you would get input from sounddevice
				# For this example, we'll use zeros
				inputs = jnp.zeros((model.num_inputs, BLOCK_SIZE))
			
			# Process block
			outputs, carry = process_block_jit(carry, inputs)
			
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


if __name__ == '__main__':
	# Try real-time audio first, fall back to offline example
	realtime_audio_example()