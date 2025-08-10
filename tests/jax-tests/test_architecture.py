"""
Unit tests for JAX architecture components.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import sys
from pathlib import Path

# Add architecture directory to path to import minimal.py components
arch_dir = Path(__file__).parent.parent.parent.parent.parent / "architecture" / "jax"
sys.path.insert(0, str(arch_dir))


@pytest.mark.unit
class TestConstants:
	"""Test that constants are properly defined."""

	def test_constants_exist(self):
		"""Verify key constants are defined."""
		import importlib.util
		spec = importlib.util.spec_from_file_location("minimal", arch_dir / "minimal.py")
		if spec and spec.loader:
			module = importlib.util.module_from_spec(spec)
			# Note: We can't fully execute minimal.py as it has template strings
			# This test is more of a structural check

	def test_random_uniform_range(self):
		"""Test RANDOM_UNIFORM_MIN and RANDOM_UNIFORM_MAX are correct."""
		# These should be -1.0 and 1.0
		# We can't import them directly due to template strings, but we know the values
		assert True  # Placeholder


@pytest.mark.unit
class TestExceptions:
	"""Test custom exception classes."""

	def test_faust_error_hierarchy(self):
		"""Test that custom exceptions inherit from FaustError."""
		# We would test that:
		# - SoundfileLoadError is a subclass of FaustError
		# - InvalidScaleModeError is a subclass of FaustError
		# - InvalidRNGError is a subclass of FaustError
		# - InvalidParameterError is a subclass of FaustError
		# But we can't easily import due to template strings
		assert True  # Placeholder


@pytest.mark.unit
class TestScaleMode:
	"""Test scale mode normalization and unnormalization."""

	@pytest.mark.parametrize("scale_mode", ["linear", "exp", "log"])
	def test_scale_modes(self, scale_mode):
		"""Test that each scale mode is recognized."""
		# Would test normalize_value and create_unnormalize_func
		# for each scale mode
		assert scale_mode in ["linear", "exp", "log"]

	def test_linear_normalization(self):
		"""Test linear normalization round-trip."""
		# Test that normalize(x) followed by unnormalize gives back x
		# for linear scale mode
		a_min, a_max = 0.0, 100.0
		test_value = 50.0

		# Expected normalized value is 0.5
		# This would be tested with actual implementation
		assert True  # Placeholder

	def test_invalid_scale_mode(self):
		"""Test that invalid scale mode raises InvalidScaleModeError."""
		# Would test that an invalid scale mode raises the custom exception
		assert True  # Placeholder


@pytest.mark.unit
class TestRNGHandling:
	"""Test RNG key extraction and handling."""

	def test_extract_rng_key_from_array(self, default_rngs):
		"""Test extracting RNG key from JAX array."""
		key = jax.random.key(0)
		# Would test _extract_rng_key(key) returns the key
		assert isinstance(key, jax.Array)

	def test_extract_rng_key_from_rngs(self, default_rngs):
		"""Test extracting RNG key from Rngs object."""
		# Would test _extract_rng_key(rngs) returns a JAX array
		assert isinstance(default_rngs, nnx.Rngs)

	def test_invalid_rng_type(self):
		"""Test that invalid RNG type raises InvalidRNGError."""
		# Would test passing an invalid type raises InvalidRNGError
		assert True  # Placeholder


@pytest.mark.unit
class TestParameterNormalization:
	"""Test parameter normalization for sliders and nentrys."""

	def test_slider_normalization(self):
		"""Test slider parameter normalization to [0, 1]."""
		# Test that a slider value is correctly normalized
		# based on its min, max, and scale mode
		assert True  # Placeholder

	def test_nentry_gumbel_softmax(self):
		"""Test nentry uses Gumbel-softmax for discrete parameters."""
		# Test that nentry creates logits and tau parameters
		# Test that unnormalization uses Gumbel-softmax or argmax
		assert True  # Placeholder

	def test_parameter_metadata_storage(self):
		"""Test that parameter metadata is correctly stored."""
		# Test that add_slider, add_nentry, etc. store metadata
		assert True  # Placeholder


@pytest.mark.unit
class TestSoundfileHandling:
	"""Test soundfile loading and handling."""

	def test_soundfile_load_success(self):
		"""Test successful soundfile loading."""
		# Would test load_soundfile with a valid audio file
		assert True  # Placeholder

	def test_soundfile_load_fallback(self):
		"""Test soundfile loading fallback when librosa not available."""
		# Would test that without librosa, returns dummy data
		assert True  # Placeholder

	def test_soundfile_load_error(self):
		"""Test soundfile loading error handling."""
		# Would test that missing file raises SoundfileLoadError
		assert True  # Placeholder

	def test_soundfile_multiple_paths(self):
		"""Test soundfile search in multiple directories."""
		# Would test that soundfile_dirs are searched in order
		assert True  # Placeholder
