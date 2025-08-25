#!/usr/bin/env python3
"""
Test script for JAX random_uniform foreign function feature.
This verifies that the random generation produces proper random values
using JAX's PRNG system instead of a linear congruential generator.
"""

import sys
import os
from os import environ
environ["JAX_PLATFORM_NAME"] = "cpu"
environ["CUDA_VISIBLE_DEVICES"] = ""
environ["JAX_PLATFORMS"] = "cpu"

import jax
from jax import numpy as jnp, random
from flax import nnx
import numpy as np

# Import the generated module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/generated')
import random_uniform as random_module

def test_random_uniform():
    """Test that random_uniform produces proper random values."""
    
    # Initialize with RNG
    rngs = nnx.Rngs(0, params=42, rng_stream=1337)
    
    # Create DSP instance
    dsp = random_module.mydsp(sample_rate=44100, rngs=rngs)
    
    # Check number of channels
    assert dsp.num_inputs == 0, f"Expected 0 inputs, got {dsp.num_inputs}"
    assert dsp.num_outputs == 2, f"Expected 2 outputs, got {dsp.num_outputs}"
    
    # Generate samples to test randomness
    carry = dsp.initialize_carry()

    block_size = 1024

    inputs = jnp.zeros((dsp.num_inputs, block_size))
    
    # Collect samples
    values_ch0 = []
    values_ch1 = []
    rng_key = random.key(0)
    
    num_samples = 100
    for i in range(num_samples):
        rng_key, next_key = random.split(rng_key)
        outputs, carry = dsp.process_block(carry, inputs, 1, rngs=next_key)
        values_ch0.append(float(outputs[0, 0]))
        values_ch1.append(float(outputs[1, 0]))
    
    # Test 1: Values should be different (random)
    unique_ch0 = len(set(values_ch0))
    unique_ch1 = len(set(values_ch1))
    
    # Should have many unique values (at least 90% for 100 samples)
    assert unique_ch0 > 90, f"Channel 0 has only {unique_ch0} unique values out of {num_samples}"
    assert unique_ch1 > 90, f"Channel 1 has only {unique_ch1} unique values out of {num_samples}"
    
    # Test 2: Channels should be independent
    # Check that the two channels don't have identical values
    identical_count = sum(1 for v0, v1 in zip(values_ch0, values_ch1) if v0 == v1)
    assert identical_count < 10, f"Channels have {identical_count} identical values, should be independent"
    
    # Test 3: Values should be in range [-1, 1]
    min_ch0, max_ch0 = min(values_ch0), max(values_ch0)
    min_ch1, max_ch1 = min(values_ch1), max(values_ch1)
    
    assert -1.0 <= min_ch0 <= 1.0, f"Channel 0 min value {min_ch0} out of range"
    assert -1.0 <= max_ch0 <= 1.0, f"Channel 0 max value {max_ch0} out of range"
    assert -1.0 <= min_ch1 <= 1.0, f"Channel 1 min value {min_ch1} out of range"
    assert -1.0 <= max_ch1 <= 1.0, f"Channel 1 max value {max_ch1} out of range"
    
    # Test 4: Distribution should be roughly uniform
    # Check that values span a good range (at least 50% of [-1, 1])
    range_ch0 = max_ch0 - min_ch0
    range_ch1 = max_ch1 - min_ch1
    
    assert range_ch0 > 1.0, f"Channel 0 range {range_ch0} is too narrow"
    assert range_ch1 > 1.0, f"Channel 1 range {range_ch1} is too narrow"
    
    # Test 5: Check mean is roughly 0 (for uniform distribution in [-1, 1])
    mean_ch0 = np.mean(values_ch0)
    mean_ch1 = np.mean(values_ch1)
    
    assert -0.2 < mean_ch0 < 0.2, f"Channel 0 mean {mean_ch0} is too far from 0"
    assert -0.2 < mean_ch1 < 0.2, f"Channel 1 mean {mean_ch1} is too far from 0"
    
    print("✅ All random_uniform tests passed!")
    print(f"  - Generated {unique_ch0}/{num_samples} unique values on channel 0")
    print(f"  - Generated {unique_ch1}/{num_samples} unique values on channel 1")
    print(f"  - Range ch0: [{min_ch0:.3f}, {max_ch0:.3f}]")
    print(f"  - Range ch1: [{min_ch1:.3f}, {max_ch1:.3f}]")
    print(f"  - Mean ch0: {mean_ch0:.3f}")
    print(f"  - Mean ch1: {mean_ch1:.3f}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_random_uniform()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)