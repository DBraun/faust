"""Tests for bargraph (hbargraph/vbargraph) support in the JAX backend."""
import pytest
import jax
import jax.numpy as jnp
from flax import nnx


SAMPLE_RATE = 44100
NUM_FRAMES = 64


@pytest.mark.integration
@pytest.mark.ci
class TestBargraphMetadata:
    """Test that bargraph metadata is properly set up."""

    def test_bargraph_detected_as_output_only(self, compile_and_load_dsp, default_rngs):
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        meta = model._parameter_metadata
        assert "fHbargraph0" in meta
        assert meta["fHbargraph0"]["output_only"] is True

    def test_bargraph_zones_list(self, compile_and_load_dsp, default_rngs):
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        zones = model._get_bargraph_zones()
        assert zones == ["fHbargraph0"]

    def test_bargraph_not_in_unnormalize(self, compile_and_load_dsp, default_rngs):
        """Bargraphs should not appear as tunable parameters."""
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        params = model.unnormalize()
        assert "fHbargraph0" not in params


@pytest.mark.integration
@pytest.mark.ci
class TestBargraphProcessBlock:
    """Test bargraph output via process_block."""

    def test_returns_three_tuple(self, compile_and_load_dsp, default_rngs):
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        carry = model.initialize_carry()
        inputs = jnp.ones((1, NUM_FRAMES)) * 0.5

        result = model.process_block(carry, inputs)
        assert len(result) == 3, "Should return (outputs, carry, bargraph_data)"
        outputs, new_carry, bargraph_data = result
        assert "fHbargraph0" in bargraph_data

    def test_bargraph_values_match_abs_input(self, compile_and_load_dsp, default_rngs):
        """bargraph_meter.dsp: level = abs(input). Check per-sample values."""
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        carry = model.initialize_carry()

        signal = jnp.array([-0.3, 0.7, -1.0, 0.0, 0.5], dtype=jnp.float32)
        inputs = signal.reshape(1, -1)

        outputs, new_carry, bargraph_data = model.process_block(carry, inputs)
        level = bargraph_data["fHbargraph0"]

        expected = jnp.abs(signal)
        assert jnp.allclose(level, expected, atol=1e-6), (
            f"Expected bargraph level = abs(input), got {level} vs {expected}"
        )

    def test_audio_passthrough(self, compile_and_load_dsp, default_rngs):
        """The audio output should be the original signal (passthrough)."""
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        carry = model.initialize_carry()
        signal = jnp.linspace(-1, 1, NUM_FRAMES, dtype=jnp.float32)
        inputs = signal.reshape(1, -1)

        outputs, _, _ = model.process_block(carry, inputs)
        assert jnp.allclose(outputs[0], signal, atol=1e-6)

    def test_bargraph_shape(self, compile_and_load_dsp, default_rngs):
        """Bargraph data should have shape (block_size,)."""
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        carry = model.initialize_carry()
        inputs = jnp.ones((1, NUM_FRAMES), dtype=jnp.float32)

        _, _, bargraph_data = model.process_block(carry, inputs)
        assert bargraph_data["fHbargraph0"].shape == (NUM_FRAMES,)


@pytest.mark.integration
@pytest.mark.ci
class TestBargraphCall:
    """Test bargraph output via __call__."""

    def test_call_returns_two_tuple_with_bargraphs(self, compile_and_load_dsp, default_rngs):
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        inputs = jnp.ones((1, NUM_FRAMES), dtype=jnp.float32) * 0.5

        result = model(inputs)
        assert len(result) == 2, "Should return (outputs, bargraph_data)"
        outputs, bargraph_data = result
        assert "fHbargraph0" in bargraph_data

    def test_call_bargraph_values(self, compile_and_load_dsp, default_rngs):
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        signal = jnp.array([-0.3, 0.7, -1.0, 0.0, 0.5], dtype=jnp.float32)
        inputs = signal.reshape(1, -1)

        outputs, bargraph_data = model(inputs)
        level = bargraph_data["fHbargraph0"]
        expected = jnp.abs(signal)
        assert jnp.allclose(level, expected, atol=1e-6)


@pytest.mark.integration
@pytest.mark.ci
class TestBargraphVmap:
    """Test that bargraphs work with jax.vmap (polyphony pattern)."""

    def test_vmap_bargraph(self, compile_and_load_dsp, default_rngs):
        mydsp = compile_and_load_dsp("bargraph_meter.dsp")
        model = mydsp(sample_rate=SAMPLE_RATE, rngs=default_rngs)
        num_voices = 3

        carry = model.initialize_carry()
        carries = jax.tree.map(lambda x: jnp.stack([x] * num_voices), carry)

        # Different amplitudes per voice
        amps = jnp.array([0.2, 0.5, 0.9], dtype=jnp.float32)
        voice_inputs = jnp.stack([
            jnp.full((1, NUM_FRAMES), a, dtype=jnp.float32) for a in amps
        ])  # (num_voices, 1, NUM_FRAMES)

        rng_keys = jax.random.split(jax.random.key(0), num_voices)

        def process_one(carry, inputs, rng):
            return model.process_block(carry, inputs, rngs=rng)

        vmapped = jax.vmap(process_one)
        outputs, new_carries, bargraph_data = vmapped(carries, voice_inputs, rng_keys)

        # Each voice's bargraph should be abs(input) = the amplitude
        for i, amp in enumerate(amps):
            expected = jnp.full(NUM_FRAMES, amp, dtype=jnp.float32)
            assert jnp.allclose(bargraph_data["fHbargraph0"][i], expected, atol=1e-6), (
                f"Voice {i}: expected {amp}, got {bargraph_data['fHbargraph0'][i][:5]}"
            )
