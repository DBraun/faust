import("stdfaust.lib");

// RL Demo Synthesizer
// Inputs: frequency (Hz)
// Continuous parameters (2): cutoff, gain - learnable via Beta distribution
// Categorical parameters (1): waveform - learnable via Categorical/Gumbel-softmax

// UI Parameters
cutoff = hslider("cutoff[scale:log]", 1000, 100, 10000, 1);
gain = hslider("gain", 0.5, 0.1, 1, 0.01);
waveform_selector = nentry("waveform", 0, 0, 3, 1);

// Waveform options
saw(f) = os.sawtooth(f);
square(f) = os.square(f);
triangle(f) = os.triangle(f);
sine(f) = os.osc(f);

// Synthesizer: select waveform, filter, and apply gain
process(freq) = saw(freq),square(freq),triangle(freq),sine(freq)
                : ba.selectn(4, waveform_selector)
                : fi.lowpass(1, cutoff)
                : _ * gain;
