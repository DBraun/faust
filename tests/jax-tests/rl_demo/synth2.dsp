import("stdfaust.lib");

// Out-of-domain target synth for the cross-domain (Stage 3) demo.
// Deliberately a different timbre family from synth.dsp: a 2-operator FM
// voice through a resonant lowpass. The policy never sees or drives these
// parameters -- it only hears the rendered audio and tries to match it with
// synth.dsp's own parameters, as in SynthRL's out-of-domain stage
// (Shin & Lee, IJCAI 2025).

ratio = hslider("ratio", 2, 0.5, 8, 0.01);
index = hslider("index[scale:log]", 100, 1, 1000, 1);
cutoff = hslider("cutoff[scale:log]", 2000, 100, 10000, 1);
gain = hslider("gain", 0.5, 0.1, 1, 0.01);

fm(f) = os.osc(f + os.osc(f * ratio) * index);

process(freq) = fm(freq) : fi.resonlp(cutoff, 1.5, 1) : _ * gain;
