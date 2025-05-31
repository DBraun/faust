// Test edge case: itbl0 with waveform data instead of incrementing integers
wf = waveform{5.5, -2.3, 8.1, 0.0, -4.7};
process = wf, 0 : rdtable;