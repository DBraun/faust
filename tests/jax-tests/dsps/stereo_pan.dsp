// Stereo panning test
pan = hslider("pan", 0.5, 0, 1, 0.01);
stereo_pan = _ <: *(1-pan), *(pan);
process = hgroup("Stereo Pan", stereo_pan);
