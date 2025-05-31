// Test edge case: rwtable without any waveform initialization
// This should trigger the fallback patterns

// Simple counter-based table without waveform
rw1 = rwtable(10, +(1)~_, windex, input, rindex)
with {
    SIZE = 10;
    windex = (+(1)~_) % SIZE;
    rindex = (windex + 2) % SIZE;
    input = +(1)~_ * 2;
};

// Float counter without waveform
rw2 = rwtable(7, +(0.5)~_, windex, input, rindex)
with {
    SIZE = 7;
    windex = (+(1)~_) % SIZE;
    rindex = (windex + 2) % SIZE;
    input = +(0.5)~_ * 1.5;
};

process = rw1, rw2;