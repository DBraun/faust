// Test edge case: Multiple rwtables with non-standard patterns
// This should create itbl0, ftbl1, itbl2, ftbl3 with custom patterns

// Pattern 1: Fibonacci-like sequence for itbl0
rw1 = rwtable(8, fib_gen, windex, input, rindex)
with {
    SIZE = 8;
    fib_gen = waveform{1, 1, 2, 3, 5, 8, 13, 21} : !,_;
    windex = (+(1)~_) % SIZE;
    rindex = (windex + 1) % SIZE;
    input = 0;
};

// Pattern 2: Exponential values for ftbl1  
rw2 = rwtable(6, exp_gen, windex, input, rindex)
with {
    SIZE = 6;
    exp_gen = waveform{1.0, 2.0, 4.0, 8.0, 16.0, 32.0} : !,_;
    windex = (+(1)~_) % SIZE;
    rindex = (windex + 1) % SIZE;
    input = 0;
};

// Pattern 3: Negative sequence for itbl2
rw3 = rwtable(5, neg_gen, windex, input, rindex)
with {
    SIZE = 5;
    neg_gen = waveform{-10, -20, -30, -40, -50} : !,_;
    windex = (+(1)~_) % SIZE;
    rindex = (windex + 1) % SIZE;
    input = 0;
};

// Pattern 4: Sine-like values for ftbl3
rw4 = rwtable(8, sine_gen, windex, input, rindex)
with {
    SIZE = 8;
    sine_gen = waveform{0.0, 0.707, 1.0, 0.707, 0.0, -0.707, -1.0, -0.707} : !,_;
    windex = (+(1)~_) % SIZE;
    rindex = (windex + 1) % SIZE;
    input = 0;
};

process = rw1, rw2, rw3, rw4;