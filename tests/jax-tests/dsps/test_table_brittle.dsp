// Test all brittle assumptions:
// 1. itbl0 that doesn't use incrementing integers
// 2. ftbl1 that doesn't use 0.1 increments  
// 3. Tables without corresponding waveform data

// itbl0 with decreasing values (not incrementing)
rw1 = rwtable(8, 100-(10*_)~+(1), 0, 0, rindex)
with {
    rindex = (+(1)~_) % 8;
};

// ftbl1 with 0.333 increments (not 0.1)
rw2 = rwtable(6, +(0.333)~_, 0, 0, rindex)
with {
    rindex = (+(1)~_) % 6;
};

// itbl2 without any SIG waveform data
rw3 = rwtable(5, (*(2))~+(1), 0, 0, rindex)
with {
    rindex = (+(1)~_) % 5;
};

// ftbl3 without any SIG waveform data
rw4 = rwtable(4, +(3.14159)~_, 0, 0, rindex)
with {
    rindex = (+(1)~_) % 4;
};

process = rw1, rw2, rw3, rw4;