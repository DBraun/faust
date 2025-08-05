// Test edge case: ftbl1 with custom pattern instead of 0.1 increments
// Using rwtable to ensure it's a read-write table
size = 10;
init_val = waveform{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} : !,_;
idx = (+(1)~_) % size;
process = rwtable(size, init_val, idx, 0, idx);