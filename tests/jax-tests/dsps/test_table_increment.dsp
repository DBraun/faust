// Test: rwtable with 0.25 increment pattern (not 0.1)
// This should break the fallback assumption

rw = rwtable(10, +(0.25)~_, windex, input, rindex)
with {
    SIZE = 10;
    windex = 0;  // Always write at index 0
    rindex = (+(1)~_) % SIZE;  // Read sequentially
    input = 0;  // Don't modify values
};

process = rw;