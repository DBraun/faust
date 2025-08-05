import("stdfaust.lib");

// Test with foreign function approach
random_uniform = ffunction(float self.random_uniform(), <stdlib.h>, "");

// Create two channels of random noise
process = random_uniform, random_uniform;