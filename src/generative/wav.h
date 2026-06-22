#pragma once
#include <string>
#include <vector>

// Write a 16-bit PCM WAV from channel-major float audio (ch0[0..len), ch1[...]).
// Ports save_wav() from optimized/mlx/scripts/sa3_mlx.py: clips to [-1,1],
// scales by 32767 -> int16, interleaves (L0,R0,L1,R1,...). Returns false if the
// audio contains non-finite samples or the file cannot be written.
bool write_wav(const std::string &path, const std::vector<float> &audio,
               int channels, int length, int sample_rate);

// Read a WAV into channel-major float audio (all of ch0, then ch1, …) — the same
// layout write_wav consumes and the host's buffer-input preprocessing expects.
// Supports PCM16 and IEEE float32 (the common init-clip formats). Fills
// channels/frames/sample_rate. Returns false on a missing/malformed file or an
// unsupported encoding (and prints why).
bool read_wav(const std::string &path, std::vector<float> &audio, int &channels,
              int &frames, int &sample_rate);
