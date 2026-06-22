#pragma once
#include <cstdint>
#include <string>

// Derive an independent 64-bit seed for a named noise input. Each noise input
// gets its own RNG stream keyed by (base seed, input name), so the values drawn
// for one input do not depend on input order or on whether sibling noise inputs
// are host-supplied (e.g. driven from a Jitter matrix). FNV-1a over the name,
// mixed with the seed via the splitmix64 finalizer.
//
// Shared by the generative runner (gen: re-derives the stream each one-shot
// generate() call) and the streaming Backend (live: seeds one persistent stream
// per noise input once, then advances it per block) so both stay identical. See
// EXECUTORCH_PROTOCOL.md.
inline uint64_t noise_substream_seed(uint64_t seed, const std::string &name) {
  uint64_t h = 1469598103934665603ull; // FNV-1a offset basis
  for (unsigned char c : name) {
    h ^= c;
    h *= 1099511628211ull; // FNV-1a prime
  }
  uint64_t z = seed ^ h;
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
