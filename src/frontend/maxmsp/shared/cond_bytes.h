#pragma once
// Shared conditioning-ingest helper for the neural~ family. Converts Max atoms
// into the raw little-endian bytes of a model condition input's declared dtype,
// so both neural.gen~ (offline) and neural.live~ (streaming) convert list/dict
// conditioning the same way. The includer must already have the ExecuTorch
// ScalarType in scope (e.g. via backend.h / gen_runner.h).
#include <cstdint>
#include <vector>

// Convert `n` numeric atoms into raw little-endian bytes of `dt`'s element type
// (Long => int64, Int => int32, else float). Integer dtypes read each value via
// `get_long(k)` (full precision); float via `get_float(k)`. Shared by the
// dictionary (by-name) and list (by-position) conditioning paths so the dtype
// handling lives in one place.
template <typename GetLong, typename GetFloat>
static std::vector<char> cond_atoms_to_bytes(executorch::aten::ScalarType dt,
                                             long n, GetLong get_long,
                                             GetFloat get_float) {
  using executorch::aten::ScalarType;
  const size_t esize = (dt == ScalarType::Long) ? 8 : 4;
  std::vector<char> buf((size_t)n * esize);
  switch (dt) {
  case ScalarType::Long: {
    auto *p = reinterpret_cast<int64_t *>(buf.data());
    for (long k = 0; k < n; k++)
      p[k] = (int64_t)get_long(k);
    break;
  }
  case ScalarType::Int: {
    auto *p = reinterpret_cast<int32_t *>(buf.data());
    for (long k = 0; k < n; k++)
      p[k] = (int32_t)get_long(k);
    break;
  }
  default: {
    auto *p = reinterpret_cast<float *>(buf.data());
    for (long k = 0; k < n; k++)
      p[k] = (float)get_float(k);
    break;
  }
  }
  return buf;
}
