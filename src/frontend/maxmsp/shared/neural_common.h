#pragma once
// Helpers shared by the neural.live~ family of externals (neural.live~,
// mc.neural.live~, mcs.neural.live~). Previously these were copy-pasted into
// each .cpp; they are gathered here as `inline` free functions.
#include "c74_min.h"
#include <string>

// Resolve a model argument to the ABSOLUTE path of its `.pte` program. A usable
// model is a `<stem>.pte` program + a `<stem>.json` sidecar living together; the
// sidecar is mandatory (it declares the I/O) while the `.pte` MAY be absent — in
// which case the backend loads metadata-only and reports the program as missing.
// We therefore locate the *sidecar* via Max's search path and return its sibling
// `.pte` path (whether or not that `.pte` exists on disk); the backend probes for
// it. Returns "" when no sidecar can be found. `arg` may end in `.pte`, `.json`,
// or carry no extension. Resolving the sidecar (which exists for any usable
// model) avoids the failed `.pte` lookup that aborted loading when it was absent.
inline std::string resolve_model_pte(std::string arg) {
  auto ends_with = [](const std::string &s, const std::string &e) {
    return s.size() >= e.size() &&
           s.compare(s.size() - e.size(), e.size(), e) == 0;
  };
  if (ends_with(arg, ".pte"))
    arg.resize(arg.size() - 4);
  else if (ends_with(arg, ".json"))
    arg.resize(arg.size() - 5);

  c74::min::path sidecar(arg + ".json"); // search Max's file paths for the sidecar
  if (!sidecar)                          // operator bool: not found
    return std::string();
  std::string json_abs = std::string(sidecar); // absolute path to the .json
  if (ends_with(json_abs, ".json"))
    json_abs.resize(json_abs.size() - 5);
  return json_abs + ".pte"; // sibling .pte (may or may not exist on disk)
}

// Resolve a model argument to the ABSOLUTE path of its `.json` sidecar (the
// companion of resolve_model_pte, which returns the sibling `.pte`). Strips a
// trailing `.pte`/`.json`, locates "<stem>.json" via Max's search path, and
// returns its absolute path, or "" if not found.
inline std::string resolve_sidecar_json(std::string arg) {
  auto ends_with = [](const std::string &s, const std::string &e) {
    return s.size() >= e.size() &&
           s.compare(s.size() - e.size(), e.size(), e) == 0;
  };
  if (ends_with(arg, ".pte"))
    arg.resize(arg.size() - 4);
  else if (ends_with(arg, ".json"))
    arg.resize(arg.size() - 5);

  c74::min::path sidecar(arg + ".json"); // search Max's file paths for the sidecar
  if (!sidecar)                          // operator bool: not found
    return std::string();
  return std::string(sidecar); // absolute path to the .json
}

// Smallest power of two >= x (>= 1). Used to round a requested host block size
// up to a power of two.
inline unsigned power_ceil(unsigned x) {
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}

// Zero every sample of every channel of an output bundle (model disabled / not
// yet loaded / vector-size mismatch).
inline void fill_with_zero(c74::min::audio_bundle output) {
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    for (int i(0); i < output.frame_count(); i++) {
      out[i] = 0.;
    }
  }
}

// Reconcile the user-requested internal buffer size with the model's exported
// size and ratios, returning the size to use and (via `use_thread`) whether the
// worker thread should run. Mirrors the ladder that the three externals' ctors
// used inline:
//   - model exported a fixed buffer size  -> adopt it (a 0 request => no thread);
//   - request 0 with no exported size     -> no-thread mode at the higher ratio;
//   - request below the higher ratio       -> bump up to the higher ratio;
//   - otherwise                            -> round up to a power of two.
// Diagnostics go to the caller's Min logger so they reach the Max console.
inline int negotiate_buffer_size(int requested, int model_buffer_size,
                                 int higher_ratio, bool &use_thread,
                                 c74::min::logger &log_err) {
  const bool no_thread_requested = (requested == 0);
  if (model_buffer_size > 0) {
    if (!no_thread_requested && requested != model_buffer_size)
      log_err << "buffer size " << requested
              << " overridden by model's exported size " << model_buffer_size
              << c74::min::endl;
    if (no_thread_requested)
      use_thread = false;
    return model_buffer_size;
  }
  if (!requested) {
    // NO THREAD MODE (dynamic-shape / missing metadata)
    use_thread = false;
    return higher_ratio;
  }
  if (requested < higher_ratio) {
    log_err << "buffer size too small, switching to " << higher_ratio
            << c74::min::endl;
    return higher_ratio;
  }
  return power_ceil(requested);
}
