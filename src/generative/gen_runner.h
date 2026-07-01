#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../backend/backend.h"

// SuppliedInput (pre-computed input handed to the runner by name) is declared in
// backend.h so the streaming consumer can reuse it for conditions.

// Tokenizer-free generative runner: owns the Backend and fills each of the
// model's declared inputs by role — externally-supplied conditioning matched by
// name (e.g. token ids / attention mask from an external tokenizer), seeded
// Gaussian noise, or a clamped scalar attribute — then runs the model. Links
// `backend` only (NO tokenizer / Rust), so Rust-free consumers (neural.gen~) can
// drive a latent-diffusion model directly. Data-driven; no model-specific code.
class GenRunner {
public:
  // Load the .pte (+ sidecar). Returns 0 on success, non-zero on failure.
  int load(const std::string &pte_path);
  bool ready() const { return m_ready; }

  // Run `method`. For each declared input: if `supplied` has an entry under its
  // name, validate dtype/numel and copy it; else fill by role — Noise => seeded
  // N(0,1), Attribute => clamp(attr_values[name] if given else default, min, max).
  // A Condition role that is not supplied is an error. On success (0) `out_audio`
  // holds channel-major floats (channels * length).
  int generate(const std::string &method,
               const std::map<std::string, SuppliedInput> &supplied,
               const std::map<std::string, double> &attr_values, uint64_t seed,
               std::vector<float> &out_audio);

  // Bind externally-supplied conditioning onto the model's condition-role inputs
  // by name, ready to pass as `supplied` to generate(). `cond` maps an input's
  // name to its raw bytes already in that input's declared dtype (the role no
  // longer says which tensor — matching is purely by name, dtype comes from the
  // sidecar). The byte buffers must outlive the generate() call. Shared by every
  // caller so the name->input mapping lives in exactly one place.
  static void
  bind_condition(const std::vector<InputSpec> &inputs,
                 const std::map<std::string, std::vector<char>> &cond,
                 std::map<std::string, SuppliedInput> &supplied);

  // Bind a host-preprocessed init waveform onto a model's buffer-role inputs
  // by name (parallel to bind_condition). `init` is a channel-major float buffer
  // already at the input's target geometry (see prepare_init_audio); it must
  // outlive the generate() call. No-op if `init` is empty (=> generate fills the
  // buffer input with silence).
  static void bind_init_audio(const std::vector<InputSpec> &inputs,
                              const std::vector<float> &init,
                              std::map<std::string, SuppliedInput> &supplied);

  // Bind host-supplied noise tensors onto matching noise-role inputs by name
  // (parallel to bind_condition / bind_init_audio). `noise` maps an input's name
  // to its float32 values (already at the input's numel; e.g. snapshotted from a
  // Jitter matrix by neural.gen~). Only entries whose name matches a Noise-role
  // input are bound; the rest of the noise inputs stay seeded inside generate().
  // The float buffers must outlive the generate() call. generate()'s existing
  // numel/dtype checks validate each binding.
  static void bind_noise(const std::vector<InputSpec> &inputs,
                         const std::map<std::string, std::vector<float>> &noise,
                         std::map<std::string, SuppliedInput> &supplied);

  // Host-side init-audio preprocessing: resample (linear interp), channel-map
  // and crop/pad arbitrary source audio into the channel-major float buffer a
  // buffer-role input expects. `src` is channel-major (all of ch0, then ch1, …),
  // `src_ch` channels of `src_frames` each at `src_sr`. `out` is channel-major
  // `dst_ch * dst_len` at `dst_sr`: mono src duplicates to every dst channel,
  // >dst_ch src takes the first dst_ch; source shorter than dst_len is
  // zero-padded, longer is cropped.
  static void prepare_init_audio(const std::vector<float> &src, int src_ch,
                                 size_t src_frames, double src_sr, int dst_ch,
                                 size_t dst_len, double dst_sr,
                                 std::vector<float> &out);

  // Facade over the owned Backend so callers never reach through to it: method
  // listing, typed input/output descriptors.
  bool is_generative() { return m_backend.is_generative(); }
  // True when the .pte program is present and the model can actually generate
  // (vs. has_metadata(): the sidecar parsed, so the host can build its I/O).
  bool runnable() { return m_backend.is_loaded(); }
  bool has_metadata() { return m_backend.has_metadata(); }
  bool has_method(const std::string &method) {
    return m_backend.has_method(method);
  }
  std::vector<std::string> available_methods() {
    return m_backend.get_available_methods();
  }
  const std::vector<InputSpec> &gen_inputs(const std::string &method) {
    return m_backend.method_inputs(method);
  }
  OutputSpec output_spec(const std::string &method) {
    return m_backend.method_output(method);
  }

private:
  Backend m_backend;
  bool m_ready = false;
};
