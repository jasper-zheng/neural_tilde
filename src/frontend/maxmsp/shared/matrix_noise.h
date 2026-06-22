#pragma once
// Jitter-matrix → noise-input helper shared by the externals that let a host
// drive a `noise`-role input from a Jitter `float32`/`float64` matrix instead of
// the seed (neural.gen~ and neural.live~). float64 cells are down-cast to the
// float noise buffer. The matrix is [planes x H x W]: all tensor dims
// except the last two fold into the planecount, width(X)=shape[-1],
// height(Y)=shape[-2]. The per-object inlet wiring (creating the inlets, snapshot
// threading) stays in each external; the model-agnostic detection, matrix read,
// and the `<name>_inlet` toggle-attribute factory live here.
#include "../../../backend/backend.h" // InputSpec, Role
#include "c74_min.h"

#include <string>
#include <vector>

// A noise-role input a Jitter matrix can drive, with its folded geometry.
struct MatrixNoise {
  std::string name;
  int n0, n1, n2;  // planecount (folded leading dims), height(Y), width(X)
  size_t numel;    // n0 * n1 * n2
};

// The noise-role inputs whose shape folds to planes x H x W within Jitter's
// plane cap. Re-derive on every load so a reload tracks the current model. A
// noise that folds to > the plane cap is left seeded (warned, no matrix inlet).
inline std::vector<MatrixNoise>
detect_matrix_noise(const std::vector<InputSpec> &inputs,
                    c74::min::logger &err) {
  std::vector<MatrixNoise> out;
  for (const auto &in : inputs) {
    const auto &s = in.shape;
    if (in.role != Role::Noise || s.size() < 2)
      continue;
    int n1 = s[s.size() - 2], n2 = s[s.size() - 1];
    long long n0 = 1;
    for (size_t k = 0; k + 2 < s.size(); k++)
      n0 *= s[k];
    if (n0 > c74::max::JIT_MATRIX_MAX_PLANECOUNT) {
      err << "noise '" << in.name << "' folds to " << n0
          << " planes, exceeding Jitter's "
          << c74::max::JIT_MATRIX_MAX_PLANECOUNT
          << " plane cap, left seeded (no matrix inlet)" << c74::min::endl;
      continue;
    }
    out.push_back({in.name, (int)n0, n1, n2, (size_t)n0 * n1 * n2});
  }
  return out;
}

// Read the named Jitter float32/float64 matrix in `args[0]` into `out` (row-major,
// numel = nz.numel), validated against nz's folded geometry. Returns true on a
// clean read; otherwise logs the reason to `err` and leaves `out` untouched.
// cell(plane=p, x, y) -> out[(p*H + y)*W + x]; dimstride is in bytes (row-padded).
inline bool read_jit_matrix(const c74::min::atoms &args, const MatrixNoise &nz,
                            std::vector<float> &out, c74::min::logger &err) {
  if (args.empty()) {
    err << "jit_matrix message carried no matrix name" << c74::min::endl;
    return false;
  }
  c74::min::symbol msym = args[0];
  void *mat = c74::max::jit_object_findregistered((c74::max::t_symbol *)msym);
  if (!mat) {
    err << "matrix '" << std::string(msym) << "' not found" << c74::min::endl;
    return false;
  }
  if (c74::max::object_classname((c74::max::t_object *)mat) !=
      c74::max::_jit_sym_jit_matrix) {
    err << "'" << std::string(msym) << "' is not a jit_matrix" << c74::min::endl;
    return false;
  }

  // Lock the matrix for the read (the sender owns/reuses its data).
  void *savelock = c74::max::object_method((c74::max::t_object *)mat,
                                           c74::max::_jit_sym_lock, (void *)1);
  c74::max::t_jit_matrix_info info;
  c74::max::object_method((c74::max::t_object *)mat, c74::max::_jit_sym_getinfo,
                          &info);
  const bool is_f32 = info.type == c74::max::_jit_sym_float32;
  const bool is_f64 = info.type == c74::max::_jit_sym_float64;
  const bool shape_ok =
      (is_f32 || is_f64) && info.planecount == nz.n0 && info.dimcount == 2 &&
      info.dim[0] == nz.n2 && info.dim[1] == nz.n1;
  char *bp = nullptr;
  if (shape_ok)
    c74::max::object_method((c74::max::t_object *)mat, c74::max::_jit_sym_getdata,
                            &bp);
  bool ok = false;
  if (shape_ok && bp) {
    out.assign(nz.numel, 0.0f);
    for (int p = 0; p < nz.n0; p++)
      for (int y = 0; y < nz.n1; y++) {
        const char *row = bp + (size_t)y * info.dimstride[1];
        for (int x = 0; x < nz.n2; x++) {
          const char *cell = row + (size_t)x * info.dimstride[0];
          // float64 cells are down-cast to the float noise buffer.
          const float v = is_f32 ? reinterpret_cast<const float *>(cell)[p]
                                 : (float)reinterpret_cast<const double *>(cell)[p];
          out[((size_t)p * nz.n1 + y) * nz.n2 + x] = v;
        }
      }
    ok = true;
  }
  c74::max::object_method((c74::max::t_object *)mat, c74::max::_jit_sym_lock,
                          savelock);

  if (!shape_ok) {
    long d0 = info.dimcount >= 1 ? info.dim[0] : 0;
    long d1 = info.dimcount >= 2 ? info.dim[1] : 0;
    err << "matrix for '" << nz.name << "' must be float32 or float64, " << nz.n0
        << " plane(s), " << nz.n2 << " x " << nz.n1
        << " (got planecount=" << info.planecount << ", " << d0 << " x " << d1
        << "), ignored" << c74::min::endl;
  } else if (!bp) {
    err << "matrix for '" << nz.name << "' has no data, ignored" << c74::min::endl;
  }
  return ok;
}

// Create the boolean `<name>_inlet` toggle attribute for a matrix-drivable noise
// input (shared by neural.gen~ and neural.live~). Renders as an on/off checkbox
// with a descriptive inspector label. getter/setter are the raw Max attribute
// accessors: pass nullptr/nullptr for Max-owned storage (gen~, read once per
// generate), or custom thread-safe accessors (live~, read every audio block).
// Returns the attribute name (`<name>_inlet`) for the caller to track for
// cleanup; empty string on failure.
inline std::string
make_noise_inlet_attr(c74::max::t_object *self, const MatrixNoise &nz,
                      c74::max::method getter = nullptr,
                      c74::max::method setter = nullptr) {
  std::string an = nz.name + "_inlet";
  c74::max::t_object *attr = (c74::max::t_object *)c74::max::attribute_new(
      an.c_str(), c74::max::gensym("long"), 0, getter, setter);
  if (!attr)
    return {};
  c74::max::attr_addfilter_clip(attr, 0, 1, 1, 1);
  c74::max::object_addattr(self, attr);
  c74::max::object_attr_setlong(self, c74::max::gensym(an.c_str()), 0);
  // Render as an on/off checkbox with a descriptive inspector label.
  c74::max::object_attr_addattr_format(self, an.c_str(), "style",
                                       c74::max::gensym("symbol"), 0, "s",
                                       c74::max::gensym("onoff"));
  std::string label = "Read '" + nz.name +
                      "' noise from its matrix inlet instead of the seed.";
  c74::max::object_attr_addattr_format(self, an.c_str(), "label",
                                       c74::max::gensym("symbol"), 0, "s",
                                       c74::max::gensym(label.c_str()));
  return an;
}
