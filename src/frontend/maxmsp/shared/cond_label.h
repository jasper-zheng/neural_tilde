#pragma once
// Condition-inlet label helper shared by the externals that create one message
// inlet per `condition`-role model input (neural.gen~ and neural.live~). Renders
// an inlet's assist string from its InputSpec so a user hovering the inlet sees
// the tensor it expects — name, shape, dtype, and the optional sidecar
// description — instead of just the name. Mirrors how matrix_noise.h embeds a
// noise input's geometry in its inlet label. The per-object inlet wiring stays in
// each external; only the descriptor formatting lives here.
#include "../../../backend/backend.h" // InputSpec, Role, ScalarType

#include <string>
#include <vector>

// Short dtype name for a label (same mapping as the backend smoke test).
inline const char *cond_dtype_name(executorch::aten::ScalarType t) {
  using executorch::aten::ScalarType;
  if (t == ScalarType::Long)
    return "int64";
  if (t == ScalarType::Int)
    return "int32";
  if (t == ScalarType::Float)
    return "float32";
  return "?";
}

// "(d0, d1, ...)" for a shape; "()" for a scalar/0-d input. Kept consistent with
// shape_str in gen_smoke_test.cpp so shape rendering matches repo-wide.
inline std::string cond_shape_str(const std::vector<int> &shape) {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); i++)
    s += std::to_string(shape[i]) + (i + 1 < shape.size() ? ", " : "");
  return s + ")";
}

// The descriptor for a condition input's inlet: "<name> (<shape>) <dtype>", with
// ": <description>" appended only when the sidecar supplied one. Callers append
// their own trailer (e.g. " (list, or dictionary by name)").
inline std::string cond_inlet_label(const InputSpec &in) {
  std::string s = in.name + " " + cond_shape_str(in.shape) + " " +
                  cond_dtype_name(in.dtype);
  if (!in.description.empty())
    s += ": " + in.description;
  return s;
}
