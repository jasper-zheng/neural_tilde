#pragma once
// Posts one console error per external when the machine's Microsoft C++
// runtime is older than the one the externals were built against (Windows).
// Max itself needs an older runtime, so this can happen on a machine where Max
// runs fine. Called from the class setup of every external linking nn_platform.
#include "../../../backend/nn_platform.h"
#include "c74_min.h"
#include <string>

inline void check_vc_runtime() {
  static bool checked = false;
  if (checked)
    return;
  checked = true;
  const std::string problem = nn::vc_runtime_problem();
  if (!problem.empty())
    c74::max::error("%s", problem.c_str());
}
