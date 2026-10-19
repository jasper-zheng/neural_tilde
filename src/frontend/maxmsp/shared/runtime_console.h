#pragma once
// Max console plumbing for the externals that run models (neural.live~,
// mc.neural.live~, mcs.neural.live~, neural.gen~): the runtime's diagnostics
// sink and the load-time method preparation. stderr is invisible in Max, so
// everything the runtime says is posted to the console here.
#include "../../../backend/nn_log.h"
#include "vc_runtime_check.h"
#include "c74_min.h"
#include <string>

// The runtime's messages carry a "neural: " prefix for stderr; the object name
// Max prints in front of a posted line already says where it came from.
inline const char *strip_neural_prefix(const std::string &msg) {
  const char *s = msg.c_str();
  return msg.rfind("neural: ", 0) == 0 ? s + 8 : s;
}

// A sink posting to the Max console as `self`. Called from worker and audio
// threads too: Max's console posting is thread-safe, min's cerr is not.
inline nn::LogSink max_console_sink(c74::max::t_object *self) {
  return [self](nn::LogLevel level, const std::string &msg) {
    const char *text = strip_neural_prefix(msg);
    if (level == nn::LogLevel::Error)
      c74::max::object_error(self, "%s", text);
    else if (level == nn::LogLevel::Warning)
      c74::max::object_warn(self, "%s", text);
    else
      c74::max::object_post(self, "%s", text);
  };
}

// The process default: messages not tied to an instance (e.g. ExecuTorch
// logging outside any model call). Each external links its own copy of the
// runtime, so each installs it, from its class setup, which is also where the
// C++ runtime check runs.
inline void install_default_console_sink() {
  check_vc_runtime();
  nn::set_default_log_sink([](nn::LogLevel level, const std::string &msg) {
    if (level == nn::LogLevel::Error)
      c74::max::error("%s", msg.c_str());
    else
      c74::max::post("%s", msg.c_str());
  });
}

// Load `method` now (for CUDA: the kernel library and the weight upload)
// instead of on the first audio block or generate, and say where the model
// runs. Works with a Backend or a GenRunner. Returns false when the method
// can't be loaded; the sink has already posted the runtime's reason.
template <class Runner>
bool prepare_and_report(Runner &runner, const std::string &method,
                        const std::string &model_path, c74::max::t_object *self) {
  if (runner.prepare(method)) {
    c74::max::object_error(self, "method \"%s\" could not be loaded, inference is disabled",
                           method.c_str());
    return false;
  }
  c74::max::object_post(self, "loaded %s on %s", model_path.c_str(),
                        runner.device_description().c_str());
  return true;
}
