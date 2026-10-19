#pragma once
#include <functional>
#include <string>

// Routable diagnostics for the runtime (Backend, GenRunner). Max on Windows
// never shows stderr, so every message goes through a sink the host can point
// at its console. With no sink installed, messages print to stderr as before.
namespace nn {

enum class LogLevel { Info, Warning, Error };

using LogSink = std::function<void(LogLevel, const std::string &)>;

// Process-wide fallback sink: used by a Backend without its own sink, and for
// ExecuTorch runtime messages emitted outside any Backend call. Null => stderr.
// Thread-safe.
void set_default_log_sink(LogSink sink);

// Emit through `sink` if set, else through the default sink.
void log(const LogSink &sink, LogLevel level, const std::string &msg);

// RAII: while alive, ExecuTorch's own error messages emitted on this thread go
// to `sink` (the Backend that is calling into the runtime) instead of the
// default sink. Nests; restores the previous scope on destruction.
class LogScope {
public:
  explicit LogScope(const LogSink *sink);
  ~LogScope();
  LogScope(const LogScope &) = delete;
  LogScope &operator=(const LogScope &) = delete;

private:
  const LogSink *m_prev;
};

// Route ExecuTorch's ET_LOG Error/Fatal messages into the sinks above (Info and
// Debug are dropped). Idempotent; Backend() calls it.
void install_executorch_log_forwarding();

} // namespace nn
