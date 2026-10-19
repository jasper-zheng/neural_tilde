#include "nn_log.h"

#include <iostream>
#include <mutex>

#include <executorch/runtime/platform/platform.h>

namespace nn {

static std::mutex g_sink_mutex;
static LogSink g_default_sink;
static thread_local const LogSink *t_scope_sink = nullptr;

void set_default_log_sink(LogSink sink) {
  std::lock_guard<std::mutex> lock(g_sink_mutex);
  g_default_sink = std::move(sink);
}

static void log_default(LogLevel level, const std::string &msg) {
  LogSink sink;
  {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    sink = g_default_sink; // call outside the lock: a sink may log or re-set
  }
  if (sink)
    sink(level, msg);
  else
    std::cerr << msg << std::endl;
}

void log(const LogSink &sink, LogLevel level, const std::string &msg) {
  if (sink)
    sink(level, msg);
  else
    log_default(level, msg);
}

LogScope::LogScope(const LogSink *sink) : m_prev(t_scope_sink) {
  t_scope_sink = sink;
}

LogScope::~LogScope() { t_scope_sink = m_prev; }

static void emit_executorch_log(et_timestamp_t, et_pal_log_level_t level,
                                const char *, const char *, size_t,
                                const char *message, size_t length) {
  if (level != kError && level != kFatal)
    return;
  std::string msg = "neural: executorch: ";
  msg.append(message, length);
  while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r'))
    msg.pop_back();
  if (t_scope_sink)
    log(*t_scope_sink, LogLevel::Error, msg);
  else
    log_default(LogLevel::Error, msg);
}

void install_executorch_log_forwarding() {
  static std::once_flag once;
  std::call_once(once, [] {
    executorch::runtime::register_pal(executorch::runtime::PalImpl::create(
        emit_executorch_log, __FILE__));
  });
}

} // namespace nn
