#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Host-platform helpers for the runtime: UTF-8 paths and the CUDA device query.
// Max hands the externals UTF-8 paths; on Windows the narrow file APIs read them
// in the ANSI code page instead, so all file access goes through these.
namespace nn {

// A filesystem path from a UTF-8 string (wide on Windows).
std::filesystem::path fs_path(const std::string &utf8);
// The UTF-8 string of a filesystem path.
std::string utf8_string(const std::filesystem::path &p);
bool file_exists(const std::string &utf8);
bool read_file(const std::string &utf8, std::vector<uint8_t> &out);

// A path ExecuTorch's narrow-path file loaders can open: `utf8` unchanged when
// it is plain ASCII (always, off Windows), else its ASCII 8.3 short form if the
// volume has one. Empty when there is no such path: the caller then loads the
// file into memory instead. NN_FORCE_BUFFER_LOADER=1 forces the empty result
// for non-ASCII paths (testing).
std::string loader_path(const std::string &utf8);

// "windows" | "macos" | "linux": the value an exporter writes as `platform`.
const char *host_platform();

// The CUDA driver and device 0, queried once per process through the driver
// API loaded at runtime (no link-time CUDA dependency, so the externals still
// load on machines without an NVIDIA driver). Windows only; elsewhere
// `error` says so.
struct CudaDeviceInfo {
  bool ok = false;        // a usable device 0 was found
  int driver_version = 0; // e.g. 13000 for CUDA 13.0 (0: no driver)
  int device_count = 0;
  std::string name;       // device 0
  int cc = 0;             // compute capability as major*10 + minor (89 = sm_89)
  std::string error;      // why !ok, in plain words
};
const CudaDeviceInfo &cuda_device_info();

// The CUDA runtime extracts each method's kernel library to
// %TEMP%\executorch_cuda_<pid>_<n>.so and keeps it loaded for the process
// lifetime; Windows can't delete a loaded DLL, so the files outlive the
// process. Removes the stale ones (those of running processes are locked and
// stay). Windows only; runs once per process. Returns the number removed.
int sweep_stale_cuda_kernel_files();

// Free / total memory of device 0 in bytes (its primary context). Diagnostic.
bool cuda_mem_info(size_t &free_bytes, size_t &total_bytes);

// "13.0" from a CUDA version number such as 13000.
std::string cuda_version_string(int version);

// The Microsoft C++ runtime (msvcp140.dll and friends). Code built with a
// toolset needs a runtime at least that new, but Max itself is built with an
// older one, so a machine running Max can have a runtime too old for the
// externals. Windows only; {0, 0} elsewhere or when unknown.
struct RuntimeVersion {
  int major = 0;
  int minor = 0;
};
// The loaded msvcp140.dll. NN_FAKE_VC_RUNTIME=14.30 overrides it (testing).
RuntimeVersion vc_runtime_version();
// The oldest runtime this build supports: its toolset.
RuntimeVersion vc_runtime_required();
// Empty when the loaded runtime is new enough (or unknown, or not on Windows);
// else one line saying what to install.
std::string vc_runtime_problem();

// The command-line arguments as UTF-8 (on Windows argv is in the ANSI code
// page, so they are re-read from the wide command line). For the CLI tools.
std::vector<std::string> utf8_args(int argc, char **argv);

} // namespace nn
