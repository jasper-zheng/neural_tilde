#include "nn_platform.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h> // CommandLineToArgvW
#endif

namespace nn {

#ifdef _WIN32
static std::wstring widen(const std::string &utf8) {
  if (utf8.empty())
    return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int)utf8.size(),
                              nullptr, 0);
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int)utf8.size(), w.data(), n);
  return w;
}

static std::string narrow(const std::wstring &w) {
  if (w.empty())
    return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0,
                              nullptr, nullptr);
  std::string s((size_t)n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n,
                      nullptr, nullptr);
  return s;
}
#endif

std::filesystem::path fs_path(const std::string &utf8) {
#ifdef _WIN32
  return std::filesystem::path(widen(utf8));
#else
  return std::filesystem::path(utf8);
#endif
}

std::string utf8_string(const std::filesystem::path &p) {
#ifdef _WIN32
  return narrow(p.wstring());
#else
  return p.string();
#endif
}

bool file_exists(const std::string &utf8) {
  std::error_code ec;
  return std::filesystem::is_regular_file(fs_path(utf8), ec);
}

bool read_file(const std::string &utf8, std::vector<uint8_t> &out) {
  std::ifstream f(fs_path(utf8), std::ios::binary | std::ios::ate);
  if (!f)
    return false;
  const auto bytes = f.tellg();
  if (bytes < 0)
    return false;
  out.resize((size_t)bytes);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(out.data()), (std::streamsize)out.size());
  return (bool)f;
}

static bool is_ascii(const std::string &s) {
  for (unsigned char c : s)
    if (c >= 0x80)
      return false;
  return true;
}

std::string loader_path(const std::string &utf8) {
  if (is_ascii(utf8))
    return utf8;
#ifdef _WIN32
  const char *force = std::getenv("NN_FORCE_BUFFER_LOADER");
  if (force && *force && *force != '0')
    return {};
  const std::wstring w = widen(utf8);
  DWORD n = GetShortPathNameW(w.c_str(), nullptr, 0);
  if (n == 0)
    return {};
  std::wstring shrt((size_t)n, L'\0');
  n = GetShortPathNameW(w.c_str(), shrt.data(), n);
  if (n == 0)
    return {};
  shrt.resize(n);
  const std::string s = narrow(shrt);
  return is_ascii(s) ? s : std::string();
#else
  return utf8; // POSIX file APIs take UTF-8 bytes as they are
#endif
}

const char *host_platform() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#else
  return "linux";
#endif
}

std::vector<std::string> utf8_args(int argc, char **argv) {
  std::vector<std::string> args;
#ifdef _WIN32
  int n = 0;
  LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &n);
  if (wargv) {
    for (int i = 0; i < n; i++)
      args.push_back(narrow(wargv[i]));
    LocalFree(wargv);
    return args;
  }
#endif
  for (int i = 0; i < argc; i++)
    args.emplace_back(argv[i]);
  return args;
}

RuntimeVersion vc_runtime_version() {
  RuntimeVersion v;
#ifdef _WIN32
  const char *fake = std::getenv("NN_FAKE_VC_RUNTIME");
  if (fake && *fake) {
    std::sscanf(fake, "%d.%d", &v.major, &v.minor);
    return v;
  }
  HMODULE mod = GetModuleHandleW(L"msvcp140.dll");
  wchar_t path[MAX_PATH] = {0};
  if (!mod || !GetModuleFileNameW(mod, path, MAX_PATH))
    return v;
  DWORD handle = 0;
  const DWORD size = GetFileVersionInfoSizeW(path, &handle);
  if (size == 0)
    return v;
  std::vector<uint8_t> data(size);
  VS_FIXEDFILEINFO *ffi = nullptr;
  UINT len = 0;
  if (GetFileVersionInfoW(path, 0, size, data.data()) &&
      VerQueryValueW(data.data(), L"\\", reinterpret_cast<void **>(&ffi), &len) &&
      ffi && len >= sizeof(VS_FIXEDFILEINFO)) {
    v.major = HIWORD(ffi->dwFileVersionMS);
    v.minor = LOWORD(ffi->dwFileVersionMS);
  }
#endif
  return v;
}

RuntimeVersion vc_runtime_required() {
  RuntimeVersion v;
#ifdef _MSC_VER
  // _MSC_VER 1944 is toolset 14.44 (VS 2022 17.14); 1950 is 14.50.
  v.major = 14;
  v.minor = _MSC_VER - 1900;
#endif
  return v;
}

std::string vc_runtime_problem() {
  const RuntimeVersion have = vc_runtime_version();
  const RuntimeVersion need = vc_runtime_required();
  if (have.major == 0 || need.major == 0)
    return {};
  if (have.major > need.major ||
      (have.major == need.major && have.minor >= need.minor))
    return {};
  return "neural_tilde needs the Microsoft Visual C++ Redistributable (x64) " +
         std::to_string(need.major) + "." + std::to_string(need.minor) +
         " or newer, but this machine has " + std::to_string(have.major) + "." +
         std::to_string(have.minor) +
         ": install https://aka.ms/vs/17/release/vc_redist.x64.exe and restart Max";
}

std::string cuda_version_string(int version) {
  return std::to_string(version / 1000) + "." +
         std::to_string((version % 1000) / 10);
}

#ifdef _WIN32
// The handful of CUDA driver API entry points we need, resolved from
// nvcuda.dll (installed with the NVIDIA driver). Prototypes per cuda.h.
namespace {
using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
constexpr int kCcMajor = 75; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
constexpr int kCcMinor = 76; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

struct Driver {
  bool loaded = false;
  CUresult(__stdcall *cuInit)(unsigned int) = nullptr;
  CUresult(__stdcall *cuDriverGetVersion)(int *) = nullptr;
  CUresult(__stdcall *cuDeviceGetCount)(int *) = nullptr;
  CUresult(__stdcall *cuDeviceGet)(CUdevice *, int) = nullptr;
  CUresult(__stdcall *cuDeviceGetName)(char *, int, CUdevice) = nullptr;
  CUresult(__stdcall *cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
  CUresult(__stdcall *cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
  CUresult(__stdcall *cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
  CUresult(__stdcall *cuCtxPushCurrent)(CUcontext) = nullptr;
  CUresult(__stdcall *cuCtxPopCurrent)(CUcontext *) = nullptr;
  CUresult(__stdcall *cuMemGetInfo)(size_t *, size_t *) = nullptr;
};

template <typename F> static bool resolve(HMODULE lib, const char *name, F &fn) {
  fn = reinterpret_cast<F>(reinterpret_cast<void *>(GetProcAddress(lib, name)));
  return fn != nullptr;
}

const Driver &driver() {
  static Driver d;
  static std::once_flag once;
  std::call_once(once, [] {
    HMODULE lib = LoadLibraryW(L"nvcuda.dll");
    if (!lib)
      return;
    d.loaded = resolve(lib, "cuInit", d.cuInit) &&
               resolve(lib, "cuDriverGetVersion", d.cuDriverGetVersion) &&
               resolve(lib, "cuDeviceGetCount", d.cuDeviceGetCount) &&
               resolve(lib, "cuDeviceGet", d.cuDeviceGet) &&
               resolve(lib, "cuDeviceGetName", d.cuDeviceGetName) &&
               resolve(lib, "cuDeviceGetAttribute", d.cuDeviceGetAttribute) &&
               resolve(lib, "cuDevicePrimaryCtxRetain",
                       d.cuDevicePrimaryCtxRetain) &&
               resolve(lib, "cuDevicePrimaryCtxRelease_v2",
                       d.cuDevicePrimaryCtxRelease) &&
               resolve(lib, "cuCtxPushCurrent_v2", d.cuCtxPushCurrent) &&
               resolve(lib, "cuCtxPopCurrent_v2", d.cuCtxPopCurrent) &&
               resolve(lib, "cuMemGetInfo_v2", d.cuMemGetInfo);
  });
  return d;
}
} // namespace
#endif

const CudaDeviceInfo &cuda_device_info() {
  static CudaDeviceInfo info;
  static std::once_flag once;
  std::call_once(once, [] {
#ifdef _WIN32
    const Driver &d = driver();
    if (!d.loaded) {
      info.error = "no NVIDIA driver found (nvcuda.dll is missing or too old)";
      return;
    }
    d.cuDriverGetVersion(&info.driver_version);
    const CUresult init = d.cuInit(0);
    if (init == 0)
      d.cuDeviceGetCount(&info.device_count);
    if (init != 0 || info.device_count <= 0) {
      info.error = "no CUDA GPU found";
      if (init != 0 && init != 100) // 100: CUDA_ERROR_NO_DEVICE
        info.error += " (cuInit error " + std::to_string(init) + ")";
      return;
    }
    CUdevice dev = 0;
    d.cuDeviceGet(&dev, 0);
    char name[256] = {0};
    d.cuDeviceGetName(name, (int)sizeof(name) - 1, dev);
    info.name = name;
    int major = 0, minor = 0;
    d.cuDeviceGetAttribute(&major, kCcMajor, dev);
    d.cuDeviceGetAttribute(&minor, kCcMinor, dev);
    info.cc = major * 10 + minor;
    info.ok = true;
#else
    info.error = "CUDA models are only supported on Windows";
#endif
  });
  return info;
}

int sweep_stale_cuda_kernel_files() {
  static int removed = 0;
#ifdef _WIN32
  static std::once_flag once;
  std::call_once(once, [] {
    std::error_code ec;
    const auto dir = std::filesystem::temp_directory_path(ec);
    if (ec)
      return;
    for (auto it = std::filesystem::directory_iterator(dir, ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
      const std::wstring name = it->path().filename().wstring();
      if (name.rfind(L"executorch_cuda_", 0) != 0 || it->path().extension() != L".so")
        continue;
      std::error_code rm;
      if (std::filesystem::remove(it->path(), rm))
        removed++;
    }
  });
#endif
  return removed;
}

bool cuda_mem_info(size_t &free_bytes, size_t &total_bytes) {
#ifdef _WIN32
  if (!cuda_device_info().ok)
    return false;
  const Driver &d = driver();
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (d.cuDeviceGet(&dev, 0) != 0 || d.cuDevicePrimaryCtxRetain(&ctx, dev) != 0)
    return false;
  bool ok = false;
  if (d.cuCtxPushCurrent(ctx) == 0) {
    ok = d.cuMemGetInfo(&free_bytes, &total_bytes) == 0;
    CUcontext popped = nullptr;
    d.cuCtxPopCurrent(&popped);
  }
  d.cuDevicePrimaryCtxRelease(dev);
  return ok;
#else
  (void)free_bytes;
  (void)total_bytes;
  return false;
#endif
}

} // namespace nn
