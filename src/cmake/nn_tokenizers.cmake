# Fetch mlc-ai/tokenizers-cpp and expose its `tokenizers_cpp` static target.
#
# tokenizers-cpp loads a HuggingFace fast-tokenizer `tokenizer.json` and builds a
# Rust `tokenizers` core via cargo, which must be on PATH (macOS: `brew install
# rust`; Windows: `winget install Rustlang.Rustup`, MSVC toolchain). The Rust core
# is statically linked, so the resulting external needs no Rust at runtime.
#
# HF-only: disable the SentencePiece path (and its submodule/build) — neural.gen~
# uses tokenizer.json, not a .model spm file.

include(FetchContent)

find_program(NN_CARGO cargo HINTS "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
if(NOT NN_CARGO)
  message(FATAL_ERROR
    "NN_ENABLE_TOKENIZER needs Rust's cargo on PATH (macOS: `brew install rust`; "
    "Windows: `winget install Rustlang.Rustup`). Or configure with "
    "-DNN_ENABLE_TOKENIZER=OFF to build without neural.tokenizer.")
endif()

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
# Same dynamic MSVC runtime (/MD) as everything else (see src/CMakeLists.txt).
set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY "MD" CACHE STRING "" FORCE)

FetchContent_Declare(
  tokenizers_cpp_fetch
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG        c586c52f93f7b060753bd2388eb96a105cb7374d  # main as of 2026-10-18
)
if(WIN32)
  # tokenizers-cpp always configures its sentencepiece subproject (unused here), which
  # symlinks third_party/absl -> third_party/abseil-cpp/absl: that needs admin rights or
  # Developer Mode on Windows. Pre-create it as a directory junction (no privilege
  # needed; the target appears when sentencepiece fetches abseil), so it is skipped.
  FetchContent_GetProperties(tokenizers_cpp_fetch)
  if(NOT tokenizers_cpp_fetch_POPULATED)
    FetchContent_Populate(tokenizers_cpp_fetch)
  endif()
  set(_spm_third_party "${tokenizers_cpp_fetch_SOURCE_DIR}/sentencepiece/third_party")
  if(NOT EXISTS "${_spm_third_party}/absl")
    file(TO_NATIVE_PATH "${_spm_third_party}/absl" _link)
    file(TO_NATIVE_PATH "${_spm_third_party}/abseil-cpp/absl" _target)
    execute_process(COMMAND cmd /c mklink /J "${_link}" "${_target}"
                    OUTPUT_QUIET ERROR_QUIET)
  endif()
  add_subdirectory(${tokenizers_cpp_fetch_SOURCE_DIR} ${tokenizers_cpp_fetch_BINARY_DIR})
else()
  FetchContent_MakeAvailable(tokenizers_cpp_fetch)
endif()
