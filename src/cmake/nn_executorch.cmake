# Locate the ExecuTorch 1.3.1 runtime (built + installed out-of-tree) and provide
# a helper to link a Max external against it together with the MLX delegate.
#
# Build the runtime once (see ADAPTATION_GUIDE.md):
#   cd <executorch> && cmake --preset mlx-release -DPYTHON_EXECUTABLE=<venv python>
#   cmake --build cmake-out --target install -j8
# then configure neural with:
#   -DEXECUTORCH_ROOT=<executorch>/cmake-out

if(NOT DEFINED EXECUTORCH_ROOT AND DEFINED ENV{EXECUTORCH_ROOT})
  set(EXECUTORCH_ROOT "$ENV{EXECUTORCH_ROOT}")
endif()
if(NOT DEFINED EXECUTORCH_ROOT)
  message(FATAL_ERROR
    "EXECUTORCH_ROOT is not set. Point it at the ExecuTorch install dir, e.g.\n"
    "  -DEXECUTORCH_ROOT=/path/to/executorch/cmake-out")
endif()

list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_ROOT}")
list(APPEND CMAKE_FIND_ROOT_PATH "${EXECUTORCH_ROOT}")
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)

# Link `target` against the ExecuTorch runtime, CPU op kernels, and the MLX
# delegate. The imported ET targets already carry `-force_load` via their
# INTERFACE_LINK_OPTIONS, so op/kernel/backend static initializers register
# without any extra whole-archive handling. Everything links statically, so the
# only runtime artifact to ship is mlx.metallib (copied per-external, post-build).
function(nn_link_executorch target)
  target_link_libraries(${target} PRIVATE
    executorch
    extension_module
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    optimized_native_cpu_ops_lib   # portable/optimized ATen ops (CPU fallback)
    "-framework Foundation"
    "-framework Accelerate"
  )
  # MLX (Apple-Silicon GPU) delegate — fast, but does NOT persist cached_conv
  # streaming state across execute() (streaming models click; see plan/memory).
  if(TARGET mlxdelegate)
    target_link_libraries(${target} PRIVATE
      mlxdelegate mlx "-framework Metal" "-framework QuartzCore")
  endif()
  # XNNPACK (CPU) delegate — correctly persists streaming (cached_conv) state.
  if(TARGET xnnpack_backend)
    target_link_libraries(${target} PRIVATE
      xnnpack_backend XNNPACK xnnpack-microkernels-prod)
    if(TARGET kleidiai)
      target_link_libraries(${target} PRIVATE kleidiai)
    endif()
  endif()
  # CoreML (Apple GPU/ANE) delegate — maps cached_conv mutable buffers to native
  # Core ML state, so it streams correctly (unlike MLX). The CoreML model is
  # embedded in the .pte (no extra bundle artifact). Needs macOS 15+ at runtime.
  if(TARGET coremldelegate)
    find_library(SQLITE_LIBRARY sqlite3)
    target_link_libraries(${target} PRIVATE
      coremldelegate sqlite3 "-framework CoreML")
  endif()
endfunction()
