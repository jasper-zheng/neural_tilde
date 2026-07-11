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

# --- Metal (AOTInductor) delegate: locate a libomp.dylib to ship beside each external ---
# The Metal runtime + the compiled .so embedded in a Metal .pte both depend on libomp at
# runtime; torch's libomp has an absolute install id that won't resolve on other machines
# (see cmake/nn_metal_libomp.sh). Only relevant when EXECUTORCH_ROOT was built with
# -DEXECUTORCH_BUILD_METAL=ON. Override with -DNN_LIBOMP_PATH=/path/to/libomp.dylib; for an
# exact ABI match with the AOTI .so, point it at the SAME libomp the ET runtime (torch) used.
if(TARGET metal_backend)
  if(NOT NN_LIBOMP_PATH)
    find_file(NN_LIBOMP_PATH NAMES libomp.dylib
      PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
  endif()
  if(NN_LIBOMP_PATH)
    message(STATUS "Metal delegate: will bundle libomp from ${NN_LIBOMP_PATH}")
  else()
    message(WARNING "Metal delegate is available but NN_LIBOMP_PATH is unset and no libomp.dylib "
                    "was found (try `brew install libomp` or -DNN_LIBOMP_PATH=...). Metal .pte files "
                    "will fail to load until libomp is bundled beside the external.")
  endif()
endif()

# Absolute path to the bundling helper, captured HERE (CMAKE_CURRENT_LIST_DIR resolves to the
# caller's dir inside a function, so we must record it at include time).
set(NN_METAL_LIBOMP_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/nn_metal_libomp.sh" CACHE INTERNAL "")

# POST_BUILD: ship libomp beside ``target`` and repoint its libomp dependency to @rpath.
# No-op unless the Metal backend is present and a libomp was located.
function(nn_bundle_metal_libomp target)
  if(NOT TARGET metal_backend OR NOT NN_LIBOMP_PATH)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND sh "${NN_METAL_LIBOMP_SCRIPT}"
            "$<TARGET_FILE:${target}>" "${NN_LIBOMP_PATH}"
    COMMENT "Bundle libomp.dylib for the Metal (AOTI) delegate")
endfunction()

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
  # MLX (Apple-Silicon GPU) delegate — fast, and DOES persist cached_conv streaming
  # state across execute() (verified bit-identical to XNNPACK on conv models incl.
  # multi-rate encoders; the old "MLX clicks" note was an untested assumption).
  # Still experimental: narrower op coverage (e.g. complex/FFT unsupported).
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
  # MPS (Apple GPU via MPSGraph) delegate. Persists cached_conv streaming state
  # across execute() (verified bit-identical to XNNPACK), so it streams correctly —
  # but its op/shape coverage is incomplete and some models abort at runtime in
  # MPSGraph verification, so validate each model. Deprecated in ExecuTorch >=1.4
  # (use CoreML/Metal there). The imported target already carries the Metal*
  # frameworks + -force_load via its interface; the frameworks are re-listed here
  # to match the CoreML/MLX blocks.
  if(TARGET mpsdelegate)
    target_link_libraries(${target} PRIVATE
      mpsdelegate
      "-framework Metal"
      "-framework MetalPerformanceShaders"
      "-framework MetalPerformanceShadersGraph")
  endif()
  # Metal (AOTInductor) delegate — EXPERIMENTAL. Whole-graph torch._inductor AOT compile is
  # embedded as an ad-hoc-signed .so in the .pte; the ExecuTorch Metal runtime extracts it to
  # a temp file and dlopen()s it at load (no CoreML-style .mlmodelc compile at load). The
  # imported metal_backend target carries -Wl,-export_dynamic via its interface — so the
  # dlopen'd .so resolves the aoti_torch_* shims from the host external — plus -force_load;
  # aoti_common provides the ETensor AOTI shims. NOTE: libomp is a runtime dep of the compiled
  # .so and is bundled beside each external (see the frontend CMakeLists, like mlx.metallib).
  # cached_conv streaming persistence across the monolithic AOTI fn is UNVERIFIED — smoke-test
  # neural.live~ before relying on it. Only active once EXECUTORCH_ROOT was built with
  # -DEXECUTORCH_BUILD_METAL=ON (older installs won't define the target -> this no-ops).
  if(TARGET metal_backend)
    target_link_libraries(${target} PRIVATE
      metal_backend
      aoti_common
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
      "-framework MetalPerformanceShadersGraph")
    # Ship libomp beside the external + repoint its libomp dep to @rpath (runtime dep of the
    # AOTI .so). No-op if NN_LIBOMP_PATH was not resolved above.
    nn_bundle_metal_libomp(${target})
  endif()
endfunction()
