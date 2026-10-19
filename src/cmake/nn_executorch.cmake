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

# --- CUDA (AOTInductor) delegate, Windows: the driver must support at least the CUDA
# version the runtime was built with (the ET config already found CUDAToolkit). Encoded
# like cudaDriverGetVersion (13.0 -> 13000); Backend refuses CUDA models on older drivers.
if(TARGET aoti_cuda_backend AND CUDAToolkit_VERSION_MAJOR)
  math(EXPR NN_CUDA_REQUIRED_VERSION
       "${CUDAToolkit_VERSION_MAJOR} * 1000 + ${CUDAToolkit_VERSION_MINOR} * 10")
  message(STATUS "CUDA delegate: models need a driver supporting CUDA "
                 "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}+")
endif()

# Link `target` against the ExecuTorch runtime, CPU op kernels, and the MLX
# delegate. The imported ET targets already carry `-force_load` via their
# INTERFACE_LINK_OPTIONS, so op/kernel/backend static initializers register
# without any extra whole-archive handling. Everything links statically, so the
# only runtime artifact to ship is mlx.metallib (copied per-external, post-build).
# Windows: XNNPACK and the CUDA delegate. Static registration survives MSVC linking
# because the imported ET targets carry /WHOLEARCHIVE in their INTERFACE_LINK_OPTIONS
# (the Windows counterpart of -force_load). The two CUDA runtime DLLs are copied
# beside an executable, or for a Max external into NN_SUPPORT_DIR (default
# <build>/support), whose contents go into the package's support/ folder, which Max
# puts on the DLL search path.
set(NN_SUPPORT_DIR "${CMAKE_BINARY_DIR}/support" CACHE PATH
    "Where the Windows runtime DLLs for the Max externals are collected")
function(nn_link_executorch target)
  target_link_libraries(${target} PRIVATE
    executorch
    extension_module
    extension_tensor
    extension_data_loader
    extension_flat_tensor
    optimized_native_cpu_ops_lib   # portable/optimized ATen ops (CPU fallback)
  )
  if(APPLE)
    target_link_libraries(${target} PRIVATE
      "-framework Foundation"
      "-framework Accelerate")
  endif()
  # CUDA (AOTInductor) delegate — Windows/NVIDIA. The kernels are a native library
  # embedded in the .pte (weights in the .ptd), extracted to %TEMP% and loaded at
  # method load; that library imports aoti_cuda_shims.dll, which imports
  # extension_cuda.dll, so both ship beside the target.
  if(TARGET aoti_cuda_backend)
    target_link_libraries(${target} PRIVATE aoti_cuda_backend)
    get_target_property(_type ${target} TYPE)
    if(_type STREQUAL "MODULE_LIBRARY")
      set(_dest "${NN_SUPPORT_DIR}")
    else()
      set(_dest "$<TARGET_FILE_DIR:${target}>")
    endif()
    foreach(_dll aoti_cuda_shims extension_cuda)
      if(TARGET ${_dll})
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "${_dest}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${_dll}>" "${_dest}"
          COMMENT "Copy ${_dll}.dll for ${target}")
      endif()
    endforeach()
  endif()
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

# --- Max externals (frontend/maxmsp/*) -------------------------------------------------
# Call right after include(min-pretarget.cmake), in the external's CMakeLists (a macro, so
# it edits that directory's scope). With the Visual Studio generator the Max SDK's
# max-pretarget forces the static MSVC runtime (/MT) through CMAKE_<LANG>_FLAGS_<CONFIG>
# and a directory-wide add_compile_options. Everything the externals link (ExecuTorch,
# the CUDA shims, Rust) uses the dynamic runtime, so put back CMake's defaults; the
# runtime then comes from CMAKE_MSVC_RUNTIME_LIBRARY (/MD, set in src/CMakeLists.txt).
macro(nn_frontend_pretarget)
  if(MSVC)
    foreach(_lang C CXX)
      set(CMAKE_${_lang}_FLAGS_DEBUG "/Zi /Ob0 /Od /RTC1")
      set(CMAKE_${_lang}_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
      set(CMAKE_${_lang}_FLAGS_MINSIZEREL "/O1 /Ob1 /DNDEBUG")
      set(CMAKE_${_lang}_FLAGS_RELWITHDEBINFO "/Zi /O2 /Ob1 /DNDEBUG")
    endforeach()
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")
  endif()
endmacro()

# Call after include(min-posttarget.cmake): the package version and the MSVC settings
# shared by every external.
function(nn_frontend_settings target)
  target_compile_definitions(${target} PRIVATE VERSION="${NN_VERSION}")
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8 /Zc:__cplusplus /permissive-)
    target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES)
  endif()
endfunction()
