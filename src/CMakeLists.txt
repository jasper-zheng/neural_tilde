cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
endif()

project(neural_tilde)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()
    message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
endif()

# Package version: single source of truth is pyproject.toml (the Python
# exporter's version); every external is stamped with it (-DVERSION).
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../pyproject.toml" _nn_version_line
     REGEX "^version *= *\"[^\"]+\"" LIMIT_COUNT 1)
string(REGEX REPLACE "^version *= *\"([^\"]+)\".*" "\\1" NN_VERSION "${_nn_version_line}")
if(NOT NN_VERSION)
    message(FATAL_ERROR "Could not read `version = \"...\"` from ../pyproject.toml")
endif()
message(STATUS "neural_tilde version ${NN_VERSION}")

# Windows: the dynamic MSVC runtime (/MD) everywhere. ExecuTorch, the CUDA
# shims, the inductor-compiled CUDA kernels and Rust all use it, and Max.exe
# itself depends on it, so it is present wherever Max runs.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Find the ExecuTorch runtime and define nn_link_executorch() /
# nn_frontend_settings().
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nn_executorch.cmake)

# Text tokenizer for generative models (neural.tokenizer, gen_cli). Fetches
# tokenizers-cpp, which builds a Rust core via cargo (macOS: `brew install
# rust`; Windows: `winget install Rustlang.Rustup`). Off => Rust-free build
# without those two. The Rust core links statically.
option(NN_ENABLE_TOKENIZER "Build the tokenizers-cpp text tokenizer (requires cargo/rustc)" ON)

add_subdirectory(backend) # DEEP LEARNING BACKEND (ExecuTorch)
# Generative core (gen_runner: backend only, no tokenizer/Rust) for neural.gen~;
# its text->audio CLI (gen_cli) is added only with the tokenizer.
add_subdirectory(generative)

if(NN_ENABLE_TOKENIZER)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nn_tokenizers.cmake)
    add_subdirectory(tokenizer)
endif()

# Max externals: macOS (.mxo) and Windows (.mxe64). The Pd external is not yet
# migrated to ExecuTorch.
if(APPLE OR WIN32)
    add_subdirectory(frontend/maxmsp/neural.live_tilde)     # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mc.neural.live_tilde)  # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mcs.neural.live_tilde) # MAX MSP EXTERNAL
    # Pure Max/Jitter helper (uniform->N(0,1) probit). No ExecuTorch/Rust.
    add_subdirectory(frontend/maxmsp/neural.gaussianize)
    # Pure Max helper: reads a model .json metadata file and emits its contents
    # as a Max dictionary. No ExecuTorch/Rust.
    add_subdirectory(frontend/maxmsp/neural.info)
    # Generative latent-diffusion runner (needs gen_runner = backend only).
    add_subdirectory(frontend/maxmsp/neural.gen_tilde)
    if(NN_ENABLE_TOKENIZER)
        # Text tokenizer object (needs the tokenizer/Rust). Emits a dictionary
        # (input_ids + attention_mask) for neural.gen~.
        add_subdirectory(frontend/maxmsp/neural.tokenizer)
    endif()
endif()

# Standalone smoke test: drives Backend on a .pte without Max (build target
# `et_smoke_test`). Useful for validating the runtime + a model end-to-end.
add_executable(et_smoke_test backend/et_smoke_test.cpp)
target_link_libraries(et_smoke_test PRIVATE backend)
nn_link_executorch(et_smoke_test)
set_property(TARGET et_smoke_test PROPERTY CXX_STANDARD 17)
if(MLX_METALLIB_PATH)
    add_custom_command(TARGET et_smoke_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MLX_METALLIB_PATH}"
                "$<TARGET_FILE_DIR:et_smoke_test>/mlx.metallib"
        COMMENT "Copy mlx.metallib next to et_smoke_test")
endif()

# Standalone smoke test for the generative (kind: "generative") path. Drives
# Backend::generate() on a .pte without Max; supports a golden-parity --io mode
# (build target `gen_smoke_test`). See src/backend/gen_smoke_test.cpp.
add_executable(gen_smoke_test backend/gen_smoke_test.cpp)
target_link_libraries(gen_smoke_test PRIVATE backend)
nn_link_executorch(gen_smoke_test)
set_property(TARGET gen_smoke_test PROPERTY CXX_STANDARD 17)
if(MLX_METALLIB_PATH)
    add_custom_command(TARGET gen_smoke_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MLX_METALLIB_PATH}"
                "$<TARGET_FILE_DIR:gen_smoke_test>/mlx.metallib"
        COMMENT "Copy mlx.metallib next to gen_smoke_test")
endif()
