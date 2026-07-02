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

# Find the ExecuTorch 1.3.1 runtime and define nn_link_executorch().
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nn_executorch.cmake)

add_subdirectory(backend) # DEEP LEARNING BACKEND (ExecuTorch)

# Text tokenizer for generative models (neural.gen~). Fetches tokenizers-cpp,
# which builds a Rust core via cargo (`brew install rust`). Off => signal-only,
# Rust-free build. The Rust core links statically into the .mxo.
option(NN_ENABLE_TOKENIZER "Build the tokenizers-cpp text tokenizer (requires cargo/rustc)" ON)
if(NN_ENABLE_TOKENIZER)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nn_tokenizers.cmake)
    add_subdirectory(tokenizer)
    add_subdirectory(generative) # host orchestration + gen_cli (needs the tokenizer)
endif()

# macOS only. Windows is dropped; the Pd external is not yet migrated to ExecuTorch.
if(APPLE)
    add_subdirectory(frontend/maxmsp/neural.live_tilde)     # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mc.neural.live_tilde)  # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mcs.neural.live_tilde) # MAX MSP EXTERNAL
    # Pure Max/Jitter helper (uniform->N(0,1) probit). No ExecuTorch/Rust, so it
    # lives outside the tokenizer guard.
    add_subdirectory(frontend/maxmsp/neural.gaussianize)
    # Pure Max helper: reads a model .json metadata file and emits its contents
    # as a Max dictionary. No ExecuTorch/Rust, so it also lives outside the guard.
    add_subdirectory(frontend/maxmsp/neural.info)
    if(NN_ENABLE_TOKENIZER)
        # Generative latent-diffusion runner (needs gen_runner = backend only).
        add_subdirectory(frontend/maxmsp/neural.gen_tilde)
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
