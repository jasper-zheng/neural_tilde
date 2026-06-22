cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(generative)

# gen_runner: the tokenizer-FREE generative core — role dispatcher only (supplied
# conditioning by name / seeded noise / clamped params). Links `backend`
# (ExecuTorch runtime) only, NO tokenizer/Rust. This is what the Rust-free Max
# object neural.gen~ links. (The WAV writer is CLI-only; it lives with gen_cli.)
add_library(gen_runner STATIC gen_runner.cpp)
set_property(TARGET gen_runner PROPERTY CXX_STANDARD 17)
target_include_directories(gen_runner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gen_runner PUBLIC backend)

# Standalone text->audio CLI (prompt -> WAV), no Max needed. The only consumer
# that combines the tokenizer (Rust) with gen_runner, so the prompt->audio
# orchestration lives in gen_cli itself; it links both directly. The Max split
# is neural.tokenizer (tokenizer) + neural.gen~ (gen_runner).
add_executable(gen_cli gen_cli.cpp wav.cpp)
set_property(TARGET gen_cli PROPERTY CXX_STANDARD 17)
target_link_libraries(gen_cli PRIVATE gen_runner tokenizer)
nn_link_executorch(gen_cli)
if(MLX_METALLIB_PATH)
    add_custom_command(TARGET gen_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MLX_METALLIB_PATH}"
                "$<TARGET_FILE_DIR:gen_cli>/mlx.metallib"
        COMMENT "Copy mlx.metallib next to gen_cli")
endif()
