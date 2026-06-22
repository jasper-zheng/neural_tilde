cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(tokenizer)

# Thin HF-tokenizer wrapper (NnTokenizer) over tokenizers-cpp. Kept as its own
# static lib so only generative consumers (neural.gen~, the host) pull in the
# Rust tokenizer core — the signal externals stay lean and Rust-free.
add_library(tokenizer STATIC tokenizer.cpp tokenizer_config.cpp)
set_property(TARGET tokenizer PROPERTY CXX_STANDARD 17)
target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tokenizer PUBLIC tokenizers_cpp)

# Standalone token-for-token parity test against the HF golden (no Max needed).
add_executable(tok_test tok_test.cpp)
set_property(TARGET tok_test PROPERTY CXX_STANDARD 17)
target_link_libraries(tok_test PRIVATE tokenizer)
