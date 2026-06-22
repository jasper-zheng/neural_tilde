# Fetch mlc-ai/tokenizers-cpp and expose its `tokenizers_cpp` static target.
#
# tokenizers-cpp loads a HuggingFace fast-tokenizer `tokenizer.json` and builds a
# Rust `tokenizers` core via cargo (must be on PATH; `brew install rust`). The
# Rust core is statically linked, so the resulting .mxo needs no Rust at runtime.
#
# HF-only: disable the SentencePiece path (and its submodule/build) — neural.gen~
# uses tokenizer.json, not a .model spm file.

include(FetchContent)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  tokenizers_cpp_fetch
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG        main          # TODO: pin to a specific commit once verified
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp_fetch)
