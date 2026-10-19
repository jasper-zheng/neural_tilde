cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

project(backend)

# ExecuTorch is located by the parent (cmake/nn_executorch.cmake). The install
# ships vendored c10 headers, so no external libtorch is needed.
# Host-platform helpers (UTF-8 paths, CUDA device query) with no ExecuTorch
# dependency, so the tokenizer and neural.info can use them without the runtime.
add_library(nn_platform STATIC nn_platform.cpp)
set_property(TARGET nn_platform PROPERTY CXX_STANDARD 17)
target_include_directories(nn_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
	target_compile_options(nn_platform PUBLIC /utf-8 /Zc:__cplusplus)
endif()
if(WIN32)
	target_link_libraries(nn_platform PRIVATE
		shell32   # CommandLineToArgvW (nn::utf8_args)
		version)  # GetFileVersionInfoW (nn::vc_runtime_version)
endif()

add_library(backend STATIC backend.cpp nn_log.cpp)
set_property(TARGET backend PROPERTY CXX_STANDARD 17)
target_link_libraries(backend PUBLIC nn_platform)

# PUBLIC so the ExecuTorch include dirs (Module/Tensor API) and the runtime's
# -force_load link options propagate to the Max externals that link `backend`.
target_link_libraries(backend PUBLIC
	executorch
	extension_module
	extension_tensor
	extension_data_loader
)

# Minimum CUDA version the driver must support for this build's CUDA runtime
# (set by nn_executorch.cmake when the CUDA delegate is present).
if(NN_CUDA_REQUIRED_VERSION)
	target_compile_definitions(backend PRIVATE
		NN_CUDA_REQUIRED_VERSION=${NN_CUDA_REQUIRED_VERSION})
endif()
