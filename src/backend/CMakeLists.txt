cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

project(backend)

# ExecuTorch is located by the parent (cmake/nn_executorch.cmake). The install
# ships vendored c10 headers, so no external libtorch is needed.
add_library(backend STATIC backend.cpp)
set_property(TARGET backend PROPERTY CXX_STANDARD 17)

# PUBLIC so the ExecuTorch include dirs (Module/Tensor API) and the runtime's
# -force_load link options propagate to the Max externals that link `backend`.
target_link_libraries(backend PUBLIC
	executorch
	extension_module
	extension_tensor
)
