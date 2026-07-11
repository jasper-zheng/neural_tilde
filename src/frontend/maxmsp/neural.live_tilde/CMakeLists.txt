# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.24)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

#############################################################
# MAX EXTERNAL
#############################################################

include_directories(
	"${C74_INCLUDES}"
)

set(SOURCE_FILES neural.live_tilde.cpp)

# Version pinned for the neural package (was: git describe --tags).
set(VERSION "0.0.4")
add_definitions(-DVERSION="${VERSION}")

add_library(
	${PROJECT_NAME}
	MODULE
	${SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE backend)
nn_link_executorch(${PROJECT_NAME})           # ExecuTorch runtime + MLX delegate

if (APPLE)
set_target_properties(${PROJECT_NAME} PROPERTIES
	BUILD_WITH_INSTALL_RPATH FALSE
	LINK_FLAGS "-Wl,-rpath,@loader_path/"
)
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

# min-posttarget forces CXX_STANDARD 17; override AFTER it because
# neural.live_tilde.cpp uses <semaphore> (C++20).
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

if (APPLE)
# Ship the MLX Metal shader library next to the external binary so the MLX
# runtime can locate it at load time. Everything else links statically.
if (MLX_METALLIB_PATH)
add_custom_command(
		TARGET ${PROJECT_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MLX_METALLIB_PATH}"
				"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/mlx.metallib"
		COMMENT "Copy mlx.metallib"
	)
endif()

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
	add_custom_command(
		TARGET ${PROJECT_NAME}
		POST_BUILD
		COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
		COMMENT "Codesign external"
	)
endif()
endif()
