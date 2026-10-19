# neural.info — read a model's .json metadata and emit it as a Max dictionary,
# plus what this host offers (platform, CUDA GPU). Links only nn_platform (no
# ExecuTorch, no MLX, no tokenizer/Rust). The Max dictionary C API symbols
# resolve at load from the Max host, like the other objects.

cmake_minimum_required(VERSION 3.24)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)
nn_frontend_pretarget()             # dynamic MSVC runtime (see nn_executorch.cmake)

if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

#############################################################
# MAX EXTERNAL
#############################################################

include_directories(
	"${C74_INCLUDES}"
)

set(SOURCE_FILES neural.info.cpp)

add_library(
	${PROJECT_NAME}
	MODULE
	${SOURCE_FILES}
)

# UTF-8 file access + the host/CUDA device query (no ExecuTorch).
target_link_libraries(${PROJECT_NAME} PRIVATE nn_platform)

if (APPLE)
set_target_properties(${PROJECT_NAME} PROPERTIES
	BUILD_WITH_INSTALL_RPATH FALSE
	LINK_FLAGS "-Wl,-rpath,@loader_path/"
)
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)
nn_frontend_settings(${PROJECT_NAME})  # VERSION + MSVC flags

if (APPLE)
if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
	add_custom_command(
		TARGET ${PROJECT_NAME}
		POST_BUILD
		COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
		COMMENT "Codesign external"
	)
endif()
endif()
