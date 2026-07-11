# neural.gen~ — generative (latent-diffusion) Max external. Mirrors neural_tilde
# but links gen_runner (backend only, Rust-FREE) — tokenization lives in the
# separate neural.tokenizer external, so this object pulls in no tokenizer/Rust.

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

set(SOURCE_FILES neural.gen_tilde.cpp)

set(VERSION "0.0.4")
add_definitions(-DVERSION="${VERSION}")

add_library(
	${PROJECT_NAME}
	MODULE
	${SOURCE_FILES}
)

# gen_runner pulls in backend (ExecuTorch) only — no tokenizer/Rust.
target_link_libraries(${PROJECT_NAME} PRIVATE gen_runner)
nn_link_executorch(${PROJECT_NAME})           # ExecuTorch runtime + MLX delegate

if (APPLE)
set_target_properties(${PROJECT_NAME} PROPERTIES
	BUILD_WITH_INSTALL_RPATH FALSE
	LINK_FLAGS "-Wl,-rpath,@loader_path/"
)
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

# min-posttarget forces CXX_STANDARD 17; override AFTER it because
# neural.gen_tilde.cpp uses <semaphore> (C++20).
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

if (APPLE)
# Ship the MLX Metal shader library next to the external binary.
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
