cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CUDA_BUILD_CUBIN OFF)
set(CUDA_64_BIT_DEVICE_CODE "ON")
project(gstl_tests LANGUAGES CXX CUDA)

# C++ related
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# CUDA related
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# General stuff

set(GSTL_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/")
set(THIRDPARTY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ " "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cuda-api-wrappers/")
set(TEST_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/tests/")

include_directories(${GSTL_INCLUDES} ${THIRDPARTY_INCLUDES} ${TEST_FOLDER})

file(GLOB SOURCES
		${TEST_FOLDER}main.cpp
		${TEST_FOLDER}algorithms/**.cpp
		${TEST_FOLDER}algorithms/**.cu
		${TEST_FOLDER}allocators/**.cpp
		${TEST_FOLDER}allocators/**.cu
		${TEST_FOLDER}containers/**.cpp
		${TEST_FOLDER}containers/**.cu
		${TEST_FOLDER}concurrent/**.cpp
		${TEST_FOLDER}concurrent/**.cu
		${TEST_FOLDER}numeric/**.cpp
		${TEST_FOLDER}numeric/**.cu
		${TEST_FOLDER}utility/**.cpp
		${TEST_FOLDER}utility/**.cu
)

add_executable(gstl_tests ${SOURCES})
