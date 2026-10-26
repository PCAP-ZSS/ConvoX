cmake_minimum_required(VERSION 3.20)
project(convox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Release builds use -O2 so the serial reference is not over-optimised.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenMP REQUIRED)

add_library(convox_common STATIC src/common/image.cpp src/common/filter.cpp)
target_include_directories(convox_common PUBLIC
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(convox_common PUBLIC OpenMP::OpenMP_CXX)

option(CONVOX_CLAMP_EDGE "Clamp-to-edge borders instead of zero-padding" OFF)
if(CONVOX_CLAMP_EDGE)
  add_compile_definitions(CONVOX_CLAMP_EDGE)
endif()

# Serial reference: no auto-vectorisation, so it stays the plain-loop baseline.
add_library(convox_serial STATIC src/serial/serial.cpp)
target_include_directories(convox_serial PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(convox_serial PUBLIC convox_common)
target_compile_options(convox_serial PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)

add_executable(convox src/common/main.cpp)
target_link_libraries(convox PRIVATE convox_common convox_serial)

# Test executables, built on demand with `cmake --build build --target test_image`.
add_executable(test_image EXCLUDE_FROM_ALL tests/test_image.cpp)
target_link_libraries(test_image PRIVATE convox_common)
add_executable(test_filter EXCLUDE_FROM_ALL tests/test_filter.cpp)
target_link_libraries(test_filter PRIVATE convox_common)
add_executable(test_serial EXCLUDE_FROM_ALL tests/test_serial.cpp)
target_link_libraries(test_serial PRIVATE convox_serial)

# Optional CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_library(convox_cuda STATIC src/cuda/device_info.cu)
  target_include_directories(convox_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(convox PRIVATE convox_cuda)
  target_compile_definitions(convox PRIVATE CONVOX_HAVE_CUDA=1)
  message(STATUS "CUDA: enabled (${CMAKE_CUDA_COMPILER_VERSION})")
else()
  message(STATUS "CUDA: not found, building CPU-only")
endif()

# Optional MPI
find_package(MPI QUIET COMPONENTS CXX)
if(MPI_CXX_FOUND)
  message(STATUS "MPI: enabled")
else()
  message(STATUS "MPI: not found, skipping")
endif()
