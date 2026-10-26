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

option(CONVOX_CLAMP_EDGE "Clamp-to-edge borders instead of zero-padding" OFF)
if(CONVOX_CLAMP_EDGE)
  add_compile_definitions(CONVOX_CLAMP_EDGE)
endif()

execute_process(COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE CONVOX_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT CONVOX_GIT_COMMIT)
  set(CONVOX_GIT_COMMIT unknown)
endif()

# Optional CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
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

# ---- libraries ----
add_library(convox_common STATIC
  src/common/image.cpp
  src/common/filter.cpp
  src/common/validator.cpp
  src/common/timing.cpp
  src/common/env.cpp
  src/common/csv.cpp)
target_include_directories(convox_common PUBLIC
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(convox_common PUBLIC OpenMP::OpenMP_CXX)
target_compile_definitions(convox_common PRIVATE CONVOX_GIT_COMMIT="${CONVOX_GIT_COMMIT}")

# Serial reference: no auto-vectorisation, so it stays the plain-loop baseline.
add_library(convox_serial STATIC src/serial/serial.cpp)
target_include_directories(convox_serial PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(convox_serial PUBLIC convox_common)
target_compile_options(convox_serial PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)

set(CONVOX_IMPL_LIBS convox_serial)

if(CMAKE_CUDA_COMPILER)
  add_library(convox_cuda STATIC src/cuda/device_info.cu)
  target_include_directories(convox_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(convox_common PUBLIC CONVOX_HAVE_CUDA=1)
  target_link_libraries(convox_common PUBLIC convox_cuda)
  list(APPEND CONVOX_IMPL_LIBS convox_cuda)
endif()

# Dispatcher: maps an implementation name to its function.
add_library(convox_runner STATIC src/common/runner.cpp)
target_link_libraries(convox_runner PUBLIC convox_common ${CONVOX_IMPL_LIBS})

# ---- executable ----
add_executable(convox src/common/main.cpp)
target_link_libraries(convox PRIVATE convox_runner)

# ---- tests, built on demand: cmake --build build --target <name> ----
foreach(t image filter validator serial timing)
  add_executable(test_${t} EXCLUDE_FROM_ALL tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE convox_runner)
endforeach()
