project(CasHash-CUDA)
cmake_minimum_required(VERSION 3.1)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a seperate directory and run cmake from there.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if( NOT cmake_build_type_tolower STREQUAL "debug"
    AND NOT cmake_build_type_tolower STREQUAL "release"
    AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()

if(cmake_build_type_tolower STREQUAL "debug")
  list(APPEND CMAKE_CXX_FLAGS "-Wall -Wno-long-long")
  #add_definitions(-DDEBUG_HASH_MATCHER)
  #add_definitions(-DDEBUG_HASH_CONVERTER2)
endif()

find_package(CUDA REQUIRED)

# Kepler
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")

# C++0x support
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)

# export compile commands so that our auto-completion system can index the source files
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)

#cuda_add_executable(KeyMatchCUDA
#  src/main.cpp
#  src/KeyFileReader.cpp
#  )

cuda_add_executable(TestKeyFileReader
  src/TestKeyFileReader.cpp
  src/KeyFileReader.cpp
  )

cuda_add_executable(TestHashConverter
  src/TestHashConverter.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu
  )

target_link_libraries(
  TestHashConverter
  curand)

cuda_add_executable(TestHashMatcher
  src/TestHashMatcher.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu
  src/HashMatcher.cpp
  src/HashMatcher.cu
  )

target_link_libraries(
   TestHashMatcher
   curand)

cuda_add_executable(KeyMatchCUDA
  src/main.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu
  src/HashMatcher.cpp
  src/HashMatcher.cu
  )

target_link_libraries(
   KeyMatchCUDA
   curand)

# For HHLR
configure_file(job.sh.in job.sh @ONLY)
