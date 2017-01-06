project(CasHash-CUDA)
cmake_minimum_required(VERSION 2.8)

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()

find_package(CUDA REQUIRED)

# Kepler
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

# export compile commands so that our auto-completion system can index the source files
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories(${PROJECT_SOURCE_DIR}/include)

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

cuda_add_executable(TestCasHashMatcher
  src/TestCasHashMatcher.cpp
  src/KeyFileReader.cpp
  src/HashConverter.cpp
  src/HashConverter.cu
  src/CasHashMatcher.cpp
  )

target_link_libraries(
  TestHashConverter
  curand)

target_link_libraries(
  TestCasHashMatcher
  curand)

# Activate (and require) C++11 support
set_property(TARGET TestKeyFileReader PROPERTY CXX_STANDARD 11)
set_property(TARGET TestKeyFileReader PROPERTY CXX_STANDARD_REQUIRED ON)

# For HHLR
configure_file(job.sh.in job.sh @ONLY)
