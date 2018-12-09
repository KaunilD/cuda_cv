﻿# CMakeList.txt : CMake project for cuda_cv, include source and define
# project specific logic here.
cmake_minimum_required (VERSION 3.8)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    VERSION: ${OpenCV_VERSION}")
message(STATUS "    LIBS: ${OpenCV_LIBS}")
message(STATUS "    INCLUDE_PATH: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  
  # Complied for 1060 architecture. Add more from here:
  # http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/clang; 
  -gencode;arch=compute_61,code=sm_61;"
  )
  
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

# Include sub-projects.
add_subdirectory ("kernels")
# Add source to this project's executable.
add_executable (cuda_cv "cuda_cv.cpp" "cuda_cv.h")
target_link_libraries( cuda_cv ${OpenCV_LIBS} )