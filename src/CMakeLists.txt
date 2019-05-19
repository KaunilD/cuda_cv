cmake_minimum_required (VERSION 3.8)
set( CMAKE_CXX_STANDARD 11 )

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
	message(STATUS "CUDA library status:")
	message(STATUS "    NVCC_EXECUTABLE: ${CUDA_NVCC_EXECUTABLE}")
	message(STATUS "    INCLUDE_PATH: ${CUDA_INCLUDE_DIRS}")
	message(STATUS "    TOOLKIT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

	# compared to class settings, we let NVidia's FindCUDA CMake detect 
	# whether to build x64.  
	# Complied for 1060 architecture. Add more from here:
	# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
	set(CUDA_NVCC_FLAGS "
		-gencode;arch=compute_61,code=sm_61;"
	)
  
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
	message("CUDA is not installed on this system.")
endif()

set ( KERNELS
	kernels/color.cu
	kernels/edge_detect.cu
	kernels/filter.cu
	kernels/matcher.cu
	kernels/utils.cu
)

set ( HEADERS
	${CMAKE_SOURCE_DIR}/include/constants.h
	${CMAKE_SOURCE_DIR}/include/cuda_cv.h
)

set ( SRCS
	cuda_cv.cpp
)

add_library( libcudacv STATIC
	${KERNELS}
	${HEADERS}
	${SRCS}
)

target_include_directories( libcudacv PUBLIC ${CMAKE_SOURCE_DIR}/include )
target_include_directories( libcudacv PUBLIC ${CUDA_INCLUDE_DIRS})

target_link_libraries( libcudacv ${OpenCV_LIBS})