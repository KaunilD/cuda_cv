/*********************************/
// Common utilities for CUDA memeory
// allocation and deallocation.
//
// TRIGGER WARNING: DUPLICATE CODE 
// SINCE TEMPLATES PROTOTYPES 
// CANT BE SEPERATED INTO HEADERS AND
// SOURCE.
/*********************************/
#include "utils.h"

void error(const char *s) {
	perror(s);
	assert(0);
	exit(-1);
}

void check_error(cudaError_t status) {
	cudaError_t prev_status = cudaGetLastError();

	if (status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}

	if (prev_status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}

}

uchar3* cuda_make_array(uchar3* host_src_data, size_t data_size) {
	uchar3* gpu_dst_data = NULL;
	// allocate space on the gpu for the data
	cudaError_t status = cudaMalloc(&gpu_dst_data, data_size);
	check_error(status);
	// copy from host to gpu
	status = cudaMemcpy(gpu_dst_data, host_src_data, data_size, cudaMemcpyHostToDevice);
	check_error(status);
	
	if (!gpu_dst_data) {
		error("GPU malloc failed\n");
	}

	return gpu_dst_data;
}

unsigned char* cuda_set_array(size_t data_size) {
	unsigned char* gpu_dst_data = NULL;
	// allocate space on the gpu for the data
	cudaError_t status = cudaMalloc(&gpu_dst_data, data_size);
	check_error(status);
	// set new allocated space to 0 to avoide garbage.
	status = cudaMemset(gpu_dst_data, 0, data_size);
	check_error(status);

	if (!gpu_dst_data) {
		error("GPU malloc failed!\n");
	}

	return gpu_dst_data;
}