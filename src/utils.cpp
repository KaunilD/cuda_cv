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

uchar3* cuda_make_array(uchar3* host_src_data, size_t data_size) {
	uchar3* gpu_dst_data;
	// allocate space on the gpu for the data
	cudaMalloc(&gpu_dst_data, data_size);
	// copy from host to gpu
	cudaMemcpy(gpu_dst_data, host_src_data, data_size, cudaMemcpyHostToDevice);
	
	if (!gpu_dst_data) {
		printf("GPU malloc failed!");
	}

	return gpu_dst_data;
}

unsigned char* cuda_set_array(size_t data_size) {
	unsigned char* gpu_dst_data;
	// allocate space on the gpu for the data
	cudaMalloc(&gpu_dst_data, data_size);
	// set new allocated space to 0 to avoide garbage.
	cudaMemset(gpu_dst_data, 0, data_size);

	if (!gpu_dst_data) {
		printf("GPU malloc failed!");
	}

	return gpu_dst_data;
}