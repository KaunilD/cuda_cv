#include "utils.cu"

__global__ void d_filter_BOX_GRAY(const uchar1* const src, unsigned char * const dst, int width, int height ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	Neighbours3x3<uchar1> neighbours;
	
	d_getNeighbours_8(src, width, height, x, y, neighbours);
	
	dst[x+width*y] = d_avgNeighbours_8(neighbours);

}


void h_filter_BOX_GRAY(const uchar1 * src, unsigned char * dst, int width, int height) {
	
	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil(width/threadsPerBlock.x), ceil(height/threadsPerBlock.y), 1);
	
	d_filter_BOX_GRAY <<< blocks , threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
}


__global__ void d_filter_MEDIAN_GRAY(const uchar1* const src, unsigned char * const dst, int width, int height ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	dst[x+width*y] = 0;

}


void h_filter_MEDIAN_GRAY(const uchar1 * src, unsigned char * dst, int width, int height) {
	
	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil( (float) width/threadsPerBlock.x), ceil( (float) height/threadsPerBlock.y), 1);
	
	d_filter_MEDIAN_GRAY <<< blocks , threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
}