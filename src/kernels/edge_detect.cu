#include "utils.cu"

__global__ void d_calculateGradientMag(unsigned char * diffX, unsigned char * const diffY, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	diffY[x+width*y] = sqrtf(diffY[x+width*y]*diffY[x+width*y] + diffX[x+width*y]*diffX[x+width*y]);
}


__global__ void d_partialDiffX_SIMPLE(const uchar1* const src, unsigned char * const dst, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x == 0){
		dst[x+width*y] = -src[x+width*y].x + src[(x+1)+width*y].x;
	}else if (x == width-1){
		dst[x+width*y] = src[x+width*y].x - src[(x-1)+width*y].x;
	}else{
		dst[x+width*y] = src[(x-1)+width*y].x - src[(x)+width*y].x;
	}
}


__global__ void d_partialDiffY_SIMPLE(const uchar1* const src, unsigned char * const dst, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y == 0){
		dst[x+width*y] = -src[x+width*y].x + src[x+width*(y+1)].x;
	}else if (y == height-1){
		dst[x+width*y] = src[x+width*y].x - src[x+width*(y-1)].x;
	}else{
		dst[x+width*y] = src[x+width*(y-1)].x - src[x+width*(y+1)].x;
	}
}


__global__ void d_partialDiffX_PREWITT(const uchar1* const src, unsigned char * const dst, int width, int height) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	
	Neighbours3x3<uchar1> neighbours;

	d_getNeighbours_8(src, width, height, x, y, neighbours);

	dst[x+width*y] = ((neighbours.p13.x+neighbours.p23.x+neighbours.p33.x)-(neighbours.p11.x+neighbours.p21.x+neighbours.p31.x))/6.0;
}


__global__ void d_partialDiffY_PREWITT(const uchar1* const src, unsigned char * const dst, int width, int height) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	Neighbours3x3<uchar1> neighbours;

	d_getNeighbours_8(src, width, height, x, y, neighbours);

	dst[x+width*y] = ((neighbours.p13.x+neighbours.p23.x+neighbours.p33.x)-(neighbours.p11.x+neighbours.p21.x+neighbours.p31.x))/6.0;
}


__global__ void d_partialDiffX_SOBEL(const uchar1* const src, unsigned char * const dst, int width, int height) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	
	Neighbours3x3<uchar1> neighbours;

	d_getNeighbours_8(src, width, height, x, y, neighbours);

	dst[x+width*y] = (-neighbours.p11.x+neighbours.p13.x-2*neighbours.p21.x+2*neighbours.p23.x-neighbours.p31.x+neighbours.p33.x)/8.0;
}


__global__ void d_partialDiffY_SOBEL(const uchar1* const src, unsigned char * const dst, int width, int height) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	Neighbours3x3<uchar1> neighbours;

	d_getNeighbours_8(src, width, height, x, y, neighbours);

	dst[x+width*y] = (neighbours.p11.x+2*neighbours.p12.x+neighbours.p13.x-neighbours.p31.x-2*neighbours.p32.x-neighbours.p33.x)/8.0;
}


void h_edgeDetect_SOBEL(const uchar1 * src, unsigned char * dst, unsigned char * temp, int width, int height) {

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil(width/threadsPerBlock.x), ceil(height/threadsPerBlock.y), 1);
	
	d_partialDiffX_SOBEL <<< blocks, threadsPerBlock >>> (src, temp, width, height);
	cudaDeviceSynchronize();
	
	d_partialDiffY_SOBEL <<< blocks, threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
	
	d_calculateGradientMag <<< blocks, threadsPerBlock >>> (temp, dst, width, height);
	cudaDeviceSynchronize();
}


void h_edgeDetect_PREWITT(const uchar1 * src, unsigned char * dst, unsigned char * temp, int width, int height) {

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil(width/threadsPerBlock.x), ceil(height/threadsPerBlock.y), 1);
	
	d_partialDiffX_PREWITT <<< blocks, threadsPerBlock >>> (src, temp, width, height);
	cudaDeviceSynchronize();
	
	d_partialDiffY_PREWITT <<< blocks, threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
	
	d_calculateGradientMag <<< blocks, threadsPerBlock >>> (temp, dst, width, height);
	cudaDeviceSynchronize();
}


void h_edgeDetect_SIMPLE(const uchar1 * src, unsigned char * dst, unsigned char * temp, int width, int height) {

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil(width/threadsPerBlock.x), ceil(height/threadsPerBlock.y), 1);
	
	d_partialDiffX_SIMPLE <<< blocks, threadsPerBlock >>> (src, temp, width, height);
	cudaDeviceSynchronize();
	
	d_partialDiffY_SIMPLE <<< blocks, threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
	
	d_calculateGradientMag <<< blocks, threadsPerBlock >>> (temp, dst, width, height);
	cudaDeviceSynchronize();
}