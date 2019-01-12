__global__ void d_edgeDetectSIMPLE(unsigned char * diffX, unsigned char * const diffY, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	diffY[x+width*y] = diffY[x+width*y]*diffY[x+width*y] + diffX[x+width*y]*diffX[x+width*y];
}

__global__ void d_partialDiffX(const uchar1* const src, unsigned char * const dst, int width, int height) {

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

__global__ void d_partialDiffY(const uchar1* const src, unsigned char * const dst, int width, int height) {
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

void h_edgeDetectSIMPLE(const uchar1 * src, unsigned char * dst, unsigned char * temp, int width, int height) {

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil(width/threadsPerBlock.x), ceil(height/threadsPerBlock.y), 1);
	
	d_partialDiffX <<< blocks, threadsPerBlock >>> (src, temp, width, height);
	cudaDeviceSynchronize();
	d_partialDiffY <<< blocks, threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
	d_edgeDetectSIMPLE <<< blocks, threadsPerBlock >>> (temp, dst, width, height);
	cudaDeviceSynchronize();
	
}