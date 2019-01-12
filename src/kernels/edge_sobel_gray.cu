__global__ void d_edgeDetectSOBEL(unsigned char * diffX, unsigned char * const diffY, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	diffY[x+width*y] = diffY[x+width*y]*diffY[x+width*y] + diffX[x+width*y]*diffX[x+width*y];
}

__device__ float d_getPixelIntensity(const uchar1* const src, int x, int y, int width, int height){
	int colaced_loc = x + y*width;
	if (colaced_loc < 0) {
		return 0;
	}
	
	if (colaced_loc > width*height){
		return src[width*height].x;
	}

	return src[colaced_loc].x;
}

__global__ void d_partialDiffSOBELX(const uchar1* const src, unsigned char * const dst, int width, int height) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	float p11 = d_getPixelIntensity(src, x-1, y-1, width, height);
	float p12 = d_getPixelIntensity(src, x, y-1, width, height);
	float p13 = d_getPixelIntensity(src, x+1, y-1, width, height);

	float p21 = d_getPixelIntensity(src, x-1, y, width, height);
	float p22 = d_getPixelIntensity(src, x, y, width, height);
	float p23 = d_getPixelIntensity(src, x+1, y, width, height);

	float p31 = d_getPixelIntensity(src, x-1, y+1, width, height);
	float p32 = d_getPixelIntensity(src, x, y+1, width, height);
	float p33 = d_getPixelIntensity(src, x+1, y+1, width, height);

	dst[x+width*y] = (-p11+p13-2*p21+2*p23-p31+p33)/8.0;
}

__global__ void d_partialDiffSOBELY(const uchar1* const src, unsigned char * const dst, int width, int height) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	float p11 = d_getPixelIntensity(src, x-1, y-1, width, height);
	float p12 = d_getPixelIntensity(src, x, y-1, width, height);
	float p13 = d_getPixelIntensity(src, x+1, y-1, width, height);

	float p21 = d_getPixelIntensity(src, x-1, y, width, height);
	float p22 = d_getPixelIntensity(src, x, y, width, height);
	float p23 = d_getPixelIntensity(src, x+1, y, width, height);

	float p31 = d_getPixelIntensity(src, x-1, y+1, width, height);
	float p32 = d_getPixelIntensity(src, x, y+1, width, height);
	float p33 = d_getPixelIntensity(src, x+1, y+1, width, height);

	dst[x+width*y] = (p11+2*p12+p13-p31-2*p32-p33)/8.0;
}

void h_edgeDetectSOBEL(const uchar1 * src, unsigned char * dst, unsigned char * temp, int width, int height) {

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil(width/threadsPerBlock.x), ceil(height/threadsPerBlock.y), 1);
	
	d_partialDiffSOBELX <<< blocks, threadsPerBlock >>> (src, temp, width, height);
	cudaDeviceSynchronize();
	d_partialDiffSOBELY <<< blocks, threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
	d_edgeDetectSOBEL <<< blocks, threadsPerBlock >>> (temp, dst, width, height);
	cudaDeviceSynchronize();
}