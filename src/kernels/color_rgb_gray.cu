__global__ void d_cvtColorRGB2GRAY(const uchar3* const src, unsigned char * const dst, int width, int height ) {
	// converts rgb image to grayscale 
	// uses weighted sum formula instead 
	// of mean of r, g, b pixels
	// formula: 0.299f*R + 0.587f*G + 0.114f*B

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	uchar3 pixel = src[x+width*y];
	
	dst[x+width*y] = 0.299*pixel.x+0.587*pixel.y+0.114*pixel.z;

}


void h_cvtColorRGB2GRAY(const uchar3 * src, unsigned char * dst, int width, int height) {
	

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil( width/threadsPerBlock.x), ceil( height/threadsPerBlock.y), 1);
	
	d_cvtColorRGB2GRAY <<< blocks , threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
}