__global__ void d_cvtColorRGBA2GRAY(const uchar4* const rgbaImage, unsigned char * const grayImage, int width, int height ) {
	// converts rgb image to grayscale 
	// uses weighted sum formula instead 
	// of mean of r, g, b pixels
	// formula: 0.299f*R + 0.587f*G + 0.114f*B

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	uchar4 pixel = rgbaImage[x+width*y];
	
	grayImage[x+width*y] = 0.299*pixel.x+0.587*pixel.y+0.114*pixel.z;

}


void h_cvtColorRGBA2GRAY(const uchar4 * rgbaImage, unsigned char * grayImage, int width, int height) {
	

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil( (float) width/threadsPerBlock.x), ceil( (float) height/threadsPerBlock.y), 1);
	
	d_cvtColorRGBA2GRAY <<< blocks , threadsPerBlock >>> (rgbaImage, grayImage, width, height);
	cudaDeviceSynchronize();
}