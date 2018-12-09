__global__ void rgb_grayscale(const uchar* const rgbaImage, unsigned char * const greyImage, int rows, int cols ) {
	// converts rgb image to grayscale 
	// uses weighted sum formula instead 
	// of mean of r, g, b pixels
	// formula: 0.299f*R + 0.587f*G + 0.114f*B
	int bIdx = blockIdx.x;
	int tIdx = threadIdx.x;
	

}