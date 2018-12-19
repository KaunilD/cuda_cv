__global__ void d_cvt_rgba2gray(const uchar4* const rgbaImage, unsigned char * const grayImage, int rows, int cols ) {
	// converts rgb image to grayscale 
	// uses weighted sum formula instead 
	// of mean of r, g, b pixels
	// formula: 0.299f*R + 0.587f*G + 0.114f*B
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	uchar4 pixel = rgbaImage[pos];
	grayImage[pos] = 0.299*pixel.x+0.587*pixel.y+0.114*pixel.z;

}


void h_cvt_rgba2gray(const uchar4 * rgbaImage, unsigned char * grayImage, int rows, int cols) {
	d_cvt_rgba2gray <<< dim3(ceil((rows*cols)/1024), 1, 1) , dim3(1024, 1, 1) >>> (rgbaImage, grayImage, rows, cols);
	cudaDeviceSynchronize();
}