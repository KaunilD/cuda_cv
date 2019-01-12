__global__ void d_filterMedianGRAY(const uchar1* const src, unsigned char * const dst, int width, int height ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	uchar1 pixel = src[x+width*y];
	
	dst[x+width*y] = 0;

}


void h_filterMedianGRAY(const uchar1 * src, unsigned char * dst, int width, int height) {
	

	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil( (float) width/threadsPerBlock.x), ceil( (float) height/threadsPerBlock.y), 1);
	
	d_filterMedianGRAY <<< blocks , threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
}