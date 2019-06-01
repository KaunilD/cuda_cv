/*
#########################################
#										#
#		RGB -> GRAYSCALE CONVERSION		#
#										#
#########################################
*/
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


void h_cvtColor_RGB2GRAY(const uchar3 * src, unsigned char * dst, int width, int height) {
	
	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil( width/threadsPerBlock.x), ceil( height/threadsPerBlock.y), 1);
	
	d_cvtColorRGB2GRAY <<< blocks , threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
}

/*
#########################################
#										#
#		RGB -> HSV CONVERSION			#
#										#
#########################################
*/
__global__ void d_cvtColorRGB2HSV(const uchar3* const src, uchar3 * const dst, int width, int height ) {
	// converts rgb image to hsv
	// V = max(RGB)
	// S = (V - min(R, G, B))/V
	// H = https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	uchar3 pixel = src[x+width*y];
	
	int R = pixel.x;
	int G = pixel.y;
	int B = pixel.z;

	int V = max(max(R, G), B);
	int C = V - min(min(R, G), B);
	
	int H = 0;
	if (C == 0){
		H = 0;
	}else if(V == R){
		H = (G-B)/C;
	}else if(V == G){
		H = (B-R)/C + 2;
	}else if(V == B){
		H = (R-G)/C + 4;
	}
	H/=6;

	if (H <= 0){
		H+=1;
	}

	dst[x+width*y].z = V;
	dst[x+width*y].y = C/V;
	dst[x+width*y].x = H;

}

void h_cvtColor_RGB2HSV(const uchar3 * src, uchar3 * dst, int width, int height) {
	
	dim3 threadsPerBlock(16, 16, 1);
	dim3 blocks(ceil( width/threadsPerBlock.x), ceil( height/threadsPerBlock.y), 1);
	
	d_cvtColorRGB2HSV <<< blocks , threadsPerBlock >>> (src, dst, width, height);
	cudaDeviceSynchronize();
}
