#include "algorithms.h"

using namespace std;

Algorithms::Algorithms()
{
}

Algorithms::~Algorithms()
{
}

void h_cvt_rgba2gray(
	const uchar4 * rgbaImage,
	unsigned char * grayImage,
	int rows, int cols
);

int Algorithms::rgba2gray(cv::Mat src, cv::Mat& output) {
	cv::Mat gray, src_rgba;
	uchar4 * d_rgba_data, * h_rgba_data;
	unsigned char * d_gray_data, * h_gray_data;
	
	// convert to RGBA since cuda requires uchar4
	cv::cvtColor(src, src_rgba, cv::COLOR_BGR2RGBA);
	// create an empty 1 channel image to processed 
	// data from GPU
	gray.create(src.rows, src.cols, CV_16UC1);
	// point to the src rgba data in CPU.
	// to be able to copy data from CPU to GPU
	h_rgba_data = (uchar4 *)src_rgba.ptr<unsigned char>(0);
	// point to the host gray image in CPU.
	// this will be used to copy data from GPU to CPU.
	h_gray_data = gray.ptr<unsigned char>(0);

	// number of pixels to allocate space in the GPU memory.
	const size_t numPixels = src.rows * src.cols;
	// allocate space in the GPU for RGBA data from CPU.
	cudaMalloc(&d_rgba_data, sizeof(uchar4) * numPixels);
	// allocate space in GPU memory for storing the data
	// generated while computing gray image.
	cudaMalloc(&d_gray_data, sizeof(unsigned char) * numPixels);
	// clean up memory block.
	cudaMemset(d_gray_data, 0, sizeof(unsigned char) * numPixels);
	// copy scr RGBA data from CPU to GPU at hrgba_data location till
	// numpixels location.
	cudaMemcpy(d_rgba_data, h_rgba_data, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	// call the cuda kernel
	h_cvt_rgba2gray(d_rgba_data, d_gray_data, src.rows, src.cols);
	// copy data back from the GPU to the CPU.
	cudaMemcpy(h_gray_data, d_gray_data, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	// cleanup GPU memory.
	cudaFree(d_rgba_data);
	cudaFree(d_gray_data);

	cv::Mat _output(src.rows, src.cols, CV_8UC1, (void*)h_gray_data);
	output.create(_output.rows, _output.cols, _output.type);
	_output.copyTo(output);
	return 0;
}