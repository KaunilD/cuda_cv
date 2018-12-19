// cuda_cv.cpp : Defines the entry point for the application.
//
#include "cuda_cv.h"

using namespace std;

cv::Mat image, imageRGBA;
cv::Mat imageGray;

uchar4 * d_rgba_data, * h_rgba_data;
unsigned char * d_gray_data, * h_gray_data;

void h_cvt_rgba2gray(
	const uchar4 * rgbaImage, 
	unsigned char * grayImage, 
	int rows, int cols
);

int main()
{
	// an integer pointer.
	// points to the address
	const string& img_filename = "C:\\Users\\dhruv\\Development\\cuda\\cuda_cv\\src\\len_full.jpg";
	image = cv::imread(img_filename);
	// convert to RGBA becuase cuda uses uchar4.
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
	// allocate space on host for grayscale data
	imageGray.create(image.rows, image.cols, CV_16UC1);

	h_rgba_data = (uchar4 *) imageRGBA.ptr<unsigned char>(0);
	h_gray_data = imageGray.ptr<unsigned char>(0);

	const size_t numPixels = image.rows * image.cols;

	cudaMalloc(&d_rgba_data, sizeof(uchar4) * numPixels);

	cudaMalloc(&d_gray_data, sizeof(unsigned char) * numPixels);
	cudaMemset(d_gray_data, 0, sizeof(unsigned char) * numPixels);
	
	cudaMemcpy(d_rgba_data, h_rgba_data, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);


	h_cvt_rgba2gray(d_rgba_data, d_gray_data, image.rows, image.cols);

	cudaMemcpy(h_gray_data, d_gray_data, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);


	cv::Mat output(image.rows, image.cols, CV_16UC1, (void*)h_gray_data);

	cv::imwrite("gray.png", output);

	cudaFree(d_rgba_data);
	cudaFree(d_gray_data);

	// cin.get();
	return 0;
}
