#pragma once
// Benchmarking
#include <iostream>
#include <chrono>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

void h_filterBoxBlurGRAY(
	const uchar1 * src,
	unsigned char * dst,
	unsigned char * temp,
	int width, int height
);


void h_cvtColorRGBA2GRAY(
	const uchar4 * src,
	unsigned char * dst,
	int width, int height
);

namespace cucv {

	
	int cvtColorRGBA2GRAY(cv::Mat src, cv::Mat& dst) {
		cv::Mat gray, src_rgba;
		uchar4 *d_rgba_data, *h_rgba_data;
		unsigned char * d_gray_data, * h_gray_data;
	
		// convert to RGBA since cuda requires uchar4
		cv::cvtColor(src, src_rgba, cv::COLOR_BGR2RGBA);
		// create an empty 1 channel image to copy processed 
		// data from GPU
		gray.create(src.rows, src.cols, CV_8UC1);
		// point to the src rgba data in CPU.
		// to be able to copy data from CPU to GPU
		h_rgba_data = (uchar4 *)src_rgba.ptr<unsigned char>(0);
		// point to the host gray image in CPU.
		// this will be used to copy data from GPU to CPU.
		h_gray_data = gray.ptr<unsigned char>(0);

		// number of pixels to allocate space in the GPU memory.
		const size_t num_pixels = src.rows * src.cols;
		// allocate space in the GPU for RGBA data from CPU.
		cudaMalloc(&d_rgba_data, sizeof(uchar4) * num_pixels);
		// allocate space in GPU memory for storing the data
		// generated while computing gray image.
		cudaMalloc(&d_gray_data, sizeof(unsigned char) * num_pixels);
		// clean up memory block.
		cudaMemset(d_gray_data, 0, sizeof(unsigned char) * num_pixels);
		// copy scr RGBA data from CPU to GPU at hrgba_data location till
		// num_pixels location.
		cudaMemcpy(d_rgba_data, h_rgba_data, sizeof(uchar4) * num_pixels, cudaMemcpyHostToDevice);
		// call the cuda kernel
		h_cvtColorRGBA2GRAY(d_rgba_data, d_gray_data, src.cols, src.rows);
		// copy data back from the GPU to the CPU.
		cudaMemcpy(h_gray_data, d_gray_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);
		// cleanup GPU memory.
		cudaFree(d_rgba_data);
		cudaFree(d_gray_data);
	
		gray.copyTo(dst);
		return 0;
	}



	int filterBoxBlurGRAY(cv::Mat src, cv::Mat& dst) {
		cv::Mat gray, gray_src;
	
		uchar1 *d_src_data, *h_src_data;
		unsigned char *h_dst_data, *d_dst_data, *d_temp_data;

		cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
		gray.create(gray_src.rows, gray_src.cols, CV_8UC1);

		h_src_data = (uchar1 *)gray_src.ptr<unsigned char>(0);
		h_dst_data = gray.ptr<unsigned char>(0);

		size_t num_pixels = gray_src.rows * gray_src.cols;

		cudaMalloc(&d_src_data, sizeof(uchar1) * num_pixels);
	
		cudaMalloc(&d_dst_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_dst_data, 0, sizeof(unsigned char) * num_pixels);
	
		cudaMalloc(&d_temp_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_temp_data, 0, sizeof(unsigned char) * num_pixels);
	
		cudaMemcpy(d_src_data, h_src_data, sizeof(uchar1) * num_pixels, cudaMemcpyHostToDevice);

		h_filterBoxBlurGRAY(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);

		cudaMemcpy(h_dst_data, d_dst_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);

		gray.copyTo(dst);

		cudaFree(d_dst_data);
		cudaFree(d_src_data);

		return 0;
	}

} // namespace cucv