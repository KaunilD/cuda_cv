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
// Constants
#include "constants.h"

void h_edgeDetect_SIMPLE(
	const uchar1 * src,
	unsigned char * dst,
	unsigned char * temp,
	int width, int height
);

void h_edgeDetect_SOBEL(
	const uchar1 * src,
	unsigned char * dst,
	unsigned char * temp,
	int width, int height
);

void h_edgeDetect_PREWITT(
	const uchar1 * src,
	unsigned char * dst,
	unsigned char * temp,
	int width, int height
);


void h_cvtColor_RGB2GRAY( 
	const uchar3 * src,
	unsigned char * dst,
	int width, int height
);

void h_filter_BOX_GRAY(
	const uchar1 * src,
	unsigned char * dst,
	int width, int height
);

namespace cucv {

	template <typename ucharT>
	int cvtColor (cv::Mat src, cv::Mat& dst, int code) {

		ucharT *d_src_data, *h_src_data;
		unsigned char *d_dst_data, *h_dst_data;
		size_t num_pixels = src.rows*src.cols;

		switch (code) {
			case ChannelConversionCodes::RGB2GRAY: {
				dst.create(src.rows, src.cols, CV_8UC1);
				break;
			}
			default: {
				std::cout << "Operation not yet supported." << std::endl;
				return -1;
				break;
			}
		}

		h_src_data = (ucharT *)src.ptr<unsigned char>(0);
		h_dst_data = dst.ptr<unsigned char>(0);

		cudaMalloc(&d_src_data, sizeof(ucharT) * num_pixels);
		cudaMemcpy(d_src_data, h_src_data, sizeof(ucharT) * num_pixels, cudaMemcpyHostToDevice);
		
		cudaMalloc(&d_dst_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_dst_data, 0, sizeof(unsigned char) * num_pixels);
		
		std::clock_t start;
		double duration;

		switch (code) {
			case ChannelConversionCodes::RGB2GRAY: {

				start = std::clock();
				h_cvtColor_RGB2GRAY(d_src_data, d_dst_data, src.cols, src.rows);
				duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "printf: " << duration << '\n';
				break;
			}
			default: {
				std::cout << "Operation not yet supported." << std::endl;
				return -1;
				break;
			}
		}

		
		cudaMemcpy(h_dst_data, d_dst_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);
		
		cudaFree(d_src_data);
		cudaFree(d_dst_data);

		return 0;
	}
	

	int edgeDetect(cv::Mat src, cv::Mat& dst, int code) {


		cv::Mat gray_src;
		uchar1 *d_src_data, *h_src_data;
		unsigned char *h_dst_data, *d_dst_data, *d_temp_data;
		size_t num_pixels = src.rows * src.cols;

		cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);

		dst.create(src.rows, src.cols, CV_8UC1);

		h_src_data = (uchar1 *)gray_src.ptr<unsigned char>(0);
		h_dst_data = dst.ptr<unsigned char>(0);

		cudaMalloc(&d_src_data, sizeof(uchar1) * num_pixels);

		cudaMalloc(&d_dst_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_dst_data, 0, sizeof(unsigned char) * num_pixels);

		cudaMalloc(&d_temp_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_temp_data, 0, sizeof(unsigned char) * num_pixels);

		cudaMemcpy(d_src_data, h_src_data, sizeof(uchar1) * num_pixels, cudaMemcpyHostToDevice);


		std::clock_t start;
		double duration;

		switch (code) {
			case EdgeCodes::SIMPLE: {
				start = std::clock();
				h_edgeDetect_SIMPLE(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);
				duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "printf: " << duration << '\n';
				break;
			}
			case EdgeCodes::SOBEL: {
				start = std::clock();
				h_edgeDetect_SOBEL(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);
				duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "printf: " << duration << '\n';
				break;
			}
			case EdgeCodes::PREWITT: {
				start = std::clock();
				h_edgeDetect_PREWITT(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);
				duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "printf: " << duration << '\n';
				break;
			}
			case EdgeCodes::CANNY: {
				std::cout << "Operation not supported yet!" << std::endl;
				break;
			}
			default: {
				std::cout << "Operation not supported yet!" << std::endl;
				return -1;
				break;
			}
		}

		cudaMemcpy(h_dst_data, d_dst_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);

		cudaFree(d_dst_data);
		cudaFree(d_src_data);

		return 0;
	}

	int filter(cv::Mat src, cv::Mat &dst, int code) {
		cv::Mat gray_src;
		uchar1 *d_src_data, *h_src_data;
		unsigned char *h_dst_data, *d_dst_data;
		
		size_t num_pixels = src.rows * src.cols;
		// currently this is only supported for 
		// 1 channel image.
		cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);

		dst.create(src.rows, src.cols, CV_8UC1);

		h_src_data = (uchar1 *)gray_src.ptr<unsigned char>(0);
		h_dst_data = dst.ptr<unsigned char>(0);

		cudaMalloc(&d_src_data, sizeof(uchar1) * num_pixels);

		cudaMalloc(&d_dst_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_dst_data, 0, sizeof(unsigned char) * num_pixels);

		cudaMemcpy(d_src_data, h_src_data, sizeof(uchar1) * num_pixels, cudaMemcpyHostToDevice);

		std::clock_t start;
		double duration;
		start = std::clock();
		h_filter_BOX_GRAY(d_src_data, d_dst_data, gray_src.cols, gray_src.rows);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "printf: " << duration << '\n';

		cudaMemcpy(h_dst_data, d_dst_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);

		cudaFree(d_dst_data);
		cudaFree(d_src_data);

		return 0;

	}

} // namespace cucv