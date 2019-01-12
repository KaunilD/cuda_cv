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

void h_cvtColorRGB2GRAY(
	const uchar3 * src,
	unsigned char * dst,
	int width, int height
);

namespace cucv {

	int cvtColor_RGB2GRAY(cv::Mat src, cv::Mat& dst) {
		
		if (src.channels() != 3) {
			return -1;
		}

		uchar3 *d_src_data, *h_src_data;

		unsigned char *d_dst_data, *h_dst_data;

		dst.create(src.rows, src.cols, CV_8UC1);

		h_src_data = (uchar3 *)src.ptr<unsigned char>(0);
		h_dst_data = dst.ptr<unsigned char>(0);

		size_t num_pixels = src.rows*src.cols;
		
		cudaMalloc(&d_src_data, sizeof(uchar3) * num_pixels);
		
		cudaMalloc(&d_dst_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_dst_data, 0, sizeof(unsigned char) * num_pixels);
		
		cudaMemcpy(d_src_data, h_src_data, sizeof(uchar3) * num_pixels, cudaMemcpyHostToDevice);
		
		h_cvtColorRGB2GRAY(d_src_data, d_dst_data, src.cols, src.rows);
		
		cudaMemcpy(h_dst_data, d_dst_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);
		
		cudaFree(d_src_data);
		cudaFree(d_dst_data);

		return 0;
	}
	
	int cvtColor(cv::Mat src, cv::Mat& dst, int code) {
		cv::Mat gray, src_rgba;
		
		switch (code) {
			case ChannelConversionCodes::RGB2GRAY: {
				return cucv::cvtColor_RGB2GRAY(src, dst);
				break;
			}
			default: {
				std::cout << "Operation not yet supported." << std::endl;
				return -1;
				break;
			}
		}

	}
	

	int edgeDetect(cv::Mat src, cv::Mat& dst, int code) {
		cv::Mat gray_src;

		uchar1 *d_src_data, *h_src_data;
		unsigned char *h_dst_data, *d_dst_data, *d_temp_data;

		cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);

		dst.create(gray_src.rows, gray_src.cols, CV_8UC1);

		h_src_data = (uchar1 *)gray_src.ptr<unsigned char>(0);
		h_dst_data = dst.ptr<unsigned char>(0);

		size_t num_pixels = gray_src.rows * gray_src.cols;

		cudaMalloc(&d_src_data, sizeof(uchar1) * num_pixels);

		cudaMalloc(&d_dst_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_dst_data, 0, sizeof(unsigned char) * num_pixels);

		cudaMalloc(&d_temp_data, sizeof(unsigned char) * num_pixels);
		cudaMemset(d_temp_data, 0, sizeof(unsigned char) * num_pixels);

		cudaMemcpy(d_src_data, h_src_data, sizeof(uchar1) * num_pixels, cudaMemcpyHostToDevice);

		switch (code) {
			case EdgeCodes::SIMPLE: {
				h_edgeDetect_SIMPLE(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);
				break;
			};
			case EdgeCodes::SOBEL: {
				h_edgeDetect_SOBEL(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);
				break;
			};
			case EdgeCodes::PREWITT: {
				h_edgeDetect_PREWITT(d_src_data, d_dst_data, d_temp_data, gray_src.cols, gray_src.rows);
				break;
			};
			case EdgeCodes::CANNY: {
				std::cout << "Operation not supported yet!" << std::endl;
				break;
			};
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

} // namespace cucv