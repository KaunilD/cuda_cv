#include "edge.h"

namespace cucv {
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
}