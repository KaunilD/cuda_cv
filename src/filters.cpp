#include "filters.h"

namespace cucv {

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
}
