#include "color.h"

namespace cucv {

	int cvtColor(cv::Mat src, cv::Mat& dst, int code) {
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

		// make space and copy source image into the GPU
		// return the pointer to data copied into the GPU
		uchar3 *h_src_data = (uchar3 *)src.ptr<unsigned char>(0);
		uchar3 *d_src_data = cuda_make_array(h_src_data, sizeof(uchar3) * num_pixels);

		// hook for result of the operation on the GPU
		// ceate space in the GPU for result image 
		// and initialize it to zero
		unsigned char *d_dst_data = cuda_set_array(sizeof(unsigned char) * num_pixels);

		switch (code) {
		case ChannelConversionCodes::RGB2GRAY: {
			h_cvtColor_RGB2GRAY(d_src_data, d_dst_data, src.cols, src.rows);
			std::cout << "done" << '\n';
			break;
		}
		default: {
			std::cout << "Operation not yet supported." << std::endl;
			return -1;
			break;
		}
		}

		// hook for the result of the operation on the host
		unsigned char *h_dst_data = dst.ptr<unsigned char>(0);
		// copy from GPU to host for writing to file.
		cudaMemcpy(h_dst_data, d_dst_data, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost);

		cudaFree(d_src_data);
		cudaFree(d_dst_data);

		return 0;
	}

}