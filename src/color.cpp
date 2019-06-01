#include "color.h"

namespace cucv {

	int cvtColor(cv::Mat src, cv::Mat& dst, int code) {
		size_t num_pixels = src.rows*src.cols;

		switch (code) {
		case ChannelConversionCodes::RGB2GRAY: {
			dst.create(src.rows, src.cols, CV_8UC1);
			break;
		}
		case ChannelConversionCodes::RGB2HSV: {
			dst.create(src.rows, src.cols, CV_8UC3);
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
		
		switch (code) {
		
		case ChannelConversionCodes::RGB2GRAY: {
			unsigned char *d_dst_data = cuda_set_array(sizeof(uchar3) * num_pixels);

			h_cvtColor_RGB2GRAY(d_src_data, d_dst_data, src.cols, src.rows);
			// hook for the result of the operation on the host
			uchar3 *h_dst_data = (uchar3 *)dst.ptr<unsigned char>(0);
			// copy from GPU to host for writing to file.
			cudaMemcpy(h_dst_data, d_dst_data, sizeof(uchar3) * num_pixels, cudaMemcpyDeviceToHost);
			cudaFree(d_dst_data);

			std::cout << "done" << '\n';

			break;
		}
		case ChannelConversionCodes::RGB2HSV: {
			uchar3 *d_dst_data = (uchar3 *)cuda_set_array(sizeof(uchar3) * num_pixels);

			h_cvtColor_RGB2HSV(d_src_data, d_dst_data, src.cols, src.rows);
			// hook for the result of the operation on the host
			uchar3 *h_dst_data = (uchar3 *)dst.ptr<unsigned char>(0);
			// copy from GPU to host for writing to file.
			cudaMemcpy(h_dst_data, d_dst_data, sizeof(uchar3) * num_pixels, cudaMemcpyDeviceToHost);
			cudaFree(d_dst_data);

			std::cout << "done" << '\n';
			break;
		}
		default: {
			std::cout << "Operation not yet supported." << std::endl;
			return -1;
			break;
		}
		}

		cudaFree(d_src_data);
		
		return 0;
	}

}