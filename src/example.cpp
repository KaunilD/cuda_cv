// cuda_cv.cpp : Defines the entry point for the application.
//
#include "example.h"

using namespace std;
using namespace cucv;

int main()
{
	cv::Mat src, dst;

	const string& img_filename = "C:\\Users\\dhruv\\Development\\cuda\\cuda_cv\\src\\data\\21kx5k.jpg";

	src = cv::imread(img_filename);
	
	cucv::cvtColor<uchar3>(src, dst, ChannelConversionCodes::RGB2GRAY);
	
	cv::imwrite("exp.png", dst);
	
	// cin.get();
	return 0;
}
