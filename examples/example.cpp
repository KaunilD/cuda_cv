// cuda_cv.cpp : Defines the entry point for the application.
//
#include "example.h"

using namespace std;

int main()
{
	cv::Mat src, dst;
	const string& img_filename = "exp.png";
	src = cv::imread(img_filename);
	cucv::cvtColor(src, dst, cucv::ChannelConversionCodes::RGB2GRAY);
	cv::imwrite("exp_.png", dst);
	return 0;
}
