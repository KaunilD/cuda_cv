// cuda_cv.cpp : Defines the entry point for the application.
//
#include "example.h"

using namespace std;
using namespace cucv;

int main()
{
	double duration;
	std::clock_t start;
	cv::Mat src, dst;
	const string& img_filename = "C:\\Users\\dhruv\\Development\\cuda\\cuda_cv\\src\\data\\21kx5k.jpg";

	src = cv::imread(img_filename);
	
	/*
	// BENCHMARKING CODE
	start = std::clock();
	cv::boxFilter(src, dst, src.depth(), cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "printf: " << duration << '\n';
	*/
	cv::imwrite("exp.png", dst);
	
	cin.get();
	
	return 0;
}
