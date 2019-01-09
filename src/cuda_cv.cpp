// cuda_cv.cpp : Defines the entry point for the application.
//
#include "cuda_cv.h"

using namespace std;



int main()
{
	cv::Mat image, dst;

	const string& img_filename = "C:\\Users\\dhruv\\Development\\cuda\\cuda_cv\\src\\data\\400x800.jpg";
	
	image = cv::imread(img_filename);

	Algorithms::rgba2gray(image, dst);
	cv::imwrite("exp_.png", dst);
	
	// cin.get();
	return 0;
}
