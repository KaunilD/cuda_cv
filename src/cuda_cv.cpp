// cuda_cv.cpp : Defines the entry point for the application.
//
#include "cuda_cv.h"

using namespace std;

cv::Mat image;
cv::Mat imageGray;

int main()
{
	const string& img_filename = "C:\\Users\\dhruv\\Development\\cuda\\cuda_cv\\src\\len_full.jpg";
	image = cv::imread(img_filename);

	cv::cvtColor(image, image, CV_BGR2RGBA);

	imageGray.create(image.rows, image.cols, CV_16UC1);

	cv::imwrite("lena_rgba.png", image);
	cv::imwrite("lena_grey.png", imageGray);
	return 0;
}
