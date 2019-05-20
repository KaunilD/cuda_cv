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
#include "utils.h"


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


namespace cucv {
	int edgeDetect(cv::Mat, cv::Mat&, int);
}