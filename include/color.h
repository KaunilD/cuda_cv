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

void h_cvtColor_RGB2GRAY(
	const uchar3 * src,
	unsigned char * dst,
	int width, int height
);

void h_cvtColor_RGB2HSV(
	const uchar3 * src,
	uchar3 * dst,
	int width, int height
);

namespace cucv {
	int cvtColor(cv::Mat src, cv::Mat &dst, int code);
}