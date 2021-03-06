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


void h_filter_BOX_GRAY(
	const uchar1 * src,
	unsigned char * dst,
	int width, int height
);

namespace cucv {
	int filter(cv::Mat , cv::Mat , int );
}