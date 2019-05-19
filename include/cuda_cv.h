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


void h_cvtColor_RGB2GRAY(
	const uchar3 * src,
	unsigned char * dst,
	int width, int height
);

void h_filter_BOX_GRAY(
	const uchar1 * src,
	unsigned char * dst,
	int width, int height
);
