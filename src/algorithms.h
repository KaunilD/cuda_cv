#pragma once
// CV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

class Algorithms
{
public:
	Algorithms();
	~Algorithms();

	static int rgba2gray(cv::Mat, cv::Mat&);

private:

};