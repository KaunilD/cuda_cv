### CUDA-CV

Implementation of Computer Vision algorithms with Nvidia CUDA support.

![execution chart](comparision.png)

### Table of contents

1. [Dependencies](### 1.Libraries)
2. [Setup](### 2. Setup)

### 1. Dependencies

1. OpenCV 3.4 - For converting images to Mat. That's it.
2. CUDA 

### 2. Setup

Entire project is built using CMake (3.9) with MSVC 2017 generator rules on Windows. I will be adding *ix for support after core algorithms have been implemented. 

#### 2.1 Build and Install OpenCV on Windows:

1. Download OpenCV 3.4 source from the [github](https://github.com/opencv/opencv/releases/tag/3.4.4) repo. Lets call this directory as `OPENCV`.
2. Start CMake GUI (I use GUI on Windows because many dependencies have to be manually linked).
3. Point Source to `OPENCV/src`
4. Point Build to `OPENCV/build`
5. Configure. (Here after the config file is generated, you can ignore the modules that you dont need. This project only needs opencv core and highgui.)
6. Set Compiler to MSVC 2017 64. (Mine is 64bit OS.)
7. Generate.
8. Add `OPENCV/build/install/<your_platform>/vc15/lib` to system `Path` variable.
9. Set `OpenCV_DIR` to `OPENCV/build/install`

#### 2.2 Install CUDA 9.2.

1. The .exe downloaded from website is just a compressed file. So, after the setup files have been extracted make a copy of the folder (lets call this `CUDA`). You'll need a few files from this later for CUDA support in MSVC 2017.
2. Install all the components except __Visual Studio Integration__.
3. After the installation is complete copy files under `CUDA/CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions` to `C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations`
4. Now you have successfully installed and integrated CUDA support in Visual Studio.
5. You might want to edit host_config.h and change the upper limit for MSC_VER greatr than 1911 like this: `#if _MSC_VER < 1600 || _MSC_VER > 1955`

#### NOTE:
1. Do not include any `.cu` files in the cpp header files `.h`. NVCC mistakes `.cu` as cpp files on account of it being called int he header files and uses native c compiler for compilation instead of nvcc itself compiling it.
