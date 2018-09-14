#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include <cuda_runtime.h>

using namespace std;

// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
// grayWidthStep - number of gray bytes
__global__ void bgr_to_blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int blue = 0;
  int red = 0;
  int green = 0;

	if ((xIndex < width) && (yIndex < height))
	{
    for (int i = -2; i<=2; i++) {
      for (int j = -2; j<=2; j++) {
        if ((xIndex + j < width) && (yIndex + i < height) && (xIndex + j > 0) && (yIndex + i > 0))
        {
          blue += input[(yIndex+i) * colorWidthStep + (3 * (xIndex+j))]/25;
          green += input[((yIndex+i) * colorWidthStep + (3 * (xIndex+j)))+1]/25;
          red += input[((yIndex+i) * colorWidthStep + (3 * (xIndex+j)))+2]/25;
        }
      }
    }

    output[yIndex * colorWidthStep + (3 * xIndex)] = static_cast<unsigned char>(blue);
    output[(yIndex * colorWidthStep + (3 * xIndex))+1] = static_cast<unsigned char>(green);
    output[(yIndex * colorWidthStep + (3 * xIndex))+2] = static_cast<unsigned char>(red);
	}
}

void convert_blur(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors
	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

  auto start_cpu =  chrono::high_resolution_clock::now();
	// Launch the color conversion kernel
	bgr_to_blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));
	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  auto end_cpu =  chrono::high_resolution_clock::now();

  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

  printf("bgr_to_blur_kernel <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
         grid.y,
         block.x, block.y, duration_ms.count());

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "image.jpg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC3);

	//Call the wrapper function
	convert_blur(input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
