#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include "omp.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

void bgr_to_blur_kernel(const cv::Mat& input, cv::Mat& output, int width, int height)
{
  int yIndex, xIndex, i, j, blue, red, green;
  #pragma omp parallel for private(yIndex, xIndex, i, j, blue, red, green) shared(input, output)
    for (yIndex=0; yIndex<height; yIndex++) {
      for (xIndex=0; xIndex<width; xIndex++) {
        blue = 0;
        red = 0;
        green = 0;

        for (i = -2; i<=2; i++) {
          for (j = -2; j<=2; j++) {
            if ((xIndex + j < width) && (yIndex + i < height) && (xIndex + j > 0) && (yIndex + i > 0)) {
              blue += input.at<cv::Vec3b>(yIndex+i, xIndex+j)[0]/25;
              green += input.at<cv::Vec3b>(yIndex+i, xIndex+j)[1]/25;
              red += input.at<cv::Vec3b>(yIndex+i, xIndex+j)[2]/25;
            }
          }
        }
        output.at<cv::Vec3b>(yIndex, xIndex)[0] = blue;
        output.at<cv::Vec3b>(yIndex, xIndex)[1] = green;
        output.at<cv::Vec3b>(yIndex, xIndex)[2] = red;
      }
    }
}

  void convert_blur(const cv::Mat& input, cv::Mat& output)
  {
  	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

    omp_set_num_threads(6);

    auto start_cpu =  chrono::high_resolution_clock::now();
  	bgr_to_blur_kernel(input, output, input.cols, input.rows);
    auto end_cpu =  chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("bgr_to_blur_kernel elapsed %f ms\n", duration_ms.count());
  }

int main(int argc, char **argv)
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
