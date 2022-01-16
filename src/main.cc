#include "timer.hpp"
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>

void funcToPreventOptimization(float*) {}

void run_cuda_1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost);
void run_vlkn_1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost);
void run_cpu1_1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost); // single thread
void run_cpu4_1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost); // 4 threads

constexpr int W = 512;
constexpr int H = 512;
constexpr int C = 1;
constexpr int N = 10000;


int main(int argc, char** argv) {

#ifdef USE_VALIDATION
	setenv("VK_INSTANCE_LAYERS", "VK_LAYER_LUNARG_api_dump:VK_LAYER_KHRONOS_validation", true);
	printf(" - Using Vulkan validation layers.\n");
#else
	printf(" - Not using Vulkan validation layers.\n");
#endif


	if (argc < 2) { printf(" - Must provide path to image.\n"); return 1; }

	/*
	double t = 1;
	for (int i=0; i<10; i++) {
		std::cout << " - " << prettyPrintSeconds(t) << "\n";
		t *= .1;
	}
	*/

	float* outVlkn = (float*) malloc(4*H*W*C);
	float* outCpu1 = (float*) malloc(4*H*W*C);
	float* outCpu4 = (float*) malloc(4*H*W*C);
	float* outCuda = (float*) malloc(4*H*W*C);

	float* in = (float*) malloc(4*H*W*C);
	{
		cv::Mat inMat { H, W, CV_32F, in };
		cv::Mat tmp = cv::imread(argv[1], 0);
		if (tmp.rows != 512 or tmp.cols != 512)
			cv::resize(tmp,tmp, {512,512});
		tmp.convertTo(inMat, CV_32F);
	}

	printf(" - Running GaussianBlur Test\n");
	{
		Timer cudaTimer { "cudaBlur" };
		Timer vlknTimer { "vlknBlur" };
		Timer cpu1Timer { "cpu1Blur" };
		Timer cpu4Timer { "cpu4Blur" };
		for (int i=0; i<1; i++) {
		//for (int i=0; i<10; i++) {
			run_cpu1_1(cpu1Timer, N, W,H,C, outCpu1, in);
			run_cpu4_1(cpu4Timer, N, W,H,C, outCpu4, in);
			run_vlkn_1(vlknTimer, N, W,H,C, outVlkn, in);
			run_cuda_1(cudaTimer, N, W,H,C, outCuda, in);
		}
	}

	cv::Mat matInpt { H, W, CV_32FC1, in      }; cv::imwrite("outInpt.png", matInpt);
	cv::Mat matVlkn { H, W, CV_32FC1, outVlkn }; cv::imwrite("outVlkn.png", matVlkn);
	cv::Mat matCuda { H, W, CV_32FC1, outCuda }; cv::imwrite("outCuda.png", matCuda);
	cv::Mat matCpu1 { H, W, CV_32FC1, outCpu1 }; cv::imwrite("outCpu1.png", matCpu1);
	cv::Mat matCpu4 { H, W, CV_32FC1, outCpu4 }; cv::imwrite("outCpu4.png", matCpu4);

	free(outVlkn);
	free(outCuda);
	free(outCpu1);
	free(outCpu4);

	return 0;
}
