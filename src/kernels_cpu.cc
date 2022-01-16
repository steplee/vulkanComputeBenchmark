#include "timer.hpp"

#include <cassert>
#include <cmath>
#include <unistd.h>
#include <cstdlib>
#include <cstring>


// Called after every iter to ensure compiler doesn't optimize loop away.
void funcToPreventOptimization(float*);

static inline int my_clamp(int x, int a, int b) { return x < a ? a : x > b ? b : x; }

void run_cpu4_1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost) {

	float *out = (float*) aligned_alloc(16, 4 * H * W * C);
	float *in  = (float*) aligned_alloc(16, 4 * H * W * C);

	memcpy(in, inHost, 4*H*W*C);

	const float K[25] = {
		0.0124776415432326, 0.026415167354310425, 0.033917746268994874, 0.026415167354310425, 0.0124776415432326, 
		0.026415167354310425, 0.05592090972790175, 0.07180386941492664, 0.05592090972790175, 0.026415167354310425, 
		0.033917746268994874, 0.07180386941492664, 0.09219799334529334, 0.07180386941492664, 0.033917746268994874, 
		0.026415167354310425, 0.05592090972790175, 0.07180386941492664, 0.05592090972790175, 0.026415167354310425, 
		0.0124776415432326, 0.026415167354310425, 0.033917746268994874, 0.026415167354310425, 0.0124776415432326
	};

	{
		TimerMeasurement<> tm(t, N);
		for (int i=0; i<N; i++) {
			#pragma omp parallel for schedule(static, 4) num_threads(4)
			for (int y=0; y<H; y++)
			for (int x=0; x<W; x++) {
				int c = 0;


				float val = 0.f;

				int k = 0;
				for (int j=-2; j<3; j++)
				for (int i=-2; i<3; i++) {
					int yy = j+y, xx = i+x;
					yy = my_clamp(yy, 0, H-1);
					xx = my_clamp(xx, 0, W-1);
					val += in[yy*W*C+xx*C+c] * K[k++];
				}

				out[y*512+x] = val;

			}

			funcToPreventOptimization(out);
		}
	}

	memcpy(outHost, out, 4*H*W*C);

	free(out); free(in);

}

void run_cpu1_1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost) {

	float *out = (float*) aligned_alloc(16, 4 * H * W * C);
	float *in  = (float*) aligned_alloc(16, 4 * H * W * C);

	memcpy(in, inHost, 4*H*W*C);

	const float K[25] = {
		0.0124776415432326, 0.026415167354310425, 0.033917746268994874, 0.026415167354310425, 0.0124776415432326, 
		0.026415167354310425, 0.05592090972790175, 0.07180386941492664, 0.05592090972790175, 0.026415167354310425, 
		0.033917746268994874, 0.07180386941492664, 0.09219799334529334, 0.07180386941492664, 0.033917746268994874, 
		0.026415167354310425, 0.05592090972790175, 0.07180386941492664, 0.05592090972790175, 0.026415167354310425, 
		0.0124776415432326, 0.026415167354310425, 0.033917746268994874, 0.026415167354310425, 0.0124776415432326
	};

	{
		TimerMeasurement<> tm(t, N);
		for (int i=0; i<N; i++) {
			for (int y=0; y<H; y++)
			for (int x=0; x<W; x++) {
				int c = 0;


				float val = 0.f;

				int k = 0;
				for (int j=-2; j<3; j++)
				for (int i=-2; i<3; i++) {
					int yy = j+y, xx = i+x;
					yy = my_clamp(yy, 0, H-1);
					xx = my_clamp(xx, 0, W-1);
					val += in[yy*W*C+xx*C+c] * K[k++];
				}

				out[y*512+x] = val;

			}

			funcToPreventOptimization(out);
		}
	}

	memcpy(outHost, out, 4*H*W*C);

	free(out); free(in);

}
