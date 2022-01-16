#include "timer.hpp"

//#define CU_GRP_SZ 16
#define CU_GRP_SZ 8

static __device__ int clamp(int x, int a, int b) {
	return x < a ? a : x > b ? b : x;
}

// Optimized version would
//     1) Fetch global memory to shared memory.
//     2) Use two passes.
//     3) Use device-dependent row pitch.
static __global__ void cuda_naiveGaussianBlur_5(
		float* out, const float* in,
		int W, int H, int C) {

	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int c = threadIdx.z;

	if (y>=H or x>=W) return;

	// x = np.exp(-np.linalg.norm(np.stack(np.meshgrid(*(np.linspace(-1,1,5),)*2),-1), axis=-1)**2 / 1.); x = x / x.sum()
	const float K[25] = {
		0.0124776415432326, 0.026415167354310425, 0.033917746268994874, 0.026415167354310425, 0.0124776415432326, 
		0.026415167354310425, 0.05592090972790175, 0.07180386941492664, 0.05592090972790175, 0.026415167354310425, 
		0.033917746268994874, 0.07180386941492664, 0.09219799334529334, 0.07180386941492664, 0.033917746268994874, 
		0.026415167354310425, 0.05592090972790175, 0.07180386941492664, 0.05592090972790175, 0.026415167354310425, 
		0.0124776415432326, 0.026415167354310425, 0.033917746268994874, 0.026415167354310425, 0.0124776415432326
	};


	float val = 0.f;

	int k = 0;
	for (int j=-2; j<3; j++)
	for (int i=-2; i<3; i++) {
		int yy = j+y, xx = i+x;
		yy = clamp(yy, 0, H-1);
		xx = clamp(xx, 0, W-1);
		val += in[yy*W*C+xx*C+c] * K[k++];
	}

	out[y*W*C+x*C+c] = val;
}


void run_cuda_naiveGaussianBlur_5(
		float* out, const float* in,
		int W, int H, int C) {

	dim3 blk ( (H+CU_GRP_SZ-1)/CU_GRP_SZ, (W+CU_GRP_SZ-1)/CU_GRP_SZ, 1 );
	dim3 thr ( CU_GRP_SZ, CU_GRP_SZ, C );
	cuda_naiveGaussianBlur_5<<<blk,thr>>>(out,in,W,H,C);
}


void run_cuda_1(Timer& t, int N,
		int W, int H, int C, float* outHost, const float* inHost) {
	float *in, *out;
	cudaMalloc(&in, sizeof(float)*W*H*C);
	cudaMalloc(&out, sizeof(float)*W*H*C);

	cudaMemcpy(in, inHost, 4*H*W*C, cudaMemcpyHostToDevice);

	{
		TimerMeasurement<> tm(t,N);
		for (int i=0; i<N; i++) {
			run_cuda_naiveGaussianBlur_5(out,in,W,H,C);
			cudaDeviceSynchronize();
		}
		cudaDeviceSynchronize();
		cudaMemcpy(outHost, out, 4*H*W*C, cudaMemcpyDeviceToHost);
	}


	cudaFree(in);
	cudaFree(out);
}
