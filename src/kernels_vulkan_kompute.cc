
//#undef KOMPUTE_LOG_OVERRIDE 1

#include "kompute/Core.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/Sequence.hpp"
#include "kompute/Manager.hpp"
#include "kompute/operations/OpAlgoDispatch.hpp"
#include "kompute/operations/OpTensorSyncDevice.hpp"
#include "kompute/operations/OpTensorSyncLocal.hpp"
#include <fstream>

#include "timer.hpp"

static std::vector<uint32_t> blur_src;

static const std::string blur_shader(R"(
#version 450

// The execution structure
layout (local_size_x = 8, local_size_y = 8) in;
//layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// The buffers are provided via the tensors
layout(binding = 0) buffer bufA { float a[]; };
layout(binding = 1) buffer bufOut { float b[]; };

layout (constant_id = 0) const int W = 0;
layout (constant_id = 1) const int H = 0;
layout (constant_id = 2) const int C = 0;

int my_clamp(int x, int a, int b) {
	return x < a ? a : x > b ? b : x;
}

void main() {
	int y = int(gl_GlobalInvocationID.x);
	int x = int(gl_GlobalInvocationID.y);
	int c = 0;

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
		yy = my_clamp(yy, 0, H-1);
		xx = my_clamp(xx, 0, W-1);
		val += a[yy*W*C+xx*C+c] * K[k++];
	}

	// Incorrect and still not faster :|
	/*
	int loy = y <= 2   ? 0   : y-2;
	int hiy = y >= H-3 ? H-1 : y+3;
	int lox = x <= 2   ? 0   : x-2;
	int hix = x >= W-3 ? W-1 : x+3;
	for (int yy=loy; yy<hiy; yy++)
	for (int xx=lox; xx<hix; xx++) {
		val += a[yy*W*C+xx*C+c] * K[k++];
	}
	*/

	b[y*512*C+x*C+c] = val;
}
)");

static std::vector<uint32_t>
compileSource(
		const std::string& source)
{
	if (system(std::string("glslangValidator --stdin -S comp -V -o tmp_kp_shader.comp.spv << END\n" + source + "\nEND").c_str()))
		throw std::runtime_error("Error running glslangValidator command");
	std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
	std::vector<char> buffer;
	buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
	return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}


void run_vlkn_1(Timer& t, int N, int W, int H, int C, float* outHost, const float* inHost) {
	kp::Manager mgr;
	if (blur_src.empty())
		blur_src = compileSource(blur_shader);

	std::vector<float> in_vec(inHost, inHost+H*W*C);
	std::vector<float> zeros2; zeros2.resize(W*H*C, 0);
	/*
	auto in_  = mgr.tensor(zeros1);
	auto out_ = mgr.tensor(zeros2);
	auto in = mgr.tensor(in_->rawData(), H*W*C, 4, kp::Tensor::TensorDataTypes::eFloat);
	auto out = mgr.tensor(out_->rawData(), H*W*C, 4, kp::Tensor::TensorDataTypes::eFloat);
	*/
	auto in  = mgr.tensor(in_vec);
	auto out = mgr.tensor(zeros2);

	std::vector<std::shared_ptr<kp::Tensor>> params = { in, out };
	//std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, blur_src, kp::Workgroup{(uint32_t)H,(uint32_t)W,(uint32_t)C});
	//std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, blur_src, kp::Workgroup{(uint32_t)((H+16-1)/16),(uint32_t)((W+16-1)/16),(uint32_t)C});
	std::vector<int> sizes = {H,W,C};
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<int>(params, blur_src, kp::Workgroup{(uint32_t)((H+8-1)/8),(uint32_t)((W+8-1)/8),(uint32_t)C}, sizes, {});


	{
		//TimerMeasurement<> tm(t, N);

		//mgr.sequence()->record<kp::OpTensorSyncDevice>(params)->eval();
		mgr.sequence()->eval<kp::OpTensorSyncDevice>(params);

		auto runAlgo = mgr.sequence()->record<kp::OpAlgoDispatch>(algo);

		{
		TimerMeasurement<> tm(t, N);
		for (int i=0; i<N; i++) {
			runAlgo->eval();
			//mgr.sequence()->eval<kp::OpAlgoDispatch>(algo);
			//
		}
		}

		//mgr.sequence()->record<kp::OpTensorSyncLocal>(params)->eval();
		mgr.sequence()->eval<kp::OpTensorSyncLocal>(params);

		memcpy(outHost, out->data(), 4*H*W*C);
	}

}

// Called after every iter to ensure compiler doesn't optimize loop away.
void funcToPreventOptimization(float*);

static inline int my_clamp(int x, int a, int b) { return x < a ? a : x > b ? b : x; }
void run_cpu__1(Timer& t, int N, int W, int H, int C, float *outHost, const float* inHost) {

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
			#pragma omp parallel for schedule(static, 4)
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
