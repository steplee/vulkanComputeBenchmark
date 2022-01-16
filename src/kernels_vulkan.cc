#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include "utils/utils2.hpp"

#include "timer.hpp"

#include <vector>
#include <cassert>
#include <cmath>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <fstream>

/*
 *
 * My god, I've never had to write so much boilerplate!
 *
 */

// Contains SPIR-V bitcode, only compiled once.
static std::vector<uint32_t> blur_spv;

static const std::string blur_shader(R"(
#version 450

// The execution structure
layout (local_size_x = 8, local_size_y = 8) in;
//layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// The buffers are provided via the tensors
layout(binding = 0) buffer bufA { float a[]; };
layout(binding = 1) buffer bufOut { float b[]; };

//layout (constant_id = 0) const int W = 0;
//layout (constant_id = 1) const int H = 0;
//layout (constant_id = 2) const int C = 0;

int my_clamp(int x, int a, int b) {
	return x < a ? a : x > b ? b : x;
}

void main() {
	int W=512;
	int H=512;
	int C=1;
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

//using namespace vk = vk::raii;

void run_vlkn_1(Timer& t, int N, int W, int H, int C, float* outHost, const float* inHost) {
	vk::raii::Context ctx;
	//vk::raii::Instance instance();
	vk::raii::Instance instance = vk::raii::su::makeInstance(ctx, "Bench", "Mark");

	auto pds = vk::raii::PhysicalDevices( instance );
	for (auto& pd : pds) {
		auto props = pd.getProperties();
		char* name = props.deviceName;
		fmt::print(" - Found device: {}\n", name);
		fmt::print("       - has {} queues\n", pd.getQueueFamilyProperties().size());
	}

	vk::raii::PhysicalDevice physicalDevice = std::move(pds.front());

    uint32_t graphicsQueueFamilyIndex = vk::su::findGraphicsQueueFamilyIndex( physicalDevice.getQueueFamilyProperties() );

	float                     queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo( {}, graphicsQueueFamilyIndex, 1, &queuePriority );
    vk::DeviceCreateInfo      deviceCreateInfo( {}, deviceQueueCreateInfo );

	// Make device and q
    vk::raii::Device device( physicalDevice, deviceCreateInfo );
	vk::raii::Queue graphicsQueue( device, graphicsQueueFamilyIndex, 0 );
    vk::raii::CommandPool commandPool = vk::raii::CommandPool(
      device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueFamilyIndex } );

    vk::raii::CommandBuffer commandBuffer_copy = vk::raii::su::makeCommandBuffer( device, commandPool );
    vk::raii::CommandBuffer commandBuffer_blur = vk::raii::su::makeCommandBuffer( device, commandPool );

	// Compile and load spv
	if (blur_spv.empty())
		blur_spv = compileSource(blur_shader);
	vk::ShaderModuleCreateInfo shaderCreateInfo( {}, blur_spv );
    vk::raii::ShaderModule     blurShader( device, shaderCreateInfo );

	// Create buffers
	auto bufSize = 4*W*H*C;
	vk::BufferCreateInfo bufferInfoHost({},
						 bufSize,
						 vk::BufferUsageFlagBits::eTransferSrc |
						 vk::BufferUsageFlagBits::eTransferDst |
						 vk::BufferUsageFlagBits::eUniformBuffer |
						 vk::BufferUsageFlagBits::eStorageBuffer);
	vk::BufferCreateInfo bufferInfo12({},
						 bufSize,
						 vk::BufferUsageFlagBits::eTransferSrc |
						 vk::BufferUsageFlagBits::eTransferDst |
						 vk::BufferUsageFlagBits::eUniformBuffer |
						 vk::BufferUsageFlagBits::eStorageBuffer
						 );
	vk::raii::Buffer bufferHost(device, bufferInfoHost);
	vk::raii::Buffer buffer1(device, bufferInfo12);
	vk::raii::Buffer buffer2(device, bufferInfo12);

    vk::PhysicalDeviceMemoryProperties memoryProperties   = physicalDevice.getMemoryProperties();
    vk::MemoryRequirements             memoryRequirements = bufferHost.getMemoryRequirements();
    uint32_t                           memoryTypeIndexHost    = vk::su::findMemoryType(
      memoryProperties, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible );
    uint32_t                           memoryTypeIndex12    = vk::su::findMemoryType(
      memoryProperties, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal );

    vk::MemoryAllocateInfo memoryAllocateInfoHost( memoryRequirements.size, memoryTypeIndexHost );
    vk::MemoryAllocateInfo memoryAllocateInfo12( memoryRequirements.size, memoryTypeIndex12 );
    vk::raii::DeviceMemory deviceMemory_host( device, memoryAllocateInfoHost );
    vk::raii::DeviceMemory deviceMemory_1( device, memoryAllocateInfo12 );
    vk::raii::DeviceMemory deviceMemory_2( device, memoryAllocateInfo12 );
    bufferHost.bindMemory( *deviceMemory_host, 0 );
    buffer1.bindMemory( *deviceMemory_1, 0 );
    buffer2.bindMemory( *deviceMemory_2, 0 );

	// Map host buffer, copy, unmap
	void* dstBuf = deviceMemory_host.mapMemory(0, 4*H*W*C);
	memcpy(dstBuf, inHost, 4*H*W*C);
	deviceMemory_host.unmapMemory();

	// Send command to init buffer
	vk::BufferCopy region { 0, 0, (vk::DeviceSize) 4*H*W*C };
	commandBuffer_copy.begin( vk::CommandBufferBeginInfo() );
	commandBuffer_copy.copyBuffer(*bufferHost, *buffer1, {1, &region});
	commandBuffer_copy.end();

	vk::raii::Semaphore sema( device, vk::SemaphoreCreateInfo() );
    vk::raii::Fence        commandFence( device, vk::FenceCreateInfo() );
    vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eComputeShader );
    vk::SubmitInfo         submitInfo( {}, {}, *commandBuffer_copy, *sema );

	{
		//TimerMeasurement<> tm(t, N);
		graphicsQueue.submit( submitInfo, *commandFence );
		while ( device.waitForFences( { *commandFence }, true, vk::su::FenceTimeout ) == vk::Result::eTimeout ) {
			printf(" - sleep.\n");
			sched_yield();
			//usleep(200'000);
		}
	}

	// Send command to run algo

	// I copied all necessary structs, enums, and functions here to have a quicker reference.
	// These come from vulkna_raii.hpp, vulkan_structs.hpp, and vulkan_enums.hpp
	// {{{
#if 0
			VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateInfo(
			VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateFlags flags_  = {},
			VULKAN_HPP_NAMESPACE::ShaderStageFlagBits            stage_  = VULKAN_HPP_NAMESPACE::ShaderStageFlagBits::eVertex,
			VULKAN_HPP_NAMESPACE::ShaderModule                   module_ = {},
			const char *                                         pName_  = {},
			const VULKAN_HPP_NAMESPACE::SpecializationInfo *     pSpecializationInfo_ = {} ) VULKAN_HPP_NOEXCEPT

    VULKAN_HPP_CONSTEXPR ComputePipelineCreateInfo( VULKAN_HPP_NAMESPACE::PipelineCreateFlags           flags_  = {},
                                                    VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo stage_  = {},
                                                    VULKAN_HPP_NAMESPACE::PipelineLayout                layout_ = {},
                                                    VULKAN_HPP_NAMESPACE::Pipeline basePipelineHandle_          = {},
                                                    int32_t basePipelineIndex_ = {} ) VULKAN_HPP_NOEXCEPT


		Pipeline(
			VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
			VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
																							pipelineCache,
			VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo const &                         createInfo,
			VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )

		Device::
      VULKAN_HPP_NODISCARD VULKAN_HPP_RAII_NAMESPACE::Pipeline createComputePipeline(
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                        pipelineCache,
        VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo const &                         createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr ) const;

			CommandBuffer::
      void bindPipeline( VULKAN_HPP_NAMESPACE::PipelineBindPoint pipelineBindPoint,
                         VULKAN_HPP_NAMESPACE::Pipeline          pipeline ) const VULKAN_HPP_NOEXCEPT;

			CommandBuffer::
      void dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT;

		PipelineBindPoint::eCompute;

		// For binding uniform buffers:
				Device::
			VULKAN_HPP_NODISCARD VULKAN_HPP_RAII_NAMESPACE::DescriptorSetLayout createDescriptorSetLayout(
				VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo const &                     createInfo,
				VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr ) const;

			VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateInfo(
			VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateFlags     flags_        = {},
			uint32_t                                                 bindingCount_ = {},
			const VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding * pBindings_    = {} ) VULKAN_HPP_NOEXCEPT

			VULKAN_HPP_CONSTEXPR DescriptorSetLayoutBinding(
			uint32_t                               binding_            = {},
			VULKAN_HPP_NAMESPACE::DescriptorType   descriptorType_     = VULKAN_HPP_NAMESPACE::DescriptorType::eSampler,
			uint32_t                               descriptorCount_    = {},
			VULKAN_HPP_NAMESPACE::ShaderStageFlags stageFlags_         = {},
			const VULKAN_HPP_NAMESPACE::Sampler *  pImmutableSamplers_ = {} ) VULKAN_HPP_NOEXCEPT

		DescriptorType::eUniformBuffer;
		DescriptorType::eStorageBuffer;
		ShaderStageFlagBits::eCompute;

		Device::
      VULKAN_HPP_NODISCARD VULKAN_HPP_RAII_NAMESPACE::DescriptorPool createDescriptorPool(
        VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo const &                          createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr ) const;

		DescriptorPoolCreateInfo( VULKAN_HPP_NAMESPACE::DescriptorPoolCreateFlags  flags_         = {},
									uint32_t                                         maxSets_       = {},
									uint32_t                                         poolSizeCount_ = {},
									const VULKAN_HPP_NAMESPACE::DescriptorPoolSize * pPoolSizes_ = {} ) VULKAN_HPP_NOEXCEPT

		Device::
			VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_RAII_NAMESPACE::DescriptorSet>
			Device::allocateDescriptorSets( VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo const & allocateInfo ) const
		VULKAN_HPP_CONSTEXPR DescriptorSetAllocateInfo(
		VULKAN_HPP_NAMESPACE::DescriptorPool              descriptorPool_     = {},
		uint32_t                                          descriptorSetCount_ = {},
		const VULKAN_HPP_NAMESPACE::DescriptorSetLayout * pSetLayouts_        = {} ) VULKAN_HPP_NOEXCEPT

				Device::
			void updateDescriptorSets(
				ArrayProxy<const VULKAN_HPP_NAMESPACE::WriteDescriptorSet> const & descriptorWrites,
				ArrayProxy<const VULKAN_HPP_NAMESPACE::CopyDescriptorSet> const &  descriptorCopies ) const VULKAN_HPP_NOEXCEPT;


			VULKAN_HPP_CONSTEXPR WriteDescriptorSet(
			VULKAN_HPP_NAMESPACE::DescriptorSet  dstSet_                    = {},
			uint32_t                             dstBinding_                = {},
			uint32_t                             dstArrayElement_           = {},
			uint32_t                             descriptorCount_           = {},
			VULKAN_HPP_NAMESPACE::DescriptorType descriptorType_            = VULKAN_HPP_NAMESPACE::DescriptorType::eSampler,
			const VULKAN_HPP_NAMESPACE::DescriptorImageInfo *  pImageInfo_  = {},
			const VULKAN_HPP_NAMESPACE::DescriptorBufferInfo * pBufferInfo_ = {},
			const VULKAN_HPP_NAMESPACE::BufferView *           pTexelBufferView_ = {} ) VULKAN_HPP_NOEXCEPT

			VULKAN_HPP_CONSTEXPR DescriptorBufferInfo( VULKAN_HPP_NAMESPACE::Buffer     buffer_ = {},
													VULKAN_HPP_NAMESPACE::DeviceSize offset_ = {},
													VULKAN_HPP_NAMESPACE::DeviceSize range_  = {} ) VULKAN_HPP_NOEXCEPT

      void bindDescriptorSets( VULKAN_HPP_NAMESPACE::PipelineBindPoint                       pipelineBindPoint,
                               VULKAN_HPP_NAMESPACE::PipelineLayout                          layout,
                               uint32_t                                                      firstSet,
                               ArrayProxy<const VULKAN_HPP_NAMESPACE::DescriptorSet> const & descriptorSets,
                               ArrayProxy<const uint32_t> const & dynamicOffsets ) const VULKAN_HPP_NOEXCEPT;
#endif
			// }}}


	vk::DescriptorSetLayoutBinding bindings[2] = {
		{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
		{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
	};
	vk::DescriptorSetLayoutCreateInfo layoutInfo({}, 2, bindings);
	//vk::raii::DescriptorSetLayout layout0(layoutInfo0);
	auto layout0 = device.createDescriptorSetLayout(layoutInfo);

	vk::DescriptorPoolSize uniBufSize(vk::DescriptorType::eStorageBuffer, 2);
	vk::DescriptorPoolCreateInfo dpoolInfo (vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 1, &uniBufSize);
	auto dpool = device.createDescriptorPool(dpoolInfo);

	auto dsets = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{*dpool, 1, &*layout0});

	vk::DescriptorBufferInfo bufferInfo0(*buffer1, 0, VK_WHOLE_SIZE);
	vk::DescriptorBufferInfo bufferInfo1(*buffer2, 0, VK_WHOLE_SIZE);
	vk::WriteDescriptorSet writeSets[2] = {
		{ *dsets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfo0 },
		{ *dsets[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfo1 }
	};
	device.updateDescriptorSets({2, writeSets}, nullptr);

	auto playout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{{}, 1, &*layout0, 0, nullptr});

	vk::PipelineShaderStageCreateInfo pipeShaderInfo({}, vk::ShaderStageFlagBits::eCompute, *blurShader, "main");
	vk::ComputePipelineCreateInfo computePipelineInfo({}, pipeShaderInfo, *playout);
	auto pipeline = device.createComputePipeline(nullptr, computePipelineInfo);

	commandBuffer_blur.begin( vk::CommandBufferBeginInfo() );
	commandBuffer_blur.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *playout, 0, *dsets[0], {});
	commandBuffer_blur.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
	commandBuffer_blur.dispatch(H/8,W/8,C);
	commandBuffer_blur.end();

	{
		TimerMeasurement<> tm(t, N);
		vk::SubmitInfo         submitInfo( {}, {}, *commandBuffer_blur, {} );

		for (int i=0; i<N; i++) {
			graphicsQueue.submit( submitInfo, nullptr );
			device.waitIdle();
		}
	}

	// Gather output
	{
		vk::BufferCopy region { 0, 0, (vk::DeviceSize) 4*H*W*C };
		commandBuffer_copy.reset({});
		commandBuffer_copy.begin( vk::CommandBufferBeginInfo() );
		commandBuffer_copy.copyBuffer(*buffer2, *bufferHost, {1, &region});
		commandBuffer_copy.end();

		vk::SubmitInfo         submitInfo( {}, {}, *commandBuffer_copy, {} );
		graphicsQueue.submit( submitInfo, {} );
		device.waitIdle();

		// Map host buffer, copy, unmap
		void* srcBuf = deviceMemory_host.mapMemory(0, 4*H*W*C);
		memcpy(outHost, srcBuf, 4*H*W*C);
		deviceMemory_host.unmapMemory();
	}

}


