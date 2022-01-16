cmake_minimum_required(VERSION 3.18)
project(benchmark VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_CXX_FLAGS " -g -O3 ")

find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)
3find_package(Fmt REQUIRED)

option(USE_VALIDATION OFF)
if (USE_VALIDATION)
	set(VALIDATION_LIBS VkLayer_utils VkLayer_khronos_validation)
	add_definitions(-DUSE_VALIDATION)
else()
	set(VALIDATION_LIBS "")
endif()

add_executable(runBench
    src/main.cc

	src/kernels.cu
	src/kernels_vulkan.cc
	src/kernels_cpu.cc

	src/utils/utils1.cc
	)

message(STATUS "validation libs ${VALIDATION_LIBS}")

target_link_libraries(runBench
	${VALIDATION_LIBS}
    Vulkan::Vulkan
	#kompute::kompute
	glfw
	#fmt::fmt
	opencv_imgcodecs
	opencv_core
	)

include_directories(
        ../../single_include/
		${OpenCV_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIR}
		${OpenCV_INCLUDE_DIRECTORY}
		${OpenCV_INCLUDE_DIRECTORIES}
		)
