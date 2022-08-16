cmake_minimum_required(VERSION 3.10)

project(plugin_build_example)

add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-8.0.1.6/include)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-8.0.1.6/lib)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

cuda_add_library(yolo SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)

target_link_libraries(yolo nvinfer cudart)

add_definitions(-O2 -pthread)

