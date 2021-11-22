cmake_minimum_required(VERSION 3.0)

project(plugin_build_example)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda/tensorrt
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# # tensorrt
# set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-7.2.2/TensorRT-7.2.2.3/include/)
# set(TENSORRT_LIBRARY_INFER /usr/local/TensorRT-7.2.2/TensorRT-7.2.2.3/lib/libnvinfer.so)
# set(TENSORRT_LIBRARY_MYELIN /usr/local/TensorRT-7.2.2/TensorRT-7.2.2.3/lib/libmyelin.so)
# set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-7.2.2/TensorRT-7.2.2.3/lib/)

include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolo SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu
                                                    )
target_link_libraries(yolo nvinfer cudart)

add_definitions(-O2 -pthread)

