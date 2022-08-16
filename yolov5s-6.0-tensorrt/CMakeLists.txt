cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-8.0.1.6/include)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-8.0.1.6/lib)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(yolov5 yolov5.cpp yololayer.cu)

target_link_libraries(yolov5 nvinfer cudart ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

