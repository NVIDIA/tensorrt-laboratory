#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#define CHECK_CUDA(code) { CHECK_EQ((code), cudaSuccess) << cudaGetErrorString((code)); }
