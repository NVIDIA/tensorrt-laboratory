/* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "tensorrt/laboratory/core/utils.h"

#include <NvInfer.h>

#include <glog/logging.h>

#include <numeric>

namespace trtlab {
namespace TensorRT {

std::size_t SizeofDataType(::nvinfer1::DataType dtype)
{
    switch(dtype)
    {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            LOG(FATAL) << "Unknown TensorRT DataType";
    }
}

static int RoundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

std::size_t ElementsofVolume(const ::nvinfer1::Dims& d, const ::nvinfer1::TensorFormat& format)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    switch(format)
    {
        case nvinfer1::TensorFormat::kCHW2: spv = 2; break;
        case nvinfer1::TensorFormat::kCHW4: spv = 4; break;
        case nvinfer1::TensorFormat::kHWC8: spv = 8; break;
        case nvinfer1::TensorFormat::kCHW16: spv = 16; break;
        case nvinfer1::TensorFormat::kCHW32: spv = 32; break;
        case nvinfer1::TensorFormat::kLINEAR:
        default: spv = 1; break;
    }
    if (spv > 1)
    {
        if (d.nbDims < 3) LOG(FATAL) << "Vectorized format only makes sense when nbDims>=3";
        d_new.d[d_new.nbDims - 3] = RoundUp(d_new.d[d_new.nbDims - 3], spv);
    }
    return std::accumulate(d_new.d, d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

} // namespace TensorRT
} // namespace trtlab
