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
#include "trtlab/tensorrt/buffers.h"
#include "trtlab/tensorrt/inference_manager.h"

#include "trtlab/core/memory/malloc.h"
#include "trtlab/cuda/memory/cuda_device.h"
#include "trtlab/cuda/memory/cuda_managed.h"
#include "trtlab/cuda/memory/cuda_pinned_host.h"
#include "gtest/gtest.h"

#include <list>

using namespace trtlab;
using namespace trtlab;
using namespace trtlab::TensorRT;

namespace {

static size_t one_mb = 1024 * 1024;

/*
template<typename T>
class TestBuffers : public ::testing::Test
{
};

using MemoryTypes =
    ::testing::Types<Malloc, CudaDeviceMemory, CudaManagedMemory, CudaPinnedHostMemory>;

TYPED_TEST_CASE(TestBuffers, MemoryTypes);

TYPED_TEST(TestBuffers, make_shared)
{
    auto buffers = std::make_shared<FixedBuffers>(one_mb, one_mb);
}
*/

class TestCyclicBuffers : public ::testing::Test
{
};

TEST_F(TestCyclicBuffers, CyclicBuffers)
{
    auto host = std::make_unique<CyclicAllocator<CudaPinnedHostMemory>>(5, 1024 * 1024);
    auto device = std::make_unique<CyclicAllocator<CudaDeviceMemory>>(5, 1024 * 1024);

    auto buffers = std::make_shared<CyclicBuffers<CudaPinnedHostMemory, CudaDeviceMemory>>(
        std::move(host), std::move(device));

    auto b0 = buffers->AllocateHost(1024);
    auto b1 = buffers->AllocateDevice(1024);
}

} // namespace
