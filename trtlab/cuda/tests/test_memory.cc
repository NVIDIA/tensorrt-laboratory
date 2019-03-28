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
#include <list>
#include <memory>

#include "trtlab/core/memory/allocator.h"
#include "trtlab/cuda/memory/cuda_device.h"
#include "trtlab/cuda/memory/cuda_managed.h"
#include "trtlab/cuda/memory/cuda_pinned_host.h"

#include "gtest/gtest.h"

using namespace trtlab;
using namespace trtlab;

namespace {

static size_t one_mb = 1024 * 1024;

template<typename T>
class TestMemory : public ::testing::Test
{
};

using MemoryTypes = ::testing::Types<CudaDeviceMemory, CudaManagedMemory, CudaPinnedHostMemory>;

TYPED_TEST_CASE(TestMemory, MemoryTypes);

TYPED_TEST(TestMemory, make_shared)
{
    auto shared = std::make_shared<Allocator<TypeParam>>(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());

    if(std::dynamic_pointer_cast<DeviceMemory>(shared))
    {
        EXPECT_EQ(shared->DeviceInfo().device_type, kDLGPU);
    }
    else
    {
        EXPECT_EQ(shared->DeviceInfo().device_type, kDLCPUPinned);
    }

    shared.reset();
    EXPECT_FALSE(shared);
}

TYPED_TEST(TestMemory, make_unique)
{
    auto unique = std::make_unique<Allocator<TypeParam>>(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    unique.reset();
    EXPECT_FALSE(unique);
}

TYPED_TEST(TestMemory, ctor)
{
    Allocator<TypeParam> memory(one_mb);
    EXPECT_TRUE(memory.Data());
    EXPECT_EQ(one_mb, memory.Size());
}

/*
TYPED_TEST(TestMemory, move_ctor)
{
    Allocator<TypeParam> memory(one_mb);
    Allocator<TypeParam> host(std::move(memory));

    EXPECT_TRUE(host.Data());
    EXPECT_EQ(one_mb, host.Size());

    EXPECT_FALSE(memory.Data());
    EXPECT_EQ(0, memory.Size());
}

TYPED_TEST(TestMemory, move_to_shared_ptr)
{
    Allocator<TypeParam> memory(one_mb);
    auto ptr = std::make_shared<Allocator<TypeParam>>(std::move(memory));
    EXPECT_TRUE(ptr);
    EXPECT_TRUE(ptr->Data());
}
*/

} // namespace
