/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "YAIS/Memory.h"
#include "YAIS/MemoryStack.h"
#include "gtest/gtest.h"

using namespace yais;

namespace
{

static size_t one_mb = 1024 * 1024;

class TestMemoryStack : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        stack = MemoryStack<CudaDeviceAllocator>::make_shared(one_mb);
    }

    virtual void TearDown()
    {
        stack->Reset();
    }

    std::shared_ptr<MemoryStack<CudaDeviceAllocator>> stack;
};

TEST_F(TestMemoryStack, EmptyOnCreate)
{
    ASSERT_EQ(one_mb, stack->Size());
    ASSERT_EQ(one_mb, stack->Available());
    ASSERT_EQ(0, stack->Allocated());
}

TEST_F(TestMemoryStack, AllocateAndReset)
{
    auto p0 = stack->Allocate(128 * 1024);
    ASSERT_TRUE(p0);
    EXPECT_EQ(128 * 1024, stack->Allocated());
    stack->Reset();
    EXPECT_EQ(0, stack->Allocated());
    auto p1 = stack->Allocate(1);
    EXPECT_EQ(p0, p1);
}

TEST_F(TestMemoryStack, Unaligned)
{
    auto p0 = stack->Allocate(1);
    ASSERT_TRUE(p0);
    EXPECT_EQ(stack->Alignment(), stack->Allocated());

    auto p1 = stack->Allocate(1);
    ASSERT_TRUE(p1);
    EXPECT_EQ(2 * stack->Alignment(), stack->Allocated());

    auto len = (char *)p1 - (char *)p0;
    EXPECT_EQ(len, stack->Alignment());
}

} // namespace