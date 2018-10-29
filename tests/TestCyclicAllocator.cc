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
#include "YAIS/CyclicStacks.h"
#include "gtest/gtest.h"

using namespace yais;

namespace
{

static size_t one_mb = 1024 * 1024;

class TestCyclicStacks : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        stack = std::make_shared<CyclicStacks<CudaHostAllocator>>(5, one_mb);
    }

    virtual void TearDown()
    {
        stack.reset();
    }

    std::shared_ptr<CyclicStacks<CudaHostAllocator>> stack;
};

TEST_F(TestCyclicStacks, EmptyOnCreate)
{
    EXPECT_EQ(5,stack->AvailableSegments());
    EXPECT_EQ(5*one_mb,stack->AvailableBytes());
}

TEST_F(TestCyclicStacks, AllocateBuffer)
{
    {
        auto seg_0_0 = stack->AllocateBuffer(1);
        EXPECT_EQ(5*one_mb - stack->Alignment(),stack->AvailableBytes());
    }
    // even though seg_0_0 is released, the current segment does not get reset/recycled
    // because the stack still owns reference to seg_0
    // a segment is only recycled after it becomes detached and its reference count goes to 0
    EXPECT_EQ(5*one_mb - stack->Alignment(), stack->AvailableBytes());
    
    {
        auto seg_0_1 = stack->AllocateBuffer(1024);
        EXPECT_EQ(5,stack->AvailableSegments()); // seg_0 is still active
        auto seg_1_0 = stack->AllocateBuffer(one_mb);
        EXPECT_EQ(4,stack->AvailableSegments()); // seg_0 is detached; seg_1 active, but 0 free space
        auto seg_2_0 = stack->AllocateBuffer(1024);
        EXPECT_EQ(3,stack->AvailableSegments()); // seg_2 is now active; 0 and 1 are detached
        seg_1_0.reset(); // we can release seg_0/1 in any order
        EXPECT_EQ(4,stack->AvailableSegments());
        seg_0_1.reset(); // we can release seg_0/1 in any order
        EXPECT_EQ(5,stack->AvailableSegments());
    }
    // seg_0 has had 2 allocation, then failed to have capacity for the 3rd allocation
    // seg_1 is completely used from the 3rd allocation, but is still the active segment
    // until seg_2 is allocated
    EXPECT_EQ(5,stack->AvailableSegments());

    {
        // everything has been released, so we can grab 5 x one_mb buffers
        // but we will OOM on our 6th
        auto b0 = stack->AllocateBuffer(one_mb);
        auto b1 = stack->AllocateBuffer(one_mb);
        auto b2 = stack->AllocateBuffer(one_mb);
        auto b3 = stack->AllocateBuffer(one_mb);
        auto b4 = stack->AllocateBuffer(one_mb);
        EXPECT_DEATH(stack->AllocateBuffer(one_mb), "");
    }

    EXPECT_DEATH(stack->AllocateBuffer(one_mb+1), "");
}

} // namespace
