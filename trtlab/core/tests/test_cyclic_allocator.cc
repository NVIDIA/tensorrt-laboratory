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
#include "trtlab/core/memory/cyclic_allocator.h"
#include "trtlab/core/memory/malloc.h"
#include "trtlab/core/memory/sysv_allocator.h"
#include "gtest/gtest.h"

using namespace trtlab;

namespace {

static size_t one_mb = 1024 * 1024;

template<typename T>
class TestCyclicStacks : public ::testing::Test
{
};

using MemoryTypes = ::testing::Types<Malloc, SystemV>;

TYPED_TEST_CASE(TestCyclicStacks, MemoryTypes);

TYPED_TEST(TestCyclicStacks, EmptyOnCreate)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    EXPECT_EQ(5, stack->AvailableSegments());
    EXPECT_EQ(5 * one_mb, stack->AvailableBytes());
}

TYPED_TEST(TestCyclicStacks, AddSegment)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    stack->AddSegment();
    EXPECT_EQ(6, stack->AvailableSegments());
}

TYPED_TEST(TestCyclicStacks, DropSegment)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    stack->DropSegment();
    EXPECT_EQ(4, stack->AvailableSegments());
}

TYPED_TEST(TestCyclicStacks, Allocate)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    {
        auto seg_0_0 = stack->Allocate(1);
        EXPECT_EQ(5 * one_mb - stack->Alignment(), stack->AvailableBytes());
    }
    // even though seg_0_0 is released, the current segment does not get reset/recycled
    // because the stack still owns reference to seg_0
    // a segment is only recycled after it becomes detached and its reference count goes to 0
    EXPECT_EQ(5 * one_mb - stack->Alignment(), stack->AvailableBytes());

    {
        auto seg_0_1 = stack->Allocate(1024);
        EXPECT_EQ(5, stack->AvailableSegments()); // seg_0 is still active
        auto seg_1_0 = stack->Allocate(one_mb);
        EXPECT_EQ(3,
                  stack->AvailableSegments()); // seg_0 is detached; seg_1 detaches if capacity is 0
        auto seg_2_0 = stack->Allocate(1024);
        EXPECT_EQ(3, stack->AvailableSegments()); // seg_2 is now active; 0 and 1 are detached
        seg_1_0.reset(); // we can release seg_0/1 in any order
        EXPECT_EQ(4, stack->AvailableSegments());
        seg_0_1.reset(); // we can release seg_0/1 in any order
        EXPECT_EQ(5, stack->AvailableSegments());
    }
    // seg_0 has had 2 allocation, then failed to have capacity for the 3rd allocation
    // seg_1 is completely used from the 3rd allocation, but is still the active segment
    // until seg_2 is allocated
    EXPECT_EQ(5, stack->AvailableSegments());

    {
        // everything has been released, so we can grab 5 x one_mb buffers
        // but we will OOM on our 6th
        auto b0 = stack->Allocate(one_mb);
        auto b1 = stack->Allocate(one_mb);
        auto b2 = stack->Allocate(one_mb);
        auto b3 = stack->Allocate(one_mb);
        auto b4 = stack->Allocate(one_mb);
        EXPECT_EQ(0, stack->AvailableSegments());

        // The following will hang and deadlock the test
        // stack->Allocate(one_mb);
    }
}

TYPED_TEST(TestCyclicStacks, AllocateThenReleaseStack)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    auto buf = stack->Allocate(1024);
    stack.reset();
    EXPECT_EQ(buf->Size(), 1024);
    DLOG(INFO) << "Deallocation should hppen after this statement";
}

/*
TYPED_TEST(TestCyclicStacks, CastToMemoryType)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    std::unique_ptr<TypeParam> buf = std::move(stack->Allocate(1024));
    EXPECT_EQ(buf->Size(), 1024);
    stack.reset();
    EXPECT_EQ(buf->Size(), 1024);
    DLOG(INFO) << "Deallocation should hppen after this statement";
}
*/

TYPED_TEST(TestCyclicStacks, AllocateShouldFail)
{
    auto stack = std::make_unique<CyclicAllocator<TypeParam>>(5, one_mb);
    EXPECT_DEATH(stack->Allocate(one_mb + 1), "");
}

} // namespace
