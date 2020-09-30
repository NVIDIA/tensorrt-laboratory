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
#include "trtlab/core/memory/memory_stack.h"
#include "trtlab/core/memory/smart_stack.h"
#include "trtlab/core/memory/sysv_allocator.h"

#include "gtest/gtest.h"

using namespace trtlab;
using namespace trtlab;

namespace {

static size_t one_mb = 1024 * 1024;

// template <typename T>
class TestMemoryStack : public ::testing::Test
{
  protected:
    virtual void SetUp() { stack = std::make_shared<MemoryStack<Malloc>>(one_mb); }

    virtual void TearDown() { stack->Reset(); }

    std::shared_ptr<MemoryStack<Malloc>> stack;
};

class TestSmartStack : public ::testing::Test
{
  protected:
    virtual void SetUp() { stack = SmartStack<SystemV>::Create(one_mb); }

    virtual void TearDown()
    {
        if(stack) stack->Reset();
    }

    std::shared_ptr<SmartStack<SystemV>> stack;
};

// using MemoryTypes = ::testing::Types<Malloc>;
// TYPED_TEST_CASE(TestMemoryStack, MemoryTypes);

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

    auto len = (char*)p1 - (char*)p0;
    EXPECT_EQ(len, stack->Alignment());

    EXPECT_EQ(stack->Offset(p0), 0);
    EXPECT_EQ(stack->Offset(p1), stack->Alignment());
}

TEST_F(TestSmartStack, EmptyOnCreate)
{
    ASSERT_EQ(one_mb, stack->Size());
    ASSERT_EQ(one_mb, stack->Available());
    ASSERT_EQ(0, stack->Allocated());
}

TEST_F(TestSmartStack, AllocateAndReset)
{
    auto p0 = stack->Allocate(128 * 1024);
    ASSERT_TRUE(p0);
    EXPECT_EQ(128 * 1024, stack->Allocated());
    EXPECT_EQ(p0->DataType(), types::bytes);
    stack->Reset();
    EXPECT_EQ(0, stack->Allocated());
    auto p1 = stack->Allocate(1);
    EXPECT_EQ(p0->Data(), p1->Data());
}

TEST_F(TestSmartStack, Unaligned)
{
    auto p0 = stack->Allocate(1);
    ASSERT_TRUE(p0->Data());
    EXPECT_EQ(stack->Alignment(), stack->Allocated());

    auto p1 = stack->Allocate(1);
    ASSERT_TRUE(p1);
    EXPECT_EQ(2 * stack->Alignment(), stack->Allocated());

    auto len = (char*)p1->Data() - (char*)p0->Data();
    EXPECT_EQ(len, stack->Alignment());

    EXPECT_EQ(stack->Offset(p0->Data()), 0);
    EXPECT_EQ(stack->Offset(p1->Data()), stack->Alignment());

    EXPECT_EQ(p0->Offset(), 0);
    EXPECT_EQ(p1->Offset(), stack->Alignment());

    EXPECT_GE(p0->Stack().Memory().ShmID(), 0);
    // EXPECT_EQ(p0->ShmID(), -1);

    DLOG(INFO) << "delete descriptors";

    p0.reset();
    p1.reset();

    DLOG(INFO) << "delete stack";

    stack.reset();
}

TEST_F(TestSmartStack, PassMemory)
{
    auto memory = std::make_unique<Allocator<Malloc>>(one_mb);
    auto s = SmartStack<Malloc>::Create(std::move(memory));
}

TEST_F(TestSmartStack, PassSpecializedMemory)
{
    auto memory = std::make_unique<Allocator<Malloc>>(one_mb);
    memory->Reshape({512, 512}, types::fp32);
    EXPECT_EQ(memory->DataType(), types::fp32);

    auto s = SmartStack<Malloc>::Create(std::move(memory));
    // MemoryStack will Reshape any CoreMemory object to bytes at full capacity
    // This is ok since the MemoryStack is taking ownership of the object.
    EXPECT_EQ(s->Memory().DataType(), types::bytes);

    // This no longer fails because MemoryStack is converting our memory object
    // back into a useable form
    // EXPECT_THROW(auto p0 = s->Allocate(1), std::invalid_argument);
}

} // namespace