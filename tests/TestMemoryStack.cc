
#include "YAIS/Memory.h"
#include "YAIS/MemoryStack.h"
#include "gtest/gtest.h"

using namespace yais;

namespace
{

static size_t one_mb = 1024*1024;

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
    auto p0 = stack->Allocate(128*1024);
    ASSERT_TRUE(p0);
    EXPECT_EQ(128*1024, stack->Allocated());
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