
#include "YAIS/Memory.h"
#include "gtest/gtest.h"

using namespace yais;

namespace
{

static size_t one_mb = 1024*1024;

TEST(MemoryStack, CudaMalloc)
{
    auto stack = MemoryStack<CudaDeviceAllocator>::make_shared(one_mb);
    ASSERT_EQ(one_mb, stack->Size());
    ASSERT_EQ(one_mb, stack->Available());
    ASSERT_EQ(0, stack->Allocated());
}

} // namespace