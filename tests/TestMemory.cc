
#include "YAIS/Memory.h"
#include "gtest/gtest.h"

#include <list>

using namespace yais;

namespace
{

static size_t one_mb = 1024*1024;

template <typename T>
class TestMemory : public ::testing::Test
{
  public:
    typedef std::list<T> List;
};

using AllocatorTypes = ::testing::Types<
    CudaDeviceAllocator, CudaManagedAllocator, CudaHostAllocator, SystemMallocAllocator>;

TYPED_TEST_CASE(TestMemory, AllocatorTypes);

TYPED_TEST(TestMemory, make_shared)
{
    auto shared = TypeParam::make_shared(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    shared.reset();
    EXPECT_FALSE(shared);
}

TYPED_TEST(TestMemory, make_unique)
{
    auto unique = TypeParam::make_unique(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    unique.reset();
    EXPECT_FALSE(unique);
}

} // namespace