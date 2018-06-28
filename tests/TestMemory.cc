
#include "YAIS/Memory.h"
#include "gtest/gtest.h"

using namespace yais;

namespace
{

static size_t one_mb = 1024*1024;

TEST(Memory, CudaMalloc)
{
    auto shared = CudaDeviceAllocator::make_shared(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    EXPECT_EQ(256, shared->DefaultAlignment());
    shared.reset();
    EXPECT_FALSE(shared);

    auto unique = CudaDeviceAllocator::make_unique(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    EXPECT_EQ(256, unique->DefaultAlignment());
    unique.reset();
    EXPECT_FALSE(unique);
}

TEST(Memory, CudaHostMallocManaged)
{
    auto shared = CudaManagedAllocator::make_shared(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    EXPECT_EQ(256, shared->DefaultAlignment());
    shared.reset();
    EXPECT_FALSE(shared);

    auto unique = CudaManagedAllocator::make_unique(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    EXPECT_EQ(256, unique->DefaultAlignment());
    unique.reset();
    EXPECT_FALSE(unique);
}

TEST(Memory, CudaHostMalloc)
{
    auto shared = CudaHostAllocator::make_shared(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    EXPECT_EQ(64, shared->DefaultAlignment());
    shared.reset();
    EXPECT_FALSE(shared);

    auto unique = CudaHostAllocator::make_unique(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    EXPECT_EQ(64, unique->DefaultAlignment());
    unique.reset();
    EXPECT_FALSE(unique);
}

TEST(Memory, SystemMalloc)
{
    auto shared = SystemMallocAllocator::make_shared(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    EXPECT_EQ(64, shared->DefaultAlignment());
    shared.reset();
    EXPECT_FALSE(shared);

    auto unique = SystemMallocAllocator::make_unique(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    EXPECT_EQ(64, unique->DefaultAlignment());
    unique.reset();
    EXPECT_FALSE(unique);
}

} // namespace