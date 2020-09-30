

#include "trtlab/core/memory/cyclic_allocator.h"
#include "trtlab/core/memory/malloc.h"
#include "trtlab/core/memory/bytes.h"

#include "trtlab/core/memory/stl_allocator.h"

#include <gtest/gtest.h>

using namespace trtlab;

static const uint64_t one_mb = 1024 * 1024;

class TestCustomAllocator : public ::testing::Test
{
};

template<typename T>
using custom_vector = std::vector<T, stl::temporary_allocator<T, Malloc>>;

TEST_F(TestCustomAllocator, CustomVector)
{
    auto v1 = custom_vector<int> {1, 2, 4, 8};
    ASSERT_EQ(v1[0], 1);
    ASSERT_EQ(v1[1], 2);
    ASSERT_EQ(v1[2], 4);
    ASSERT_EQ(v1[3], 8);

    auto v2 = v1;
    ASSERT_EQ(v2[0], 1);
    ASSERT_EQ(v2[1], 2);
    ASSERT_EQ(v2[2], 4);
    ASSERT_EQ(v2[3], 8);

    v2[0] = 10;
    ASSERT_NE(v1[0], v2[0]);

    void* before = v2.data();
    v2.resize(1024);
    void* after = v2.data();
    ASSERT_NE(before, after);
    DLOG(INFO) << "finished resize";
    ASSERT_EQ(v1[0], 1);
    ASSERT_EQ(v1[1], 2);
    ASSERT_EQ(v1[2], 4);
    ASSERT_EQ(v1[3], 8);

    auto v3 = std::move(v2);
}

