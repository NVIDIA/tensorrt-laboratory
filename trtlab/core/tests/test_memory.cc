#include <gtest/gtest.h>

#include <trtlab/core/memory/first_touch_allocator.h>
#include <trtlab/memory/allocator.h>
#include <trtlab/memory/malloc_allocator.h>

class TestMemory : public ::testing::Test {};

using namespace trtlab::memory;

TEST_F(TestMemory, FirstTouchAllocator)
{
    auto raw = first_touch_allocator<malloc_allocator, 0x42>(0);
    auto alloc = make_allocator(std::move(raw));

    auto md = alloc.allocate_descriptor(16);

    char *data = (char *) md.data();
    for(int i=0; i<16; i++)
    {
        EXPECT_EQ(data[i], 0x42);
    }
}