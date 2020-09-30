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
#include <gtest/gtest.h>

#include <cstring>
#include <mutex>
#include <vector>

#include "trtlab/memory/allocator.h"
#include "trtlab/memory/block_allocators.h"
#include "trtlab/memory/block_arena.h"
#include "trtlab/memory/std_allocator.h"
#include "trtlab/memory/tracking.h"
#include "trtlab/memory/literals.h"
#include "trtlab/memory/smart_ptr.h"

#include "trtlab/memory/detail/memory_stack.h"
#include "trtlab/memory/transactional_allocator.h"

using namespace trtlab::memory;
using namespace trtlab::memory::literals;

class TestMemory : public ::testing::Test
{
};

#define ALIAS_TEMPLATE(Name, ...) using Name = __VA_ARGS__

namespace trtlab
{
    namespace memory
    {
        template <typename T, class RawAllocator>
        ALIAS_TEMPLATE(vector, std::vector<T, std_allocator<T, RawAllocator>>);
    }
} // namespace trtlab

template <typename T, typename RawAllocator>
auto make_vector(RawAllocator&& alloc)
{
    return vector<T, RawAllocator>(alloc);
}

template <typename T, typename RawAllocator, typename Mutex>
auto make_vector(allocator<RawAllocator, Mutex> alloc)
{
    return vector<T, trtlab::memory::allocator<RawAllocator, Mutex>>(alloc);
}

struct log_tracker
{
    void on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << name << ": node allocated: " << ptr << "; size: " << size << "; alignment: " << alignment;
    }

    void on_node_deallocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << name << ": node deallocated: " << ptr << "; " << size << "; " << alignment;
    }

    void on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << name << ": array allocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
    }

    void on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << name << ": array deallocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
    }

    const char* name;
};

struct empty_tracker
{
    void on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << ": node allocated: " << ptr << "; size: " << size << "; alignment: " << alignment;
    }

    void on_node_deallocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << ": node deallocated: " << ptr << "; " << size << "; " << alignment;
    }

    void on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << ": array allocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
    }

    void on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
    {
        LOG(INFO) << ": array deallocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
    }
};

struct counting_tracker
{
    counting_tracker(std::string n) : name(n), m_node_count(0), m_node_bytes(0), m_array_count(0), m_array_bytes(0) {}
    ~counting_tracker() {}

    counting_tracker(const counting_tracker&) = default;
    counting_tracker& operator=(const counting_tracker&) = default;

    counting_tracker(counting_tracker&&) noexcept = default;
    counting_tracker& operator=(counting_tracker&&) noexcept = default;

    void on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
    {
        DLOG(INFO) << name << ": node allocated: " << ptr << "; size: " << size << "; alignment: " << alignment;
        m_node_count++;
        m_node_bytes += size;
        DLOG(INFO) << name << ": node allocations: " << m_node_count << "; node bytes: " << m_node_bytes;
    }

    void on_node_deallocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
    {
        DLOG(INFO) << name << ": node deallocated: " << ptr << "; " << size << "; " << alignment;
        m_node_count--;
        m_node_bytes -= size;
        DLOG(INFO) << name << ": node allocations: " << m_node_count << "; node bytes: " << m_node_bytes;
    }

    void on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
    {
        DLOG(INFO) << name << ": array allocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
        m_array_count++;
        m_array_bytes += count * size;
        DLOG(INFO) << name << ": array allocations: " << m_array_count << "; array bytes: " << m_array_bytes;
    }

    void on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
    {
        DLOG(INFO) << name << ": array deallocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
        m_array_count--;
        m_array_bytes -= count * size;
        DLOG(INFO) << name << ": array allocations: " << m_array_count << "; array bytes: " << m_array_bytes;
    }

    std::size_t count() const noexcept
    {
        return m_node_count + m_array_count;
    }

    std::size_t bytes() const noexcept
    {
        return m_node_bytes + m_array_bytes;
    }

    std::size_t node_count() const noexcept
    {
        return m_node_count;
    }
    std::size_t node_bytes() const noexcept
    {
        return m_node_bytes;
    }
    std::size_t array_count() const noexcept
    {
        return m_array_count;
    }
    std::size_t array_bytes() const noexcept
    {
        return m_array_bytes;
    }

private:
    std::string name;
    std::size_t m_node_count;
    std::size_t m_node_bytes;
    std::size_t m_array_count;
    std::size_t m_array_bytes;
};

template <typename Allocator>
static void test_alloc_x10(Allocator& alloc)
{
    auto p0 = alloc.allocate_node(33_MiB, 8UL);
    auto p1 = alloc.allocate_node(44_MiB, 8UL);
    auto p2 = alloc.allocate_node(78_MiB, 8UL);
    auto p3 = alloc.allocate_node(12_MiB, 8UL);
    auto p4 = alloc.allocate_node(32_MiB, 8UL);
    auto p5 = alloc.allocate_node(100_MiB, 8UL);
    auto p6 = alloc.allocate_node(18_MiB, 8UL);
    auto p7 = alloc.allocate_node(21_MiB, 8UL);
    auto p8 = alloc.allocate_node(15_MiB, 8UL);
    auto p9 = alloc.allocate_node(71_MiB, 8UL);

    alloc.deallocate_node(p3, 12_MiB, 8UL);
    alloc.deallocate_node(p1, 44_MiB, 8UL);
    alloc.deallocate_node(p7, 21_MiB, 8UL);
    alloc.deallocate_node(p8, 15_MiB, 8UL);
    alloc.deallocate_node(p0, 33_MiB, 8UL);
    alloc.deallocate_node(p4, 32_MiB, 8UL);
    alloc.deallocate_node(p6, 18_MiB, 8UL);
    alloc.deallocate_node(p5, 100_MiB, 8UL);
    alloc.deallocate_node(p9, 71_MiB, 8UL);
    alloc.deallocate_node(p2, 78_MiB, 8UL);
}

struct TestRawMalloc
{
    using memory_type = host_memory;
    using is_stateful = std::false_type;

    static void* allocate_node(std::size_t size, std::size_t)
    {
        return std::malloc(size);
    }

    static void deallocate_node(void* ptr, std::size_t, std::size_t) noexcept
    {
        return std::free(ptr);
    }
};

TEST_F(TestMemory, BasicTraits)
{
    auto raw = TestRawMalloc();
    static_assert(std::is_same<typename decltype(raw)::is_stateful, std::false_type>::value, "");
    auto alloc = make_allocator(std::move(raw));
    //static_assert(std::is_same<typename decltype(alloc)::is_stateful, std::false_type>::value, "");

    void* p0 = alloc.allocate(1_MiB);
    ASSERT_NE(p0, nullptr);
    std::memset(p0, 1, 1_MiB);
    alloc.deallocate(p0, 1_MiB);
}

TEST_F(TestMemory, ReferenceStorage)
{
    auto raw = TestRawMalloc();
    static_assert(std::is_same<typename decltype(raw)::is_stateful, std::false_type>::value, "");

    auto alloc = make_allocator(std::move(raw));

    ASSERT_EQ(alloc.use_count(), 1);
    auto ref = make_allocator_reference(alloc);
    ASSERT_EQ(alloc.use_count(), 2);
}

#include "malloc_allocator.h"

TEST_F(TestMemory, VectorWithTracking)
{
    auto raw     = malloc_allocator();
    auto adpt    = make_allocator_adapter(std::move(raw));
    ASSERT_EQ(adpt.device_context().device_type, kDLCPU);

    auto tracked = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(adpt));
    ASSERT_EQ(tracked.device_context().device_type, kDLCPU);

    auto alloc   = make_allocator(std::move(tracked));

    ASSERT_EQ(alloc.device_context().device_type, kDLCPU);

    ASSERT_EQ(alloc.use_count(), 1);
    auto vec1 = make_vector<int>(alloc);
    ASSERT_EQ(alloc.use_count(), 2);

    vec1.reserve(128);

    LOG(INFO) << vec1.get_allocator().get_allocator().min_alignment();
    LOG(INFO) << vec1.get_allocator().get_allocator().max_alignment();

    for (int i = 0; i < 10; i++)
    {
        vec1.push_back(i);
    }

    auto vec2 = vec1;

    for (int i = 0; i < 10; i++)
    {
        vec2[i] += 2;
    }

    ASSERT_EQ(alloc.use_count(), 3);
}

// block_allocator from raw type
TEST_F(TestMemory, SingleBlockAllocatorFromType)
{
    auto block_alloc = make_block_allocator<single_block_allocator, TestRawMalloc>(1_MiB);
    auto block       = block_alloc.allocate_block();
    ASSERT_EQ(block_alloc.next_block_size(), 0);
    block_alloc.deallocate_block(block);
    ASSERT_EQ(block_alloc.next_block_size(), 1_MiB);
}

TEST_F(TestMemory, SingleBlockAllocatorFromObj)
{
    auto raw         = TestRawMalloc();
    auto tracked     = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(raw));
    auto block_alloc = make_block_allocator<single_block_allocator>(std::move(tracked), 1_MiB);

    auto block = block_alloc.allocate_block();
    block_alloc.deallocate_block(block);

    block = block_alloc.allocate_block();
    EXPECT_ANY_THROW(block_alloc.allocate_block());
    block_alloc.deallocate_block(block);
}

TEST_F(TestMemory, FixedSizedBlockAllocatorFromType)
{
    auto block_alloc = make_block_allocator<fixed_size_block_allocator, TestRawMalloc>(1_MiB);
    auto block       = block_alloc.allocate_block();
    ASSERT_EQ(block_alloc.next_block_size(), 1_MiB);
    block_alloc.deallocate_block(block);
}

TEST_F(TestMemory, FixedSizedBlockAllocatorFromObj)
{
    auto raw         = TestRawMalloc();
    auto tracked     = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(raw));
    auto block_alloc = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);

    auto block0 = block_alloc.allocate_block();
    auto block1 = block_alloc.allocate_block();
    ASSERT_EQ(block0.size, 1_MiB);
    ASSERT_EQ(block1.size, 1_MiB);
    block_alloc.deallocate_block(block0);
    block_alloc.deallocate_block(block1);
}

TEST_F(TestMemory, GrowingdBlockAllocatorFromType)
{
    auto block_alloc = make_block_allocator<growing_block_allocator, TestRawMalloc>(1_MiB);
    auto block       = block_alloc.allocate_block();
    ASSERT_EQ(block_alloc.next_block_size(), 2_MiB);
    block_alloc.deallocate_block(block);
}

TEST_F(TestMemory, GrowingdBlockAllocatorFromObj)
{
    auto raw         = TestRawMalloc();
    auto tracked     = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(raw));
    auto block_alloc = make_block_allocator<growing_block_allocator>(std::move(tracked), 1_MiB, 2_MiB, 2.0);

    auto block0 = block_alloc.allocate_block();
    auto block1 = block_alloc.allocate_block();
    ASSERT_EQ(block0.size, 1_MiB);
    ASSERT_EQ(block1.size, 2_MiB);
    ASSERT_EQ(block_alloc.next_block_size(), 2_MiB);
    block_alloc.deallocate_block(block0);
    block_alloc.deallocate_block(block1);
}

TEST_F(TestMemory, CountLimitedFixedSizeBlockAllocatorFromType)
{
    auto block_alloc = make_block_allocator<fixed_size_block_allocator, TestRawMalloc>(1_MiB);
    auto block       = block_alloc.allocate_block();
    ASSERT_EQ(block_alloc.next_block_size(), 1_MiB);
    block_alloc.deallocate_block(block);
}

TEST_F(TestMemory, CountLimitedFixedSizeBlockAllocatorFromObj)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(raw));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);
    auto alloc   = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 2UL);

    auto block0 = alloc.allocate_block();
    auto block1 = alloc.allocate_block();
    ASSERT_EQ(block0.size, 1_MiB);
    ASSERT_EQ(block1.size, 1_MiB);
    ASSERT_EQ(alloc.block_count(), 2);
    ASSERT_ANY_THROW(alloc.allocate_block());
    alloc.deallocate_block(block0);
    alloc.deallocate_block(block1);
}

TEST_F(TestMemory, SizeLimitedFixedSizeBlockAllocatorFromObj)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(raw));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);
    auto alloc   = make_extended_block_allocator<size_limited_block_allocator>(std::move(block), 2_MiB);

    auto block0 = alloc.allocate_block();
    auto block1 = alloc.allocate_block();
    ASSERT_EQ(block0.size, 1_MiB);
    ASSERT_EQ(block1.size, 1_MiB);
    ASSERT_EQ(alloc.bytes_allocated(), 2_MiB);
    ASSERT_ANY_THROW(alloc.allocate_block());
    alloc.deallocate_block(block0);
    alloc.deallocate_block(block1);
}

TEST_F(TestMemory, SmartPtrsWithStatelessRawAllocator)
{
    auto raw = TestRawMalloc();
    static_assert(std::is_same<typename decltype(raw)::is_stateful, std::false_type>::value, "");
    static_assert(is_thread_safe_allocator<decltype(raw)>::value, "should be true");

    auto tracked = make_tracked_allocator(empty_tracker{}, std::move(raw));
    static_assert(std::is_same<typename decltype(tracked)::is_stateful, std::false_type>::value, "");
    static_assert(is_thread_safe_allocator<decltype(tracked)>::value, "should be true");

    auto alloc = make_allocator(std::move(tracked));
    static_assert(is_thread_safe_allocator<decltype(alloc)>::value, "should be true");
    static_assert(std::is_same<typename decltype(alloc)::mutex, no_mutex>::value, "");

    ASSERT_EQ(alloc.use_count(), 1);

    auto i0 = allocate_unique<int>(alloc, 1);
    ASSERT_EQ(alloc.use_count(), 2);
}

TEST_F(TestMemory, SmartPtrsWithStatefulRawAllocator)
{
    auto raw = TestRawMalloc();
    static_assert(std::is_same<typename decltype(raw)::is_stateful, std::false_type>::value, "");
    static_assert(is_thread_safe_allocator<decltype(raw)>::value, "should be true");

    auto tracked = make_tracked_allocator(log_tracker{"** malloc **"}, std::move(raw));
    static_assert(std::is_same<typename decltype(tracked)::is_stateful, std::true_type>::value, "");
    static_assert(!is_thread_safe_allocator<decltype(tracked)>::value, "should be false");

    auto alloc = make_allocator(std::move(tracked));
    static_assert(is_thread_safe_allocator<decltype(alloc)>::value, "should be true");
    static_assert(std::is_same<typename decltype(alloc)::mutex, std::mutex>::value, "");

    ASSERT_EQ(alloc.use_count(), 1);

    auto i0 = allocate_unique<int>(alloc, 1);
    ASSERT_EQ(alloc.use_count(), 2);
}

TEST_F(TestMemory, MemoryDescriptors)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(empty_tracker{}, std::move(raw));
    auto alloc   = make_allocator(std::move(tracked));

    ASSERT_EQ(alloc.use_count(), 1);

    auto ref = make_allocator_reference(alloc);
    ASSERT_EQ(alloc.use_count(), 2);

    LOG(INFO) << "testing any";

    auto any_ref = make_any_allocator_reference(alloc);
    ASSERT_EQ(alloc.use_count(), 3);

    auto md = alloc.allocate_descriptor(1_MiB);
    /*
    auto md = alloc.allocate_mdesc(1_MiB);
    ASSERT_EQ(alloc.use_count(), 4);

    auto md3 = alloc.allocate_mdesc_v3(1_MiB);
    ASSERT_EQ(alloc.use_count(), 5);

    mdesc_v3 v3;
    v3 = std::move(md3);
    ASSERT_EQ(alloc.use_count(), 5);
*/
    LOG(INFO) << "finished testing any";
}

TEST_F(TestMemory, AllocatorTraits)
{
    auto raw   = TestRawMalloc();
    auto alloc = make_allocator(std::move(raw));

    using traits = allocator_traits<decltype(alloc)>;

    auto min = traits::min_alignment(alloc);
    auto ctx = traits::device_context(alloc);
}

TEST_F(TestMemory, CachedBlockArena)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(counting_tracker("** tracker: malloc **"), std::move(raw));
    // protect the counters by making tracked a thread-safe allocator
    auto safe  = make_thread_safe_allocator<std::mutex>(std::move(tracked));
    auto block = make_block_allocator<fixed_size_block_allocator>(std::move(safe), 1_MiB);
    auto alloc = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 2UL);
    auto arena = make_cached_block_arena(std::move(alloc));

    /*                          v thread-safe v  v tracked v                */
    const auto& tracker = arena.get_allocator().get_allocator().get_tracker();

    auto block0 = arena.allocate_block();
    ASSERT_EQ(tracker.array_count(), 1);
    ASSERT_EQ(tracker.array_bytes(), 1_MiB);

    auto block1 = arena.allocate_block();
    ASSERT_EQ(tracker.array_count(), 2);
    ASSERT_EQ(tracker.array_bytes(), 2_MiB);

    ASSERT_EQ(arena.get_block_allocator().block_count(), 2);
    ASSERT_ANY_THROW(arena.allocate_block());

    // caching arena will hold the block and not actually deallocate it
    arena.deallocate_block(std::move(block0));
    ASSERT_EQ(tracker.array_count(), 2);
    ASSERT_EQ(tracker.array_bytes(), 2_MiB);

    // deallocate any used cached blocks
    arena.shrink_to_fit();
    ASSERT_EQ(tracker.array_count(), 1);
    ASSERT_EQ(tracker.array_bytes(), 1_MiB);

    // add a block back to the cache
    arena.deallocate_block(std::move(block1));
    ASSERT_EQ(tracker.array_count(), 1);
    ASSERT_EQ(tracker.array_bytes(), 1_MiB);

    // this allocation should pull from the cache
    block0 = arena.allocate_block();
    ASSERT_EQ(tracker.array_count(), 1);
    ASSERT_EQ(tracker.array_bytes(), 1_MiB);
    arena.deallocate_block(std::move(block0));
}

TEST_F(TestMemory, DetailStack)
{
    detail::fixed_memory_stack stack;
}

TEST_F(TestMemory, TransactionalAllocator)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(counting_tracker("** tracker: malloc **"), std::move(raw));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);
    auto counted = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 3);
    auto arena   = make_cached_block_arena(std::move(counted));
    auto txalloc = make_transactional_allocator(std::move(arena));
    auto alloc   = make_allocator(std::move(txalloc));

    /*                           v txalloc v     v tracked v  */
    const auto& tracker = alloc.get_allocator().get_allocator().get_tracker();

    // the transactional allocator will allocate one block on instantiate
    // you can avoid this overhead if you've pre-instantiated the arena by
    // calling reserve_blocks on the arena allocator prior to moving it into
    // the transactional allocator
    ASSERT_EQ(tracker.count(), 1);
    ASSERT_EQ(tracker.bytes(), 1_MiB);

    {
        // this is allocated on the initial stack
        auto md = alloc.allocate_descriptor(1_KiB);
        ASSERT_EQ(tracker.count(), 1UL);
        ASSERT_EQ(tracker.bytes(), 1_MiB);
    }

    {
        // since the previous memory descriptor is released
        // the next allocation can force recycling of that block
        // rather than allocating a new one
        auto md1 = alloc.allocate_descriptor(1_MiB);
        ASSERT_EQ(tracker.count(), 1);
        ASSERT_EQ(tracker.bytes(), 1_MiB);
    }

    {
        auto on_block_0 = alloc.allocate_descriptor(1_KiB);
        ASSERT_EQ(tracker.count(), 1UL);
        ASSERT_EQ(tracker.bytes(), 1_MiB);

        // not enough room on the original stack
        // since we have not pre-reserved arena blocks
        // this will allocate a new block

        auto on_block_1 = alloc.allocate_descriptor(1_MiB);
        ASSERT_EQ(tracker.count(), 2UL);
        ASSERT_EQ(tracker.bytes(), 2_MiB);
    }
}

#include "huge_page_allocator.h"
#include "detail/page_info.h"
#include "linux/kernel-page-flags.h"

#include <thread>

TEST_F(TestMemory, TransparentHugePages)
{
    constexpr std::size_t size = 20_MiB;

    auto thp_2m = transparent_huge_page_allocator<2_MiB>();

    auto alloc = make_allocator(std::move(thp_2m));

    ASSERT_EQ(2_MiB, alloc.min_alignment());
    ASSERT_EQ(2_MiB, alloc.max_alignment());

    auto md = alloc.allocate_descriptor(size);
    std::memset(md.data(), 0, size);

    // this breaks down the array into system page-sized pages
    // on linux, 4k pages
    // in this example, we will test each of those 4k pages to see if they belong
    // to a larger transparent huge page
    page_info_array pinfo     = get_info_for_range(md.data(), ((char*)md.data()) + size);
    flag_count      thp_count = get_flag_count(pinfo, KPF_THP);

    EXPECT_TRUE(thp_count.pages_available) << "hugepage info not available; probably not running as root";

    if (thp_count.pages_available)
    {
        EXPECT_EQ(thp_count.pages_set, thp_count.pages_total);
    }

    // DLOG(INFO) << "grep -e AnonHugePages  /proc/" << getpid() << "/smaps";
    // std::this_thread::sleep_for(std::chrono::seconds(60));
}

#include "detail/free_list.h"

bool is_equal(void* a, void* b)
{
    return reinterpret_cast<std::uintptr_t>(a) == reinterpret_cast<std::uintptr_t>(b);
}

TEST_F(TestMemory, TestFreeList)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(counting_tracker("** tracker: malloc **"), std::move(raw));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);

    auto b0 = block.allocate_block();
    auto b1 = block.allocate_block();

    auto mem0 = b0.memory;
    auto mem1 = b1.memory;

    ASSERT_FALSE(is_equal(b0.memory, b1.memory));

    auto list = detail::block_list();

    list.insert(std::move(b0));
    list.insert(std::move(b1));
    ASSERT_EQ(list.size(), 2);

    auto l0 = list.allocate();
    ASSERT_EQ(list.size(), 1);
    EXPECT_TRUE(is_equal(l0.memory, mem1));

    list.deallocate(std::move(l0));
    ASSERT_EQ(list.size(), 2);

    // a second allocation and deallocation should look similar
    l0 = list.allocate();
    ASSERT_EQ(list.size(), 1);
    EXPECT_TRUE(is_equal(l0.memory, mem1));
    auto l1 = list.allocate();
    ASSERT_EQ(list.size(), 0);
    EXPECT_TRUE(is_equal(l1.memory, mem0));

    block.deallocate_block(std::move(l0));
    block.deallocate_block(std::move(l1));
    ASSERT_EQ(list.size(), 0);
}

#include "block_stack.h"

TEST_F(TestMemory, MemoryArenaUncached)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(counting_tracker("** tracker: malloc **"), std::move(raw));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);
    auto count   = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 2UL);
    auto stack   = make_block_stack<uncached>(std::move(count));

    /*                            v count v       v tracked v                */
    const auto& tracker = stack.get_allocator().get_allocator().get_tracker();

    ASSERT_LT(stack.next_block_size(), 1_MiB);
    auto actual_size = stack.next_block_size();

    // we can only grow memory_arena one block at a time
    // the memory_arena is a stack of blocks
    // when we deallocate_block, it pops the stack back
    // the stack is not necessarily continuous

    stack.allocate_block();
    auto block0 = stack.current_block().memory;
    ASSERT_EQ(stack.current_block().size, actual_size);
    ASSERT_EQ(tracker.count(), 1);

    stack.allocate_block();
    auto block1 = stack.current_block().memory;
    ASSERT_FALSE(is_equal(block0, block1));
    ASSERT_EQ(stack.current_block().size, actual_size);
    ASSERT_EQ(tracker.count(), 2);

    stack.deallocate_block();
    ASSERT_TRUE(is_equal(stack.current_block().memory, block0));
    ASSERT_EQ(tracker.count(), 1);

    stack.deallocate_block();
    ASSERT_TRUE(stack.empty());
    ASSERT_EQ(tracker.count(), 0);
}

#include "memory_pool.h"
#include "detail/container_node_sizes.h"
#include <list>

TEST_F(TestMemory, MemoryPool)
{
    auto raw     = TestRawMalloc();
    auto tracked = make_tracked_allocator(counting_tracker("** tracker: malloc **"), std::move(raw));
    auto single  = make_block_allocator<single_block_allocator>(std::move(tracked), 1_MiB);
    auto stack   = make_block_stack<uncached>(std::move(single));
    auto pool    = memory_pool<decltype(single)>(list_node_size<int>::value, std::move(stack));

    auto node_size  = list_node_size<int>::value;
    auto node_count = pool.capacity_left() / node_size;
    LOG(INFO) << "pool node_size      : " << node_size;
    LOG(INFO) << "pool node capacity  : " << node_count;
    auto ctx = pool.device_context();

    auto list  = std::list<int, std_allocator<int, decltype(pool)>>(pool);
    auto queue = std::queue<int, decltype(list)>(std::move(list));

    for (int i = 0; i < node_count; i++)
    {
        queue.push(i);
    }

    ASSERT_ANY_THROW(queue.push(42));
}

#include "memory_type.h"

TEST_F(TestMemory, IsMemoryType)
{
    static_assert(is_memory_type<host_memory>::value, "");

    struct empty_memory
    {
    };
    static_assert(!is_memory_type<empty_memory>::value, "");

    struct with_valid_impl
    {
        constexpr static DLDeviceType device_type()
        {
            return kDLCPU;
        }
        constexpr static std::size_t min_allocation_alignment()
        {
            return 8UL;
        }
        constexpr static std::size_t max_access_alignment()
        {
            return 8UL;
        }
        static std::size_t access_alignment_for(std::size_t)
        {
            return with_valid_impl::max_access_alignment();
        }
    };
    static_assert(!is_memory_type<with_valid_impl>::value, "");
    static_assert(decltype(detail::is_memory_type_impl<with_valid_impl>(0))::value, "");

    struct with_valid_base : detail::any_memory
    {
    };
    static_assert(!is_memory_type<with_valid_base>::value, "");

    struct valid_memory_type : public with_valid_impl, public with_valid_base
    {
    };

    //static_assert(std::is_base_of<detail::any_memory, valid_memory_type>::value, "");
    static_assert(decltype(detail::is_memory_type_impl<valid_memory_type>(0))::value, "");

    static_assert(is_memory_type<valid_memory_type>::value, "");

    struct pinned_host_memory : host_memory
    {
        constexpr static DLDeviceType device_type() noexcept
        {
            return kDLCPUPinned;
        }
    };

    static_assert(is_memory_type<pinned_host_memory>::value, "");
    static_assert(is_host_memory<pinned_host_memory>::value, "");
}

TEST_F(TestMemory, HostMemory)
{
    static_assert(is_memory_type<host_memory>::value, "");
    static_assert(is_host_memory<host_memory>::value, "");
    static_assert(host_memory::max_access_alignment() == 8UL, "");
    ASSERT_EQ(host_memory::access_alignment_for(1), 1);
    ASSERT_EQ(host_memory::access_alignment_for(2), 2);
    ASSERT_EQ(host_memory::access_alignment_for(3), 2);
    ASSERT_EQ(host_memory::access_alignment_for(4), 4);
    ASSERT_EQ(host_memory::access_alignment_for(5), 4);
    ASSERT_EQ(host_memory::access_alignment_for(6), 4);
    ASSERT_EQ(host_memory::access_alignment_for(7), 4);
    ASSERT_EQ(host_memory::access_alignment_for(8), 8);
    ASSERT_EQ(host_memory::access_alignment_for(9), 8);
    ASSERT_EQ(host_memory::access_alignment_for(100), 8);
}

/*
#include "affinity.h"

TEST_F(TestMemory, Topology)
{
    auto logical_cpus = [](const cpuaff::cpu& cpu) { return int(cpu.id().get()); };

    auto initial_affinity = affinity::this_thread::get_affinity();
    LOG(INFO) << "affinity: this_thread: " << initial_affinity;

    {
        affinity_guard guard(cpu_set::from_string("0"));
        auto           scoped_affinity = affinity::this_thread::get_affinity();
        EXPECT_EQ(scoped_affinity.size(), 1);
    }

    auto after_guard_affinity = affinity::this_thread::get_affinity();
    EXPECT_EQ(initial_affinity, after_guard_affinity);

    auto numa_nodes = affinity::system::topology();
    for(const auto& n : numa_nodes)
    {
        LOG(INFO) << n;
    }
}
*/

#include "malloc_allocator.h"

TEST_F(TestMemory, FirstTouchMallocAllocator)
{
    constexpr std::size_t size = 20_MiB;

    //auto raw = first_touch_allocator<malloc_allocator>();

    auto alloc = make_allocator(malloc_allocator());

    ASSERT_EQ(8, alloc.min_alignment());
    ASSERT_EQ(8, alloc.max_alignment());

    auto md = alloc.allocate_descriptor(size);

    // this breaks down the array into system page-sized pages; on linux, 4k pages
    // in this example, we will test each of those 4k pages to see if they belong
    // to a larger transparent huge page
    page_info_array pinfo     = get_info_for_range(md.data(), ((char*)md.data()) + size);
    flag_count      thp_count = get_flag_count(pinfo, KPF_THP);

    EXPECT_TRUE(thp_count.pages_available) << "hugepage info not available; probably not running as root";

    if (thp_count.pages_available)
    {
        EXPECT_EQ(thp_count.pages_set, 0);
    }

    // DLOG(INFO) << "grep -e AnonHugePages  /proc/" << getpid() << "/smaps";
    // std::this_thread::sleep_for(std::chrono::seconds(60));
}

#include "detail/ranges.h"

using detail::find_ranges;
using detail::print_ranges;

TEST_F(TestMemory, FindRanges0)
{
    std::vector<int>                 a{1};
    std::vector<std::pair<int, int>> a_ranges{{1, 1}};
    auto                             ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1");
}

TEST_F(TestMemory, FindRanges1)
{
    std::vector<int>                 a{1, 2};
    std::vector<std::pair<int, int>> a_ranges{{1, 2}};
    auto                             ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-2");
}

TEST_F(TestMemory, FindRanges2)
{
    std::vector<int>                 a{1, 2, 3};
    std::vector<std::pair<int, int>> a_ranges{{1, 3}};
    auto                             ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-3");
}

TEST_F(TestMemory, FindRanges3)
{
    std::vector<int>                 a{1, 3};
    std::vector<std::pair<int, int>> a_ranges{{1, 1}, {3, 3}};
    auto                             ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1,3");
}

TEST_F(TestMemory, FindRanges4)
{
    std::vector<int>                 a{1, 2, 4, 5, 6, 10};
    std::vector<std::pair<int, int>> a_ranges{{1, 2}, {4, 6}, {10, 10}};
    auto                             ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-2,4-6,10");
}

TEST_F(TestMemory, FindRanges5)
{
    std::vector<int>                 a{0, 1, 2, 3, 4, 5, 6};
    std::vector<std::pair<int, int>> a_ranges{{0, 6}};
    auto                             ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "0-6");
}

template <typename Key, typename Value, typename BlockAllocator>
auto make_map(BlockAllocator&& block_alloc)
{
    static_assert(is_block_allocator<BlockAllocator>::value, "");

    using node_type = std::pair<Key, Value>;
    auto node_size  = alignof(node_type) + sizeof(node_type) + 64;

    auto stack = make_block_stack<uncached>(std::move(block_alloc));
    auto pool  = memory_pool<BlockAllocator>(node_size, std::move(stack));
    auto alloc = make_thread_unsafe_allocator(std::move(pool));

    return std::map<Key, Value, std::less<Key>, std_allocator<node_type, decltype(alloc)>>(alloc);
}

TEST_F(TestMemory, MapWithCustomAllocator)
{
    auto huge  = transparent_huge_page_allocator<2_MiB>();
    auto track = make_tracked_allocator(counting_tracker{"** tracker: huge **"}, std::move(huge));
    auto block = make_block_allocator<fixed_size_block_allocator>(std::move(track), 2_MiB);
    auto alloc = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 4);
    auto arena = make_cached_block_arena(std::move(alloc));

    // populate cache
    arena.reserve_blocks(4);

    // create map
    auto m = make_map<int, int>(std::move(arena));

    // unwind the allocator stack to get the tracker
    //                      v  std_alloc  v v memory_pool v v block_stack v v block_arena v  v tracker v
    const auto& tracker = m.get_allocator().get_allocator().get_allocator().get_allocator().get_allocator().get_tracker();

    // our system allocator has been used 4 times to populate the block cache
    EXPECT_EQ(tracker.count(), 4);

    LOG(INFO) << "Start Using Map";

    for (int i = 0; i < 10240; i++)
    {
        m[i] = i;
    }

    EXPECT_EQ(tracker.count(), 4);
    LOG(INFO) << "total system allocations: " << tracker.count();
}

TEST_F(TestMemory, MapWithTracedMalloc)
{
    auto track = make_tracked_allocator(counting_tracker{"** tracker: malloc **"}, malloc_allocator());

    // create map
    auto m = std::map<int, int, std::less<int>, std_allocator<std::pair<int, int>, decltype(track)>>(track);

    // get tracker          v  std_alloc  v   v tracker v
    const auto& tracker = m.get_allocator().get_allocator().get_tracker();

    LOG(INFO) << "system allocation count on init: " << tracker.count();
    LOG(INFO) << "start using map";

    for (int i = 0; i < 10240; i++)
    {
        m[i] = i;
    }

    LOG(INFO) << "finish using map";
    LOG(INFO) << "system allocation count on fini: " << tracker.count();
}

bool equiv_ptr(void* lhs, void* rhs)
{
    return reinterpret_cast<addr_t>(lhs) == reinterpret_cast<addr_t>(rhs);
}

TEST_F(TestMemory, RBTree_Set)
{
    memory_block b1{reinterpret_cast<void*>(0x00000001), 128};
    memory_block b2{reinterpret_cast<void*>(0xDEADBEEF), 1024};
    memory_block b3{reinterpret_cast<void*>(0xFACEBAD1), 1024};
    memory_block b4{reinterpret_cast<void*>(0xA0000000), 2048};

    std::set<memory_block, memory_block_compare_size<>> blocks(memory_block_compare_size<>{});

    blocks.insert(b1);
    blocks.insert(b2);
    blocks.insert(b3);
    blocks.insert(b4);

    // using is_transparent -> we can find blocks by size

    // we can find blocks by actual block or by size
    auto search = blocks.find(1024);
    ASSERT_NE(search, blocks.end());
    ASSERT_TRUE(equiv_ptr(search->memory, b2.memory));

    // we can find blocks that meet certin size requirements by using lower bound
    search = blocks.lower_bound(129);
    ASSERT_NE(search, blocks.end());
    ASSERT_TRUE(equiv_ptr(search->memory, b2.memory));

    // this is how we woudl have to do it without is_transparent
    search = blocks.lower_bound(memory_block{nullptr, 127});
    ASSERT_NE(search, blocks.end());
    ASSERT_TRUE(equiv_ptr(search->memory, b1.memory));

    search = blocks.lower_bound(memory_block{nullptr, 128});
    ASSERT_NE(search, blocks.end());
    ASSERT_TRUE(equiv_ptr(search->memory, b1.memory));

    search = blocks.lower_bound(memory_block{nullptr, 129});
    ASSERT_NE(search, blocks.end());
    ASSERT_TRUE(equiv_ptr(search->memory, b2.memory));

    // we can also find blocks directly by content
    search = blocks.find(b3);
    ASSERT_NE(search, blocks.end());
    ASSERT_TRUE(equiv_ptr(search->memory, b3.memory));
}

#include "bfit_allocator.h"
TEST_F(TestMemory, bfit)
{
    auto track = make_tracked_allocator(log_tracker{"** tracker: malloc **"}, malloc_allocator());
    auto alloc = make_bfit_allocator(6 * 128_MiB, std::move(track));
    //auto alloc = make_allocator(std::move(bfit));

    test_alloc_x10(alloc);
    ASSERT_EQ(alloc.free_nodes(), 1);
    ASSERT_EQ(alloc.used_nodes(), 0);
    test_alloc_x10(alloc);
    ASSERT_EQ(alloc.free_nodes(), 1);
    ASSERT_EQ(alloc.used_nodes(), 0);
    test_alloc_x10(alloc);
    ASSERT_EQ(alloc.free_nodes(), 1);
    ASSERT_EQ(alloc.used_nodes(), 0);
}

#include <boost/histogram.hpp> // make_histogram, regular, weight, indexed
#include <boost/format.hpp>
#include "utils.h"

TEST_F(TestMemory, histogram)
{
    using namespace boost::histogram; // strip the boost::histogram prefix
    auto h = make_histogram(axis::regular<>(6, 12, 36, "x"));

    /*
    Let's fill a histogram with data, typically this happens in a loop.

    STL algorithms are supported. std::for_each is very convenient to fill a
    histogram from an iterator range. Use std::ref in the call, if you don't
    want std::for_each to make a copy of your histogram.
  */
    std::vector<std::size_t> data = {1_KiB, 3_MiB, 512, 256, 128};

    auto log_hist = [&h](std::size_t size) { h(ilog2_ceil(size)); };

    std::for_each(data.begin(), data.end(), log_hist);
    //h(-1.5); // is placed in underflow bin -1
    //h(-1.0); // is placed in bin 0, bin interval is semi-open
    //h(2.0);  // is placed in overflow bin 6, bin interval is semi-open
    //h(20.0); // is placed in overflow bin 6

    /*
    This does a weighted fill using the `weight` function as an additional
    argument. It may appear at the beginning or end of the argument list. C++
    doesn't have keyword arguments like Python, this is the next-best thing.
  */

    std::ostringstream os;
    os << std::endl;
    for (auto&& x : indexed(h, coverage::all))
    {
        //os << boost::format("bin %2i [%4.1f, %4.1f): %i\n") % x.index() % x.bin().lower() % x.bin().upper() % *x;
        os << boost::format("bin %2i [%10s, %10s): %i\n") % x.index() % bytes_to_string(std::pow(2, x.bin().lower()))
                  % bytes_to_string(std::pow(2, x.bin().upper())) % *x;
    }

    LOG(INFO) << os.str();
}

#include <trtlab/memory/trackers.h>
#include <trtlab/memory/raii_allocator.h>

TEST_F(TestMemory, TrackHighLevelAllocator)
{
    auto high_lvl = make_allocator(make_bfit_allocator(256_MiB, malloc_allocator()));
    auto tracker1 = make_tracked_allocator(size_tracker{}, high_lvl.copy());
    auto tracker2 = make_tracked_allocator(size_tracker{}, high_lvl.copy());
    auto alloc1 = make_allocator(std::move(tracker1));
    auto alloc2 = make_allocator(std::move(tracker2));

    const auto& t1 = alloc1.get_allocator().get_tracker();
    const auto& t2 = alloc2.get_allocator().get_tracker();

    EXPECT_EQ(t1.bytes(), 0);
    EXPECT_EQ(t2.bytes(), 0);


    auto md0 = alloc1.allocate_descriptor(3_MiB);
    EXPECT_EQ(t1.bytes(), 3_MiB);

    auto md1 = alloc2.allocate_descriptor(128_MiB);
    EXPECT_EQ(t2.bytes(), 128_MiB);

    descriptor md2;
    EXPECT_ANY_THROW(md2 = alloc1.allocate_descriptor(128_MiB));

    // the state of the trackers should not change
    EXPECT_EQ(t1.bytes(), 3_MiB);
    EXPECT_EQ(t2.bytes(), 128_MiB);

    auto raii = make_raii_allocator(alloc1);
    auto md3 = raii.allocate_descriptor(45_MiB);
    EXPECT_EQ(t1.bytes(), 48_MiB);

    void* p0  = raii.allocate(2_MiB);
    void* p1  = raii.allocate(4_MiB);
    EXPECT_EQ(t1.bytes(), 54_MiB);

    raii.deallocate(p1);
    // this would normally be a memory leak,
    // but because the allocator owns a descriptor for all non-descriptor allocations
    // this will get cleaned up by the destructor.
    // todo: add a warning for all
    // raii.deallocate(p0);
    EXPECT_EQ(t1.bytes(), 50_MiB);

    auto iraii = raii.shared();
    auto md4 = iraii->allocate_descriptor(8_MiB);
    EXPECT_EQ(t1.bytes(), 58_MiB);

}

#include <trtlab/memory/memory_typed_allocator.h>

TEST_F(TestMemory, IAllocator)
{
    auto high_lvl = make_allocator(make_bfit_allocator(256_MiB, malloc_allocator()));

    CHECK_EQ(high_lvl.device_context().device_type, kDLCPU);

    auto tracker1 = make_tracked_allocator(log_tracker{"** tracker: high-level #1 **"}, high_lvl.copy());
    auto tracker2 = make_tracked_allocator(log_tracker{"** tracker: high-level #2 **"}, high_lvl.copy());
    auto alloc1 = make_allocator(std::move(tracker1));
    auto alloc2 = make_allocator(std::move(tracker2));


    LOG(INFO) << "init host_allocator";
    auto host = host_allocator(alloc1.shared());

    //auto tracker3 = make_tracked_allocator(log_tracker{"** tracker: high-level #3 **"}, std::move(host)););

}