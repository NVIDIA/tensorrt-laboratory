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

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/block_allocators.h"
#include "trtlab/core/memory/containers.h"
#include "trtlab/core/memory/transactional_allocator.h"
#include "trtlab/core/memory/host_first_touch_allocator.h"

#include <foonathan/memory/aligned_allocator.hpp>
#include <foonathan/memory/container.hpp>
#include <foonathan/memory/smart_ptr.hpp>
#include <foonathan/memory/tracking.hpp>
#include <foonathan/memory/namespace_alias.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>

#include "test_common.h"

using namespace memory::literals;



template<typename T, typename RawAllocator>
auto make_vector(RawAllocator&& alloc)
{
    return memory::vector<T, RawAllocator>(alloc);
}

template<typename T, typename RawAllocator, typename Mutex>
auto make_vector(memory::trtlab::allocator<RawAllocator, Mutex> alloc)
{
    return memory::vector<T, memory::trtlab::allocator<RawAllocator, Mutex>>(alloc);
}

class TestFooMemory : public TrackedTest {};

TEST_F(TestFooMemory, BlankTest) 
{
    DVLOG(1) << "FooMemory testing";
    ASSERT_TRUE(true); 
}

TEST_F(TestFooMemory, Malloc)
{
    auto raw = memory::malloc_allocator();
    auto alloc = memory::make_allocator_adapter(std::move(raw));
    auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(alloc));

    auto p0 = tracked.allocate_node(1024, 8);

    ASSERT_EQ(log_tracker::node_total, 1024);
    ASSERT_EQ(log_tracker::node_count, 1);

    tracked.deallocate_node(p0, 1024, 8);

    EndTest();
}

TEST_F(TestFooMemory, MallocTraits)
{
    auto raw = memory::malloc_allocator();
    auto adp = memory::make_allocator_adapter(std::move(memory::malloc_allocator()));
    auto ref = memory::make_allocator_reference(raw);
    auto mt_ref = memory::make_allocator_reference<std::mutex>(raw);
    auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, ref);

    // default malloc implementation does not provide a memory_type
    static_assert(std::is_same<decltype(adp)::memory_type, memory::host_memory>::value, "should be host_memory");
    static_assert(std::is_same<decltype(ref)::memory_type, memory::host_memory>::value, "should be host_memory");
    static_assert(std::is_same<decltype(mt_ref)::memory_type, memory::host_memory>::value, "should be host_memory");
    static_assert(std::is_same<decltype(tracked)::memory_type, memory::host_memory>::value, "should be host_memory");

    DVLOG(1) << "access allocator traits from adapter";
    auto context_adp = adp.device_context();
    ASSERT_EQ(context_adp.device_type, kDLCPU);
    ASSERT_EQ(context_adp.device_id, 0);

    DVLOG(1) << "access allocator traits from reference";
    auto context_ref = ref.device_context();
    ASSERT_EQ(context_ref.device_type, kDLCPU);
    ASSERT_EQ(context_ref.device_id, 0);

    DVLOG(1) << "access allocator traits from mt reference";
    auto context_mt_ref = mt_ref.device_context();
    ASSERT_EQ(context_mt_ref.device_type, kDLCPU);
    ASSERT_EQ(context_mt_ref.device_id, 0);

    DVLOG(1) << "access allocator traits from tracker";
    auto context_trk = tracked.device_context();
    ASSERT_EQ(context_trk.device_type, kDLCPU);
    ASSERT_EQ(context_trk.device_id, 0);

    EndTest();
}

TEST_F(TestFooMemory, MallocAsStdAllocator)
{
    auto raw = memory::malloc_allocator();
    auto ref = memory::make_allocator_reference(raw);
    auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, ref);

    std::size_t count = 64;
    auto vec = memory::vector<std::size_t, decltype(tracked)>(tracked);
    vec.reserve(count);

    ASSERT_GE(log_tracker::node_total, count * sizeof(std::size_t));
    ASSERT_GE(log_tracker::node_count, count);

    std::size_t tmp_total = log_tracker::node_total;

    for(std::size_t i=0; i<count; i++)
    {
        vec.push_back(i);
    }

    ASSERT_EQ(log_tracker::node_total, tmp_total);

    DVLOG(1) << "pushing beyond vector capacity - expect new alloc/dealloc/copy";
    vec.push_back(count+1);
    DVLOG(1) << "^^^ should see alloc/dealloc messages ^^^";

    ASSERT_GT(log_tracker::node_total, tmp_total);

    EndTest();
}

TEST_F(TestFooMemory, MallocThreadSafe)
{
    auto raw = memory::malloc_allocator();
    auto ref = memory::make_allocator_reference<std::mutex>(raw);
    auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, ref);

    std::size_t count = 64;
    auto vec = memory::vector<std::size_t, decltype(tracked)>(tracked);
    vec.reserve(count);

    ASSERT_GE(log_tracker::node_total, count * sizeof(std::size_t));
    ASSERT_GE(log_tracker::node_count, count);

    std::size_t tmp_total = log_tracker::node_total;

    for(std::size_t i=0; i<count; i++)
    {
        vec.push_back(i);
    }

    ASSERT_EQ(log_tracker::node_total, tmp_total);

    DVLOG(1) << "pushing beyond vector capacity - expect new alloc/dealloc/copy";
    vec.push_back(count+1);
    DVLOG(1) << "^^^ should see alloc/dealloc messages ^^^";

    ASSERT_GT(log_tracker::node_total, tmp_total);

    EndTest();
}

TEST_F(TestFooMemory, GrowthCappedBlockAllocator)
{

    // base allocator
    auto malloc_raw = memory::malloc_allocator();
    auto malloc_ref = memory::make_allocator_reference(malloc_raw);
    auto malloc_tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, malloc_ref);

    // malloc block allocator
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 2, malloc_tracked);

    auto context = block_alloc.device_context();
    ASSERT_EQ(context.device_type, kDLCPU);
    ASSERT_EQ(context.device_id, 0);

    DVLOG(1) << "allocate first block - should pass";
    auto block_0 = block_alloc.allocate_block();
    ASSERT_NE(block_0.memory, nullptr);
    ASSERT_EQ(block_0.size, 1_MiB);

    DVLOG(1) << "allocate second block - should pass";
    auto block_1 = block_alloc.allocate_block();
    ASSERT_NE(block_1.memory, nullptr);
    ASSERT_EQ(block_1.size, 1_MiB);

    DVLOG(1) << "allocate third block - should throw an exception";
    ASSERT_ANY_THROW(auto block_2 = block_alloc.allocate_block());

    DVLOG(1) << "increasing limit to 3; allocate third block - should pass";
    block_alloc.set_max_block_count(3);
    auto block_2 = block_alloc.allocate_block();

    DVLOG(1) << "deallocate blocks";
    block_alloc.deallocate_block(block_0);
    block_alloc.deallocate_block(block_1);
    block_alloc.deallocate_block(block_2);

    EndTest();
}

TEST_F(TestFooMemory, BlockArena)
{

    // base allocator
    auto malloc_raw = memory::malloc_allocator();
    auto malloc_ref = memory::make_allocator_reference(malloc_raw);
    auto malloc_tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, malloc_ref);

    // malloc block allocator
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 2, malloc_tracked);

    ASSERT_EQ(log_tracker::node_count, 0);
    
    // arena with caching
    auto arena = memory::trtlab::make_block_arena(std::move(block_alloc));

    arena.reserve_blocks(2);
    ASSERT_EQ(log_tracker::node_count, 2);

    DVLOG(1) << "allocate first block - should pass";
    auto block_0 = arena.allocate_block();
    ASSERT_NE(block_0.memory, nullptr);
    ASSERT_EQ(block_0.size, 1_MiB);
    ASSERT_EQ(log_tracker::node_count, 2);

    DVLOG(1) << "allocate second block - should pass";
    auto block_1 = arena.allocate_block();
    ASSERT_NE(block_1.memory, nullptr);
    ASSERT_EQ(block_1.size, 1_MiB);
    ASSERT_EQ(log_tracker::node_count, 2);

    DVLOG(1) << "allocate third block - should throw an exception";
    ASSERT_ANY_THROW(auto block_2 = arena.allocate_block());
    ASSERT_EQ(log_tracker::node_count, 2);

    DVLOG(1) << "increasing limit to 3; allocate third block - should pass";
    arena.get_block_allocator().set_max_block_count(3);
    auto block_2 = arena.allocate_block();
    ASSERT_EQ(log_tracker::node_count, 3);

    DVLOG(1) << "deallocate blocks";

    DVLOG(1) << "deallocate block_0; then force the cache to shrink_to_fit";
    arena.deallocate_block(block_0);
    ASSERT_EQ(log_tracker::node_count, 3);
    arena.shrink_to_fit();
    ASSERT_EQ(log_tracker::node_count, 2);

    DVLOG(1) << "deallocate remaining blocks";
    arena.deallocate_block(block_1);
    arena.deallocate_block(block_2);
    ASSERT_EQ(log_tracker::node_count, 2);
    DVLOG(1) << "there should be two deallocations after the * end of test * from the block_arena destructor";

    EndTest();
}

template<typename StatelessAllocator>
auto make_raw_transactional_allocator(std::size_t block_size, std::size_t block_count = 2)
{
    // base allocator
    auto raw = StatelessAllocator();

    // convert to full fledged allocator - use direct_storage which optimizes out mutexes for stateless allocators
    auto alloc = memory::make_allocator_adapter(std::move(raw));

    static_assert(!decltype(alloc)::is_stateful::value, "should be stateless");
    static_assert(std::is_same<memory::no_mutex, typename decltype(alloc)::mutex>::value, "should use memory::no_mutex");

    // create a tracker for calls to the malloc allocator
    auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: base **"}, std::move(alloc));

    // malloc block allocator
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(block_size, block_count, std::move(tracked));

    // transactional allocator
    return memory::trtlab::make_transactional_allocator(std::move(block_alloc));
}

template<typename StatelessAllocator = memory::malloc_allocator>
auto make_smart_transactional_allocator(std::size_t block_size, std::size_t block_count = 2)
{
    // transactional allocator
    auto alloc = make_raw_transactional_allocator<StatelessAllocator>(block_size, block_count);

    // populate the cache
    alloc.reserve_blocks(block_count);

    // smart allocator
    // use a special timeout_mutex - throws an exception if the lock is not obtained in MUTEX_TIMEOUT_MS
    return memory::trtlab::make_allocator<timeout_mutex>(std::move(alloc));
}

TEST_F(TestFooMemory, TransactionalLifeCycle)
{

    auto smart = make_smart_transactional_allocator(1_MiB);

    static_assert(decltype(smart)::is_stateful::value, "should be stateful");
    static_assert(std::is_same<decltype(smart)::mutex, timeout_mutex>::value, "should be a timeout mutex");

    // we can safely make copies of allocators
    // because all allocators hold a shared_ptr to the underlying allocator storage (raw allocator + mutex)
    auto tracker = memory::make_tracked_allocator(log_tracker{"** transactional **"}, smart.copy());

    static_assert(decltype(tracker)::is_stateful::value, "should be stateful");

    {
        DVLOG(1) << "make some shared_ptr<int> objects with the reference";
        auto int_0 = memory::allocate_shared<int>(smart, 2);
        auto int_1 = memory::allocate_shared<int>(smart, 4);
    }

    {
        auto lock = smart.lock();
        DVLOG(1) << "got lock - should hang if trying to use the allocator";
        ASSERT_THROW(auto int_0 = memory::allocate_shared<int>(smart, 2), timeout_error);
    }

    {
        DVLOG(1) << "make some shared_ptr<int> objects with the tracker";
        auto int_0 = memory::allocate_shared<int>(tracker, 2);
        auto int_1 = memory::allocate_shared<int>(tracker, 4);
    }

    {
        auto lock = smart.lock();
        ASSERT_THROW(auto int_0 = memory::allocate_shared<int>(tracker, 2), timeout_error);
    }

    EndTest();
}


TEST_F(TestFooMemory, TransactionalAllocatorFullyCaptured)
{
    auto trans = make_raw_transactional_allocator<memory::malloc_allocator>(1_MiB);

    auto ptr = trans.allocate_node(1024*1024, 8);

    DVLOG(1) << ptr;

    trans.deallocate_node(ptr, 1024*1024, 8);

    EndTest();
}



TEST_F(TestFooMemory, SmartTransactionalLifeCycle)
{
    auto smart = make_smart_transactional_allocator(1_MiB, 2);

    // smart allocators should be shared
    static_assert(memory::is_shared_allocator<decltype(smart)>::value, "not shared");

    // ensure we can reach the raw allocator
    smart.get_raw_allocator().get_block_allocator().set_max_block_count(3);

    // basic allocation
    auto ptr = smart.allocate_node(1024, 8);
    ASSERT_NE(ptr, nullptr) << "should have thrown an exception or a valid ptr";
    smart.deallocate_node(ptr, 1024, 8);

    EndTest();
}

TEST_F(TestFooMemory, SmartTransactionalBase)
{
    auto smart = make_smart_transactional_allocator(1_MiB, 2);

    // smart allocators should be shared
    static_assert(memory::is_shared_allocator<decltype(smart)>::value, "not shared");

    EndTest();
}

TEST_F(TestFooMemory, SmartTransactionalDescriptor)
{
    auto smart = make_smart_transactional_allocator(1_MiB, 2);

    // smart allocators should be shared
    static_assert(memory::is_shared_allocator<decltype(smart)>::value, "not shared");

    // a smart descriptor holds a shared_ptr to the allocator
    DVLOG(1) << "create descriptor";
    auto md = smart.allocate_descriptor(1024, 8);
    DVLOG(1) << "created descriptor";
    ASSERT_EQ(smart.use_count(), 2);
    ASSERT_NE(md.data(), nullptr);
    ASSERT_EQ(md.size(), 1024);
    ASSERT_EQ(md.device_context().device_type, kDLCPU);

    // a smart descriptor is only moveable, not copyable
    DVLOG(1) << "move descriptor";
    auto moved_md = std::move(md);
    ASSERT_EQ(md.data(), nullptr);
    ASSERT_EQ(smart.use_count(), 2);
    static_assert(!std::is_copy_constructible<decltype(md)>::value, "should not be copyable");
    static_assert(!std::is_copy_assignable<decltype(md)>::value, "should not be copyable");

    // to make a smart descriptor copyable, you can create a shared_ptr to it
    DVLOG(1) << "convert to shared descriptor";
    auto shared_md = moved_md.make_shared();
    ASSERT_EQ(smart.use_count(), 2);
    ASSERT_EQ(shared_md.use_count(), 1);

    // copying the descriptor does not increment the ref count of the allocator
    // the descriptor holds a shared_ptr to the allocator,
    // but the descriptor is not copied, rather it becomes a shared pointer
    auto copied_md = shared_md;
    ASSERT_EQ(smart.use_count(), 2);
    ASSERT_EQ(shared_md.use_count(), 2);

    EndTest();
}

TEST_F(TestFooMemory, AlignedSmartAllocator)
{

    // base allocator
    auto malloc_raw = memory::malloc_allocator();

    // convert to full fledged allocator - use direct_storage which optimizes out mutexes for stateless allocators
    auto malloc_alloc = memory::make_allocator_adapter(std::move(malloc_raw));

    static_assert(!decltype(malloc_alloc)::is_stateful::value, "should be stateless");
    static_assert(std::is_same<memory::no_mutex, typename decltype(malloc_alloc)::mutex>::value, "should use memory::no_mutex");

    // create a tracker for calls to the malloc allocator
    auto malloc_tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(malloc_alloc));

    // malloc block allocator
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 2, std::move(malloc_tracked));

    // transactional allocator
    auto alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    alloc.reserve_blocks(2);

    auto aligned = memory::make_aligned_allocator(256, std::move(alloc));

    auto smart = memory::trtlab::make_allocator(std::move(aligned));

    auto p0 = smart.allocate_node(64, 8);
    auto p1 = smart.allocate_node(64, 8);

    auto i0 = std::uintptr_t(p0);
    auto i1 = std::uintptr_t(p1);

    ASSERT_EQ(i0 % 256, 0);
    ASSERT_EQ(i1 % 256, 0);

    ASSERT_GE(i1 - i0, 256);

    smart.deallocate_node(p0, 64, 0);
    smart.deallocate_node(p1, 64, 0);

    EndTest();
}

TEST_F(TestFooMemory, SmartAllocatorStateful)
{

    void *ptr;

    // base allocator
    auto malloc_raw = memory::malloc_allocator();

    // convert to full fledged allocator - use direct_storage which optimizes out mutexes for stateless allocators
    auto malloc_alloc = memory::make_allocator_adapter(std::move(malloc_raw));

    static_assert(!decltype(malloc_alloc)::is_stateful::value, "should be stateless");
    static_assert(std::is_same<memory::no_mutex, typename decltype(malloc_alloc)::mutex>::value, "should use memory::no_mutex");

    // create a tracker for calls to the malloc allocator
    auto malloc_tracked = memory::make_tracked_allocator(log_tracker{"** tracker: malloc **"}, std::move(malloc_alloc));

    // malloc block allocator
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 2, std::move(malloc_tracked));

    // transactional allocator
    auto alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    alloc.reserve_blocks(2);

    // smart allocator
    // use a special timeout_mutex - throws an exception if the lock is not obtained in MUTEX_TIMEOUT_MS
    auto smart = memory::trtlab::make_allocator<timeout_mutex>(std::move(alloc));
    ASSERT_EQ(smart.use_count(), 1);

    // smart allocators should be shared
    static_assert(memory::is_shared_allocator<decltype(smart)>::value, "not shared");

    // ensure we can reach the raw allocator
    smart.get_raw_allocator().get_block_allocator().set_max_block_count(3);

    // basic allocation
    ptr = smart.allocate_node(1024, 8);
    ASSERT_NE(ptr, nullptr) << "should have thrown an exception or a valid ptr";
    smart.deallocate_node(ptr, 1024, 8);

    // smart descriptor
    {
        // a smart descriptor holds a shared_ptr to the allocator
        auto md = smart.allocate_descriptor(1024, 8);
        ASSERT_EQ(smart.use_count(), 2);
        ASSERT_NE(md.data(), nullptr);
        ASSERT_EQ(md.size(), 1024);

        // a smart descriptor is only moveable, not copyable
        auto moved_md = std::move(md);
        ASSERT_EQ(md.data(), nullptr);
        ASSERT_EQ(smart.use_count(), 2);
        static_assert(!std::is_copy_constructible<decltype(md)>::value, "should not be copyable");
        static_assert(!std::is_copy_assignable<decltype(md)>::value, "should not be copyable");

        // to make a smart descriptor copyable, you can create a shared_ptr to it
        auto shared_md = moved_md.make_shared();
        ASSERT_EQ(smart.use_count(), 2);
        ASSERT_EQ(shared_md.use_count(), 1);

        // copying the descriptor does not increment the ref count of the allocator
        // the descriptor holds a shared_ptr to the allocator,
        // but the descriptor is not copied, rather it becomes a shared pointer
        auto copied_md = shared_md;
        ASSERT_EQ(smart.use_count(), 2);
        ASSERT_EQ(shared_md.use_count(), 2);
    }

    // smart allocators are copyable
    auto smart_copy = smart;
    ASSERT_EQ(smart.use_count(), 2);
    ASSERT_EQ(smart_copy.use_count(), 2);

    {
        auto lock = smart.lock();
        ASSERT_EQ(smart.use_count(), 2);
        ASSERT_EQ(smart_copy.use_count(), 2);

        ptr = lock->allocate_node(1024, 8);
        lock->deallocate_node(ptr, 1024, 8);

        ASSERT_THROW(ptr = smart_copy.allocate_node(1024, 8), timeout_error);
    }

    // smart allocators are also moveable
    auto smart_move = std::move(smart_copy);
    ASSERT_EQ(smart.use_count(), 2);
    ASSERT_EQ(smart_move.use_count(), 2);
    ASSERT_EQ(smart_copy.use_count(), 0);

    // create a tracked allocator
    // if we were not explicity about the copy, smart would have been moved
    {
        auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: smart **"}, smart.copy());
        ASSERT_EQ(smart.use_count(), 3);

        ptr = tracked.allocate_node(1024, 8);
        tracked.deallocate_node(ptr, 1024, 8);

        auto lock = smart.lock();
        ASSERT_THROW(ptr = tracked.allocate_node(1024, 8), timeout_error);
    }

    ASSERT_EQ(smart.use_count(), 2);

    // without the explicit copy, the passed allocator is moved into the tracker
    {
        auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: smart **"}, smart_move);
        ASSERT_EQ(smart.use_count(), 2);
        ASSERT_EQ(smart_move.use_count(), 0);

        ptr = tracked.allocate_node(1024, 8);
        tracked.deallocate_node(ptr, 1024, 8);

        // tracked.lock() is inaccessible
        auto lock = tracked.get_allocator().lock();
        ASSERT_THROW(ptr = smart.allocate_node(1024, 8), timeout_error);
    }

    ASSERT_EQ(smart.use_count(), 1);

    //

    {
        // pass by reference which for a shared allocator would call the copy constructor
        auto v = memory::trtlab::make_vector<int>(smart);
        ASSERT_EQ(smart.use_count(), 2);
        v.reserve(1024);
    }

    ASSERT_EQ(smart.use_count(), 1);

    {
        // this fails to move the smart allocator into the std_allocator owned by the vector
        auto v = memory::trtlab::make_vector<int>(std::move(smart));
        ASSERT_EQ(smart.use_count(), 2);
        v.reserve(1024);
    }

    ASSERT_EQ(smart.use_count(), 1);

    // test smart pointers

    {
        auto shared = memory::allocate_shared<int>(smart, 1);
        ASSERT_EQ(smart.use_count(), 2);
    }

    ASSERT_EQ(smart.use_count(), 1);


    EndTest();
}

TEST_F(TestFooMemory, SmartAllocatorStateless)
{
    void *ptr;

    // base allocator
    auto malloc_raw = memory::malloc_allocator();
    auto malloc_smart = memory::trtlab::make_allocator(std::move(malloc_raw));

    ASSERT_EQ(malloc_smart.use_count(), 1);

    ptr = malloc_smart.allocate_node(1024, 8);
    malloc_smart.deallocate_node(ptr, 1024, 8);

    auto smart_copy = malloc_smart;
    ASSERT_EQ(malloc_smart.use_count(), 2);
    ASSERT_EQ(smart_copy.use_count(), 2);

    {
        auto lock = malloc_smart.lock();
        ASSERT_EQ(malloc_smart.use_count(), 2);
        ASSERT_EQ(smart_copy.use_count(), 2);

        ptr = lock->allocate_node(1024, 8);
        lock->deallocate_node(ptr, 1024, 8);

        DVLOG(1) << "this should not hang - stateless allocators, even if requesting std::mutex use no_mutex";
        ptr = smart_copy.allocate_node(1024, 8);
        smart_copy.deallocate_node(ptr, 1024, 8);
    }

    EndTest();
}


struct backend {};
struct backend_a : backend {};
struct backend_b : backend_a {};
struct backend_c : backend {};

template<typename BackendType>
class md : public BackendType
{
  public:
    md() : m_ptr(nullptr) {}
    md(void* ptr) : m_ptr(ptr) {}

  private:
    void* m_ptr;
};


template<typename BackendType>
void do_a(md<BackendType>& require_a_or_derived_from_a)
{
    static_assert(std::is_base_of<backend_a, md<BackendType>>::value, "Backend needs to be derived from backend_a");
    DVLOG(1) << "yep";
}


TEST_F(TestFooMemory, TemplateInheritance)
{
    static_assert(std::is_base_of<backend, md<backend>>::value, "should be true");
    static_assert(std::is_base_of<backend_a, md<backend_b>>::value, "should be true");

    auto a = std::move(md<backend_a>());
    auto b = std::move(md<backend_b>());
    auto c = std::move(md<backend_c>());

    do_a(a);
    do_a(b);

    // will fail to compile
    //do_a(c);
}

struct pinned_memory : memory::host_memory
{
    static constexpr DLDeviceType device_type() { return kDLCPUPinned; }
};

#include <typeinfo>

TEST_F(TestFooMemory, OtherMemoryTypes)
{
    ASSERT_EQ(memory::host_memory::device_type(), kDLCPU);
    ASSERT_EQ(pinned_memory::device_type(), kDLCPUPinned);
    ASSERT_EQ(pinned_memory::min_alignment(), 8UL);
}


/*

min_alignment CAN be overridden, but ONLY IF the value is GREATER than the base memory_type.

In this test, malloc_allocator has memory_type == host_memory which has a default min_alignment of 8.

1024 CAN be used as a min_alignment override.
1 CANNOT be used as a min_alignment override.

*/

struct malloc_allocator_1024 : public memory::malloc_allocator
{
    constexpr static std::size_t min_alignment() { return 1024UL; }
};

struct malloc_allocator_1 : public memory::malloc_allocator
{
    constexpr static std::size_t min_alignment() { return 1UL; }
};

TEST_F(TestFooMemory, MinAlignment)
{
    auto raw_1 = malloc_allocator_1();
    auto malloc_1 = memory::trtlab::make_allocator(std::move(raw_1));

    auto raw = memory::malloc_allocator();
    auto malloc_8 = memory::trtlab::make_allocator(std::move(raw));

    auto raw_1024 = malloc_allocator_1024();
    auto malloc_1024 = memory::trtlab::make_allocator(std::move(raw_1024));

    ASSERT_EQ(malloc_1.min_alignment(), 8UL);
    ASSERT_EQ(malloc_8.min_alignment(), 8UL);
    ASSERT_EQ(malloc_1024.min_alignment(), 1024UL);
}


TEST_F(TestFooMemory, FirstTouch)
{
    trtlab::CpuSet cpus = trtlab::Affinity::GetCpusFromString("0,1");
    auto raw = memory::malloc_allocator();
    auto ft = memory::trtlab::make_first_touch_allocator(cpus, std::move(raw));
    auto alloc = memory::trtlab::make_allocator(std::move(ft));

    auto md = alloc.allocate_descriptor(1_MiB);
}