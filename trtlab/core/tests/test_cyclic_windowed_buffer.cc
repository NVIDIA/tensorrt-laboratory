/* Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "gtest/gtest.h"

#include <cstdlib>

#include <chrono>
#include <future>
#include <numeric>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/malloc_allocator.h>

#include <trtlab/core/thread_pool.h>
#include <trtlab/core/cyclic_buffer.h>
#include <trtlab/core/standard_threads.h>

using namespace trtlab;
using namespace trtlab::memory;

class TestWindowedBuffer : public ::testing::Test
{
};

TEST_F(TestWindowedBuffer, SynchronousNonOverlapping)
{
    auto alloc    = make_allocator(malloc_allocator());
    auto int_x100 = alloc.allocate_descriptor(100 * sizeof(int));

    // external state will the buffer will modify
    // no mutex needed in this case, but could be captured and passed
    auto sums = std::vector<long>();

    HostCyclicBuffer buffer(std::move(int_x100), 40, 0, [&sums](std::size_t id, const void* data, std::size_t bytes) -> auto {
        DCHECK_EQ(bytes % sizeof(int), 0);
        EXPECT_EQ(id, sums.size());
        auto array = static_cast<const int*>(data);
        auto count = bytes / sizeof(int);
        long sum   = 0;
        for (int i = 0; i < count; i++)
        {
            sum += array[i];
        }
        sums.push_back(sum);
        return [](bool) { return true; };
    });

    ASSERT_EQ(sums.size(), 0);

    std::vector<int> vec;
    vec.resize(10);

    for (int i = 0; i < 20; i++)
    {
        std::iota(vec.begin(), vec.end(), i);
        buffer.AppendData(vec.data(), vec.size() * sizeof(int));

        EXPECT_EQ(sums.size(), i + 1);
        EXPECT_EQ(sums[i], 45 + 10 * i);
    }
}

TEST_F(TestWindowedBuffer, SynchronousOverlapping)
{
    auto alloc    = make_allocator(malloc_allocator());
    auto int_x100 = alloc.allocate_descriptor(110 * sizeof(int));

    // external state will the buffer will modify
    // no mutex needed in this case, but could be captured and passed
    auto sums = std::vector<long>();

    HostCyclicBuffer buffer(std::move(int_x100), 40, 20, [&sums](std::size_t id, const void* data, std::size_t bytes) -> auto {
        DCHECK_EQ(bytes % sizeof(int), 0);
        EXPECT_EQ(id, sums.size());
        auto array = static_cast<const int*>(data);
        auto count = bytes / sizeof(int);
        long sum   = 0;
        for (int i = 0; i < count; i++)
        {
            sum += array[i];
        }
        sums.push_back(sum);
        return [](bool) { return true; };
    });

    ASSERT_EQ(sums.size(), 0);

    std::vector<int> vec;
    vec.resize(10);

    for (int i = 0; i < 40; i++)
    {
        std::iota(vec.begin(), vec.end(), i);
        buffer.AppendData(vec.data(), vec.size() * sizeof(int));
    }

    for (int i = 0; i < sums.size(); i++)
    {
        EXPECT_EQ(sums[i], 45 + 5 * i);
    }
}

template <typename T>
std::function<bool(bool)> SyncFnFromFuture(std::shared_future<T>&& shared)
{
    return [f = std::move(shared)](bool wait) -> bool {
        if (wait)
        {
            f.wait();
            return true;
        }
        auto rc = f.wait_for(std::chrono::nanoseconds(100));
        if (rc == std::future_status::ready)
        {
            return true;
        }
        return false;
    };
}

TEST_F(TestWindowedBuffer, AsynchronousNonOverlapping)
{
    auto alloc    = make_allocator(malloc_allocator());
    auto int_x100 = alloc.allocate_descriptor(100 * sizeof(int));

    ThreadPool thread_pool(5);

    // external state will the buffer will modify
    // no mutex needed in this case, but could be captured and passed
    auto sums = std::vector<long>();

    HostCyclicBuffer buffer(std::move(int_x100), 40, 0,
                            [&sums, &thread_pool ](std::size_t id, const void* data, std::size_t bytes) -> auto {
                                EXPECT_EQ(id, sums.size());
                                sums.push_back(-1);
                                auto index = id;
                                DVLOG(2) << "callback " << index << " handing off to thread pool";
                                auto future = thread_pool.enqueue([&sums, index, data, bytes]() {
                                    auto rand = 1 + std::rand() / ((RAND_MAX + 1u) / 10);
                                    DVLOG(2) << "sleep for " << rand << " millseconds";
                                    std::this_thread::sleep_for(std::chrono::milliseconds(rand));
                                    DCHECK_EQ(bytes % sizeof(int), 0);
                                    auto array = static_cast<const int*>(data);
                                    auto count = bytes / sizeof(int);
                                    long sum   = 0;
                                    for (int i = 0; i < count; i++)
                                    {
                                        sum += array[i];
                                    }
                                    DVLOG(2) << "sum[ " << index << " ]: " << sum;
                                    sums[index] = sum;
                                });
                                return SyncFnFromFuture(std::move(future.share()));
                            });

    ASSERT_EQ(sums.size(), 0);

    std::vector<int> vec;
    vec.resize(10);

    for (int i = 0; i < 20; i++)
    {
        std::iota(vec.begin(), vec.end(), i);
        buffer.AppendData(vec.data(), vec.size() * sizeof(int));
    }

    buffer.Sync();

    for (int i = 0; i < 20; i++)
    {
        EXPECT_EQ(sums[i], 45 + 10 * i);
    }
}

TEST_F(TestWindowedBuffer, AsynchronousOverlapping)
{
    auto alloc    = make_allocator(malloc_allocator());
    auto int_x100 = alloc.allocate_descriptor(100 * sizeof(int));

    ThreadPool thread_pool(5);

    // external state will the buffer will modify
    // no mutex needed in this case, but could be captured and passed
    auto sums = std::vector<long>();

    HostCyclicBuffer buffer(std::move(int_x100), 40, 20,
                            [&sums, &thread_pool ](std::size_t id, const void* data, std::size_t bytes) -> auto {
                                EXPECT_EQ(id, sums.size());
                                sums.push_back(-1);
                                auto index = id;
                                DVLOG(2) << "callback " << index << " handing off to thread pool";
                                auto future = thread_pool.enqueue([&sums, index, data, bytes]() {
                                    auto rand = 1 + std::rand() / ((RAND_MAX + 1u) / 10);
                                    DVLOG(2) << "sleep for " << rand << " millseconds";
                                    std::this_thread::sleep_for(std::chrono::milliseconds(rand));
                                    DCHECK_EQ(bytes % sizeof(int), 0);
                                    auto array = static_cast<const int*>(data);
                                    auto count = bytes / sizeof(int);
                                    long sum   = 0;
                                    for (int i = 0; i < count; i++)
                                    {
                                        sum += array[i];
                                    }
                                    DVLOG(2) << "sum[ " << index << " ]: " << sum;
                                    sums[index] = sum;
                                });
                                return SyncFnFromFuture(std::move(future.share()));
                            });

    ASSERT_EQ(sums.size(), 0);

    std::vector<int> vec;
    vec.resize(10);

    for (int i = 0; i < 20; i++)
    {
        std::iota(vec.begin(), vec.end(), i);
        buffer.AppendData(vec.data(), vec.size() * sizeof(int));
    }

    buffer.Sync();

    for (int i = 0; i < 20; i++)
    {
        EXPECT_EQ(sums[i], 45 + 5 * i);
    }
}

#include <trtlab/core/cyclic_windowed_buffer.h>

// make protected members public for testing
struct test_cw_stack : public ::trtlab::cyclic_windowed_stack<memory::host_memory, standard_threads>
{
    using cyclic_windowed_stack = ::trtlab::cyclic_windowed_stack<memory::host_memory, standard_threads>;

public:
    using cyclic_windowed_stack::available;
    using cyclic_windowed_stack::cyclic_windowed_stack;
    using cyclic_windowed_stack::push_window;
};

TEST_F(TestWindowedBuffer, Stack)
{
    auto alloc = make_allocator(malloc_allocator());
    auto md    = alloc.allocate_descriptor(100 * sizeof(int));

    test_cw_stack stack(std::move(md), 40, 20);

    EXPECT_EQ(stack.buffer().window_count(), 19);

    std::size_t sync_count = 0;
    auto        syncfn     = [&sync_count] {
        ++sync_count;
        DVLOG(1) << "syncfn called " << sync_count << " times";
    };

    EXPECT_EQ(stack.available(), 40);

    for (int i = 0; i < stack.buffer().window_count() - 1; i++)
    {
        stack.push_window(syncfn);
        EXPECT_EQ(sync_count, 0);
        EXPECT_EQ(stack.available(), 20);
    }

    // the push_window causes the stack to recycle
    // in this scenario, two syncs are needed:
    // 1) to reserve space for the 20 bytes of replicated data
    // 2) a second sync to free the remainder of the first window
    stack.push_window(syncfn);
    EXPECT_EQ(sync_count, 2);

    stack.push_window(syncfn);
    EXPECT_EQ(sync_count, 3);

    for (int i = 0; i < stack.buffer().window_count() * 3; i++)
    {
        stack.push_window(syncfn);
        EXPECT_EQ(stack.available(), 20);
    }
}

/*
TEST_F(TestWindowedBuffer, TaskExecutor)
{
    auto alloc = make_allocator(malloc_allocator());
    auto md    = alloc.allocate_descriptor(100 * sizeof(int));

    cyclic_windowed_stack<memory::host_memory, standard_threads> stack(std::move(md), 10*sizeof(int), 5*sizeof(int));

    // clang-format off
    cyclic_windowed_task_executor<memory::host_memory, standard_threads> buffer(std::move(stack),
        [](std::size_t id, const void* data, std::size_t count) -> auto {
            auto f = std::async([id] {
                std::this_thread::sleep_for(std::chrono::milliseconds(100*id));
                LOG(INFO) << "task " << id << " complete";
            }).share();
            return f;
        });
    // clang-format on 

    std::vector<int> source(100);
    std::iota(source.begin(), source.end(), 0);

    buffer.append_data(&source[0], 100*sizeof(int));
    buffer.append_data(&source[0], 100*sizeof(int));
}
*/

#include <trtlab/core/userspace_threads.h>

TEST_F(TestWindowedBuffer, Reservation)
{
    auto alloc = make_allocator(malloc_allocator());
    auto md    = alloc.allocate_descriptor(20 * sizeof(int));

    cyclic_windowed_stack<memory::host_memory, userspace_threads> base(std::move(md), 10 * sizeof(int), 8 * sizeof(int));

    cyclic_windowed_reserved_stack<memory::host_memory, userspace_threads> stack(std::move(base)); 

    for (int i = 0; i < stack.buffer().window_count() * 3; i++)
    {
        auto r = stack.reserve_window();
        int *top = static_cast<int*>(r.data_start);
        auto count = r.data_size / sizeof(int);
        LOG(INFO) << "reservation " << i << ": filling " << count << " entries";
        std::fill(top, top + count, i);
        top = static_cast<int*>(r.window_start);
        for(int i=0; i<10; i++)
        {
            LOG(INFO) << *(top + i);
        }
        r.release();
    }
}