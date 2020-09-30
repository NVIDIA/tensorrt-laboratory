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
#include <benchmark/benchmark.h>

#include <map>

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/block_arena.h>
#include <trtlab/memory/malloc_allocator.h>
#include <trtlab/memory/memory_pool.h>
#include <trtlab/memory/std_allocator.h>
#include <trtlab/memory/literals.h>

using namespace trtlab::memory;
using namespace literals;

static void memory_pool_map_default(benchmark::State& state)
{
    std::map<std::size_t, std::size_t> m;

    for(auto _ : state)
    {
        auto upper = state.range(0);
        for(std::size_t i=0; i<upper; i++)
        {
            m[i] = i;
        }
        m.clear();
    }
}

static void memory_pool_map_malloc_raw(benchmark::State& state)
{
    auto malloc = malloc_allocator();

    // create map
    auto m = std::map<std::size_t, std::size_t, std::less<std::size_t>, std_allocator<std::pair<std::size_t, std::size_t>, decltype(malloc)>>(malloc);

    for(auto _ : state)
    {
        auto upper = state.range(0);
        for(std::size_t i=0; i<upper; i++)
        {
            m[i] = i;
        }
        m.clear();
    }
}

template<typename Key, typename Value, typename BlockAllocator>
auto make_map(BlockAllocator&& block_alloc)
{
    static_assert(is_block_allocator<BlockAllocator>::value, "");

    using node_type = std::pair<Key, Value>;
    auto node_size = alignof(node_type) + sizeof(node_type) + 64;

    auto stack = make_block_stack<uncached>(std::move(block_alloc));
    auto pool  = memory_pool<BlockAllocator>(node_size, std::move(stack));
    return pool;
    //auto alloc = make_thread_unsafe_allocator(std::move(pool));

    //return std::map<Key, Value, std::less<Key>, std_allocator<node_type, decltype(alloc)>>(alloc);
}

static void memory_pool_map_malloc_pooled(benchmark::State& state)
{
    auto malloc = make_allocator_adapter(malloc_allocator());
    auto block  = make_block_allocator<single_block_allocator>(std::move(malloc), 4_MiB);

    // create map
    auto p = make_map<std::size_t, std::size_t>(std::move(block));

    // create map
    auto m = std::map<std::size_t, std::size_t, std::less<std::size_t>, std_allocator<std::pair<std::size_t, std::size_t>, decltype(p)>>(p);


    for(auto _ : state)
    {
        auto upper = state.range(0);
        for(std::size_t i=0; i<upper; i++)
        {
            m[i] = i;
        }
        m.clear();
    }
}

BENCHMARK(memory_pool_map_default)->RangeMultiplier(2)->Range(1<<0, 1<<12);
BENCHMARK(memory_pool_map_malloc_raw)->RangeMultiplier(2)->Range(1<<0, 1<<12);
BENCHMARK(memory_pool_map_malloc_pooled)->RangeMultiplier(2)->Range(1<<0, 1<<12);