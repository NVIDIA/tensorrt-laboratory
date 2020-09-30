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

#include "trtlab/core/hybrid_mutex.h"

#include "trtlab/core/memory.h"

#include <foonathan/memory/namespace_alias.hpp>

using namespace trtlab;
using namespace trtlab;

static void allocators_transactional_raw(benchmark::State& state)
{
    using namespace memory::literals;

    auto alloc = memory::make_allocator_adapter(memory::malloc_allocator());
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 4u, std::move(alloc));
    auto trans_alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    trans_alloc.reserve_blocks(4u);

    for(auto _ : state)
    {
        for(int i=0; i < state.range(0); i++)
        {
            auto ptr = trans_alloc.allocate_node(1024, 64);
            trans_alloc.deallocate_node(ptr, 0u, 0u);
        }
    }
}

static void allocators_transactional_std(benchmark::State& state)
{
    using namespace memory::literals;

    auto alloc = memory::make_allocator_adapter(memory::malloc_allocator());
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 4u, std::move(alloc));
    auto trans_alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    trans_alloc.reserve_blocks(4u);

    auto smart = memory::trtlab::make_allocator(std::move(trans_alloc));

    for(auto _ : state)
    {
        for(int i=0; i < state.range(0); i++)
        {
            auto ptr = smart.allocate_node(1024, 64);
            smart.deallocate_node(ptr, 0u, 0u);
        }
    }
}

static void allocators_transactional_md(benchmark::State& state)
{
    using namespace memory::literals;

    auto alloc = memory::make_allocator_adapter(memory::malloc_allocator());
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(1_MiB, 4u, std::move(alloc));
    auto trans_alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    trans_alloc.reserve_blocks(4u);

    auto smart = memory::trtlab::make_allocator(std::move(trans_alloc));

    for(auto _ : state)
    {
        for(int i=0; i < state.range(0); i++)
        {
            auto md = smart.allocate_descriptor(1024, 64);
        }
    }
}

#if 0
template<typename T>
using custom_vector = std::vector<T, stl::temporary_allocator<T, Malloc>>;

template<typename T, typename RawAllocator>
auto make_vector(RawAllocator& alloc)
{
    using std_allocator = memory::std_allocator<T, RawAllocator>;
    return std::vector<T, std_allocator>(std_allocator(alloc));
}


static void BM_vector_transactional(benchmark::State& state)
{
    using namespace memory::literals;

    auto malloc = memory::make_allocator_reference(memory::MallocAllocator());
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(128_MiB, 8, std::move(malloc));
    auto trans_alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    trans_alloc.reserve_blocks(8);

    for(auto _ : state)
    {
        auto vec = make_vector<int>(trans_alloc);
        vec.reserve(1024*1024*8);
    }
}

static void BM_vector_smart_transactional(benchmark::State& state)
{
    using namespace memory::literals;

    auto malloc = memory::make_allocator_reference(memory::MallocAllocator());
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(128_MiB, 8, std::move(malloc));
    auto trans_alloc = memory::trtlab::make_transactional_allocator(std::move(block_alloc));
    trans_alloc.reserve_blocks(8);

    auto smart = memory::trtlab::make_allocator<std::mutex>(std::move(trans_alloc));
    
    for(auto _ : state)
    {
        auto vec = memory::trtlab::make_vector<int>(smart);
        vec.reserve(1024*1024*8);
    }
}


static void BM_CyclicAllocator_stl_allocator(benchmark::State& state)
{
    {
        auto v0 = custom_vector<int>(1024);
    }
    for(auto _ : state)
    {
        custom_vector<int> vector;
        vector.reserve(1024*1024*8);
    }
}

static void BM_CyclicAllocator_stl_allocator2(benchmark::State& state)
{
    size_t ctr = 1024;
    for(auto _ : state)
    {
        custom_vector<int> v3;
        v3.reserve(ctr*ctr*8);
    }
}

static void BM_vector_default(benchmark::State& state)
{
    size_t ctr = 1024;
    for(auto _ : state)
    {
        std::vector<int> vec;
        vec.reserve(1024*1024*8);
    }
}

static void BM_stl_allocator_ctor(benchmark::State& state)
{
    for(auto _ : state)
    {
        auto a = stl::temporary_allocator<int, Malloc>();
    }
}

static void BM_stl_allocator_allocate_lifecycle(benchmark::State& state)
{
    for(auto _ : state)
    {
        auto a = stl::temporary_allocator<int, Malloc>();
        auto i = a.allocate(1024);
        a.deallocate(i, 1024);
    }
}
#endif

BENCHMARK(allocators_transactional_raw)->RangeMultiplier(2)->Range(1, 1 << 2);
BENCHMARK(allocators_transactional_std)->RangeMultiplier(2)->Range(1, 1 << 0);
BENCHMARK(allocators_transactional_md)->RangeMultiplier(2)->Range(1, 1 << 0);
