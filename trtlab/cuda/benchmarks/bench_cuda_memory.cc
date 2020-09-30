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

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/transactional_allocator.h>
#include <trtlab/memory/literals.h>

#include <trtlab/cuda/common.h>
#include <trtlab/cuda/memory/cuda_allocators.h>

using namespace trtlab::memory;
using namespace literals;

namespace bench
{

template <typename Allocator>
static void run_allocate_node_x1(benchmark::State& state, Allocator& alloc)
{
    for(auto _ : state)
    {
        auto p0 = (int*)alloc.allocate_node(33_MiB, 256);
        benchmark::DoNotOptimize(p0);
        alloc.deallocate_node(p0, 33_MiB, 256);
    }
}

template <typename Allocator>
static void run_allocate_node_x10(benchmark::State& state, Allocator& alloc)
{
    for(auto _ : state)
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

        benchmark::DoNotOptimize(p0);
        benchmark::DoNotOptimize(p1);
        benchmark::DoNotOptimize(p2);
        benchmark::DoNotOptimize(p3);
        benchmark::DoNotOptimize(p4);
        benchmark::DoNotOptimize(p5);
        benchmark::DoNotOptimize(p6);
        benchmark::DoNotOptimize(p7);
        benchmark::DoNotOptimize(p8);
        benchmark::DoNotOptimize(p9);

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
}

template <typename Allocator>
static void run_allocate_descriptor_x1(benchmark::State& state, Allocator& alloc)
{
    for(auto _ : state)
    {
        auto p0 = alloc.allocate_descriptor(33_MiB, 8UL);
    }
}

template <typename Allocator>
static void run_allocate_descriptor_x10(benchmark::State& state, Allocator& alloc)
{
    for(auto _ : state)
    {
        auto p0 = alloc.allocate_descriptor(33_MiB, 8UL);
        auto p1 = alloc.allocate_descriptor(44_MiB, 8UL);
        auto p2 = alloc.allocate_descriptor(78_MiB, 8UL);
        auto p3 = alloc.allocate_descriptor(12_MiB, 8UL);
        auto p4 = alloc.allocate_descriptor(32_MiB, 8UL);
        auto p5 = alloc.allocate_descriptor(100_MiB, 8UL);
        auto p6 = alloc.allocate_descriptor(18_MiB, 8UL);
        auto p7 = alloc.allocate_descriptor(21_MiB, 8UL);
        auto p8 = alloc.allocate_descriptor(15_MiB, 8UL);
        auto p9 = alloc.allocate_descriptor(71_MiB, 8UL);
    }
}

static auto make_transactional_allocator()
{
    constexpr std::size_t block_size = 128_MiB;
    constexpr std::size_t block_count = 6;

    auto malloc  = make_allocator_adapter(cuda_malloc_allocator(0));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(malloc), block_size);
    auto counted = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), block_count);
    auto arena   = make_cached_block_arena(std::move(counted));
    auto alloc   = make_transactional_allocator(std::move(arena));    

    alloc.reserve_blocks(block_count);

    return alloc;
}
}

// malloc

static void memory_cuda_malloc_raw_x1(benchmark::State& state)
{
    auto alloc = make_allocator(cuda_malloc_allocator(0));
    bench::run_allocate_node_x1(state, alloc);
    CHECK_CUDA(cudaDeviceSynchronize());
}

static void memory_cuda_malloc_raw_x10(benchmark::State& state)
{
    auto alloc = make_allocator(cuda_malloc_allocator(0));
    bench::run_allocate_descriptor_x10(state, alloc);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// transactional

static void memory_transactional_raw_x1(benchmark::State& state)
{
    auto alloc = bench::make_transactional_allocator();
    bench::run_allocate_node_x1(state, alloc);
}

static void memory_transactional_raw_x10(benchmark::State& state)
{
    auto alloc = bench::make_transactional_allocator();
    bench::run_allocate_node_x10(state, alloc);
}

static void memory_transactional_allocator_x1(benchmark::State& state)
{
    auto txalloc = bench::make_transactional_allocator();
    auto alloc =   make_allocator(std::move(txalloc));
    bench::run_allocate_node_x1(state, alloc);
}

static void memory_transactional_allocator_x10(benchmark::State& state)
{
    auto txalloc = bench::make_transactional_allocator();
    auto alloc =   make_allocator(std::move(txalloc));
    bench::run_allocate_node_x10(state, alloc);
}

static void memory_transactional_descriptor_x1(benchmark::State& state)
{
    auto txalloc = bench::make_transactional_allocator();
    auto alloc =   make_allocator(std::move(txalloc));
    bench::run_allocate_descriptor_x1(state, alloc);
}

static void memory_transactional_descriptor_x10(benchmark::State& state)
{
    auto txalloc = bench::make_transactional_allocator();
    auto alloc =   make_allocator(std::move(txalloc));
    bench::run_allocate_descriptor_x10(state, alloc);
}

BENCHMARK(memory_cuda_malloc_raw_x1);
BENCHMARK(memory_cuda_malloc_raw_x10);

BENCHMARK(memory_transactional_raw_x1);
BENCHMARK(memory_transactional_allocator_x1);
BENCHMARK(memory_transactional_descriptor_x1);

BENCHMARK(memory_transactional_raw_x10);
BENCHMARK(memory_transactional_allocator_x10);
BENCHMARK(memory_transactional_descriptor_x10);