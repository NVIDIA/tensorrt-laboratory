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
#include "trtlab/core/hybrid_condition.h"
#include "trtlab/core/hybrid_mutex.h"
#include "trtlab/core/thread_pool.h"
#include <benchmark/benchmark.h>

static void BM_ThreadPool_Enqueue(benchmark::State& state)
{
    using trtlab::ThreadPool;
    auto pool = std::make_unique<ThreadPool>(1);

    for(auto _ : state)
    {
        CHECK(pool);
        // enqueue only
        auto future = pool->enqueue([] {});
        //future.get();
    }
}
BENCHMARK(BM_ThreadPool_Enqueue)->UseRealTime();

static void BM_HybridThreadPool_Enqueue(benchmark::State& state)
{
    using trtlab::BaseThreadPool;
    auto pool = std::make_unique<BaseThreadPool<hybrid_mutex, hybrid_condition>>(1);

    for(auto _ : state)
    {
        auto future = pool->enqueue([] {});
    }
}
BENCHMARK(BM_HybridThreadPool_Enqueue)->UseRealTime();