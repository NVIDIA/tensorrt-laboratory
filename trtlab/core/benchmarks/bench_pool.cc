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
#include "trtlab/core/pool.h"
#include "trtlab/core/userspace_threads.h"
#include <benchmark/benchmark.h>

static void BM_Pool_v1_Pop(benchmark::State& state)
{
    using trtlab::v1::Pool;
    struct Object
    {
    };
    auto pool = Pool<Object>::Create();
    pool->EmplacePush(new Object);

    for(auto _ : state)
    {
        auto obj = pool->Pop();
    }
}

static void BM_Pool_v2_Pop(benchmark::State& state)
{
    using trtlab::v2::Pool;
    struct Object
    {
    };
    auto pool = Pool<Object>::Create();
    pool->EmplacePush();

    for(auto _ : state)
    {
        auto obj = pool->Pop();
    }
}

static void BM_Pool_v3_Pop(benchmark::State& state)
{
    using trtlab::v3::Pool;
    struct Object
    {
    };
    auto pool = Pool<Object>::Create();
    pool->emplace_push();

    for(auto _ : state)
    {
        auto obj = std::move(pool->pop());
    }
}

static void BM_Pool_v4_Pop(benchmark::State& state)
{
    using trtlab::v4::Pool;
    struct Object
    {
    };
    auto pool = Pool<Object>::Create();
    pool->EmplacePush();

    for(auto _ : state)
    {
        auto obj = std::move(pool->pop_unique());
    }
}

static void BM_Pool_v4_Pop_Shared(benchmark::State& state)
{
    using trtlab::v4::Pool;
    struct Object
    {
    };
    auto pool = Pool<Object>::Create();
    pool->emplace_push();

    for(auto _ : state)
    {
        auto obj = std::move(pool->pop_shared());
    }
}

static void BM_Pool_v3_Pop_Userspace(benchmark::State& state)
{
    using trtlab::v3::Pool;
    struct Object
    {
    };
    auto pool = Pool<Object, trtlab::userspace_threads>::Create();
    pool->emplace_push();

    for(auto _ : state)
    {
        auto obj = std::move(pool->pop());
    }
}

BENCHMARK(BM_Pool_v1_Pop);
BENCHMARK(BM_Pool_v2_Pop);
BENCHMARK(BM_Pool_v3_Pop);
BENCHMARK(BM_Pool_v4_Pop);
BENCHMARK(BM_Pool_v4_Pop_Shared);
BENCHMARK(BM_Pool_v3_Pop_Userspace);
