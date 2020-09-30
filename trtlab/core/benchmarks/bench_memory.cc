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

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/malloc.h"
#include "trtlab/core/memory/sysv_allocator.h"

using namespace trtlab;
using namespace trtlab;


static void BM_Memory_SystemMalloc(benchmark::State& state)
{
    for(auto _ : state)
    {
        //auto unique = std::make_unique<Allocator<Malloc>>(1024 * 1024);
        //auto shared = std::make_shared<Allocator<Malloc>>(1024 * 1024);
        Allocator<Malloc> memory1(1024 * 1024);
        Allocator<Malloc> memory2(1024 * 1024);
        Allocator<Malloc> memory3(1024 * 1024);
    }
}

static void BM_Memory_SystemV_descriptor(benchmark::State& state)
{
    auto master = std::make_unique<Allocator<SystemV>>(1024 * 1024);
    for(auto _ : state)
    {
        auto mdesc = SystemV::Attach(master->ShmID());
    }
}

/*
static void BM_Memory_HostDescriptor(benchmark::State& state)
{
    void *ptr = (void*)0xDEADBEEF;
    mem_size_t size = 1337;

    for(auto _ : state)
    {
        nextgen::HostDescriptor hdesc(ptr, size, []{});
    }
}
*/

/*
static void BM_Memory_SharedHostDescriptor(benchmark::State& state)
{
    void *ptr = (void*)0xDEADBEEF;
    mem_size_t size = 1337;

    for(auto _ : state)
    {
        nextgen::Descriptor<HostMemory> hdesc(ptr, size, []{});
        auto shared = std::make_shared<nextgen::SharedDescriptor<HostMemory>>(std::move(hdesc));
    }
}
*/

/*
static void BM_Memory_NextGenMalloc(benchmark::State& state)
{
    static mem_size_t one_mb = 1024*1024;

    for(auto _ : state)
    {
        auto hdesc0 = nextgen::Malloc::Allocate(one_mb);
        auto hdesc1 = nextgen::Malloc::Allocate(one_mb);
    }
}
*/

BENCHMARK(BM_Memory_SystemMalloc);
BENCHMARK(BM_Memory_SystemV_descriptor);
// BENCHMARK(BM_Memory_HostDescriptor);
// BENCHMARK(BM_Memory_SharedHostDescriptor);
// BENCHMARK(BM_Memory_NextGenMalloc);