/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "tensorrt/playground/core/allocator.h"
#include "tensorrt/playground/core/memory.h"
#include "tensorrt/playground/core/memory_stack.h"
#include "tensorrt/playground/core/cyclic_allocator.h"

using namespace yais;
using namespace yais::Memory;

static void BM_MemoryStack_Allocate(benchmark::State &state)
{
    auto stack = std::make_shared<MemoryStack<SystemMallocMemory>>(1024*1024);
    for (auto _ : state)
    {
        auto ptr = stack->Allocate(1024);
        stack->Reset();
    }
}

static void BM_MemoryDescriptorStack_Allocate(benchmark::State &state)
{
    auto stack = std::make_shared<MemoryDescriptorStack<SystemMallocMemory>>(1024*1024);
    for (auto _ : state)
    {
        auto ptr = stack->Allocate(1024);
        stack->Reset();
    }
}

static void BM_CyclicAllocator_Malloc_Allocate(benchmark::State &state)
{
    auto stack = std::make_unique<CyclicAllocator<SystemMallocMemory>>(10, 1024*1024);
    for (auto _ : state)
    {
        auto ptr = stack->Allocate(1024);
    }
}

static void BM_CyclicAllocator_SystemV_Allocate(benchmark::State &state)
{
    auto stack = std::make_unique<CyclicAllocator<SystemV>>(10, 1024*1024);
    for (auto _ : state)
    {
        auto ptr = stack->Allocate(1024);
    }
}

BENCHMARK(BM_MemoryStack_Allocate);
BENCHMARK(BM_MemoryDescriptorStack_Allocate);
BENCHMARK(BM_CyclicAllocator_Malloc_Allocate);
BENCHMARK(BM_CyclicAllocator_SystemV_Allocate);