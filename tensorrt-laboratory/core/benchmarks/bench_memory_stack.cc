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

#include "tensorrt/laboratory/core/memory/cyclic_allocator.h"
#include "tensorrt/laboratory/core/memory/malloc.h"
#include "tensorrt/laboratory/core/memory/memory_stack.h"
#include "tensorrt/laboratory/core/memory/smart_stack.h"
#include "tensorrt/laboratory/core/memory/system_v.h"

using namespace trtlab;
using namespace trtlab;

template<typename MemoryType>
struct StackWithInternalDescriptor
{
    StackWithInternalDescriptor(size_t size) : m_Stack(size) {}
    class InternalDescriptor final : public Descriptor<MemoryType>
    {
      public:
        InternalDescriptor(void* ptr, size_t size, const MemoryType& parent)
            : Descriptor<MemoryType>(ptr, size, parent, []{}, "BenchmarkInternalDesc")
        {
        }
        ~InternalDescriptor() final override {}
    };

    DescriptorHandle<typename MemoryType::BaseType> Allocate(size_t size)
    {
        return std::move(
            std::make_unique<InternalDescriptor>(m_Stack.Allocate(size), size, m_Stack.Memory()));
    }

    typename MemoryType::BaseType&& Allocate2(size_t size)
    {
        return std::move(InternalDescriptor(m_Stack.Allocate(size), size, m_Stack.Memory()));
    }

    void Reset() { m_Stack.Reset(); }

  private:
    MemoryStack<MemoryType> m_Stack;
};

static void BM_MemoryStack_Allocate(benchmark::State& state)
{
    auto stack = std::make_shared<MemoryStack<Malloc>>(1024 * 1024);
    for(auto _ : state)
    {
        for(int i = 0; i < state.range(0); i++)
        {
            auto ptr = stack->Allocate(1024);
        }
        stack->Reset();
    }
}

static void BM_MemoryStackWithDescriptor_Allocate(benchmark::State& state)
{
    auto stack = std::make_shared<StackWithInternalDescriptor<Malloc>>(1024 * 1024);
    for(auto _ : state)
    {
        for(int i = 0; i < state.range(0); i++)
        {
            auto ptr = stack->Allocate(1024);
        }
        stack->Reset();
    }
}

static void BM_MemoryStackWithDescriptor_Allocate2(benchmark::State& state)
{
    auto stack = std::make_shared<StackWithInternalDescriptor<Malloc>>(1024 * 1024);
    for(auto _ : state)
    {
        auto ptr = stack->Allocate2(1024);
        stack->Reset();
    }
}

static void BM_SmartStack_Allocate(benchmark::State& state)
{
    auto stack = std::make_shared<SmartStack<Malloc>>(1024 * 1024);
    for(auto _ : state)
    {
        for(int i = 0; i < state.range(0); i++)
        {
            auto ptr = stack->Allocate(1024);
        }
        stack->Reset();
    }
}

static void BM_CyclicAllocator_Malloc_Allocate(benchmark::State& state)
{
    auto stack = std::make_unique<CyclicAllocator<Malloc>>(10, 1024 * 1024);
    for(auto _ : state)
    {
        for(int i = 0; i < state.range(0); i++)
        {
            auto ptr = stack->Allocate(1024);
        }
    }
}

static void BM_CyclicAllocator_SystemV_Allocate(benchmark::State& state)
{
    auto stack = std::make_unique<CyclicAllocator<SystemV>>(10, 1024 * 1024);
    for(auto _ : state)
    {
        auto ptr = stack->Allocate(1024);
    }
}

BENCHMARK(BM_MemoryStack_Allocate)->RangeMultiplier(2)->Range(1, 8 << 4);
BENCHMARK(BM_MemoryStackWithDescriptor_Allocate)->RangeMultiplier(2)->Range(1, 8 << 4);
BENCHMARK(BM_MemoryStackWithDescriptor_Allocate2);
BENCHMARK(BM_SmartStack_Allocate)->RangeMultiplier(2)->Range(1, 8 << 4);
BENCHMARK(BM_CyclicAllocator_Malloc_Allocate)->RangeMultiplier(2)->Range(1, 8 << 4);
BENCHMARK(BM_CyclicAllocator_SystemV_Allocate);