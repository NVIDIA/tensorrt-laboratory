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

#include <trtlab/memory/allocator.h>
#include <trtlab/memory/block_allocators.h>
#include <trtlab/memory/literals.h>
#include <trtlab/memory/transactional_allocator.h>

#include <trtlab/cuda/memory/cuda_allocators.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>

using namespace trtlab;
using namespace memory;
using namespace literals;

class TestCudaAllocators : public ::testing::Test
{
};

TEST_F(TestCudaAllocators, cudaMalloc)
{
    auto raw   = cuda_malloc_allocator(0);
    auto alloc = make_allocator(std::move(raw));

    auto p0 = alloc.allocate_node(1024, 256);
    alloc.deallocate_node(p0, 1024, 256);

    auto info = alloc.device_context();
    ASSERT_EQ(info.device_type, kDLGPU);

    auto md = alloc.allocate_descriptor(1024);
}

TEST_F(TestCudaAllocators, TrasactionalCudaMalloc)
{
    auto raw     = cuda_malloc_allocator(0);
    auto tracked = make_allocator_adapter(std::move(raw));
    auto block   = make_block_allocator<fixed_size_block_allocator>(std::move(tracked), 1_MiB);
    auto counted = make_extended_block_allocator<count_limited_block_allocator>(std::move(block), 3);
    auto arena   = make_cached_block_arena(std::move(counted));
    auto txalloc = make_transactional_allocator(std::move(arena));
    auto alloc   = make_allocator(std::move(txalloc));

    alloc.get_allocator().reserve_blocks(3);

    auto p0 = alloc.allocate_descriptor(1024);
}