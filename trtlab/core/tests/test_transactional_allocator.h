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
#pragma once

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/block_allocators.h"
#include "trtlab/core/memory/transactional_allocator.h"

#include <foonathan/memory/tracking.hpp>

#include "test_common.h"

template<typename StatelessAllocator>
auto make_raw_transactional_allocator(std::size_t block_size, std::size_t block_count = 2)
{
    namespace memory = foonathan::memory;

    // base allocator
    auto raw = StatelessAllocator();

    // convert to full fledged allocator - use direct_storage which optimizes out mutexes for stateless allocators
    auto alloc = memory::make_allocator_adapter(std::move(raw));

    static_assert(!decltype(alloc)::is_stateful::value, "should be stateless");
    static_assert(std::is_same<memory::no_mutex, typename decltype(alloc)::mutex>::value, "should use memory::no_mutex");

    // create a tracker for calls to the malloc allocator
    auto tracked = memory::make_tracked_allocator(log_tracker{"** tracker: base **"}, std::move(alloc));

    // malloc block allocator
    auto block_alloc = memory::trtlab::make_growth_capped_block_allocator(block_size, block_count, std::move(tracked));

    // transactional allocator
    return memory::trtlab::make_transactional_allocator(std::move(block_alloc));
}

template<typename StatelessAllocator = memory::malloc_allocator>
auto make_smart_transactional_allocator(std::size_t block_size, std::size_t block_count = 2)
{
    namespace memory = foonathan::memory;

    // transactional allocator
    auto alloc = make_raw_transactional_allocator<StatelessAllocator>(block_size, block_count);

    // populate the cache
    alloc.reserve_blocks(block_count);

    // smart allocator
    // use a special timeout_mutex - throws an exception if the lock is not obtained in MUTEX_TIMEOUT_MS
    return memory::trtlab::make_allocator<timeout_mutex>(std::move(alloc));
}