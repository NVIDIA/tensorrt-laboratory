// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_DETAIL_MEMORY_STACK_H_INCLUDED
#define TRTLAB_MEMORY_DETAIL_MEMORY_STACK_H_INCLUDED

#include <cstddef>

#include "../config.h"
#include "../align.h"
#include "debug_helpers.h"
#include "../debugging.h"

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            // simple memory stack implementation that does not support growing
            class fixed_memory_stack
            {
            public:
                fixed_memory_stack() noexcept : fixed_memory_stack(nullptr)
                {
                }

                // gives it the current pointer, the end pointer must be maintained seperataly
                explicit fixed_memory_stack(void* memory) noexcept
                    : cur_(static_cast<char*>(memory))
                {
                }

                fixed_memory_stack(fixed_memory_stack&& other) noexcept : cur_(other.cur_)
                {
                    other.cur_ = nullptr;
                }

                ~fixed_memory_stack() noexcept = default;

                fixed_memory_stack& operator=(fixed_memory_stack&& other) noexcept
                {
                    cur_       = other.cur_;
                    other.cur_ = nullptr;
                    return *this;
                }

                // bumps the top pointer without filling it
                void bump(std::size_t offset) noexcept
                {
                    cur_ += offset;
                }

                // bumps the top pointer by offset and fills
                void bump(std::size_t offset, debug_magic m) noexcept
                {
                    detail::debug_fill(cur_, offset, m);
                    bump(offset);
                }

                // same as bump(offset, m) but returns old value
                void* bump_return(std::size_t offset,
                                  debug_magic m = debug_magic::new_memory) noexcept
                {
                    auto memory = cur_;
                    detail::debug_fill(memory, offset, m);
                    cur_ += offset;
                    return memory;
                }

                // allocates memory by advancing the stack, returns nullptr if insufficient
                // debug: mark memory as new_memory, put fence in front and back
                void* allocate(const char* end, std::size_t size, std::size_t alignment,
                               std::size_t fence_size = debug_fence_size) noexcept
                {
                    if (cur_ == nullptr)
                        return nullptr;

                    auto remaining = std::size_t(end - cur_);
                    auto offset    = align_offset(cur_ + fence_size, alignment);
                    if (fence_size + offset + size + fence_size > remaining)
                        return nullptr;

                    return allocate_unchecked(size, offset, fence_size);
                }

                // same as allocate() but does not check the size
                // note: pass it the align OFFSET, not the alignment
                void* allocate_unchecked(std::size_t size, std::size_t align_offset,
                                         std::size_t fence_size = debug_fence_size)
                    noexcept
                {
                    bump(fence_size, debug_magic::fence_memory);
                    bump(align_offset, debug_magic::alignment_memory);
                    auto mem = bump_return(size);
                    bump(fence_size, debug_magic::fence_memory);
                    return mem;
                }

                // unwindws the stack to a certain older position
                // debug: marks memory from new top to old top as freed
                // doesn't check for invalid pointer
                void unwind(char* top) noexcept
                {
                    debug_fill(top, std::size_t(cur_ - top), debug_magic::freed_memory);
                    cur_ = top;
                }

                // returns the current top
                char* top() const noexcept
                {
                    return cur_;
                }

            private:
                char* cur_;
            };
        } // namespace detail
    }
} // namespace trtlab::memory

#endif // TRTLAB_MEMORY_DETAIL_MEMORY_STACK_H_INCLUDED
