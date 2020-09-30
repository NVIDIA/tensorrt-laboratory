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
#include <cstddef>
#include <map>
#include <queue>
#include <utility>

#include <glog/logging.h>

#include "block_allocators.h"
#include "block_arena.h"
#include "block_manager.h"
#include "error.h"

#include "detail/memory_stack.h"
//#include <foonathan/memory/allocator_storage.hpp>

namespace trtlab
{
    namespace memory
    {
        namespace transactional_detail
        {
            class basic_stack : public memory_block, private detail::fixed_memory_stack
            {
                using stack = detail::fixed_memory_stack;

            public:
                basic_stack() noexcept : memory_block(), fixed_memory_stack(nullptr), m_end(nullptr) {}

                basic_stack(void* ptr, std::size_t size) noexcept : memory_block(ptr, size), fixed_memory_stack(ptr), m_end(top() + size) {}

                basic_stack(basic_stack&& other) noexcept
                : memory_block(std::move(other)), fixed_memory_stack(std::move(other)), m_end(std::exchange(other.m_end, nullptr))
                {
                }

                basic_stack& operator=(basic_stack&& other) noexcept
                {
                    m_end                               = std::exchange(other.m_end, nullptr);
                    detail::fixed_memory_stack::operator=(std::move(other));
                    memory_block::              operator=(std::move(other));
                    return *this;
                }

                void* allocate(std::size_t size_, std::size_t alignment) noexcept
                {
                    return detail::fixed_memory_stack::allocate(m_end, size_, alignment);
                }

                std::size_t available() const noexcept
                {
                    return std::size_t(m_end - top());
                }

                std::size_t capacity() const noexcept
                {
                    return size;
                };

                bool contains(const void* ptr) const noexcept
                {
                    if (!ptr && !m_end)
                    {
                        return true;
                    }
                    return memory_block::contains(ptr);
                }

                const memory_block& get_memory_block() const noexcept
                {
                    return *this;
                }

            private:
                char* m_end;
            };

            class ref_counted_stack : public basic_stack
            {
            public:
                ref_counted_stack() noexcept : basic_stack(nullptr, 0u), m_count(0u) {}
                explicit ref_counted_stack(const memory_block& block) noexcept : basic_stack(block.memory, block.size), m_count(0u) {}
                explicit ref_counted_stack(void* ptr, std::size_t size) noexcept : basic_stack(ptr, size), m_count(0u) {}

                ref_counted_stack(ref_counted_stack&& other) noexcept
                : basic_stack(std::move(other)), m_count(std::exchange(other.m_count, 0u))
                {
                }

                ref_counted_stack& operator=(ref_counted_stack&& other) noexcept
                {
                    basic_stack::operator=(std::move(other));
                    m_count              = std::exchange(other.m_count, 0u);
                    return *this;
                }

                void* allocate(std::size_t size, std::size_t alignment) noexcept
                {
                    auto ptr = basic_stack::allocate(size, alignment);
                    if (ptr)
                    {
                        m_count++;
                    }
                    return ptr;
                }

                bool should_release_after_deallocate(void* ptr) noexcept
                {
                    DCHECK(contains(ptr));
                    DCHECK(m_count) << "Caught more deallocates than allocates";
                    if (m_count && --m_count)
                    {
                        return false;
                    }
                    return true;
                }

                std::size_t use_count() const noexcept
                {
                    return m_count;
                }

            private:
                std::size_t m_count;
            };

        } // namespace transactional_detail

        template <typename BlockAllocator, typename ListType>
        class transactional_allocator
        : TRTLAB_EBO(block_arena<BlockAllocator, cached_arena, ListType>)
        {
            static_assert(is_block_allocator<BlockAllocator>::value, "BlockAllocator is not a BlockAllocator!");

            using list_type  = ListType;
            using stack_type = transactional_detail::ref_counted_stack;
            using arena_type = block_arena<BlockAllocator, cached_arena, list_type>;

            stack_type                m_current_stack;
            block_manager<stack_type> m_in_use_stacks;
            std::size_t               m_max_node_size;

        public:
            using allocator_type       = typename arena_type::allocator_type;
            using block_allocator_type = typename arena_type::block_allocator_type;
            using memory_type          = typename arena_type::memory_type;
            using is_stateful          = std::true_type;

            explicit transactional_allocator(arena_type&& arena) : arena_type(std::move(arena))
            {
                DVLOG(1) << "allocate initial block/stack";
                m_current_stack = allocate_stack();
            }

            ~transactional_allocator() noexcept
            {
                release_current_stack();
                release_in_use_stacks();
                shrink_to_fit();
            }

            transactional_allocator(transactional_allocator&& other) noexcept
            : arena_type(std::move(other)),
              m_current_stack(std::move(other.m_current_stack)),
              m_in_use_stacks(std::move(other.m_in_use_stacks)),
              m_max_node_size(std::exchange(other.m_max_node_size, 0u))
            {
            }

            transactional_allocator& operator=(transactional_allocator&& other) noexcept
            {
                arena_type::operator=(std::move(other));
                m_in_use_stacks     = std::move(other.m_in_use_stacks);
                m_current_stack     = std::move(other.m_current_stack);
                m_max_node_size     = std::exchange(other.m_max_node_size, 0u);
            }

            transactional_allocator(const transactional_allocator&) = delete;
            transactional_allocator& operator=(const transactional_allocator&) = delete;

            void* allocate_node(std::size_t size, std::size_t alignment)
            {
                // allocate off current stack if possible
                // otherwise, allocate a new stack
                DVLOG(2) << this << ": allocate_node " << size << "; " << alignment;

                // check size
                // todo: check alignment
                if (size > m_max_node_size)
                {
                    throw bad_allocation_size(info(), size, m_max_node_size);
                }

                void* ptr = m_current_stack.allocate(size, alignment);
                if (__builtin_expect((!ptr), 0)) // unlikely macro
                {
                    DVLOG(3) << "current stack exhaused - rotate stacks";
                    release_current_stack();
                    m_current_stack = allocate_stack();
                    ptr             = m_current_stack.allocate(size, alignment);
                }
                if (__builtin_expect((!ptr), 0))
                {
                    throw bad_node_size(info(), size, m_max_node_size);
                }
                DVLOG(1) << this << ": allocated " << ptr << "; size=" << size;
                return ptr;
            }

            void deallocate_node(void* ptr, std::size_t size, std::size_t alignment) noexcept
            {
                // deallocate stack if its not the current stack
                // and the reference count goes to 0
                DVLOG(1) << this << ": deallocate_node " << ptr << "; " << size << "; " << alignment;
                auto stack = find_stack(ptr);
                DCHECK(stack);
                if (stack->should_release_after_deallocate(ptr) && !m_current_stack.contains(ptr))
                {
                    DVLOG(3) << "deallocate dropping block";
                    drop_stack_containing_address(ptr);
                }
            }

            std::size_t max_node_size() const noexcept
            {
                return m_max_node_size;
            }

            DLContext device_context() const noexcept
            {
                return arena_type::device_context();
            }

            // access the arena

            void shrink_to_fit() noexcept
            {
                arena_type::shrink_to_fit();
            }

            void reserve_blocks(std::size_t block_count) noexcept
            {
                DVLOG(2) << "reserve blocks: " << block_count;
                auto count = block_count - m_in_use_stacks.size() - 1 /* m_current_stack */;
                DVLOG(3) << "reserve blocks: " << block_count << "; needed: " << count;
                arena_type::reserve_blocks(count);
            }

            allocator_type& get_allocator() noexcept
            {
                return arena_type::get_allocator();
            }

            block_allocator_type& get_block_allocator() noexcept
            {
                return arena_type::get_block_allocator();
            }

        private:
            allocator_info info() noexcept
            {
                return {"trtlab::transactional_allocator", this};
            }

            stack_type allocate_stack()
            {
                auto block      = arena_type::allocate_block();
                m_max_node_size = std::max(m_max_node_size, block.size);
                DVLOG(3) << "allocated new stack";
                return stack_type(block);
            }

            void PushStack(stack_type&& stack)
            {
                DVLOG(3) << "pushing stack to block_manager";
                m_in_use_stacks.add_block(std::move(stack));
            }

            void drop_stack_containing_address(void* ptr) noexcept
            {
                // find_stack search both the in-use block pool and teh current stack
                // We only drop in-use stacks
                auto stack = m_in_use_stacks.find_block(ptr);
                if (stack)
                {
                    DVLOG(3) << "dropping stack " << stack << " from in-use stack";
                    arena_type::deallocate_block(std::move(*stack));
                    m_in_use_stacks.drop_block(ptr);
                }
            }

            void release_current_stack() noexcept
            {
                if (m_current_stack.use_count())
                {
                    DVLOG(3) << "current stack is in-use; move to block_manager";
                    PushStack(std::move(m_current_stack));
                }
                else
                {
                    DVLOG(3) << "current stack is dereferences; deallocating";
                    auto block = m_current_stack.get_memory_block();
                    if (block.memory)
                    {
                        arena_type::deallocate_block(std::move(block));
                    }
                }
                m_current_stack = stack_type();
            }

            void release_in_use_stacks() noexcept
            {
                //if(!m_Map.empty())
                if (m_in_use_stacks.size())
                {
                    DLOG(WARNING) << "transactional_allocator being released with unreleased allocations";
                    for (void* ptr : m_in_use_stacks.blocks())
                    {
                        DVLOG(3) << "force dropping stack " << ptr;
                        drop_stack_containing_address(ptr);
                    }
                    m_in_use_stacks.clear();
                }
                DCHECK_EQ(m_in_use_stacks.size(), 0);
            }

            stack_type* find_stack(const void* ptr) noexcept
            {
                if (m_current_stack.contains(ptr))
                {
                    return &m_current_stack;
                }
                return m_in_use_stacks.find_block(ptr);
            }
        };

        template <typename BlockAllocator, typename BlockList>
        auto make_transactional_allocator(block_arena<BlockAllocator, cached_arena, BlockList>&& arena)
        {
            return transactional_allocator<BlockAllocator, BlockList>(std::move(arena));
        }

    } // namespace memory
} // namespace trtlab
