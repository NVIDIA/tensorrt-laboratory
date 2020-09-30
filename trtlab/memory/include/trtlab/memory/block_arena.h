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

#include <utility>

#include <glog/logging.h>

#include "block_allocators.h"
#include "detail/block_list.h"

namespace trtlab
{
    namespace memory
    {
        constexpr bool cached_arena   = true;
        constexpr bool uncached_arena = false;

        namespace detail
        {
            template <bool Cached, typename BlockList = detail::block_list>
            class block_cache;

            template <typename BlockList>
            class block_cache<cached_arena, BlockList>
            {
            public:
                block_cache() noexcept {}

                block_cache(block_cache<cached_arena>&& other) noexcept : m_cache(std::move(other.m_cache))
                {
                    DVLOG(3) << "block_cache was moved via move constructor";
                }

                block_cache& operator=(block_cache<cached_arena>&& other) noexcept
                {
                    DVLOG(3) << "block_cache was moved via move assignment";
                    m_cache = std::move(other.m_cache);
                }

                bool cache_empty() const noexcept
                {
                    return m_cache.empty();
                }

                std::size_t cache_size() const noexcept
                {
                    return m_cache.size();
                }

                bool take_from_cache(memory_block& block) noexcept
                {
                    if (m_cache.empty())
                    {
                        return false;
                    }
                    block = m_cache.allocate();
                    DVLOG(3) << "Acquired memory_block from cache: " << block.memory << "; " << block.size;
                    return true;
                }

                template <typename BlockAllocator>
                void deallocate_block(BlockAllocator&, memory_block&& block) noexcept
                {

                    DVLOG(3) << "deallocate_block " << block.memory << " was cached instead of released";
                    m_cache.insert(std::move(block));
                    DVLOG(3) << "deallocate_block " << block.memory << " was cached instead of released";
                }

                std::size_t cache_next_block_size() const noexcept
                {
                    return m_cache.next_block_size();
                }

                template <typename BlockAllocator>
                void cache_shrink_to_fit(BlockAllocator& alloc) noexcept
                {
                    static_assert(is_block_allocator<BlockAllocator>{}, "BlockAllocator is not a BlockAllocator!");
                    while (!m_cache.empty())
                    {
                        alloc.deallocate_block(m_cache.allocate());
                    }
                }

                template <typename BlockAllocator>
                void cache_reserve(BlockAllocator& alloc, std::size_t block_count)
                {
                    static_assert(is_block_allocator<BlockAllocator>{}, "BlockAllocator is not a BlockAllocator!");
                    while (m_cache.size() < block_count)
                    {
                        DVLOG(4) << "cache size: " << m_cache.size();
                        m_cache.insert(alloc.allocate_block());
                    }
                }

            private:
                BlockList m_cache;
            };

            template <typename BlockList>
            class block_cache<uncached_arena, BlockList>
            {
            public:
                block_cache() noexcept {}

                block_cache(block_cache<uncached_arena>&& other) noexcept {}

                block_cache& operator=(block_cache<uncached_arena>&& other) noexcept {}

                bool cache_empty() const noexcept
                {
                    return true;
                }

                std::size_t cache_size() const noexcept
                {
                    return 0u;
                }

                bool take_from_cache(memory_block& block) noexcept
                {
                    return false;
                }

                template <typename BlockAllocator>
                void deallocate_block(BlockAllocator& alloc, memory_block&& block) noexcept
                {
                    static_assert(is_block_allocator<BlockAllocator>{}, "BlockAllocator is not a BlockAllocator!");
                    alloc.deallocate_block(block);
                }

                std::size_t cache_next_block_size() const noexcept
                {
                    return 0u;
                }

                template <typename BlockAllocator>
                void cache_shrink_to_fit(BlockAllocator&) noexcept
                {
                }

                template <typename BlockAllocator>
                void cache_reserve(BlockAllocator&, std::size_t)
                {
                }
            };
        } // namespace detail

        template <typename BlockAllocator, bool Cached, typename BlockList>
        class block_arena : TRTLAB_EBO(BlockAllocator), TRTLAB_EBO(detail::block_cache<Cached, BlockList>)
        {
            static_assert(is_block_allocator<BlockAllocator>::value, "BlockAllocator is not a BlockAllocator!");

            using cache_type       = detail::block_cache<Cached, BlockList>;
            using block_arena_type = block_arena<BlockAllocator, Cached, BlockList>;

        public:
            using block_allocator_type = BlockAllocator;
            using allocator_type       = typename block_allocator_type::allocator_type;
            using memory_type          = typename block_allocator_type::memory_type;
            using is_cached            = std::integral_constant<bool, Cached>;
            using is_stateful          = std::true_type;

            //explicit transactional_arena(std::size_t block_size, Args&&... args)
            //    : block_allocator_type(block_size, detail::forward<Args>(args)...) {}

            explicit block_arena(block_allocator_type&& block_alloc) noexcept : block_allocator_type(std::move(block_alloc)) {}

            block_arena(block_arena_type&& other) noexcept : block_allocator_type(std::move(other)), cache_type(std::move(other)) {}

            block_arena_type& operator=(block_arena_type&& other) noexcept
            {
                block_allocator_type::operator=(std::move(other));
                cache_type::          operator=(std::move(other));
            }

            // explicitly delete copy ctor and assignment
            block_arena(const block_arena_type&) = delete;
            block_arena_type& operator=(const block_arena_type&) = delete;

            ~block_arena()
            {
                // deallocate blocks in cache
                shrink_to_fit();
            }

            // block_allocator methods

            memory_block allocate_block()
            {
                memory_block block;
                if (!this->take_from_cache(block))
                {
                    block = block_allocator_type::allocate_block();
                }
                return block;
            }

            void deallocate_block(memory_block block)
            {
                DCHECK(block.memory);
                DCHECK(block.size);
                DVLOG(3) << "deallocate_block: " << block.memory << "; " << block.size;
                cache_type::deallocate_block(get_block_allocator(), std::move(block));
            }

            std::size_t next_block_size() const noexcept
            {
                if (!this->cache_empty())
                {
                    return this->cache_next_block_size();
                }
                return block_allocator_type::next_block_size();
            }

            // arena methods

            void shrink_to_fit() noexcept
            {
                this->cache_shrink_to_fit(get_block_allocator());
            }

            void reserve_blocks(std::size_t block_count) noexcept
            {
                this->cache_reserve(get_block_allocator(), block_count);
            }

            DLContext device_context() const noexcept
            {
                return block_allocator_type::device_context();
            } 

            // allocator access and methods

            allocator_type& get_allocator() noexcept
            {
                return block_allocator_type::get_allocator();
            }

            const allocator_type& get_allocator() const noexcept
            {
                return block_allocator_type::get_allocator();
            }

            block_allocator_type& get_block_allocator() noexcept
            {
                return *this;
            }

            const block_allocator_type& get_block_allocator() const noexcept
            {
                return *this;
            }
        };

        template <bool Cached, typename BlockAllocator>
        auto make_block_arena(BlockAllocator&& block_alloc)
        {
            using memory_type = typename BlockAllocator::memory_type;
            using list_type = typename std::conditional<std::is_base_of<host_memory, memory_type>::value, detail::block_list, detail::block_list_oob>::type;
            return block_arena<BlockAllocator, Cached, list_type>(std::move(block_alloc));
        }

        template <typename BlockAllocator>
        auto make_cached_block_arena(BlockAllocator&& block_alloc)
        {
            return make_block_arena<cached_arena>(std::move(block_alloc));
        }

        template <typename BlockAllocator>
        auto make_uncached_block_arena(BlockAllocator&& block_alloc)
        {
            return make_block_arena<uncached_arena>(std::move(block_alloc));
        }

    } // namespace memory
} // namespace trtlab