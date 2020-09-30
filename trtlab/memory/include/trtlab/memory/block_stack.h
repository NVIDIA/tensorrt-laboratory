// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_MEMORY_ARENA_H_INCLUDED
#define TRTLAB_MEMORY_MEMORY_ARENA_H_INCLUDED

#include <type_traits>

#include "detail/debug_helpers.h"
#include "detail/assert.h"
#include "detail/utility.h"
#include "allocator_traits.h"
#include "config.h"
#include "error.h"
#include "memory_block.h"
#include "block_allocators.h"
#include "block_arena.h"

namespace trtlab
{
    namespace memory
    {

        template <class BlockAllocator, bool Cached = true>
        class block_stack;

        constexpr bool cached   = true;
        constexpr bool uncached = false;

        namespace detail
        {
            // stores memory block in an intrusive linked list and allows LIFO access
            class memory_block_stack
            {
            public:
                memory_block_stack() noexcept : head_(nullptr) {}

                ~memory_block_stack() noexcept {}

                memory_block_stack(memory_block_stack&& other) noexcept : head_(other.head_)
                {
                    other.head_ = nullptr;
                }

                memory_block_stack& operator=(memory_block_stack&& other) noexcept
                {
                    memory_block_stack tmp(detail::move(other));
                    swap(*this, tmp);
                    return *this;
                }

                friend void swap(memory_block_stack& a, memory_block_stack& b) noexcept
                {
                    detail::adl_swap(a.head_, b.head_);
                }

                // the raw allocated block returned from an allocator
                using allocated_mb = memory_block;

                // the inserted block slightly smaller to allow for the fixup value
                using inserted_mb = memory_block;

                // how much an inserted block is smaller
                static const std::size_t implementation_offset;

                // pushes a memory block
                void push(allocated_mb block) noexcept;

                // pops a memory block and returns the original block
                allocated_mb pop() noexcept;

                // steals the top block from another stack
                void steal_top(memory_block_stack& other) noexcept;

                // returns the last pushed() inserted memory block
                inserted_mb top() const noexcept
                {
                    TRTLAB_MEMORY_ASSERT(head_);
                    auto mem = static_cast<void*>(head_);
                    return {static_cast<char*>(mem) + node::offset, head_->usable_size};
                }

                bool empty() const noexcept
                {
                    return head_ == nullptr;
                }

                bool owns(const void* ptr) const noexcept;

                // O(n) size
                std::size_t size() const noexcept;

            private:
                struct node
                {
                    node*       prev;
                    std::size_t usable_size;

                    node(node* p, std::size_t size) noexcept : prev(p), usable_size(size) {}

                    static const std::size_t div_alignment;
                    static const std::size_t mod_offset;
                    static const std::size_t offset;
                };

                node* head_;
            };

            template <bool Cached>
            class memory_arena_cache;

            template <>
            class memory_arena_cache<cached_arena>
            {
            protected:
                bool cache_empty() const noexcept
                {
                    return cached_.empty();
                }

                std::size_t cache_size() const noexcept
                {
                    return cached_.size();
                }

                std::size_t cached_block_size() const noexcept
                {
                    return cached_.top().size;
                }

                bool take_from_cache(detail::memory_block_stack& used) noexcept
                {
                    if (cached_.empty())
                        return false;
                    used.steal_top(cached_);
                    return true;
                }

                template <class BlockAllocator>
                void do_deallocate_block(BlockAllocator&, detail::memory_block_stack& used) noexcept
                {
                    cached_.steal_top(used);
                }

                template <class BlockAllocator>
                void do_shrink_to_fit(BlockAllocator& alloc) noexcept
                {
                    detail::memory_block_stack to_dealloc;
                    // pop from cache and push to temporary stack
                    // this revers order
                    while (!cached_.empty())
                        to_dealloc.steal_top(cached_);
                    // now dealloc everything
                    while (!to_dealloc.empty())
                        alloc.deallocate_block(to_dealloc.pop());
                }

            private:
                detail::memory_block_stack cached_;
            };

            template <>
            class memory_arena_cache<uncached_arena>
            {
            protected:
                bool cache_empty() const noexcept
                {
                    return true;
                }

                std::size_t cache_size() const noexcept
                {
                    return 0u;
                }

                std::size_t cached_block_size() const noexcept
                {
                    return 0u;
                }

                bool take_from_cache(detail::memory_block_stack&) noexcept
                {
                    return false;
                }

                template <class BlockAllocator>
                void do_deallocate_block(BlockAllocator& alloc, detail::memory_block_stack& used) noexcept
                {
                    alloc.deallocate_block(used.pop());
                }

                template <class BlockAllocator>
                void do_shrink_to_fit(BlockAllocator&) noexcept
                {
                }
            };
        } // namespace detail


        // TODO:
        // - expose the block_allocator via get_block_allocator
        // - expose the block_allocator's allocator via get_allocator

        /// A block arena that manages huge memory blocks for a higher-level allocator.
        /// Some allocators like \ref memory_stack work on huge memory blocks,
        /// this class manages them for those allocators.
        /// It uses a \concept{concept_blockallocator,BlockAllocator} for the allocation of those blocks.
        /// The memory blocks in use are put onto a stack like structure, deallocation will pop from the top,
        /// so it is only possible to deallocate the last allocated block of the arena.
        /// Block can be cached or uncached via the Cached template parameter
        /// \ref cached_arena (or \c true) enables it explicitly.
        /// \ingroup memory core
        template <class BlockAllocator, bool Cached>
        class block_stack : TRTLAB_EBO(BlockAllocator), TRTLAB_EBO(detail::memory_arena_cache<Cached>)
        {
            static_assert(is_block_allocator<BlockAllocator>::value, "BlockAllocator is not a BlockAllocator!");
            using cache = detail::memory_arena_cache<Cached>;

        public:
            using allocator_type = BlockAllocator;
            using memory_type    = typename BlockAllocator::memory_type;
            using is_cached      = std::integral_constant<bool, Cached>;

            /// \effects Creates it by giving it the size and other arguments for the \concept{concept_blockallocator,BlockAllocator}.
            /// It forwards these arguments to its constructor.
            /// \requires \c block_size must be greater than \c 0 and other requirements depending on the \concept{concept_blockallocator,BlockAllocator}.
            /// \throws Anything thrown by the constructor of the \c BlockAllocator.
            template <typename... Args>
            explicit block_stack(std::size_t block_size, Args&&... args) : allocator_type(block_size, detail::forward<Args>(args)...)
            {
            }

            block_stack(BlockAllocator&& block_alloc) noexcept : allocator_type(std::move(block_alloc)) {}

            /// \effects Deallocates all memory blocks that where requested back to the \concept{concept_blockallocator,BlockAllocator}.
            ~block_stack() noexcept
            {
                // clear cache
                shrink_to_fit();
                // now deallocate everything
                while (!used_.empty())
                    allocator_type::deallocate_block(used_.pop());
            }

            /// @{
            /// \effects Moves the arena.
            /// The new arena takes ownership over all the memory blocks from the other arena object,
            /// which is empty after that.
            /// This does not invalidate any memory blocks.
            block_stack(block_stack&& other) noexcept
            : allocator_type(detail::move(other)), cache(detail::move(other)), used_(detail::move(other.used_))
            {
            }

            block_stack& operator=(block_stack&& other) noexcept
            {
                block_stack tmp(detail::move(other));
                swap(*this, tmp);
                return *this;
            }
            /// @}

            /// \effects Swaps to memory arena objects.
            /// This does not invalidate any memory blocks.
            friend void swap(block_stack& a, block_stack& b) noexcept
            {
                detail::adl_swap(static_cast<allocator_type&>(a), static_cast<allocator_type&>(b));
                detail::adl_swap(static_cast<cache&>(a), static_cast<cache&>(b));
                detail::adl_swap(a.used_, b.used_);
            }

            /// \effects Allocates a new memory block.
            /// It first uses a cache of previously deallocated blocks, if caching is enabled,
            /// if it is empty, allocates a new one.
            /// \returns The new \ref memory_block.
            /// \throws Anything thrown by the \concept{concept_blockallocator,BlockAllocator} allocation function.
            memory_block allocate_block()
            {
                if (!cache::take_from_cache(used_))
                    used_.push(allocator_type::allocate_block());

                auto block = used_.top();
                detail::debug_fill_internal(block.memory, block.size, false);
                return block;
            }

            /// \returns The current memory block.
            /// This is the memory block that will be deallocated by the next call to \ref deallocate_block().
            memory_block current_block() const noexcept
            {
                return used_.top();
            }

            /// \effects Deallocates the current memory block.
            /// The current memory block is the block on top of the stack of blocks.
            /// If caching is enabled, it does not really deallocate it but puts it onto a cache for later use,
            /// use \ref shrink_to_fit() to purge that cache.
            void deallocate_block() noexcept
            {
                auto block = used_.top();
                detail::debug_fill_internal(block.memory, block.size, true);
                this->do_deallocate_block(get_allocator(), used_);
            }

            /// \returns If `ptr` is in memory owned by the arena.
            bool owns(const void* ptr) const noexcept
            {
                return used_.owns(ptr);
            }

            /// \effects Purges the cache of unused memory blocks by returning them.
            /// The memory blocks will be deallocated in reversed order of allocation.
            /// Does nothing if caching is disabled.
            void shrink_to_fit() noexcept
            {
                this->do_shrink_to_fit(get_allocator());
            }

            /// \returns The capacity of the arena, i.e. how many blocks are used and cached.
            std::size_t capacity() const noexcept
            {
                return size() + cache_size();
            }

            /// \returns The size of the cache, i.e. how many blocks can be allocated without allocation.
            std::size_t cache_size() const noexcept
            {
                return cache::cache_size();
            }

            /// \returns The size of the arena, i.e. how many blocks are in use.
            /// It is always smaller or equal to the \ref capacity().
            std::size_t size() const noexcept
            {
                return used_.size();
            }

            bool empty() const noexcept
            {
                return size() == 0;
            }

            /// \returns The size of the next memory block,
            /// i.e. of the next call to \ref allocate_block().
            /// If there are blocks in the cache, returns size of the next one.
            /// Otherwise forwards to the \concept{concept_blockallocator,BlockAllocator} and subtracts an implementation offset.
            std::size_t next_block_size() const noexcept
            {
                return cache::cache_empty() ? allocator_type::next_block_size() - detail::memory_block_stack::implementation_offset :
                                             cache::cached_block_size();
            }

            /// \returns A reference of the \concept{concept_blockallocator,BlockAllocator} object.
            /// \requires It is undefined behavior to move this allocator out into another object.
            allocator_type& get_allocator() noexcept
            {
                return *this;
            }

            const allocator_type& get_allocator() const noexcept
            {
                return *this;
            }

            DLContext device_context() const
            {
                return allocator_type::device_context();
            }

        private:
            detail::memory_block_stack used_;
        };

        template <typename BlockAllocator>
        using cached_block_stack = block_stack<BlockAllocator, true>;

        template <typename BlockAllocator>
        using uncached_block_stack = block_stack<BlockAllocator, false>;

        template<bool Cached, typename BlockAllocator>
        auto make_block_stack(BlockAllocator&& alloc)
        {
            static_assert(is_block_allocator<BlockAllocator>{}, "should be a block allocator");
            return block_stack<BlockAllocator, Cached>(std::move(alloc));
        }

    } // namespace memory
} // namespace trtlab

#endif // TRTLAB_MEMORY_MEMORY_ARENA_H_INCLUDED
