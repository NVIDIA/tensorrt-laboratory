// MODIFICATION MESSAGE

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#pragma once

#include <type_traits>

#include "detail/utility.h"
#include "allocator_traits.h"
#include "memory_block.h"
#include "config.h"
#include "error.h"

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            static std::size_t max_size_t = std::size_t(-1);

            template <class BlockAllocator>
            std::true_type is_block_allocator_impl(
                int, TRTLAB_SFINAE(std::declval<memory_block&>() = std::declval<BlockAllocator&>().allocate_block()),
                TRTLAB_SFINAE(std::declval<std::size_t&>() = std::declval<BlockAllocator&>().next_block_size()),
                TRTLAB_SFINAE(std::declval<BlockAllocator>().deallocate_block(memory_block{})));

            template <typename T>
            std::false_type is_block_allocator_impl(short);
        } // namespace detail

        /// Traits that check whether a type models concept \concept{concept_blockallocator,BlockAllocator}.
        /// \ingroup memory core
        template <typename T>
        struct is_block_allocator : decltype(detail::is_block_allocator_impl<T>(0))
        {
        };

        /// A \concept{concept_blockallocator,BlockAllocator} that allows only one block allocation.
        /// It can be used to prevent higher-level allocators from expanding.
        /// The one block allocation is performed through the \c allocate_array() function of the given \concept{concept_rawallocator,RawAllocator}.
        /// \ingroup memory adapter

        template <class RawAllocator>
        class single_block_allocator : TRTLAB_EBO(allocator_traits<RawAllocator>::allocator_type)
        {
            using traits = allocator_traits<RawAllocator>;

        public:
            using allocator_type = typename traits::allocator_type;
            using memory_type    = typename traits::memory_type;

            /// \effects Creates it by passing it the size of the block and the allocator object.
            /// \requires \c block_size must be greater than 0,
            explicit single_block_allocator(std::size_t block_size, allocator_type alloc = allocator_type()) noexcept
            : allocator_type(detail::move(alloc)), block_size_(block_size), initial_size_(block_size)
            {
            }

            /// \effects Allocates a new memory block or throws an exception if there was already one allocation.
            /// \returns The new \ref memory_block.
            /// \throws Anything thrown by the \c allocate_array() function of the \concept{concept_rawallocator,RawAllocator} or \ref out_of_memory if this is not the first call.
            memory_block allocate_block()
            {
                if (block_size_)
                {
                    auto alignment = traits::min_alignment(get_allocator());
                    auto memory = traits::allocate_array(get_allocator(), block_size_, 1UL, alignment);
                    block_size_ = 0u;
                    return {memory, initial_size_};
                }
                throw out_of_fixed_memory(info(), block_size_);
            }

            /// \effects Deallocates the previously allocated memory block.
            /// It also resets and allows a new call again.
            void deallocate_block(memory_block block) noexcept
            {
                DCHECK_EQ(block.size, initial_size_);
                traits::deallocate_array(get_allocator(), block.memory, block.size, 1UL);
                block_size_ = block.size;
            }

            /// \returns The size of the next block which is either the initial size or \c 0.
            std::size_t next_block_size() const noexcept
            {
                return block_size_;
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            allocator_type& get_allocator() noexcept
            {
                return *this;
            }

            const allocator_type& get_allocator() const noexcept
            {
                return *this;
            }

            DLContext device_context() const noexcept
            {
                return traits::device_context(get_allocator());
            }

        private:
            allocator_info info() noexcept
            {
                return {TRTLAB_MEMORY_LOG_PREFIX "::single_block_allocator", this};
            }

            std::size_t block_size_;
            std::size_t initial_size_;
        };

        template <class RawAllocator>
        class fixed_size_block_allocator : TRTLAB_EBO(allocator_traits<RawAllocator>::allocator_type)
        {
            using traits = allocator_traits<RawAllocator>;

        public:
            using allocator_type = typename traits::allocator_type;
            using memory_type    = typename traits::memory_type;

            /// \effects Creates it by passing it the size of the block and the allocator object.
            /// \requires \c block_size must be greater than 0,
            explicit fixed_size_block_allocator(std::size_t block_size, allocator_type alloc = allocator_type()) noexcept
            : allocator_type(detail::move(alloc)), block_size_(block_size)
            {
            }

            /// \effects Allocates a new memory block or throws an exception if there was already one allocation.
            /// \returns The new \ref memory_block.
            /// \throws Anything thrown by the \c allocate_array() function of the \concept{concept_rawallocator,RawAllocator} or \ref out_of_memory if this is not the first call.
            memory_block allocate_block()
            {
                auto alignment = traits::min_alignment(get_allocator());
                auto         mem = traits::allocate_array(get_allocator(), block_size_, 1, alignment);
                memory_block block(mem, block_size_);
                return block;
            }

            /// \effects Deallocates the previously allocated memory block.
            /// It also resets and allows a new call again.
            void deallocate_block(memory_block block) noexcept
            {
                DCHECK_EQ(block.size, block_size_);
                traits::deallocate_array(get_allocator(), block.memory, block.size, 1);
            }

            /// \returns The size of the next block which is either the initial size or \c 0.
            std::size_t next_block_size() const noexcept
            {
                return block_size_;
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            allocator_type& get_allocator() noexcept
            {
                return *this;
            }

            const allocator_type& get_allocator() const noexcept
            {
                return *this;
            }

            DLContext device_context() const noexcept
            {
                return traits::device_context(get_allocator());
            }

        private:
            allocator_info info() noexcept
            {
                return {TRTLAB_MEMORY_LOG_PREFIX "::fixed_size_block_allocator", this};
            }

            std::size_t block_size_;
        };

        template <class RawAllocator>
        class growing_block_allocator : TRTLAB_EBO(allocator_traits<RawAllocator>::allocator_type)
        {
            using traits = allocator_traits<RawAllocator>;

        public:
            using allocator_type = typename traits::allocator_type;
            using memory_type    = typename traits::memory_type;

            /// \effects Creates it by passing it the size of the block and the allocator object.
            /// \requires \c block_size must be greater than 0,
            explicit growing_block_allocator(std::size_t block_size, allocator_type alloc = allocator_type(),
                                             std::size_t max_size = detail::max_size_t, double scale_factor = 2) noexcept
            : allocator_type(detail::move(alloc)), block_size_(block_size), max_size_(max_size), scale_factor_(scale_factor)
            {
            }

            /// \effects Allocates a new memory block or throws an exception if there was already one allocation.
            /// \returns The new \ref memory_block.
            /// \throws Anything thrown by the \c allocate_array() function of the \concept{concept_rawallocator,RawAllocator} or \ref out_of_memory if this is not the first call.
            memory_block allocate_block()
            { 
                auto alignment = traits::min_alignment(get_allocator());
                auto         mem = traits::allocate_array(get_allocator(), block_size_, 1, alignment);
                memory_block block(mem, block_size_);
                std::size_t  next_size = block_size_ * scale_factor_;
                if (next_size <= max_size_)
                    block_size_ = next_size;
                return block;
            }

            /// \effects Deallocates the previously allocated memory block.
            /// It also resets and allows a new call again.
            void deallocate_block(memory_block block) noexcept
            {
                DCHECK_LE(block.size, max_size_);
                traits::deallocate_array(get_allocator(), block.memory, block.size, 1);
            }

            /// \returns The size of the next block which is either the initial size or \c 0.
            std::size_t next_block_size() const noexcept
            {
                return block_size_;
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            allocator_type& get_allocator() noexcept
            {
                return *this;
            }

            const allocator_type& get_allocator() const noexcept
            {
                return *this;
            }

            DLContext device_context() const noexcept
            {
                return traits::device_context(get_allocator());
            }

        private:
            allocator_info info() noexcept
            {
                return {TRTLAB_MEMORY_LOG_PREFIX "::growing_block_allocator", this};
            }

            std::size_t block_size_;
            std::size_t max_size_;
            double      scale_factor_;
        };

        template <typename BlockAllocator>
        class count_limited_block_allocator : TRTLAB_EBO(BlockAllocator)
        {

        public:
            using block_allocator_type = BlockAllocator;
            using allocator_type = typename block_allocator_type::allocator_type;
            using memory_type    = typename block_allocator_type::memory_type;

            count_limited_block_allocator(BlockAllocator&& alloc, std::size_t max_blocks)
            : block_allocator_type(std::move(alloc)), m_max_blocks(max_blocks), m_block_count(0)
            {
                DCHECK(max_blocks);
                static_assert(is_block_allocator<BlockAllocator>::value, "needs to be a valid block allocator");
            }

            memory_block allocate_block()
            {
                if (m_block_count < m_max_blocks)
                {
                    m_block_count++;
                    return block_allocator_type::allocate_block();
                }
                throw std::bad_alloc();
            }

            /// \effects Deallocates the previously allocated memory block.
            /// It also resets and allows a new call again.
            void deallocate_block(memory_block block) noexcept
            {
                DCHECK(m_block_count);
                block_allocator_type::deallocate_block(block);
                m_block_count--;
            }

            /// \returns The size of the next block which is either the initial size or \c 0.
            std::size_t next_block_size() const noexcept
            {
                return block_allocator_type::next_block_size();
            }

            std::size_t block_count() const noexcept
            {
                return m_block_count;
            }

            DLContext device_context() const noexcept
            {
                return block_allocator_type::device_context();
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            allocator_type& get_allocator() noexcept
            {
                return block_allocator_type::get_allocator();
            }

            const allocator_type& get_allocator() const noexcept
            {
                return block_allocator_type::get_allocator();
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            block_allocator_type& get_block_allocator() noexcept
            {
                return *this;
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            const block_allocator_type& get_block_allocator() const noexcept
            {
                return *this;
            }

        private:
            allocator_info info() noexcept
            {
                return {TRTLAB_MEMORY_LOG_PREFIX "::count_limited_block_allocator", this};
            }

            std::size_t m_max_blocks;
            std::size_t m_block_count;
        };

        template <typename BlockAllocator>
        class size_limited_block_allocator : TRTLAB_EBO(BlockAllocator)
        {
        public:
            using block_allocator_type = BlockAllocator;
            using allocator_type       = typename block_allocator_type::allocator_type;
            using memory_type          = typename block_allocator_type::memory_type;

            size_limited_block_allocator(BlockAllocator&& alloc, std::size_t max_size)
            : block_allocator_type(std::move(alloc)), m_max_size(max_size), m_size(0)
            {
                DCHECK(m_max_size);
                static_assert(is_block_allocator<BlockAllocator>::value, "needs to be a valid block allocator");
            }

            memory_block allocate_block()
            {
                if (m_size + block_allocator_type::next_block_size() <= m_max_size)
                {
                    auto block = block_allocator_type::allocate_block();
                    m_size += block.size;
                    return block;
                }
                throw std::bad_alloc();
            }

            /// \effects Deallocates the previously allocated memory block.
            /// It also resets and allows a new call again.
            void deallocate_block(memory_block block) noexcept
            {
                DCHECK_LE(block.size, m_size);
                block_allocator_type::deallocate_block(block);
                m_size -= block.size;
            }

            /// \returns The size of the next block which is either the initial size or \c 0.
            std::size_t next_block_size() const noexcept
            {
                return block_allocator_type::next_block_size();
            }

            std::size_t bytes_allocated() const noexcept
            {
                return m_size;
            }

            DLContext device_context() const noexcept
            {
                return block_allocator_type::device_context();
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            allocator_type& get_allocator() noexcept
            {
                return block_allocator_type::get_allocator();
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            const allocator_type& get_allocator() const noexcept
            {
                return block_allocator_type::get_allocator();
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            block_allocator_type& get_block_allocator() noexcept
            {
                return *this;
            }

            /// \returns A reference to the used \concept{concept_rawallocator,RawAllocator} object.
            const block_allocator_type& get_block_allocator() const noexcept
            {
                return *this;
            }

        private:
            allocator_info info() noexcept
            {
                return {TRTLAB_MEMORY_LOG_PREFIX "::size_limited_block_allocator", this};
            }

            std::size_t m_max_size;
            std::size_t m_size;
        };

        namespace detail
        {
            template <class RawAlloc>
            using default_block_wrapper = fixed_size_block_allocator<RawAlloc>;

            template <template <class...> class Wrapper, class BlockAllocator, typename... Args>
            BlockAllocator make_block_allocator(std::true_type, BlockAllocator&& block_alloc, std::size_t block_size, Args&&... args)
            {
                return Wrapper<BlockAllocator>(std::forward<Args>(args)..., std::move(block_alloc));
            }

            template <template <class...> class BlockAllocator, class RawAlloc, typename... Args>
            auto make_block_allocator(std::false_type, RawAlloc&& alloc, std::size_t block_size, Args&&... args) -> BlockAllocator<RawAlloc>
            {
                return BlockAllocator<RawAlloc>(block_size, std::move(alloc), std::forward<Args>(args)...);
            }

        } // namespace detail

        /// Takes either a \concept{concept_blockallocator,BlockAllocator} or a \concept{concept_rawallocator,RawAllocator}.
        /// In the first case simply aliases the type unchanged, in the second to \ref growing_block_allocator (or the template in `BlockAllocator`) with the \concept{concept_rawallocator,RawAllocator}.
        /// Using this allows passing normal \concept{concept_rawallocator,RawAllocators} as \concept{concept_blockallocator,BlockAllocators}.
        /// \ingroup memory core
        template <class BlockOrRawAllocator, template <typename...> class BlockAllocator>
        using make_block_allocator_t =
            TRTLAB_IMPL_DEFINED(typename std::conditional<is_block_allocator<BlockOrRawAllocator>::value, BlockOrRawAllocator,
                                                          BlockAllocator<BlockOrRawAllocator>>::type);

        /// @{
        /// Helper function make a \concept{concept_blockallocator,BlockAllocator}.
        /// \returns A \concept{concept_blockallocator,BlockAllocator} of the given type created with the given arguments.
        /// \requires Same requirements as the constructor.
        /// \ingroup memory core
        /*
        template <class BlockOrRawAllocator, typename... Args>
        make_block_allocator_t<BlockOrRawAllocator> make_block_allocator(std::size_t block_size, Args&&... args)
        {
            return detail::make_block_allocator<
                detail::default_block_wrapper,
                BlockOrRawAllocator>(is_block_allocator<BlockOrRawAllocator>{}, block_size,
                                     detail::forward<Args>(args)...);
        }
        */

        /*
        template <template <class...> class BlockAllocator, typename RawAllocator, typename... Args>
        BlockAllocator<RawAllocator> make_block_allocator(RawAllocator&& alloc, std::size_t block_size, Args&&... args)
        {
            return BlockAllocator<RawAllocator>(block_size, std::move(alloc), std::forward<Args>(args)...);
        }
*/
        template <template <class...> class BlockAllocator, class BlockOrRawAllocator, typename... Args>
        make_block_allocator_t<BlockOrRawAllocator, BlockAllocator> make_block_allocator(BlockOrRawAllocator&& alloc, Args&&... args)
        {
            return detail::make_block_allocator<BlockAllocator, BlockOrRawAllocator>(is_block_allocator<BlockOrRawAllocator>{},
                                                                                     std::move(alloc), std::forward<Args>(args)...);
        }

        template <template <class...> class BlockAllocator, class BlockOrRawAllocator, typename... Args>
        make_block_allocator_t<BlockOrRawAllocator, BlockAllocator> make_block_allocator(Args&&... args)
        {
            static_assert(!is_block_allocator<BlockOrRawAllocator>{}, "should be a raw");
            BlockOrRawAllocator raw;
            return detail::make_block_allocator<BlockAllocator, BlockOrRawAllocator>(std::false_type{}, std::move(raw),
                                                                                     std::forward<Args>(args)...);
        }

        template <template <class...> class Extension, class BlockAllocator, typename... Args>
        Extension<BlockAllocator> make_extended_block_allocator(BlockAllocator&& block_alloc, Args&&... args)
        {
            static_assert(is_block_allocator<BlockAllocator>{}, "should be a block allocator");
            return Extension<BlockAllocator>(std::move(block_alloc), std::forward<Args>(args)...);
        }

    } // namespace memory
} // namespace trtlab