// MODIFICATION MESSAGE

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#pragma once
#include <stdexcept>

#include "config.h"

namespace trtlab
{
    namespace memory
    {
        /// A memory block.
        /// It is defined by its starting address and size.
        /// \ingroup memory core
        struct memory_block
        {
            void*       memory; ///< The address of the memory block (might be \c nullptr).
            std::size_t size;   ///< The size of the memory block (might be \c 0).

            /// \effects Creates an invalid memory block with starting address \c nullptr and size \c 0.
            memory_block() noexcept : memory_block(nullptr, std::size_t(0)) {}

            /// \effects Creates a memory block from a given starting address and size.
            memory_block(void* mem, std::size_t s) noexcept : memory(mem), size(s) {}

            /// \effects Creates a memory block from a [begin,end) range.
            memory_block(void* begin, void* end) noexcept
            : memory_block(begin, static_cast<std::size_t>(static_cast<char*>(end) - static_cast<char*>(begin)))
            {
            }

            memory_block(memory_block&& other) noexcept : memory(std::exchange(other.memory, nullptr)), size(std::exchange(other.size, 0u))
            {
            }

            memory_block& operator=(memory_block&& other) noexcept
            {
                memory = std::exchange(other.memory, nullptr);
                size   = std::exchange(other.size, 0u);
                return *this;
            }

            memory_block(const memory_block&) = default;
            memory_block& operator=(const memory_block&) = default;

            /// \returns Whether or not a pointer is inside the memory.
            bool contains(const void* address) const noexcept
            {
                auto mem  = static_cast<const char*>(memory);
                auto addr = static_cast<const char*>(address);
                return addr >= mem && addr < mem + size;
            }

            std::uintptr_t distance(void* ptr)
            {
                if (!contains(ptr))
                    throw std::runtime_error("cannot compute distance - ptr not owned by block");
                auto s = reinterpret_cast<std::uintptr_t>(memory);
                auto e = reinterpret_cast<std::uintptr_t>(ptr);
                return e - s;
            }

            void* offset(std::size_t distance)
            {
                if (distance > size)
                    return nullptr;
                auto mem = static_cast<char*>(memory);
                return mem + distance;
            }
        };

        template <typename Compare = std::less<>>
        struct memory_block_compare_size
        {
            using is_transparent = void;

            constexpr bool operator()(std::size_t size, const memory_block& block, Compare compare = Compare()) const
            {
                return compare(size, block.size);
            }

            constexpr bool operator()(const memory_block& block, std::size_t size, Compare compare = Compare()) const
            {
                return compare(block.size, size);
            }

            constexpr bool operator()(const memory_block& lhs, const memory_block& rhs, Compare compare = Compare()) const
            {
                if (compare(lhs.size, rhs.size))
                    return true;
                else if (lhs.size == rhs.size && compare(reinterpret_cast<addr_t>(lhs.memory), reinterpret_cast<addr_t>(rhs.memory)))
                    return true;
                return false;
            }
        };

        template <typename Compare = std::less<>>
        struct memory_block_compare_addr
        {
            using is_transparent = void;

            constexpr bool operator()(void* addr, const memory_block& block, Compare compare = Compare()) const
            {
                return compare(reinterpret_cast<addr_t>(addr), reinterpret_cast<addr_t>(block.memory));
            }

            constexpr bool operator()(const memory_block& block, void* addr, Compare compare = Compare()) const
            {
                return compare(reinterpret_cast<addr_t>(block.memory), reinterpret_cast<addr_t>(addr));
            }

            constexpr bool operator()(const memory_block& lhs, const memory_block& rhs, Compare compare = Compare()) const
            {
                return compare(reinterpret_cast<addr_t>(lhs.memory), reinterpret_cast<addr_t>(rhs.memory));
            }
        };

    } // namespace memory
} // namespace trtlab
