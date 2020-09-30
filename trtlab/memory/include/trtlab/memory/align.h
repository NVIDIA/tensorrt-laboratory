// MODIFICATION_MESSAGE

// Modification notes:
// - alignment is no longer a detail
// - removed compat headers for alignas, alignof and max_align_t

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_DETAIL_ALIGN_H_INCLUDED
#define TRTLAB_MEMORY_DETAIL_ALIGN_H_INCLUDED

#include <cstdint>

#include "config.h"
#include "detail/assert.h"

namespace trtlab
{
    namespace memory
    {
        using addr_t = unsigned char*;

        // whether or not an alignment is valid, i.e. a power of two and not zero
        constexpr bool is_valid_alignment(std::size_t alignment) noexcept
        {
            return alignment && (alignment & (alignment - 1)) == 0u;
        }

        // align up to a power of 2, align_bytes is expected to be a nonzero
        inline std::size_t align_up(std::size_t v, std::size_t alignment) noexcept
        {
            return (v + (alignment - 1)) & ~(alignment - 1);
        }

        // returns the offset needed to align ptr for given alignment
        // alignment must be valid
        inline std::size_t align_offset(std::uintptr_t address, std::size_t alignment) noexcept
        {
            TRTLAB_MEMORY_ASSERT(is_valid_alignment(alignment));
            auto misaligned = address & (alignment - 1);
            return misaligned != 0 ? (alignment - misaligned) : 0;
        }

        inline std::size_t align_offset(void* ptr, std::size_t alignment) noexcept
        {
            return align_offset(reinterpret_cast<std::uintptr_t>(ptr), alignment);
        }

        inline void* memory_shift(void* memory, std::int64_t size)
        {
            return static_cast<void*>(reinterpret_cast<addr_t>(memory) + size);
        }

        inline std::pair<void*, std::size_t> align_shift(void* ptr, std::size_t alignment) noexcept
        {
            auto offset = align_offset(ptr, alignment);
            return std::make_pair(memory_shift(ptr, offset), offset);
        }

        // whether or not the pointer is aligned for given alignment
        // alignment must be valid
        bool is_aligned(void* ptr, std::size_t alignment) noexcept;

        // this need to be abstracted into memory_type
        // returns the minimum alignment required for a node of given size
        std::size_t alignment_for(std::size_t size) noexcept;



        std::size_t ilog2(std::size_t x);
        std::size_t ilog2_ceil(std::size_t x);

    } // namespace memory
} // namespace trtlab

#endif // TRTLAB_MEMORY_DETAIL_ALIGN_H_INCLUDED
