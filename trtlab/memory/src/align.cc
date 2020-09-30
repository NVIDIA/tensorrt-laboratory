// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#include "align.h"
#include "detail/assert.h"

#include "ilog2.h"

using namespace trtlab::memory;
using namespace detail;

bool trtlab::memory::is_aligned(void* ptr, std::size_t alignment) noexcept
{
    TRTLAB_MEMORY_ASSERT(is_valid_alignment(alignment));
    auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return address % alignment == 0u;
}

std::size_t trtlab::memory::alignment_for(std::size_t size) noexcept
{
    return (size >= 8UL ? 8UL : (std::size_t(1) << ilog2(size)));
}

std::size_t trtlab::memory::ilog2(std::size_t x)
{
    return ilog2_base(x) - 1;
}

// ceiling ilog2() implementation, adds one if part after comma
// e.g. 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 2, 5 -> 3
std::size_t trtlab::memory::ilog2_ceil(std::size_t x)
{
    // only subtract one if power of two
    return ilog2_base(x) - std::size_t(is_power_of_two(x));
}