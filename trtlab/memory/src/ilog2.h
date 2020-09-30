// MODIFICATION MESSAGE

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef TRTLAB_MEMORY_SRC_DETAIL_ILOG2_HPP_INCLUDED
#define TRTLAB_MEMORY_SRC_DETAIL_ILOG2_HPP_INCLUDED

#include "config.h"

#include <cstddef>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            // prioritized tag dispatching to choose smallest integer that fits
            struct clzll_tag
            {
            };
            struct clzl_tag : clzll_tag
            {
            };
            struct clz_tag : clzl_tag
            {
            };

            // also subtracts the number of addtional 0s that occur because the target type is smaller
            template <typename T, typename = typename std::enable_if<sizeof(T) <= sizeof(unsigned int)>::type>
            constexpr unsigned clz(clz_tag, T x)
            {
                return __builtin_clz(x) - (sizeof(unsigned int) * CHAR_BIT - sizeof(T) * CHAR_BIT);
            }

            template <typename T, typename = typename std::enable_if<sizeof(T) <= sizeof(unsigned long)>::type>
            constexpr unsigned clz(clzl_tag, T x)
            {
                return __builtin_clzl(x) - (sizeof(unsigned long) * CHAR_BIT - sizeof(T) * CHAR_BIT);
            }

            template <typename T, typename = typename std::enable_if<sizeof(T) <= sizeof(unsigned long long)>::type>
            constexpr unsigned clz(clzll_tag, T x)
            {
                return __builtin_clzll(x) - (sizeof(unsigned long long) * CHAR_BIT - sizeof(T) * CHAR_BIT);
            }
            constexpr unsigned clz(std::uint8_t x)
            {
                return detail::clz(detail::clz_tag{}, x);
            }

            constexpr unsigned clz(std::uint16_t x)
            {
                return detail::clz(detail::clz_tag{}, x);
                ;
            }

            constexpr unsigned clz(std::uint32_t x)
            {
                return detail::clz(detail::clz_tag{}, x);
            }

            constexpr unsigned clz(std::uint64_t x)
            {
                return detail::clz(detail::clz_tag{}, x);
            }

            // undefined for 0
            template <typename UInt>
            constexpr bool is_power_of_two(UInt x)
            {
                return (x & (x - 1)) == 0;
            }

            constexpr std::size_t ilog2_base(std::uint64_t x)
            {
                return sizeof(x) * CHAR_BIT - clz(x);
            }

            // ilog2() implementation, cuts part after the comma
            // e.g. 1 -> 0, 2 -> 1, 3 -> 1, 4 -> 2, 5 -> 2
            constexpr std::size_t ilog2(std::size_t x)
            {
                return ilog2_base(x) - 1;
            }

            // ceiling ilog2() implementation, adds one if part after comma
            // e.g. 1 -> 0, 2 -> 1, 3 -> 2, 4 -> 2, 5 -> 3
            constexpr std::size_t ilog2_ceil(std::size_t x)
            {
                // only subtract one if power of two
                return ilog2_base(x) - std::size_t(is_power_of_two(x));
            }
        } // namespace detail
    }     // namespace memory
} // namespace trtlab

#endif
