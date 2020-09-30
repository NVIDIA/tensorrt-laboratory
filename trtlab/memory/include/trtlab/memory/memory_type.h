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

#include <type_traits>
#include <cstdlib>

#include <dlpack/dlpack.h>

#include "detail/utility.h"

namespace trtlab
{
    namespace memory
    {
        namespace detail
        {
            struct any_memory;

            template <class MemoryType>
            std::true_type is_memory_type_impl(
                int, TRTLAB_SFINAE(std::declval<DLDeviceType&>() = std::declval<MemoryType&>().device_type()),
                TRTLAB_SFINAE(std::declval<std::size_t&>() = std::declval<MemoryType&>().min_allocation_alignment()),
                TRTLAB_SFINAE(std::declval<std::size_t&>() = std::declval<MemoryType&>().max_access_alignment()),
                TRTLAB_SFINAE(std::declval<std::size_t&>() = std::declval<MemoryType&>().access_alignment_for(0UL)));

            template <typename T>
            std::false_type is_memory_type_impl(short);

            template <typename T>
            struct check_memory_type
            {
                using has_base = typename std::is_base_of<any_memory, T>::type;
                using has_impl = decltype(is_memory_type_impl<T>(0));

                using valid = std::integral_constant<bool, has_base::value && has_impl::value>;
            };

        }

        template <typename T>
        struct is_memory_type : detail::check_memory_type<T>::valid
        {
        };

        namespace detail
        {
            class any_memory
            {
              protected:
                template <typename MemoryType>
                std::size_t static alignment_for(std::size_t size)
                {
                    static_assert(is_memory_type<MemoryType>::value, "");
                    auto max_alignment = MemoryType::max_access_alignment();
                    return (size >= max_alignment ? max_alignment : (std::size_t(1) << ilog2(size)));
                }

              private:
                static std::size_t ilog2(std::size_t) noexcept;
            };

        } // namespace detail



        // we can define a policy memory_type
        // this can hold methods to fill memory, etc.
        struct host_memory : detail::any_memory
        {
            constexpr static DLDeviceType device_type() noexcept
            {
                return kDLCPU;
            }
            constexpr static std::size_t min_allocation_alignment() noexcept
            {
                return 8UL;
            }
            constexpr static std::size_t max_access_alignment() noexcept
            {
                return 8UL;
            }
            static std::size_t access_alignment_for(std::size_t size)
            {
                using impl = detail::any_memory;
                return impl::alignment_for<host_memory>(size);
            }
        };

        namespace detail
        {
            template <typename T>
            struct check_host_memory
            {
                using has_base = typename std::is_base_of<host_memory, T>::type;
                using has_impl = decltype(is_memory_type_impl<T>(0));

                using valid = std::integral_constant<bool, has_base::value && has_impl::value>;
            };            
        }

        template <typename T>
        struct is_host_memory : detail::check_memory_type<T>::valid
        {
        };

    } // namespace memory
} // namespace trtlab