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

#include <trtlab/memory/error.h>
#include <trtlab/memory/memory_type.h>

namespace trtlab
{
    namespace memory
    {
        struct device_memory : detail::any_memory
        {
            constexpr static DLDeviceType device_type() noexcept
            {
                return kDLGPU;
            }
            constexpr static std::size_t min_allocation_alignment() noexcept
            {
                return 256UL;
            }
            constexpr static std::size_t max_access_alignment() noexcept
            {
                return 64UL;
            }
            static std::size_t access_alignment_for(std::size_t size)
            {
                using impl = detail::any_memory;
                return impl::alignment_for<device_memory>(size);
            }
        };

        namespace detail
        {
            template <typename T>
            struct check_device_memory
            {
                using has_base = typename std::is_base_of<device_memory, T>::type;
                using has_impl = decltype(is_memory_type_impl<T>(0));

                using valid = std::integral_constant<bool, has_base::value && has_impl::value>;
            };
        } // namespace detail

        template <typename T>
        struct is_device_memory : detail::check_memory_type<T>::valid
        {
        };

        struct device_managed_memory : device_memory
        {
        };

        struct host_pinned_memory : host_memory
        {
            constexpr static DLDeviceType device_type()
            {
                return kDLCPUPinned;
            }
        };

    } // namespace memory
} // namespace trtlab