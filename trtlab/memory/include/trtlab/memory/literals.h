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

// MODIFICATION MESSAGE

// Modification Notes:
// - added TiB and TB
// - taken from foonathan/memory (memory_arena.hpp)

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#pragma once

namespace trtlab
{
    namespace memory
    {
        namespace literals
        {
            constexpr std::size_t operator"" _KiB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1024);
            }

            constexpr std::size_t operator"" _KB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1000);
            }

            constexpr std::size_t operator"" _MiB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1024 * 1024);
            }

            constexpr std::size_t operator"" _MB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1000 * 1000);
            }

            constexpr std::size_t operator"" _GiB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1024 * 1024 * 1024);
            }

            constexpr std::size_t operator"" _GB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1000 * 1000 * 1000);
            }

            constexpr std::size_t operator"" _TiB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1024 * 1024 * 1024 * 1024);
            }

            constexpr std::size_t operator"" _TB(unsigned long long value) noexcept
            {
                return std::size_t(value * 1000 * 1000 * 1000 * 1000);
            }
        } // namespace literals
    }     // namespace memory
} // namespace trtlab