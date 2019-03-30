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

#include "trtlab/core/memory/common.h"

namespace trtlab {

template<class MemoryType>
class Allocator final : public MemoryType
{
  public:
    Allocator(mem_size_t size);
    virtual ~Allocator() override;

    Allocator(Allocator<MemoryType>&& other) noexcept;
    Allocator<MemoryType>& operator=(Allocator<MemoryType>&& other) noexcept = delete;

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
};

/*
namespace nextgen {

template<class MemoryType>
class Allocator final : public Descriptor<MemoryType>
{
  public:
    Allocator(mem_size_t size)
        : Descriptor<MemoryType>(MemoryType::Allocate(size), size,
                                 [](void* ptr) { MemoryType::Free(ptr); })
    {
        auto ctx = this->DeviceContext();
        this->m_Handle.ctx.device_type = ctx.device_type;
        this->m_Handle.ctx.device_id = ctx.device_id;
    }

    virtual ~Allocator() override {}

  protected:
    Allocator(Allocator&& other) noexcept : Descriptor<MemoryType>(std::move(other)) {}
    Allocator& operator=(Allocator&& other) noexcept = delete;

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
};

} // namespace nextgen
*/

} // namespace trtlab

#include "trtlab/core/impl/memory/allocator.h"