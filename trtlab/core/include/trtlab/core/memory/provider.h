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
#include <memory>

#include "trtlab/core/memory/bytes.h"

#include "trtlab/core/memory/allocator.h"

namespace trtlab {

template<typename T>
class MemoryProvider : public BytesProvider<T>
{
    struct internal_guard {};

  public:
    static Bytes<T> Allocate(mem_size_t size)
    {
        auto shared = std::make_shared<MemoryProvider<T>>(
            std::move(std::make_unique<Allocator<T>>(size)), internal_guard());
        return shared->GetBytes();
    }

    static Bytes<T> Expose(std::unique_ptr<Allocator<T>> memory)
    {
        auto shared = std::make_shared<MemoryProvider<T>>(std::move(memory), internal_guard());
        return shared->GetBytes();
    }

    static Bytes<T> Expose(Allocator<T>&& memory)
    {
        auto unique = std::make_unique<Allocator<T>>(std::move(memory));
        auto shared = std::make_shared<MemoryProvider<T>>(std::move(unique), internal_guard());
        return shared->GetBytes();
    }

    MemoryProvider(std::unique_ptr<Allocator<T>> memory, internal_guard)
        : m_Memory(std::move(memory))
    {
    }

    ~MemoryProvider() { DLOG(INFO) << "MemoryProvider deleting: " << *m_Memory; }

  protected:
    Bytes<T> GetBytes()
    {
        return this->BytesFromThis(m_Memory->Data(), m_Memory->Size());
    }

  private:
    const void* BytesProviderData() const final override { return m_Memory->Data(); }
    mem_size_t BytesProviderSize() const final override { return m_Memory->Size(); }
    const DLContext& BytesProviderDeviceInfo() const final override { return m_Memory->DeviceInfo(); }
    const T& BytesProviderMemory() const final override { return *m_Memory; }

    std::unique_ptr<Allocator<T>> m_Memory;
};

} // namespace trtlab