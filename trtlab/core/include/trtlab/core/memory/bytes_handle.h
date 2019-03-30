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
#include <utility>

#include "trtlab/core/memory/common.h"

namespace trtlab {

namespace detail {
class UnsafeBytesHandleFactory;
}

template<typename T>
class BytesHandle : private IBytesHandle
{
  public:
    virtual ~BytesHandle() override {}

    BytesHandle(BytesHandle&&) noexcept;
    BytesHandle& operator=(BytesHandle&&) noexcept = default;

    BytesHandle(const BytesHandle&) = default;
    BytesHandle& operator=(const BytesHandle&) = default;

    void* Data() final override { return m_Data; };
    const void* Data() const final override { return m_Data; }
    mem_size_t Size() const final override { return m_Size; };
    const DLContext& DeviceInfo() const final override { return m_DeviceContext; }

  protected:
    BytesHandle(void*, mem_size_t, DLDeviceType, int);
    BytesHandle(void*, mem_size_t, DLContext);

  private:
    void* m_Data;
    mem_size_t m_Size;
    DLContext m_DeviceContext;

    friend class UnsafeBytesHandleFactory;
};

template<typename T>
BytesHandle<T>::BytesHandle(void* ptr, mem_size_t size, DLDeviceType d, int id)
    : m_Data(ptr), m_Size(size), m_DeviceContext{d, id}
{
}

template<typename T>
BytesHandle<T>::BytesHandle(void* ptr, mem_size_t size, DLContext ctx)
    : m_Data(ptr), m_Size(size), m_DeviceContext{ctx}
{
}

template<typename T>
BytesHandle<T>::BytesHandle(BytesHandle&& other) noexcept
    : m_Data{std::exchange(other.m_Data, nullptr)}, m_Size{std::exchange(other.m_Size, 0)},
      m_DeviceContext(other.m_DeviceContext)
{
}

namespace detail {

class UnsafeBytesHandleFactory
{
    template<typename T>
    BytesHandle<T> Create(void* ptr, mem_size_t size, DLContext ctx)
    {
        return BytesHandle<T>(ptr, size, ctx);
    }
};


class HostBytesHandle : public BytesHandle<StorageType::Host>
{
  public:
    HostBytesHandle(void* ptr, mem_size_t size)
        : BytesHandle<StorageType::Host>(ptr, size, kDLCPU, 0)
    {
    }
};

} // namespace detail

} // namespace trtlab