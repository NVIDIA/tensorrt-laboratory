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
#include "trtlab/core/memory/array.h"

namespace trtlab {

struct BytesBase : protected Array1D<void> {
  protected:
    using Array1D<void>::Array1D;

  public:
    using Array1D::Data;
    using Array1D::Size;
};

template<typename Backend>
class BytesHandle : public BytesBase
{
  public:
    virtual ~BytesHandle() override {}

    BytesHandle(BytesHandle&&) noexcept;
    BytesHandle& operator=(BytesHandle&&) noexcept = default;

    BytesHandle(const BytesHandle&) = default;
    BytesHandle& operator=(const BytesHandle&) = default;

    const DLContext& DeviceInfo() const { return m_DeviceContext; }

    template<typename T>
    BytesHandle<T> Cast();

  protected:
    BytesHandle(void*, mem_size_t, int);
    BytesHandle(void*, mem_size_t, DLContext);

  private:
    DLContext m_DeviceContext;

    template<typename T>
    friend class BytesHandle::BytesHandle;
};

template<typename Backend>
BytesHandle<Backend>::BytesHandle(void* ptr, mem_size_t size, int id)
    : BytesBase(ptr, size), m_DeviceContext{Backend::DeviceType(), id}
{
}

template<typename Backend>
BytesHandle<Backend>::BytesHandle(void* ptr, mem_size_t size, DLContext ctx)
    : BytesBase(ptr, size), m_DeviceContext{ctx}
{
}

template<typename Backend>
BytesHandle<Backend>::BytesHandle(BytesHandle&& other) noexcept
    : BytesBase(std::move(other)),
      m_DeviceContext(other.m_DeviceContext)
{
}

template<typename Backend>
template<typename T>
BytesHandle<T> BytesHandle<Backend>::Cast()
{
    if(!std::is_convertible<Backend*, T*>::value)
    {
        throw std::bad_cast();
    }
    return BytesHandle<T>(Data(), Size(), DeviceInfo());
}

} // namespace trtlab