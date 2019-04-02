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
#include <utility>

#include "trtlab/core/memory/common.h"
#include "trtlab/core/types.h"

namespace trtlab {

template<typename T>
struct IArrayImmutable
{
    virtual const T* Data() const = 0;
    virtual uint64_t Size() const = 0;
    virtual uint64_t Bytes() const = 0;
    virtual uint64_t ItemSize() const = 0;
    virtual int NDims() const = 0;
    // virtual int64_t* Shape() const = 0;
};

template<typename T>
struct IArray : public IArrayImmutable<T>
{
    virtual T* Data() = 0;
};

template<typename T>
class Array1D : public IArray<T>
{
  public:
    virtual ~Array1D() {}

    Array1D(Array1D&&) noexcept;
    Array1D& operator=(Array1D&&) noexcept = default;

    Array1D(const Array1D&) = default;
    Array1D& operator=(const Array1D&) = default;

    T* Data() final override { return m_Data; };
    const T* Data() const final override { return m_Data; }
    uint64_t Size() const final override { return m_Size; }
    uint64_t Bytes() const final override { return m_Size * ItemSize(); }
    uint64_t ItemSize() const final override { return types::ArrayType<T>::ItemSize(); }
    int NDims() const final override { return 1; }

  protected:
    Array1D(T*, uint64_t);

  private:
    T* m_Data;
    uint64_t m_Size;
    uint64_t m_Capacity;
};

template<typename T>
Array1D<T>::Array1D(T* ptr, uint64_t size) : m_Data(ptr), m_Size(size), m_Capacity(size)
{
}

template<typename T>
Array1D<T>::Array1D(Array1D&& other) noexcept
    : m_Data{std::exchange(other.m_Data, nullptr)}, m_Size{std::exchange(other.m_Size, 0)},
      m_Capacity{std::exchange(other.m_Capacity, 0)}
{
}

/*
template<typename T, int N>
uint64_t Array1D<T, N>::Size() const
{
    return std::accumulate(&m_Shape[0], &m_Shape[N], uint64_t(1), std::multiplies<uint64_t>());
}
*/

} // namespace trtlab