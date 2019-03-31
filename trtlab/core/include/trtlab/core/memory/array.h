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

/*
template<typename T, int N>
class ArrayBase : public IArray<T>
{
  public:
    virtual ~ArrayBase() {}

    ArrayBase(ArrayBase&&) noexcept;
    ArrayBase& operator=(ArrayBase&&) noexcept = default;

    ArrayBase(const ArrayBase&);
    ArrayBase& operator=(const ArrayBase&);

    T* Data() final override { return m_Data; };
    const T* Data() const final override { return m_Data; }

    uint64_t Size() const final override;
    uint64_t Bytes() const final override;
    uint64_t ItemSize() const final override;

    int NDims() const final override { return N; }

  protected:
    ArrayBase(T*, int64_t shape[N]);

  private:
    T* m_Data;
    int64_t m_Shape[N];
    int64_t m_Strides[N];
};

template<typename T, int N>
ArrayBase<T, N>::ArrayBase(T* ptr, int64_t shape[N]) : m_Data(ptr)
{
    std::copy(shape, &shape[N], m_Shape);
}

template<typename T, int N>
uint64_t ArrayBase<T, N>::ItemSize() const
{
    return sizeof(T);
}

template<int N>
uint64_t ArrayBase<void, N>::ItemSize() const
{
    return 1;
}

template<typename T, int N>
uint64_t ArrayBase<T, N>::Size() const
{
    return std::accumulate(&m_Shape[0], &m_Shape[N], uint64_t(1), std::multiplies<uint64_t>());
}

template<typename T>
uint64_t ArrayBase<T, 1>::Size() const
{
    return m_Shape[0];
}

typename<typename T, int N> uint64_t ArrayBase<T, N>::Bytes() const { return Size() * ItemSize(); }
*/

template<typename Type>
class ArrayBase : public IArray<typename Type::NativeType>
{
  public:
    using T = typename Type::NativeType;

    virtual ~ArrayBase() {}

    ArrayBase(ArrayBase&&) noexcept;
    ArrayBase& operator=(ArrayBase&&) noexcept = default;

    ArrayBase(const ArrayBase&) = default;
    ArrayBase& operator=(const ArrayBase&) = default;

    T* Data() final override { return m_Data; };
    const T* Data() const final override { return m_Data; }
    uint64_t Size() const final override { return m_Size; }
    uint64_t Bytes() const final override { return m_Size * Type::ItemSize(); }
    uint64_t ItemSize() const final override { return Type::ItemSize(); }

    int NDims() const final override { return 1; }

  protected:
    ArrayBase(T*, uint64_t);

  private:
    T* m_Data;
    uint64_t m_Size;
};

template<typename Type>
ArrayBase<Type>::ArrayBase(T* ptr, uint64_t size) : m_Data(ptr), m_Size(size)
{
}

template<typename Type>
ArrayBase<Type>::ArrayBase(ArrayBase&& other) noexcept
    : m_Data{std::exchange(other.m_Data, nullptr)}, m_Size{std::exchange(other.m_Size, 0)}
{
}


} // namespace trtlab