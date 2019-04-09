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

#include <glog/logging.h>

#include "trtlab/core/memory/array.h"
#include "trtlab/core/memory/tensor_shape.h"

namespace trtlab {

template<typename T>
class Bytes;

class IBytesProvider
{
    virtual const void* BytesProviderData() const = 0;
    virtual mem_size_t BytesProviderSize() const = 0;
    virtual const DLContext& BytesProviderDeviceInfo() const = 0;

    template<typename T>
    friend class Bytes;

    template<typename T>
    friend class BytesBaseType;
};

template<typename T>
class BytesProvider : public IBytesProvider, public std::enable_shared_from_this<BytesProvider<T>>
{
  protected:
    virtual const T& BytesProviderMemory() const = 0;
    Bytes<T> BytesFromThis(void*, mem_size_t);
    friend class Bytes<T>;
};

struct BytesBase : public ITensorShape
{
  public:
    BytesBase(void* ptr, dims_t size) : m_Data(ptr), m_Size(size), m_NDims(1)
    {
        DCHECK_GE(size, 0);
    }
    virtual ~BytesBase() {}

    // ITensorShape Interface
    uint32_t NDims() const final override { return m_NDims; }
    const dims_t* Shape() const final override { return &m_Size; }
    const dims_t* Strides() const final override { return &m_NDims; }
    uint64_t Size() const final override { return (uint64_t)m_Size; }

    void* Data() { return m_Data; }
    const void* Data() const { return m_Data; }

  protected:
    BytesBase() : m_Data(nullptr), m_Size(0), m_NDims(0) {}

    BytesBase(BytesBase&& other) noexcept
        : m_Data(std::exchange(other.m_Data, nullptr)),
          m_Size(std::exchange(other.m_Size, 0)),
          m_NDims(std::exchange(other.m_NDims, 0))
    {
    }
    BytesBase& operator=(BytesBase&&) noexcept = default;

    BytesBase(const BytesBase&) = delete;
    BytesBase& operator=(const BytesBase&) = delete;

  private:
    void* m_Data;
    dims_t m_Size;
    int64_t m_NDims;
};

template<typename BaseType>
class BytesBaseType : public BytesBase
{
  public:
    BytesBaseType(BytesBaseType&& other) noexcept
        : BytesBase(std::move(other)), m_Provider(std::exchange(other.m_Provider, nullptr))
    {
    }
    BytesBaseType& operator=(BytesBaseType&&) noexcept = default;

    BytesBaseType(const BytesBaseType&) = delete;
    BytesBaseType& operator=(const BytesBaseType&) = delete;

    void Release() { *this = BytesBaseType(nullptr, 0, nullptr); }
    const DLContext& DeviceInfo() const { return m_Provider->BytesProviderDeviceInfo(); }

  protected:
    BytesBaseType(void* ptr, uint64_t size, std::shared_ptr<IBytesProvider> provider)
        : BytesBase(ptr, size), m_Provider(provider)
    {
        DCHECK(CheckBounds(ptr, size, *m_Provider));
    }

    // Validate the bytes array is owned by the provider
    bool CheckBounds(void* ptr, mem_size_t, const IBytesProvider&);

    // Provide derived classes access to the BytesProvider
    const IBytesProvider* BytesProvider() const { return m_Provider.get(); }

  private:
    std::shared_ptr<const IBytesProvider> m_Provider;
};

template<typename MemoryType>
class Bytes final : public BytesBaseType<typename MemoryType::BaseType>
{
  public:
    virtual ~Bytes() override{};

    Bytes(Bytes&& other) noexcept = default;
    Bytes& operator=(Bytes&&) noexcept = default;

    Bytes(const Bytes&) = delete;
    Bytes& operator=(const Bytes&) = delete;

    const MemoryType& Memory() const;

  protected:
    Bytes(void* ptr, mem_size_t size, std::shared_ptr<BytesProvider<MemoryType>> provider)
        : BytesBaseType<typename MemoryType::BaseType>(ptr, size, std::move(provider))
    {
    }

    // Allows only BytesProviders of type T to access the constructor
    friend class BytesProvider<MemoryType>;
};

// Implementation Bytes

template<typename BaseType>
bool BytesBaseType<BaseType>::CheckBounds(void* ptr, mem_size_t size,
                                          const IBytesProvider& provider)
{
    // Validate Starting Address
    uint64_t that = reinterpret_cast<uint64_t>(provider.BytesProviderData());
    uint64_t self = reinterpret_cast<uint64_t>(ptr);
    CHECK_GE(self, that) << "Starting address out of bounds for provider";
    // Vaidate Length
    that += provider.BytesProviderSize();
    self += size;
    CHECK_LE(self, that) << "Ending address out of bounds for the provider";
    return true;
}

template<typename MemoryType>
const MemoryType& Bytes<MemoryType>::Memory() const
{
    auto base = this->BytesProvider();
    auto derived = dynamic_cast<const BytesProvider<MemoryType>*>(base);
    if(derived == nullptr)
    {
        throw std::runtime_error("Unable to access Memory of downcasted Bytes");
    }
    return derived->BytesProviderMemory();
}

// Implementation BytesProvider

template<typename T>
Bytes<T> BytesProvider<T>::BytesFromThis(void* ptr, mem_size_t size)
{
    return Bytes<T>(ptr, size, std::enable_shared_from_this<BytesProvider<T>>::shared_from_this());
}

} // namespace trtlab