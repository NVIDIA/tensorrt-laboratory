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

#include "trtlab/core/memory/bytes_handle.h"

namespace trtlab {

template<typename T>
class BytesObject;

class IBytesProvider
{
    virtual const void* BytesProviderData() const = 0;
    virtual mem_size_t BytesProviderSize() const = 0;
    virtual const DLContext& BytesProviderDeviceInfo() const = 0;

    template<typename T>
    friend class BytesObject;
};

template<typename T>
class BytesProvider : public IBytesProvider, public std::enable_shared_from_this<BytesProvider<T>>
{
  protected:
    virtual const T& BytesProviderMemory() const = 0;
    BytesObject<T> BytesObjectFromThis(void*, mem_size_t);
    friend class BytesObject<T>;
};

template<typename T>
class BytesObject final : protected BytesHandle<T>
{
  public:
    static BytesObject<T> Create(void*, mem_size_t, std::shared_ptr<BytesProvider<T>>);
    virtual ~BytesObject() override{};

    BytesObject(BytesObject&& other) noexcept
        : BytesHandle<T>(std::move(other)),
          m_BytesProvider(std::exchange(other.m_BytesProvider, nullptr)) {}
    BytesObject& operator=(BytesObject&&) noexcept = default;

    BytesObject(const BytesObject&) = default;
    BytesObject& operator=(const BytesObject&) = default;

    const T& Memory() const;

    // Expose a handle for copying
    const BytesHandle<T>& Handle() const { return *this; }

    // Downcast to the BaseType, e.g. CudaDeviceMemory -> DeviceMemory
    BytesObject<typename T::BaseType> BaseObject() { return Cast<typename T::BaseType>(); }

    // Elevate the protected getters from BytesHandle to public
    using BytesHandle<T>::Data;
    using BytesHandle<T>::Size;
    using BytesHandle<T>::DeviceInfo;

  protected:
    // Inlining causes the constructor to segfault with:
    // g++ (Ubuntu 5.4.0-6ubuntu1~16.04.11) 5.4.0 20160609
    // Reduces performance of BenchStack by 5ns from 38ns -> 43ns (dgx station 20-core broadwell)
    //__attribute__((noinline)) BytesObject(void*, mem_size_t, std::shared_ptr<BytesProvider<T>>);
    BytesObject(void*ptr, mem_size_t size, std::shared_ptr<IBytesProvider> provider)
        : BytesHandle<T>(ptr, size, provider->BytesProviderDeviceInfo()), m_BytesProvider(provider)
    {
        DCHECK(CheckBounds(ptr, size, *m_BytesProvider));
    }

    // Validate the bytes array is owned by the provider
    bool CheckBounds(void* ptr, mem_size_t, const IBytesProvider&);

    // Downcast the backend storage calls to any parents of T
    template<typename U>
    BytesObject<U> Cast();

  private:
    std::shared_ptr<IBytesProvider> m_BytesProvider;

    // Allows only BytesProviders of type T to access the constructor
    friend class BytesProvider<T>;

    // Allows any BytesObject to convert to another type U via the constructor
    // In practice, we only allow Casting to a U where the converstion T* -> U* is valid
    template<typename U>
    friend class BytesObject::BytesObject;
};

template<typename T>
BytesObject<T> BytesObject<T>::Create(void* ptr, mem_size_t size,
                                      std::shared_ptr<BytesProvider<T>> provider)
{
    // This is a public method, so we need to validate that the user provided
    // values for ptr and size and owned by the provider
    CheckBounds(ptr, size, *provider);
    return BytesObject<T>(ptr, size, provider);
}

template<typename T>
bool BytesObject<T>::CheckBounds(void* ptr, mem_size_t size, const IBytesProvider& provider)
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

template<typename T>
const T& BytesObject<T>::Memory() const
{
    // auto tmp = std::dynamic_pointer_cast<BytesProvider<T>>(m_BytesProvider);
    auto base = m_BytesProvider.get();
    auto derived = dynamic_cast<T*>(base);
    if(derived == nullptr)
    {
        throw std::runtime_error("Unable to access Memory of downcasted BytesObject");
    }
    return derived->BytesProviderMemory();
}

template<typename From>
template<typename To>
BytesObject<To> BytesObject<From>::Cast()
{
    if(!std::is_convertible<From*, To*>::value)
    {
        throw std::bad_cast();
    }
    return BytesObject<To>(Data(), Size(), m_BytesProvider);
}

template<typename T>
BytesObject<T> BytesProvider<T>::BytesObjectFromThis(void* ptr, mem_size_t size)
{
    return BytesObject<T>(ptr, size,
                          std::enable_shared_from_this<BytesProvider<T>>::shared_from_this());
}

} // namespace trtlab