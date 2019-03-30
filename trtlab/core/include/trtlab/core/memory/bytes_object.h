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
#include <glog/logging.h>
#include <memory>

#include "trtlab/core/memory/bytes_handle.h"

namespace trtlab {

template<typename T>
class BytesObject;

struct BytesProvider : public virtual std::enable_shared_from_this<BytesProvider>
{
    virtual const void* BytesProviderData() const = 0;
    virtual mem_size_t BytesProviderSize() const = 0;
    virtual const DLContext& BytesProviderDeviceInfo() const = 0;

    template<typename T>
    BytesObject<T> BytesObjectFromThis(void*, mem_size_t);

    template<typename T>
    friend class BytesObject;
};

template<typename T>
class BytesObject : protected BytesHandle<T>
{
  public:
    static BytesObject<T> Create(void*, mem_size_t, std::shared_ptr<BytesProvider>);
    virtual ~BytesObject() override{};

    BytesObject(BytesObject&&) noexcept;
    BytesObject& operator=(BytesObject&&) noexcept = default;

    BytesObject(const BytesObject&) = default;
    BytesObject& operator=(const BytesObject&) = default;

    // Expose a handle for copying
    const BytesHandle<T>& Handle() const { return *this; }

    // Elevate the protected getters from BytesHandle to public
    using BytesHandle<T>::Data;
    using BytesHandle<T>::Size;
    using BytesHandle<T>::DeviceInfo;

  protected:
    // With inline, the constructor segfaults using:
    // g++ (Ubuntu 5.4.0-6ubuntu1~16.04.11) 5.4.0 20160609
    // Reduces performance by 5ns from 38ns -> 43ns (dgx station 20-core broadwell)
    __attribute__((noinline)) BytesObject(void*, mem_size_t, std::shared_ptr<BytesProvider>);

    bool CheckBounds(void* ptr, mem_size_t, const BytesProvider&);

  private:
    std::shared_ptr<BytesProvider> m_BytesProvider;

    friend class BytesProvider;
};

template<typename T>
BytesObject<T>::BytesObject(void* ptr, mem_size_t size, std::shared_ptr<BytesProvider> provider)
    : BytesHandle<T>(ptr, size, provider->BytesProviderDeviceInfo()), m_BytesProvider(provider)
{
    DCHECK(CheckBounds(ptr, size, *m_BytesProvider));
}

template<typename T>
BytesObject<T> BytesObject<T>::Create(void* ptr, mem_size_t size,
                                      std::shared_ptr<BytesProvider> provider)
{
    // This is a public method, so we need to validate that the user provided
    // values for ptr and size and owned by the provider
    CheckBounds(ptr, size, *provider);
    return BytesObject<T>(ptr, size, provider);
}

template<typename T>
bool BytesObject<T>::CheckBounds(void* ptr, mem_size_t size, const BytesProvider& provider)
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
BytesObject<T>::BytesObject(BytesObject&& other) noexcept
    : m_BytesProvider(std::exchange(other.m_BytesProvider, nullptr)), BytesHandle<T>(
                                                                          std::move(other))
{
}

template<typename T>
BytesObject<T> BytesProvider::BytesObjectFromThis(void* ptr, mem_size_t size)
{
    return std::move(BytesObject<T>(ptr, size, shared_from_this()));
}

} // namespace trtlab