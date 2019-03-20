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
#include <string>

#include "tensorrt/laboratory/core/memory/descriptor.h"
#include "tensorrt/laboratory/core/memory/memory.h"
#include "tensorrt/laboratory/core/utils.h"

namespace trtlab {

namespace nextgen {
class HostDescriptor;
}

class HostMemory : public BaseMemory<HostMemory>
{
  public:
    using BaseMemory<HostMemory>::BaseMemory;
    static size_t DefaultAlignment();

    const char* TypeName() const override { return "HostMemory"; }

  protected:
    static DLContext DeviceContext();
};

namespace nextgen {



template<typename MemoryType>
class Descriptor : public MemoryType
{
  public:
    ~Descriptor() override { if(m_Deleter) { m_Deleter(); } }

    Descriptor(Descriptor<MemoryType>&& other) noexcept
      : m_Deleter(std::exchange(other.m_Deleter, nullptr)), MemoryType(std::move(other)) {}

  protected:
    Descriptor(void* ptr, mem_size_t size, std::function<void()> deleter);
    Descriptor(const DLTensor& dltensor, std::function<void()> deleter);
    // Moving should be allowed - with caveates
    // Moving should be allowed from a Descriptor<T> to a smart pointer of a Descriptor<T>.
    // Moving should not be allowed to downcast or change descriptor types
    Descriptor& operator=(Descriptor&&) noexcept = delete;

    Descriptor(const Descriptor<MemoryType>&&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;

  private:
    std::function<void()> m_Deleter;
};

template<typename MemoryType>
class SharedDescriptor : public Descriptor<MemoryType>, public std::enable_shared_from_this<SharedDescriptor<MemoryType>>
{
  public:
    explicit SharedDescriptor(Descriptor<MemoryType>&& owner)
        : Descriptor<MemoryType>(std::move(owner))
    {
    }
    ~SharedDescriptor() override {}

    operator DLTensor() { return this->DLPackDescriptor(); }
    operator DLTensor*() { return &(this->DLPackDescriptor()); }
};

class HostDescriptor : public Descriptor<HostMemory>
{
  public:
    using Descriptor<HostMemory>::Descriptor;
};

template<typename MemoryType>
Descriptor<MemoryType>::Descriptor(void* ptr, mem_size_t size, std::function<void()> deleter)
    : MemoryType(ptr, size, false), m_Deleter(deleter)
{
  // need to remove the allocated option
}

template<typename MemoryType>
Descriptor<MemoryType>::Descriptor(const DLTensor& dltensor, std::function<void()> deleter)
    : MemoryType(dltensor)
{
    if(this->DLPackDescriptor().ctx.device_type != kDLCPU)
    {
        throw std::runtime_error("Cannot create a HostDescriptor from the DLTensor of differing type");
    }
}


} // namespace nextgen
} // namespace trtlab
