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

#include <cuda.h>
#include <cuda_runtime.h>

//#include "trtlab/tensorrt/common.h"
//#include "trtlab/core/memory/cyclic_allocator.h"
//#include "trtlab/core/memory/memory_stack.h"
//#include "trtlab/cuda/memory/cuda_device.h"
//#include "trtlab/cuda/memory/cuda_pinned_host.h"

namespace trtlab {
namespace TensorRT {

/**
 * @brief Manages inpu/output buffers and CudaStream
 *
 * Primary TensorRT resource class used to manage both a host and a device memory stacks
 * and owns the cudaStream_t that should be used for transfers or compute on these
 * resources.
 */

/*
class Buffers : public std::enable_shared_from_this<Buffers>
{
  public:
    Buffers();
    virtual ~Buffers();

    auto CreateBindings(const std::shared_ptr<Model>&) -> std::shared_ptr<Bindings>;

    inline cudaStream_t Stream() { return m_Stream; }
    void Synchronize();

  protected:
    virtual void Reset() = 0;
    void ConfigureBindings(const std::shared_ptr<Model>& model, std::shared_ptr<Bindings>);

    virtual std::unique_ptr<HostMemory> AllocateHost(size_t size) = 0;
    virtual std::unique_ptr<DeviceMemory> AllocateDevice(size_t size) = 0;

  private:
    cudaStream_t m_Stream;
    friend class InferenceManager;
};

template<typename HostMemoryType, typename DeviceMemoryType>
class FixedBuffers : public Buffers
{
  public:
    FixedBuffers(size_t host_size, size_t device_size)
        : m_HostStack(std::make_unique<MemoryStack<HostMemoryType>>(host_size)),
          m_DeviceStack(std::make_unique<MemoryStack<DeviceMemoryType>>(device_size)), Buffers()
    {
    }

    ~FixedBuffers() override {}

  protected:
    template<typename MemoryType>
    class BufferStackDescriptor final : public Descriptor<MemoryType>
    {
      public:
        BufferStackDescriptor(void* ptr, size_t size)
            : Descriptor<MemoryType>(ptr, size, []{},
                std::string("BufferStack<" + std::string(MemoryType::TypeName()) + ">").c_str())
        {
        }
        ~BufferStackDescriptor() final override {}
    };

    std::unique_ptr<HostMemory> AllocateHost(size_t size) final override
    {
        return std::move(std::make_unique<BufferStackDescriptor<HostMemoryType>>(
            m_HostStack->Allocate(size), size));
    }

    std::unique_ptr<DeviceMemory> AllocateDevice(size_t size) final override
    {
        return std::move(std::make_unique<BufferStackDescriptor<DeviceMemoryType>>(
            m_DeviceStack->Allocate(size), size));
    }

    void Reset() final override
    {
        m_HostStack->Reset();
        m_DeviceStack->Reset();
    }

  private:
    std::unique_ptr<MemoryStack<HostMemoryType>> m_HostStack;
    std::unique_ptr<MemoryStack<DeviceMemoryType>> m_DeviceStack;
};

template<typename HostMemoryType, typename DeviceMemoryType>
class CyclicBuffers : public Buffers
{
  public:
    using HostAllocatorType = std::unique_ptr<CyclicAllocator<HostMemoryType>>;
    using DeviceAllocatorType = std::unique_ptr<CyclicAllocator<DeviceMemoryType>>;

    using HostDescriptor = typename CyclicAllocator<HostMemoryType>::Descriptor;
    using DeviceDescriptor = typename CyclicAllocator<DeviceMemoryType>::Descriptor;

    CyclicBuffers(HostAllocatorType host, DeviceAllocatorType device)
        : m_HostAllocator{std::move(host)}, m_DeviceAllocator{std::move(device)}
    {
    }
    ~CyclicBuffers() override {}

    std::unique_ptr<HostMemory> AllocateHost(size_t size)
    {
        return m_HostAllocator->Allocate(size);
    }

    std::unique_ptr<DeviceMemory> AllocateDevice(size_t size)
    {
        return m_DeviceAllocator->Allocate(size);
    }

    void Reset() final override {}

  private:
    HostAllocatorType m_HostAllocator;
    DeviceAllocatorType m_DeviceAllocator;
};
*/

} // namespace TensorRT
} // namespace trtlab
