/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt/playground/core/memory/memory.h"
#include "tensorrt/playground/core/memory/host_memory.h"

namespace yais {
namespace Memory {

class DeviceMemory : public BaseMemory<DeviceMemory>
{
  public:
    using BaseMemory<DeviceMemory>::BaseMemory;
    const std::string& Type() const override;

    void Fill(char) override;
    size_t DefaultAlignment() const override;
};

/**
 * @brief Allocates CUDA Device Memory
 *
 * Derived Memory class for GPU memory management using cudaMalloc and cudaFree.
 */
class CudaDeviceMemory : public DeviceMemory, public IAllocatable
{
  public:
    using DeviceMemory::DeviceMemory;
    const std::string& Type() const final override;

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;
};

/**
 * @brief Allocates CUDA Managed Memory
 *
 * Allocates memory that will be automatically managed by the Unified Memory system.
 */
class CudaManagedMemory : public DeviceMemory, public IAllocatable
{
  public:
    using DeviceMemory::DeviceMemory;
    const std::string& Type() const final override;

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;
};

/**
 * @brief Allocates Page Locked (Pinned) Memory on the Host
 *
 * Allocated page locked host memory using cudaMallocHost.  Pinned memory can provided better
 * performance, but should be used sparingly for staging areas for H2D and D2H transfers.
 */
class CudaPinnedHostMemory : public HostMemory, public IAllocatable
{
  public:
    using HostMemory::HostMemory;
    const std::string& Type() const final override;

  protected:
    void* Allocate(size_t) final override;
    void Free() final override;
};

} // namespace Memory
} // namespace yais
