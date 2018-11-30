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
#ifndef _YAIS_TENSORRT_BUFFERS_H_
#define _YAIS_TENSORRT_BUFFERS_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "tensorrt/playground/memory.h"
#include "tensorrt/playground/memory_stack.h"
#include "YAIS/TensorRT/Common.h"

namespace yais
{
namespace TensorRT
{

/**
 * @brief Manages input/output buffers and CudaStream
 * 
 * Primary TensorRT resource class used to manage both a host and a device memory stacks
 * and owns the cudaStream_t that should be used for transfers or compute on these
 * resources.
 */
class Buffers : public std::enable_shared_from_this<Buffers>
{
    static auto Create(size_t host_size, size_t device_size) -> std::shared_ptr<Buffers>;

  public:
    virtual ~Buffers();

    void *AllocateHost(size_t size) { return m_HostStack->Allocate(size); }
    void *AllocateDevice(size_t size) { return m_DeviceStack->Allocate(size); }

    auto CreateBindings(const std::shared_ptr<Model> &model) -> std::shared_ptr<Bindings>; // todo: remove batchsize
    auto CreateAndConfigureBindings(const std::shared_ptr<Model> &model) -> std::shared_ptr<Bindings>; // todo: remove batchsize

    inline cudaStream_t Stream() { return m_Stream; }
    void Synchronize();

  private:
    Buffers(size_t host_size, size_t device_size);
    void Reset(bool writeZeros = false);

    std::shared_ptr<MemoryStack<CudaHostAllocator>> m_HostStack;
    std::shared_ptr<MemoryStack<CudaDeviceAllocator>> m_DeviceStack;
    cudaStream_t m_Stream;

    friend class ResourceManager;
};

} // namespace TensorRT
} // namespace yais

#endif // _YAIS_TENSORRT_BUFFERS_H_