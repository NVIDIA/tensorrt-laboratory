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
#include "tensorrt/playground/buffers.h"
#include "tensorrt/playground/bindings.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

using yais::Memory::MemoryStack;
using yais::Memory::CudaDeviceMemory;
using yais::Memory::CudaPinnedHostMemory;

namespace yais {
namespace TensorRT {

Buffers::Buffers()
{
    // CHECK(cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking) == cudaSuccess); <-- breaks
    CHECK_EQ(cudaStreamCreate(&m_Stream), cudaSuccess);
}

Buffers::~Buffers()
{
    DLOG(INFO) << "Buffers Deconstructor";
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS);
    CHECK_EQ(cudaStreamDestroy(m_Stream), CUDA_SUCCESS);
}

auto Buffers::CreateBindings(const std::shared_ptr<Model>& model) -> std::shared_ptr<Bindings>
{
    auto bindings = std::shared_ptr<Bindings>(new Bindings(model, shared_from_this()));
    ConfigureBindings(model, bindings);
    return bindings;
}

void Buffers::Synchronize()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS) << "Stream Sync failed";
}

FixedBuffers::FixedBuffers(size_t host_size, size_t device_size)
    : m_HostStack(std::make_unique<MemoryStack<CudaPinnedHostMemory>>(host_size)),
      m_DeviceStack(std::make_unique<MemoryStack<CudaDeviceMemory>>(device_size)), Buffers()
{
}

FixedBuffers::~FixedBuffers() {}

/**
 * @brief Pushes both Host and Device Stack Pointers for each Binding in a Model
 *
 * For each binding in the model, a stack pointer will be pushed on both host and device
 * memory stacks.  Buffers used a MemoryStackWithTracking object, so every Push is
 * recorded and the Pointer and the Size of each stack allocation can be recalled by
 * passing the index of the binding.
 *
 * @param model
 * @param batch_size
 * @return bindings
 */
void FixedBuffers::ConfigureBindings(const std::shared_ptr<Model>& model,
                                     std::shared_ptr<Bindings> bindings)
{
    for(uint32_t i = 0; i < model->GetBindingsCount(); i++)
    {
        auto binding_size = model->GetBinding(i).bytesPerBatchItem * model->GetMaxBatchSize();
        DLOG(INFO) << "Configuring Binding " << i << ": pushing " << binding_size
                   << " to host/device stacks";
        bindings->SetHostAddress(i, m_HostStack->Allocate(binding_size));
        bindings->SetDeviceAddress(i, m_DeviceStack->Allocate(binding_size));
    }
}

/**
 * @brief Resets the Host and Device Stack Pointers to their origins
 *
 * @param writeZeros
 */
void FixedBuffers::Reset(bool writeZeros)
{
    m_HostStack->Reset(writeZeros);
    m_DeviceStack->Reset(writeZeros);
}

} // namespace TensorRT
} // namespace yais
