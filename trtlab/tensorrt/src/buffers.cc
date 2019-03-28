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
#include "trtlab/tensorrt/buffers.h"
#include "trtlab/tensorrt/bindings.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

using trtlab::CudaDeviceMemory;
using trtlab::CudaPinnedHostMemory;
using trtlab::MemoryStack;

namespace trtlab {
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

void Buffers::ConfigureBindings(const std::shared_ptr<Model>& model,
                                std::shared_ptr<Bindings> bindings)
{
    for(uint32_t i = 0; i < model->GetBindingsCount(); i++)
    {
        auto binding_size = model->GetBinding(i).bytesPerBatchItem * model->GetMaxBatchSize();
        DLOG(INFO) << "Configuring Binding " << i << ": pushing " << binding_size
                   << " to host/device stacks";
        bindings->SetHostAddress(i, AllocateHost(binding_size));
        bindings->SetDeviceAddress(i, AllocateDevice(binding_size));
    }
}

void Buffers::Synchronize()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS) << "Stream Sync failed";
}

} // namespace TensorRT
} // namespace trtlab
