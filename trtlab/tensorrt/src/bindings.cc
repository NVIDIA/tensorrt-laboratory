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
#include "trtlab/tensorrt/bindings.h"

#include <glog/logging.h>

#include "trtlab/core/memory/descriptor.h"

using trtlab::Descriptor;
using trtlab::DescriptorHandle;
using trtlab::DeviceMemory;
using trtlab::HostMemory;

/*
namespace {
class RawHostMemoryDescriptor final : public Descriptor<HostMemory>
{
  public:
    RawHostMemoryDescriptor(void* ptr, size_t size)
        : Descriptor<HostMemory>(ptr, size, "BindingsRawPtr")
    {
    }
    ~RawHostMemoryDescriptor() final override {}
};
} // namespace
*/

namespace trtlab {
namespace TensorRT {

Bindings::Bindings(const std::shared_ptr<Model> model, const std::shared_ptr<Buffers> buffers)
    : m_Model(model), m_Buffers(buffers), m_BatchSize(0)
{
    auto count = model->GetBindingsCount();
    m_HostAddresses.resize(count);
    m_DeviceAddresses.resize(count);
    for(auto i = 0; i < count; i++)
    {
        m_HostAddresses[i] = m_DeviceAddresses[i] = nullptr;
    }
}

Bindings::~Bindings() {}

typename Bindings::HostDescriptor& Bindings::HostMemoryDescriptor(int binding_id)
{
    CHECK_LT(binding_id, m_HostAddresses.size());
    return m_HostDescriptors[binding_id];
}

/*
void Bindings::SetHostAddress(int binding_id, void* addr)
{
    CHECK_LT(binding_id, m_HostAddresses.size());
    auto mdesc = std::make_unique<RawHostMemoryDescriptor>(addr, BindingSize(binding_id));
    m_HostAddresses[binding_id] = addr;
    m_HostDescriptors[binding_id] = std::move(mdesc);
}


void Bindings::SetDeviceAddress(int binding_id, void* addr)
{
    CHECK_LT(binding_id, m_DeviceAddresses.size());
    m_DeviceAddresses[binding_id] = addr;
    m_DeviceDescriptors.erase(binding_id);
}
*/

void Bindings::SetHostAddress(int binding_id, DescriptorHandle<HostMemory> mdesc)
{
    CHECK_LT(binding_id, m_HostAddresses.size());
    m_HostAddresses[binding_id] = mdesc->Data();
    m_HostDescriptors[binding_id] = std::move(mdesc);
}

void Bindings::SetDeviceAddress(int binding_id, DescriptorHandle<DeviceMemory> mdesc)
{
    CHECK_LT(binding_id, m_DeviceAddresses.size());
    m_DeviceAddresses[binding_id] = mdesc->Data();
    m_DeviceDescriptors[binding_id] = std::move(mdesc);
}

void* Bindings::HostAddress(uint32_t binding_id)
{
    CHECK_LT(binding_id, m_HostAddresses.size());
    return m_HostAddresses[binding_id];
}

void* Bindings::DeviceAddress(uint32_t binding_id)
{
    CHECK_LT(binding_id, m_DeviceAddresses.size());
    return m_DeviceAddresses[binding_id];
}

void** Bindings::DeviceAddresses() { return (void**)m_DeviceAddresses.data(); }

void Bindings::CopyToDevice(uint32_t device_binding_id)
{
    auto host_src = HostAddress(device_binding_id);
    auto bytes = BindingSize(device_binding_id);
    CopyToDevice(device_binding_id, host_src, bytes);
}

void Bindings::CopyToDevice(const std::vector<uint32_t>& ids)
{
    for(auto id : ids)
    {
        CopyToDevice(id);
    }
}

void Bindings::CopyToDevice(uint32_t device_binding_id, void* src, size_t bytes)
{
    auto dst = DeviceAddress(device_binding_id);
    DLOG(INFO) << "CopyToDevice binding_id: " << device_binding_id << "; size: " << bytes;
    CHECK_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, Stream()), CUDA_SUCCESS);
}

void Bindings::CopyFromDevice(uint32_t device_binding_id)
{
    auto host_dst = HostAddress(device_binding_id);
    auto bytes = BindingSize(device_binding_id);
    CopyFromDevice(device_binding_id, host_dst, bytes);
}

void Bindings::CopyFromDevice(const std::vector<uint32_t>& ids)
{
    for(auto id : ids)
    {
        CopyFromDevice(id);
    }
}

void Bindings::CopyFromDevice(uint32_t device_binding_id, void* dst, size_t bytes)
{
    auto src = DeviceAddress(device_binding_id);
    DLOG(INFO) << "CopyFromDevice binding_id: " << device_binding_id << "; size: " << bytes;
    CHECK_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, Stream()), CUDA_SUCCESS);
}

void Bindings::SetBatchSize(uint32_t batch_size)
{
    CHECK_LE(batch_size, m_Model->GetMaxBatchSize());
    m_BatchSize = batch_size;
}

size_t Bindings::BindingSize(uint32_t binding_id) const
{
    return m_Model->GetBinding(binding_id).bytesPerBatchItem *
           (m_BatchSize ? m_BatchSize : m_Model->GetMaxBatchSize());
}

} // namespace TensorRT
} // namespace trtlab
