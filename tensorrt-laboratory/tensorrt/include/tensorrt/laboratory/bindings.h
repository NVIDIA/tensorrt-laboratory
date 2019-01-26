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
#include <map>

#include "tensorrt/laboratory/common.h"
#include "tensorrt/laboratory/model.h"
#include "tensorrt/laboratory/buffers.h"

#include "tensorrt/laboratory/core/memory/descriptor.h"
#include "tensorrt/laboratory/core/memory/host_memory.h"
#include "tensorrt/laboratory/cuda/memory/device_memory.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace trtlab {
namespace TensorRT {

class Model;
class Buffers;

/**
 * @brief Manages memory addresses and transfers for input/output tensors.
 * 
 * Bindings manages the addresses for the input/output tensors.  Bindings are created
 * from a Buffers object and maintain a reference.  All device bindings must be configured
 * before calling ExecutionContext::Infer.  Similarly, the respective host binding must
 * be set before calling an of the implicit CopyTo/CopyFromDevice methods.
 * 
 * A Bindings object holds the state of the input/output tensors over the course of an
 * inference calculation.
 */
class Bindings
{
  public:
    using HostDescriptor = Memory::DescriptorHandle<Memory::HostMemory>;
    using DeviceDescriptor = Memory::DescriptorHandle<Memory::DeviceMemory>;

    virtual ~Bindings();

    void *HostAddress(uint32_t binding_id);
    void *DeviceAddress(uint32_t binding_id);
    void **DeviceAddresses();

    [[deprecated]] void SetHostAddress(int binding_id, void *addr);
    [[deprecated]] void SetDeviceAddress(int binding_id, void *addr);

    void SetHostAddress(int binding_id, HostDescriptor);
    void SetDeviceAddress(int binding_id, DeviceDescriptor);

    HostDescriptor& HostMemoryDescriptor(int binding_id);
    // const Memory::HostMemory& HostMemory(int binding_id) const;

    void *ActivationsAddress() { return m_ActivationsAddress; }
    void SetActivationsAddress(void *addr) { m_ActivationsAddress = addr; }

    void CopyToDevice(uint32_t);
    void CopyToDevice(const std::vector<uint32_t> &);
    void CopyToDevice(uint32_t, void *, size_t);

    void CopyFromDevice(uint32_t);
    void CopyFromDevice(const std::vector<uint32_t> &);
    void CopyFromDevice(uint32_t, void *, size_t);

    auto InputBindings() const { return m_Model->GetInputBindingIds(); }
    auto OutputBindings() const { return m_Model->GetOutputBindingIds(); }

    auto GetModel() -> const std::shared_ptr<Model> & { return m_Model; }
    auto BatchSize() const { return m_BatchSize; }
    void SetBatchSize(uint32_t);

    inline cudaStream_t Stream() const { return m_Buffers->Stream(); }
    void Synchronize() const { m_Buffers->Synchronize(); }

    size_t BindingSize(uint32_t binding_id) const;
    
  private:
    Bindings(const std::shared_ptr<Model>, const std::shared_ptr<Buffers>);

    const std::shared_ptr<Model> m_Model;
    const std::shared_ptr<Buffers> m_Buffers;
    uint32_t m_BatchSize;

    std::vector<void *> m_HostAddresses;
    std::vector<void *> m_DeviceAddresses;

    std::map<int, HostDescriptor> m_HostDescriptors;
    std::map<int, DeviceDescriptor> m_DeviceDescriptors;

    void *m_ActivationsAddress;

    friend class Buffers;
};

/*
class Workspace
{
  public:
    HostMemory& HostDescriptor(const std::string&);
    const HostMemory& HostDescriptor(const std::string&) const;
    void HostDescriptor(const std::string&, DescriptorHandle<HostMemory>);

    DeviceMemory& DeviceDescriptor(const std::string&);
    const HostMemory& DeviceDescriptor(const std::string&) const;
    void DeviceDescriptor(const std::string&, DescriptorHandle<DeviceMemory>);

    std::shared_ptr<Bindings> ExportBindings(const std::string& model);

  private:
    std::map<std::string, DescriptorHandle<HostMemory>> m_HostDescriptors;
    std::map<std::string, DescriptorHandle<DeviceMemory>> m_DeviceDescriptors;
    std::shared_ptr<InferenceManager> m_Resources;
};
*/

} // namespace TensorRT
} // namespace trtlab
