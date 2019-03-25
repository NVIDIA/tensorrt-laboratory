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
#include "dlpack.h"

#include "tensorrt/laboratory/core/memory/malloc.h"
#include "tensorrt/laboratory/cuda/memory/cuda_device.h"
#include "tensorrt/laboratory/cuda/memory/cuda_pinned_host.h"

namespace trtlab {

// Anonymous namespace for the DLPack Descriptor
namespace {
template<typename MemoryType>
class DLPackDescriptor : public Descriptor<MemoryType>
{
  public:
    DLPackDescriptor(const DLTensor& dltensor, std::function<void()> deleter)
        : Descriptor<MemoryType>(dltensor, deleter, "DLPack")
    {
    }

    ~DLPackDescriptor() override {}
};
} // namespace

namespace python {

py::capsule DLPack::Export(std::shared_ptr<CoreMemory> memory)
{
    auto self = new DLPack(memory);
    return self->CreateCapsule();
}

std::shared_ptr<CoreMemory> DLPack::Import(py::capsule obj)
{
    DLOG(INFO) << "Importing DLPack Capsule @ " << obj.ptr();
    DLManagedTensor* dlm_tensor = (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    if(dlm_tensor == nullptr)
    {
        throw std::runtime_error("dltensor not found in capsule");
    }
    const auto& dltensor = dlm_tensor->dl_tensor;
    // take ownership of the capsule by incrementing the reference count on the handle
    // note: we use a py::handle object instead of a py::object becasue we need to manually
    // control the reference counting in the deleter after re-acquiring the GIL
    auto count = obj.ref_count();
    auto handle = obj.cast<py::handle>();
    handle.inc_ref();
    auto deleter = [handle] {
        DLOG(INFO) << "DLPack Wrapper Releasing Ownership of Capsule @ " << handle.ptr();
        py::gil_scoped_acquire acquire;
        handle.dec_ref();
    };
    DCHECK_EQ(obj.ref_count(), count + 1);
    PyCapsule_SetName(obj.ptr(), "used_dltensor");

    if(dltensor.ctx.device_type == kDLCPU)
    {
        return std::make_shared<DLPackDescriptor<Malloc>>(dltensor, deleter);
    }
    else if(dltensor.ctx.device_type == kDLCPUPinned)
    {
        return std::make_shared<DLPackDescriptor<CudaPinnedHostMemory>>(dltensor, deleter);
    }
    else if(dltensor.ctx.device_type == kDLGPU)
    {
        return std::make_shared<DLPackDescriptor<CudaDeviceMemory>>(dltensor, deleter);
    }
    else
    {
        throw std::runtime_error("Invalid DLPack device_type");
    }
    return nullptr;
}

// Non-static DLPack Implementation

DLPack::DLPack(std::shared_ptr<CoreMemory> shared) : m_ManagedMemory(shared)
{
    m_ManagedTensor.dl_tensor = shared->TensorInfo();
    m_ManagedTensor.manager_ctx = static_cast<void*>(this);
    m_ManagedTensor.deleter = [](DLManagedTensor* ptr) mutable {
        if(ptr)
        {
            DLPack* self = (DLPack*)ptr->manager_ctx;
            if(self)
            {
                DLOG(INFO) << "Deleting DLPack Wrapper via DLManagedTensor::deleter";
                delete self;
            }
        }
    };
}

DLPack::~DLPack() {}

py::capsule DLPack::CreateCapsule()
{
    auto capsule = py::capsule((void*)&m_ManagedTensor, "dltensor", [](PyObject* ptr) {
        auto name = PyCapsule_GetName(ptr);
        DLOG(INFO) << "Destroying Capsule " << name << " @ " << ptr;
        auto obj = PyCapsule_GetPointer(ptr, name);
        DCHECK(obj);
        auto managed_tensor = static_cast<DLManagedTensor*>(obj);
        DCHECK(managed_tensor->deleter);
        managed_tensor->deleter(managed_tensor);
    });
    DLOG(INFO) << "Creating Capsule dltensor @ " << capsule.ptr();
    return capsule;
}

} // namespace python
} // namespace trtlab
