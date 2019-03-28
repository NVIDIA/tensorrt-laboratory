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

#include "trtlab/core/memory/malloc.h"
#include "trtlab/cuda/memory/cuda_device.h"
#include "trtlab/cuda/memory/cuda_pinned_host.h"

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

py::capsule DLPack::Export(py::object wrapped_memory)
{
    // Grab the wrapped memory object from the python object
    auto memory = wrapped_memory.cast<std::shared_ptr<CoreMemory>>();

    // Acquire a stake in the ownership of the python object by increasing the ref_count
    auto handle = wrapped_memory.cast<py::handle>();
    DLOG(INFO) << "DLPack::Manager increment ref_count to obj " << handle.ptr();
    handle.inc_ref();
    auto manager = new DLPack::Manager(memory->TensorInfo(), [handle]() mutable {
        DLOG(INFO) << "DLPack::Manager acquire gil and decrement ref_count to obj " << handle.ptr();
        py::gil_scoped_acquire acquire;
        handle.dec_ref();
    });
    return manager->Capsule();
}

py::capsule DLPack::Export(std::shared_ptr<CoreMemory> memory)
{
    DLOG(INFO) << "DLPack::Manager capturing shared_ptr to " << memory.get();
    auto manager = new DLPack::Manager(memory->TensorInfo(), [memory]() mutable {
        DLOG(INFO) << "Decrement use count (" << memory.use_count()
                   << ") of shared_ptr to: " << *memory.get();
        memory.reset();
    });
    return manager->Capsule();
}

std::shared_ptr<CoreMemory> DLPack::Import(py::capsule obj)
{
    DLOG(INFO) << "Importing DLPack Capsule @ " << obj.ptr();
    DLManagedTensor* dlm_tensor = (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    if(dlm_tensor == nullptr)
    {
        throw std::runtime_error("dltensor not found in capsule");
    }

    // by renaming the capsule from "dltensor" to "used_dltensor" we assume responsibility of
    // calling the dlm_tensor->deleter method to signal we are finished with the tensor memory
    PyCapsule_SetName(obj.ptr(), "used_dltensor");

    const auto& dltensor = dlm_tensor->dl_tensor;

    auto deleter = [dlm_tensor]() mutable {
        if(dlm_tensor->deleter)
        {
            dlm_tensor->deleter(dlm_tensor);
        }
    };

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

// DLPack::Manager Implementation

DLPack::Manager::Manager(const DLTensor& dltensor, std::function<void()> releaser)
    : m_Releaser(releaser)
{
    m_ManagedTensor.dl_tensor = dltensor;
    m_ManagedTensor.manager_ctx = static_cast<void*>(this);
    m_ManagedTensor.deleter = [](DLManagedTensor* ptr) mutable {
        DCHECK(ptr);
        Manager* self = (Manager*)ptr->manager_ctx;
        DCHECK(self);
        DLOG(INFO) << "DLManagedTensor deleter - free DLPack::Manager " << self;
        delete self;
    };
}

DLPack::Manager::~Manager()
{
    DCHECK(m_Releaser);
    DLOG(INFO) << "DLPack::Manager releasing managed tensor";
    m_Releaser();
}

py::capsule DLPack::Manager::Capsule()
{
    auto capsule = py::capsule((void*)&m_ManagedTensor, "dltensor", [](PyObject* ptr) {
        std::string name = PyCapsule_GetName(ptr);
        DLOG(INFO) << "DLPack::Capsule deleter " << name << " @ " << ptr;
        if(name == "used_dltensor")
        {
            DLOG(INFO) << "DLPack::Capsule deferring call to managed_tensor->deleter";
            return;
        }
        DLOG(INFO) << "DLPack::Capsule calling managed_tensor->deleter";
        auto obj = PyCapsule_GetPointer(ptr, name.c_str());
        DCHECK(obj);
        auto managed_tensor = static_cast<DLManagedTensor*>(obj);
        DCHECK(managed_tensor->deleter);
        managed_tensor->deleter(managed_tensor);
    });
    DLOG(INFO) << "DLPack::Capsule created dltensor @ " << capsule.ptr();
    return capsule;
}

} // namespace python
} // namespace trtlab
