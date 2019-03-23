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
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "tensorrt/laboratory/core/memory/memory_stack.h"
#include "tensorrt/laboratory/cuda/memory/cuda_device.h"
#include "tensorrt/laboratory/common.h"
#include "tensorrt/laboratory/model.h"

namespace trtlab {
namespace TensorRT {

class GraphWorkspace : public std::enable_shared_from_this<GraphWorkspace>
{
  public:
    GraphWorkspace();
    virtual ~GraphWorkspace();

    void RegisterModel(const std::string&, std::shared_ptr<Model>);
    void BuildGraphs();

    bool IsModelRegistered(const std::string&) const;
    cudaGraphExec_t GraphByName(const std::string& name);
    std::vector<void*> DeviceBindingsByName(const std::string& name);

    inline cudaStream_t Stream() { return m_Stream; }
    void Synchronize();

  private:
    size_t m_DeviceStackSize;
    size_t m_ActivationsSize;

    std::unique_ptr<Memory::MemoryStack<Memory::CudaDeviceMemory>> m_BindingsStack;
    std::unique_ptr<Memory::CudaDeviceMemory> m_ActivationSpace;

    std::map<std::string, std::shared_ptr<Model>> m_Models;
    std::map<std::string, std::vector<void*>> m_DeviceBindings;
    std::map<std::string, std::shared_ptr<::nvinfer1::IExecutionContext>> m_ExecutionContexts;
    std::map<std::string, cudaGraph_t> m_Graphs;
    std::map<std::string, cudaGraphExec_t> m_GraphExecutors;
    cudaStream_t m_Stream;
};

} // namespace TensorRT
} // namespace trtlab