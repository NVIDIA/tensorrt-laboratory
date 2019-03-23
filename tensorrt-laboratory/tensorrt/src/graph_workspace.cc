/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "tensorrt/laboratory/graph_workspace.h"

#include "tensorrt/laboratory/core/memory/allocator.h"
#include "tensorrt/laboratory/core/utils.h"
#include "tensorrt/laboratory/cuda/device_info.h"

using trtlab::Memory::Allocator;
using trtlab::Memory::CudaDeviceMemory;
using trtlab::Memory::MemoryStack;

namespace {

std::size_t Align(std::size_t size, std::size_t alignment)
{
    std::size_t remainder = size % alignment;
    size = (remainder == 0) ? size : size + alignment - remainder;
    return size;
}

} // namespace

namespace trtlab {
namespace TensorRT {

GraphWorkspace::GraphWorkspace()
{
    DLOG(INFO) << "GraphWorkspace Constructor";
    CHECK_EQ(cudaStreamCreate(&m_Stream), cudaSuccess);
}

GraphWorkspace::~GraphWorkspace()
{
    DLOG(INFO) << "GraphWorkspace Deconstructor";
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS);
    CHECK_EQ(cudaStreamDestroy(m_Stream), CUDA_SUCCESS);
}

void GraphWorkspace::RegisterModel(const std::string& name, std::shared_ptr<Model> model)
{
    if(m_BindingsStack || m_ActivationSpace)
    {
        LOG(FATAL) << "Registration of new models is not allowed after graph creation";
    }

    auto item = m_Models.find(name);
    if(item != m_Models.end())
    {
        LOG(FATAL) << "Model naming collsion; Model with name=" << name
                   << " is already registered.";
    }

    // Size according to largest padding - device alignment
    size_t bindings =
        model->GetBindingMemorySize() + model->GetBindingsCount() * DeviceInfo::Alignment();
    size_t activations = Align(model->GetActivationsMemorySize(), 128 * 1024); // add a cacheline

    size_t device = Align(bindings, 128 * 1024);

    m_DeviceStackSize = std::max(m_DeviceStackSize, device);
    m_ActivationsSize = std::max(m_ActivationsSize, activations);

    LOG(INFO) << "-- Registering Model: " << name << " --";
    LOG(INFO) << "Input/Output Tensors require " << BytesToString(device);
    LOG(INFO) << "Execution Activations require " << BytesToString(activations);
    auto weights = model->GetWeightsMemorySize();
    if(weights)
    {
        LOG(INFO) << "Weights require " << BytesToString(weights);
    }

    model->SetName(name);
    m_Models[name] = model;
    m_ExecutionContexts[name] = model->CreateExecutionContext();
}

void GraphWorkspace::BuildGraphs()
{
    // Allocate memory based on registration statistics
    m_BindingsStack = std::make_unique<MemoryStack<CudaDeviceMemory>>(m_DeviceStackSize);
    m_ActivationSpace = std::make_unique<Allocator<CudaDeviceMemory>>(m_ActivationsSize);

    for(const auto& item : m_Models)
    {
        // Unpack map entry { std::string, std::shared_ptr<Model> }
        const auto& name = item.first;
        const auto& model = *(item.second);

        LOG(INFO) << "Building graph for: " << name;

        // Set the IExecutionContext to use the Workspace's activation memory
        auto& ctx = m_ExecutionContexts.at(name);
        ctx->setDeviceMemory(m_ActivationSpace->Data());

        // Push Model Bindings on Device Memory Stack
        // These pointers will get baked into the graph
        auto max_batch_size = model.GetMaxBatchSize();
        std::vector<void*> bindings;
        bindings.resize(model.GetBindingsCount());
        for(int i = 0; i < model.GetBindingsCount(); i++)
        {
            const auto& info = model.GetBinding(i);
            auto size = max_batch_size * info.bytesPerBatchItem;
            bindings[i] = m_BindingsStack->Allocate(size);
        }

        // Build Graph
        cudaGraph_t graph;
        cudaStreamBeginCapture(m_Stream, cudaStreamCaptureModeGlobal);
        ctx->enqueue(max_batch_size, (void**)bindings.data(), m_Stream, nullptr);
        cudaStreamEndCapture(m_Stream, &graph);

        // Store the Graph by Model Name
        m_Graphs[name] = graph;
        m_DeviceBindings[name] = std::move(bindings);

        // Reset the Device Memory Stack
        m_BindingsStack->Reset();
    }
}

cudaGraph_t GraphWorkspace::GraphByName(const std::string& name)
{
    auto search = m_Graphs.find(name);
    if(search == m_Graphs.end())
    {
        throw std::runtime_error("No graph for " + name);
    }
    return search->second;
}

void GraphWorkspace::Synchronize()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS);
}

} // namespace TensorRT
} // namespace trtlab