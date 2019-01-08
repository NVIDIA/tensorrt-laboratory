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
#include "tensorrt/playground/model.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

#include "tensorrt/playground/bindings.h"
#include "tensorrt/playground/execution_context.h"
#include "tensorrt/playground/utils.h"

using ::nvinfer1::ICudaEngine;
using ::nvinfer1::IExecutionContext;

namespace yais {
namespace TensorRT {

/**
 * @brief Construct a new Model object
 */
Model::Model(std::shared_ptr<ICudaEngine> engine) : m_Engine(engine)
{
    CHECK(m_Engine) << "Model required an initialzed ICudaEngine*";
    DLOG(INFO) << "Initializing Bindings from Engine";
    for(uint32_t i = 0; i < m_Engine->getNbBindings(); i++)
    {
        const auto& binding = ConfigureBinding(i);
        m_BindingsByName[binding.name] = std::move(binding);
        m_BindingIdByName[binding.name] = i;
        DLOG(INFO) << binding.name << " maps to " << i;
        m_Bindings.push_back(std::move(ConfigureBinding(i)));
        if(m_Bindings[i].isInput)
        {
            m_InputBindings.push_back(i);
        }
        else
        {
            m_OutputBindings.push_back(i);
        }
    }
    CHECK_EQ(m_Bindings.size(), m_Engine->getNbBindings());
}

Model::~Model()
{
    DLOG(INFO) << "Destroying Model: " << m_Name;
}

Model::TensorBindingInfo Model::ConfigureBinding(uint32_t i)
{
    TensorBindingInfo binding;
    auto dims = m_Engine->getBindingDimensions(i);
    size_t elements = 1;
    for(int j = 0; j < dims.nbDims; j++)
    {
        binding.dims.push_back(dims.d[j]);
        elements *= dims.d[j];
    }

    binding.name = m_Engine->getBindingName(i);
    binding.dtype = m_Engine->getBindingDataType(i);
    binding.dtypeSize = SizeofDataType(binding.dtype);
    binding.elementsPerBatchItem = elements;
    binding.bytesPerBatchItem = elements * binding.dtypeSize;
    binding.isInput = m_Engine->bindingIsInput(i);
    LOG(INFO) << "Binding: " << binding.name
              << "; isInput: " << (binding.isInput ? "true" : "false")
              << "; dtype size: " << binding.dtypeSize
              << "; bytes per batch item: " << binding.bytesPerBatchItem;
    return binding;
}

auto Model::BindingId(const std::string& name) const -> uint32_t
{
    auto search = m_BindingIdByName.find(name);
    CHECK(search != m_BindingIdByName.end());
    return search->second;
}

auto Model::GetBinding(uint32_t id) const -> const TensorBindingInfo&
{
    CHECK_LT(id, m_Bindings.size())
        << "Invalid BindingId; given: " << id << "; max: " << m_Bindings.size();
    return m_Bindings[id];
}

Model::BindingType Model::GetBindingType(const std::string& name) const
{
    auto search = m_BindingsByName.find(name);
    if(search == m_BindingsByName.end())
    {
        return BindingType::Invalid;
    }
    if(search->second.isInput)
    {
        return BindingType::Input;
    }
    else
    {
        return BindingType::Output;
    }
    return BindingType::Invalid;
}

auto Model::GetBinding(const std::string& name) const -> const TensorBindingInfo&
{
    auto search = m_BindingsByName.find(name);
    CHECK(search != m_BindingsByName.end());
    return search->second;
}

auto Model::CreateExecutionContext() const -> std::shared_ptr<IExecutionContext>
{
    return nv_shared<IExecutionContext>(m_Engine->createExecutionContextWithoutDeviceMemory(),
                                        [engine = m_Engine, name = m_Name]() mutable {
                                            DLOG(INFO) << "Destroying IExecutionContext for Model: "
                                                       << name;
                                        });
}

void Model::AddWeights(void* ptr, size_t size)
{
    m_Weights.push_back(Weights{ptr, size});
}

void Model::PrefetchWeights(cudaStream_t stream) const
{
    for(auto weights : m_Weights)
    {
        CHECK_EQ(cudaMemPrefetchAsync(weights.addr, weights.size, 0, stream), CUDA_SUCCESS)
            << "Failed to Prefetch Weights";
    }
}

auto Model::GetWeightsMemorySize() const -> const size_t
{
    size_t total = 0;
    for(auto weights : m_Weights)
    {
        total += weights.size;
    }
    return total;
}

auto Model::GetBindingMemorySize() const -> const size_t
{
    size_t bytes = 0;
    for(auto const& binding : m_Bindings)
    {
        bytes += binding.bytesPerBatchItem;
    }
    bytes *= GetMaxBatchSize();
    return bytes;
}

} // namespace TensorRT
} // namespace yais
