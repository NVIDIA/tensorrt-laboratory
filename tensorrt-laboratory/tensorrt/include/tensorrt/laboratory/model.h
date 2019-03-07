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
#include <vector>

#include <NvInfer.h>

#include "tensorrt/laboratory/runtime.h"

namespace trtlab {
namespace TensorRT {

/**
 * @brief Wrapper class for nvinfer1::ICudaEngine.
 *
 * A Model object holds an instance of ICudaEngine and extracts some basic meta data
 * from the engine to simplify pushing the input/output bindings to a memory stack.
 */
class BaseModel // TODO: inherit from IModel so we can have non-TensorRT models that conform
{
  public:
    enum BindingType
    {
        Input,
        Output,
        Invalid
    };

    struct TensorBindingInfo;

    /**
     * @brief Construct a new Model object
     *
     * @param engine
     */
    BaseModel() = default;
    virtual ~BaseModel() {}

    auto Name() const -> const std::string& { return m_Name; }

    void SetName(const std::string& name) { m_Name = name; }

    virtual int GetMaxBatchSize() const = 0;

    auto GetBindingMemorySize() const -> const size_t;

    auto BindingId(const std::string&) const -> uint32_t;

    auto GetBinding(uint32_t) const -> const TensorBindingInfo&;
    auto GetBinding(const std::string&) const -> const TensorBindingInfo&;

    BindingType GetBindingType(const std::string&) const;

    auto GetBindingsCount() const { return m_Bindings.size(); }
    auto GetInputBindingCount() const { return m_InputBindings.size(); }
    auto GetOutputBindingCount() const { return m_OutputBindings.size(); }
    auto GetInputBindingIds() const -> const std::vector<uint32_t> { return m_InputBindings; }
    auto GetOutputBindingIds() const -> const std::vector<uint32_t> { return m_OutputBindings; }

    auto DescribeInputs() const -> const std::map<std::string, TensorBindingInfo>
    {
        std::map<std::string, TensorBindingInfo> bindings;
        for(const auto& b : m_Bindings)
        {
            if(b.isInput)
            {
                bindings[b.name] = b;
            }
        }
        return bindings;
    }

    struct TensorBindingInfo
    {
        std::string name;
        nvinfer1::DataType dtype;
        bool isInput;
        int dtypeSize;
        size_t bytesPerBatchItem;
        size_t elementsPerBatchItem;
        std::vector<size_t> dims;
    };

  protected:
    void AddBinding(TensorBindingInfo&&);

  private:
    std::vector<TensorBindingInfo> m_Bindings;
    std::map<std::string, uint32_t> m_BindingIdByName;
    std::vector<uint32_t> m_InputBindings;
    std::vector<uint32_t> m_OutputBindings;
    std::string m_Name;
};

class Model final : public BaseModel
{
  public:
    enum BindingType
    {
        Input,
        Output,
        Invalid
    };

    /**
     * @brief Construct a new Model object
     *
     * @param engine
     */
    Model(std::shared_ptr<::nvinfer1::ICudaEngine> engine); // TODO: Move to protected/private
    virtual ~Model() override;

    void AddWeights(void*, size_t); // TODO: Move to protected/private
    void PrefetchWeights(cudaStream_t) const;
    auto CreateExecutionContext() const -> std::shared_ptr<::nvinfer1::IExecutionContext>;

    int GetMaxBatchSize() const final override { return m_Engine->getMaxBatchSize(); }

    auto GetActivationsMemorySize() const { return m_Engine->getDeviceMemorySize(); }

    auto GetWeightsMemorySize() const -> const size_t;

  protected:
    TensorBindingInfo ConfigureBinding(uint32_t);

  private:
    struct Weights
    {
        void* addr;
        size_t size;
    };

    std::shared_ptr<::nvinfer1::ICudaEngine> m_Engine;
    std::shared_ptr<const Runtime> m_Runtime;
    std::vector<Weights> m_Weights;
};

} // namespace TensorRT
} // namespace trtlab
