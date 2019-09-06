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
#include "tensorrt/laboratory/inference_manager.h"

#include <glog/logging.h>

#include "tensorrt/laboratory/cuda/device_info.h"
#include "tensorrt/laboratory/cuda/memory/cuda_device.h"
#include "tensorrt/laboratory/cuda/memory/cuda_pinned_host.h"

using trtlab::Memory::CudaDeviceMemory;
using trtlab::Memory::CudaPinnedHostMemory;

namespace trtlab {
namespace TensorRT {

/**
 * @brief General TensorRT Resource class
 *
 * Derived from trtlab::Resources, this InferenceManager object provides the basic memory and compute
 * resources needed for using with a TensorRT Context.  Limited quanity resources such as Buffers
 * and ExecutionContexts are managed by thead-safe Pools.  In general, the compute is always limited
 * by the number of resources. For example, limiting the number of ExecutionContexts to 1 will
 * ensure only 1 Inference calcuation is using the GPU.  This will ensure best possible latency.
 * However, if you wish to improve throughput at the cost of increased latency, you can increase the
 * number of in-flight ExecutionContexts.  This will cause competition between the multiple forward
 * passes; however, it will also allow the GPU to better utilize the compute resources of the GPU.
 *
 * Note: the number of Buffers should alway be nExec+1 or larger to ensure you are not resource
 * bound on the Buffers used for the Input and Output Tensors of the DNN.
 *
 * @see Pool for more details on how limited quantity Resources are managed.
 */
InferenceManager::InferenceManager(int max_executions, int max_buffers)
    : m_MaxExecutions(max_executions), m_MaxBuffers(max_buffers ? max_buffers : max_executions * 2), m_HostStackSize(0),
      m_DeviceStackSize(0), m_ActivationsSize(0), m_Buffers{nullptr}, m_ActiveRuntime{nullptr}, m_DeviceID(-1)
{
    // RegisterRuntime("default", std::make_unique<CustomRuntime<StandardAllocator>>());
    // SetActiveRuntime("default");
    VLOG(1) << "-- Initialzing TensorRT Resource Manager --";
    VLOG(1) << "Maximum Execution Concurrency: " << m_MaxExecutions;
    VLOG(1) << "Maximum Copy Concurrency: " << m_MaxBuffers;
}

InferenceManager::~InferenceManager()
{
    JoinAllThreads();
}

int InferenceManager::MaxExecConcurrency() const
{
    return m_MaxExecutions;
}

int InferenceManager::MaxCopyConcurrency() const
{
    return m_MaxBuffers;
}

/**
 * @brief Register a Model with the InferenceManager object
 */
void InferenceManager::RegisterModel(const std::string& name, std::shared_ptr<Model> model)
{
    RegisterModel(name, model, m_MaxExecutions);
}

/**
 * @brief Register a Model with the InferenceManager object
 *
 * This variant allows you to specify an alternate maximum concurrency for this model.  The value
 * must be 1 <= concurrency <= MaxConcurrency.  Larger values will be capped to the maximum
 * concurrency allowed by the InferenceManager object.
 */
void InferenceManager::RegisterModel(const std::string& name, std::shared_ptr<Model> model,
                                     uint32_t max_concurrency)
{
    auto item = m_Models.find(name);
    if(item != m_Models.end())
    {
        LOG(ERROR) << "Model naming collsion; Model with name=" << name
                   << " is already registered.";
        return;
    }

    if(max_concurrency > m_MaxExecutions)
    {
        LOG(WARNING) << "Requested concurrency (" << max_concurrency
                     << ") exceeds max concurrency. "
                     << "Concurrency will be capped to " << m_MaxExecutions;
        max_concurrency = m_MaxExecutions;
    }

    // Size according to largest padding - device alignment
    size_t bindings =
        model->GetBindingMemorySize() + model->GetBindingsCount() * DeviceInfo::Alignment();
    size_t activations = Align(model->GetActivationsMemorySize(), 128 * 1024); // add a cacheline

    size_t host = Align(bindings, 32 * 1024);
    size_t device = Align(bindings, 128 * 1024);

    // TODO: Check to see if m_Buffers has been allocated.  If so, we should thown an exception
    // if the registered model requirements are larger than our allocated buffers.
    if(m_Buffers)
    {
        if(host > m_HostStackSize || device > m_DeviceStackSize)
        {
            throw std::runtime_error(
                "Required binding resources are greater than allocated capacity");
        }
    }
    if(m_ExecutionContexts)
    {
        if(activations > m_ActivationsSize)
        {
            throw std::runtime_error(
                "Required activation workspace is greater than allocated capacity");
        }
    }

    m_HostStackSize = std::max(m_HostStackSize, host);
    m_DeviceStackSize = std::max(m_DeviceStackSize, device);
    m_ActivationsSize = std::max(m_ActivationsSize, activations);

    VLOG(1) << "-- Registering Model: " << name << " --";
    VLOG(1) << "Input/Output Tensors require " << BytesToString(model->GetBindingMemorySize());
    VLOG(1) << "Execution Activations require "
              << BytesToString(model->GetActivationsMemorySize());
    auto weights = model->GetWeightsMemorySize();
    if(weights)
        VLOG(1) << "Weights require " << BytesToString(weights);

    model->SetName(name);
    m_Models[name] = model;
    m_ModelExecutionContexts[model.get()] = Pool<::nvinfer1::IExecutionContext>::Create();
    for(int i = 0; i < max_concurrency; i++)
    {
        m_ModelExecutionContexts[model.get()]->Push(model->CreateExecutionContext());
    }
}

/**
 * @brief Register a ensemble of Models with the InferenceManager object
 */
void InferenceManager::RegisterModel(const std::vector<std::string>& names, std::vector<std::shared_ptr<Model>> models)
{
    RegisterModel(names, models, m_MaxExecutions);
}

/**
 * @brief Register a ensemble of Models with the InferenceManager object
 *
 * This variant allows you to specify an alternate maximum concurrency for this model.  The value
 * must be 1 <= concurrency <= MaxConcurrency.  Larger values will be capped to the maximum
 * concurrency allowed by the InferenceManager object.
 */
void InferenceManager::RegisterModel(const std::vector<std::string>& names, std::vector<std::shared_ptr<Model>> models, uint32_t max_concurrency)
{
    size_t m_HostStackSize_temp = 0;
    size_t m_DeviceStackSize_temp = 0;

    for(size_t i = 0; i < names.size(); i++)
    {
        auto name = names.at(i);
        auto model = models.at(i);

        auto item = m_Models.find(name);
        if(item != m_Models.end())
        {
            LOG(ERROR) << "Model naming collsion; Model with name=" << name
                       << " is already registered.";
            return;
        }

        if(max_concurrency > m_MaxExecutions)
        {
            LOG(WARNING) << "Requested concurrency (" << max_concurrency
                         << ") exceeds max concurrency. "
                         << "Concurrency will be capped to " << m_MaxExecutions;
            max_concurrency = m_MaxExecutions;
        }

        // Size according to largest padding - device alignment
        size_t bindings =
            model->GetBindingMemorySize() + model->GetBindingsCount() * DeviceInfo::Alignment();
        size_t activations = Align(model->GetActivationsMemorySize(), 128 * 1024); // add a cacheline

        size_t host = Align(bindings, 32 * 1024);
        size_t device = Align(bindings, 128 * 1024);

        // TODO: Check to see if m_Buffers has been allocated.  If so, we should thown an exception
        // if the registered model requirements are larger than our allocated buffers.
        m_HostStackSize_temp += host;
        m_DeviceStackSize_temp += device;

        if(m_Buffers)
        {
            if(m_HostStackSize_temp > m_HostStackSize || m_DeviceStackSize_temp > m_DeviceStackSize)
            {
                throw std::runtime_error(
                    "Required binding resources are greater than allocated capacity");
            }
        }

        if(m_ExecutionContexts)
        {
            if(activations > m_ActivationsSize)
            {
                throw std::runtime_error(
                    "Required activation workspace is greater than allocated capacity");
            }
        }

        m_ActivationsSize = std::max(m_ActivationsSize, activations);

        VLOG(1) << "-- Registering Model: " << name << " --";
        VLOG(2) << "Input/Output Tensors require " << BytesToString(model->GetBindingMemorySize());
        VLOG(2) << "Execution Activations require "
                  << BytesToString(model->GetActivationsMemorySize());
        auto weights = model->GetWeightsMemorySize();
        if(weights)
            VLOG(2) << "Weights require " << BytesToString(weights);

        model->SetName(name);
        m_Models[name] = model;
        m_ModelExecutionContexts[model.get()] = Pool<::nvinfer1::IExecutionContext>::Create();
        for(int i = 0; i < max_concurrency; i++)
        {
            m_ModelExecutionContexts[model.get()]->Push(model->CreateExecutionContext());
        }
    }

    m_HostStackSize =  std::max(m_HostStackSize_temp, m_HostStackSize);
    m_DeviceStackSize =  std::max(m_DeviceStackSize_temp, m_DeviceStackSize);
}

void InferenceManager::RegisterRuntime(const std::string& name, std::shared_ptr<Runtime> runtime)
{
    auto search = m_Runtimes.find(name);
    CHECK(search == m_Runtimes.end());
    m_Runtimes[name] = std::move(runtime);
}

void InferenceManager::SetActiveRuntime(const std::string& name)
{
    auto search = m_Runtimes.find(name);
    CHECK(search != m_Runtimes.end());
    m_ActiveRuntime = search->second.get();
}

/**
 * @brief Allocates Host and Device Resources for Inference
 *
 * Buffers are sized according to the registered models.  Models registered after
 * AllocateInferenceManager has been call that require larger buffers should throw an exception
 * (TODO).
 */
void InferenceManager::AllocateResources()
{
    CHECK_EQ(cudaGetDevice(&m_DeviceID), CUDA_SUCCESS);

    VLOG(1) << "-- Allocating TensorRT Resources on GPU " << m_DeviceID << "--";
    VLOG(1) << "Creating " << m_MaxExecutions << " TensorRT execution tokens.";
    VLOG(1) << "Creating a Pool of " << m_MaxBuffers << " Host/Device Memory Stacks";
    VLOG(1) << "Each Host Stack contains " << BytesToString(m_HostStackSize);
    VLOG(1) << "Each Device Stack contains " << BytesToString(m_DeviceStackSize);
    VLOG(1) << "Total GPU Memory: "
              << BytesToString(m_MaxBuffers * m_DeviceStackSize +
                               m_MaxExecutions * m_ActivationsSize);

    m_Buffers = Pool<Buffers>::Create();
    for(int i = 0; i < m_MaxBuffers; i++)
    {
        DVLOG(1) << "Allocating Host/Device Buffers #" << i;
        auto buffers = std::make_shared<FixedBuffers<CudaPinnedHostMemory, CudaDeviceMemory>>(
            m_HostStackSize, m_DeviceStackSize);

        DVLOG(1) << "Building Graphs for Buffer #" << i;
        for(const auto& item : m_RegisteredGraphsByModelName)
        {
            const auto& key = item.first;
            const auto& name = key.first;
            const auto& batch_size = key.second;
            buffers->m_GraphWorkspace->RegisterModel(name, GetModel(name), batch_size);
        }
        buffers->m_GraphWorkspace->BuildGraphs();

        m_Buffers->Push(buffers);
    }

    m_ExecutionContexts = Pool<ExecutionContext>::Create();
    for(int i = 0; i < m_MaxExecutions; i++)
    {
        m_ExecutionContexts->EmplacePush(new ExecutionContext(m_ActivationsSize));
    }
}

/**
 * @brief Get a registered Model by name
 *
 * @param model_name
 * @return std::shared_ptr<Model>
 */
auto InferenceManager::GetModel(std::string model_name) -> std::shared_ptr<Model>
{
    auto item = m_Models.find(model_name);
    CHECK(item != m_Models.end()) << "Unable to find entry for model: " << model_name;
    return item->second;
}

bool InferenceManager::HasBuffers()
{
    return !m_Buffers->Empty();
}

/**
 * @brief Get a Buffers from the Resource Pool (May Block!)
 *
 * This method aquires a limited quantity Buffers object from the Pool of Buffers.  This call may
 * block foward execution of the thread if no resources are available.
 *
 * Note: The resource will be returned to the resource Pool when the reference count of the
 * shared_ptr goes to zero.  No action on the user is required, unless they want to release the
 * object earlier by using the reset() function on all instances of the shared_ptr.
 *
 * @return std::shared_ptr<Buffers>
 */
auto InferenceManager::GetBuffers() -> std::shared_ptr<Buffers>
{
    CHECK(m_Buffers) << "Call AllocateResources() before trying to acquire a Buffers object.";
    return m_Buffers->Pop([](Buffers* ptr) {
        ptr->Reset();
        DVLOG(2) << "Releasing Buffers: " << ptr;
    });
}

/**
 * @brief Get an Exeuction Context object from the Resource Pool (May Block!)
 *
 * This method aquires a limited quantity ExecutionContext object from the Pool of
 * ExecutionContexts. This call may block foward execution of the thread if no resources are
 * available.
 *
 * Note: The resource will be returned to the resource Pool when the reference count of the
 * shared_ptr goes to zero.  No action on the user is required, unless they want to release the
 * object earlier by using the reset() function on all instances of the shared_ptr.
 *
 * @return std::shared_ptr<ExecutionContext>
 */
auto InferenceManager::GetExecutionContext(const Model* model) -> std::shared_ptr<ExecutionContext>
{
    CHECK(m_ExecutionContexts)
        << "Call AllocateResources() before trying to acquire an ExeuctionContext.";
    auto item = m_ModelExecutionContexts.find(model);
    CHECK(item != m_ModelExecutionContexts.end())
        << "No ExectionContext for model " << model->Name();
    // This is the global concurrency limiter - it owns the activation scratch memory
    auto ctx = m_ExecutionContexts->Pop([](ExecutionContext* ptr) {
        ptr->Reset();
        DVLOG(2) << "Returning Execution Concurrency Limiter to Pool: " << ptr;
    });
    // This is the model concurrency limiter - it owns the TensorRT IExecutionContext
    // for which the pointer to the global limiter's memory buffer will be set
    ctx->SetContext(item->second->Pop([](::nvinfer1::IExecutionContext* ptr) {
        DVLOG(2) << "Returning Model IExecutionContext to Pool: " << ptr;
    }));
    DVLOG(1) << "Acquired Concurrency Limiting Execution Context: " << ctx.get();
    return ctx;
}

/**
 * @brief Get an Exeuction Context object from the Resource Pool (May Block!)
 *
 * Convenience method for accepting a shared_ptr<Model> as input.
 *
 * @param model
 * @return std::shared_ptr<ExecutionContext>
 */
auto InferenceManager::GetExecutionContext(const std::shared_ptr<Model>& model)
    -> std::shared_ptr<ExecutionContext>
{
    return GetExecutionContext(model.get());
}

/**
 * @brief Get an Exeuction Context object from the Resource Pool (May Block!) without model
 *
 * @return std::shared_ptr<ExecutionContext>
 */
auto InferenceManager::GetExecutionContext() -> std::shared_ptr<ExecutionContext>
{
    CHECK(m_ExecutionContexts)
        << "Call AllocateResources() before trying to acquire an ExeuctionContext.";

    // This is the global concurrency limiter - it owns the activation scratch memory
    auto ctx = m_ExecutionContexts->Pop([](ExecutionContext* ptr) {
        ptr->Reset();
        DVLOG(2) << "Returning Execution Concurrency Limiter to Pool: " << ptr;
    });

    return ctx;
}

auto InferenceManager::GetSubExecutionContext(std::shared_ptr<ExecutionContext> ctx, const Model* model) -> std::shared_ptr<SubExecutionContext>
{

    auto item = m_ModelExecutionContexts.find(model);
    CHECK(item != m_ModelExecutionContexts.end())
        << "No ExectionContext for model " << model->Name();

    auto subCtx = std::make_shared<SubExecutionContext>(ctx);
    // This is the model concurrency limiter - it owns the TensorRT IExecutionContext
    // for which the pointer to the global limiter's memory buffer will be set
    subCtx->SetContext(item->second->Pop([](::nvinfer1::IExecutionContext* ptr) {
        DVLOG(2) << "Returning Model IExecutionContext to Pool: " << ptr;
    }));
    DVLOG(1) << "Acquired Concurrency Limiting Execution SubContext: " << subCtx.get();
    return subCtx;
}

auto InferenceManager::GetSubExecutionContext(std::shared_ptr<ExecutionContext> ctx, const std::shared_ptr<Model>& model) -> std::shared_ptr<SubExecutionContext>
{
    return GetSubExecutionContext(ctx, model.get());
}

auto InferenceManager::AcquireThreadPool(const std::string& name) -> ThreadPool&
{
    // std::shared_lock<std::shared_mutex> lock(m_ThreadPoolMutex);
    auto search = m_ThreadPools.find(name);
    CHECK(search != m_ThreadPools.end());
    return *(search->second);
}

void InferenceManager::RegisterThreadPool(const std::string& name,
                                          std::shared_ptr<ThreadPool> threads)
{
    // std::unique_lock<std::shared_mutex> lock(m_ThreadPoolMutex);
    DVLOG(3) << "Registering ThreadPool: " << name;
    // Old threadpools will continute to live until all threads are joined.
    // this may need a mutex
    m_ThreadPools[name].swap(threads);
}

bool InferenceManager::HasThreadPool(const std::string& name) const
{
    auto search = m_ThreadPools.find(name);
    return (bool)(search != m_ThreadPools.end());
}

void InferenceManager::JoinAllThreads()
{
    // std::unique_lock<std::shared_mutex> lock(m_ThreadPoolMutex);
    DVLOG(3) << "Joining All Threads";
    m_ThreadPools.clear();
    DVLOG(3) << "All Threads Checked-In and Joined";
}

void InferenceManager::ForEachModel(std::function<void(const Model&)> callback)
{
    for(const auto& item : m_Models)
    {
        callback(*(item.second));
    }
}

void InferenceManager::BuildGraphForModel(const std::string& name, uint32_t batch_size)
{
    auto key = MakeKey(name, batch_size);
    m_RegisteredGraphsByModelName[key] = true;
}

} // namespace TensorRT
} // namespace trtlab
