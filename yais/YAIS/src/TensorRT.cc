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
#include "YAIS/TensorRT.h"

#include <algorithm>
#include <fstream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace yais
{
namespace TensorRT
{

Runtime::Runtime()
    : m_Logger(std::make_unique<Logger>()),
      m_Runtime(make_unique(::nvinfer1::createInferRuntime(*(m_Logger.get()))))
{
    m_Logger->log(::nvinfer1::ILogger::Severity::kINFO, "IRuntime Logger Initialized");
}

Runtime *Runtime::GetSingleton()
{
    static Runtime singleton;
    return &singleton;
}

std::vector<char> Runtime::ReadEngineFile(std::string plan_file)
{
    DLOG(INFO) << "Reading Engine: " << plan_file;
    std::ifstream file(plan_file, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        LOG(FATAL) << "Unable to read engine file: " << plan_file;
    }
    return buffer;
}

std::shared_ptr<Model> Runtime::DeserializeEngine(std::string plan_file)
{
    auto singleton = Runtime::GetSingleton();
    auto buffer = singleton->ReadEngineFile(plan_file);
    // Create Engine / Deserialize Plan - need this step to be broken up plz!!
    DLOG(INFO) << "Deserializing TensorRT ICudaEngine";
    auto engine = make_shared(
        singleton->GetRuntime()->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr));
    CHECK(engine) << "Unable to create ICudaEngine";
    return std::make_shared<Model>(engine);
}

void Runtime::Logger::log(::nvinfer1::ILogger::Severity severity, const char *msg)
{
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
        LOG(FATAL) << "[TensorRT.INTERNAL_ERROR]: " << msg;
        break;
    case Severity::kERROR:
        LOG(FATAL) << "[TensorRT.ERROR]: " << msg;
        break;
    case Severity::kWARNING:
        LOG(WARNING) << "[TensorRT.WARNING]: " << msg;
        break;
    case Severity::kINFO:
        DLOG(INFO) << "[TensorRT.INFO]: " << msg;
        break;
    default:
        DLOG(INFO) << "[TensorRT.DEBUG]: " << msg;
        break;
    }
}

// #if NV_TENSORRT_MAJOR >= 4

ManagedRuntime::ManagedRuntime()
    : Runtime(), m_Allocator(std::make_unique<ManagedAllocator>())
{
    GetRuntime()->setGpuAllocator(m_Allocator.get());
}

ManagedRuntime *ManagedRuntime::GetSingleton()
{
    static ManagedRuntime singleton;
    return &singleton;
}

std::shared_ptr<Model> ManagedRuntime::DeserializeEngine(std::string plan_file)
{
    auto singleton = ManagedRuntime::GetSingleton();
    auto buffer = singleton->ReadEngineFile(plan_file);

    return singleton->UseManagedMemory([singleton, &buffer]() -> std::shared_ptr<Model> {
        DLOG(INFO) << "Deserializing TensorRT ICudaEngine";
        auto engine = make_shared(
            singleton->GetRuntime()->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr));

        auto model = std::make_shared<Model>(engine);

        for (auto ptr : singleton->GetAllocator()->GetPointers())
        {
            cudaMemAdvise(ptr.addr, ptr.size, cudaMemAdviseSetReadMostly, 0);
            model->AddWeights(ptr.addr, ptr.size);
            DLOG(INFO) << "cudaMallocManaged allocation for TensorRT weights: " << ptr.addr;
        }
        return model;
    });
}

void *ManagedRuntime::ManagedAllocator::allocate(size_t size, uint64_t alignment, uint32_t flags)
{
    void *ptr;
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_UseManagedMemory)
    {
        CHECK_EQ(cudaMallocManaged(&ptr, size), CUDA_SUCCESS)
            << "Failed to allocate TensorRT device memory (managed)";
        // CHECK_EQ(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, 0), CUDA_SUCCESS) << "Bad advise";
        // CHECK_EQ(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, 0), CUDA_SUCCESS) << "Bad advise";
        LOG(INFO) << "TensoRT cudaMallocManaged size = " << size << "; " << ptr;
        m_Pointers.push_back(Pointer{ptr, size});
    }
    else
    {
        CHECK_EQ(cudaMalloc(&ptr, size), CUDA_SUCCESS)
            << "Failed to allocate TensorRT device memory (not managed)";
        LOG(INFO) << "TensoRT cudaMalloc size = " << size;
    }
    return ptr;
}

void ManagedRuntime::ManagedAllocator::free(void *ptr)
{
    CHECK_EQ(cudaFree(ptr), CUDA_SUCCESS) << "Failed to free TensorRT device memory";
}

// #endif

Model::Model(std::shared_ptr<ICudaEngine> engine)
    : m_Engine(engine)
{
    CHECK(m_Engine) << "Model required an initialzed ICudaEngine*";
    DLOG(INFO) << "Initializing Bindings from Engine";
    m_Bindings.resize(m_Engine->getNbBindings());
    for (uint32_t i = 0; i < m_Bindings.size(); i++)
    {
        ConfigureBinding(m_Bindings[i], i);
        if (m_Bindings[i].isInput)
            m_InputBindings.push_back(i);
        else
            m_OutputBindings.push_back(i);
    }
}

void Model::ConfigureBinding(Binding &binding, uint32_t i)
{
    auto name = m_Engine->getBindingName(i);
    auto dtype = m_Engine->getBindingDataType(i);
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        binding.dtypeSize = 4;
        break;
    case nvinfer1::DataType::kHALF:
        binding.dtypeSize = 2;
        break;
    default:
        binding.dtypeSize = 1;
        break;
    }
    auto dims = m_Engine->getBindingDimensions(i);
    size_t elements = 1;
    for (int j = 0; j < dims.nbDims; j++)
    {
        binding.dims.push_back(dims.d[j]);
        elements *= dims.d[j];
    }

    binding.elementsPerBatchItem = elements;
    binding.bytesPerBatchItem = elements * binding.dtypeSize;
    binding.isInput = m_Engine->bindingIsInput(i);
    DLOG(INFO) << "Binding: " << name << "; isInput: " << (binding.isInput ? "true" : "false")
               << "; dtype size: " << binding.dtypeSize
               << "; bytes per batch item: " << binding.bytesPerBatchItem;
}

void Model::AddWeights(void *ptr, size_t size)
{
    m_Weights.push_back(Weights{ptr, size});
}

void Model::PrefetchWeights(cudaStream_t stream) const
{
    for (auto weights : m_Weights)
    {
        CHECK_EQ(cudaMemPrefetchAsync(weights.addr, weights.size, 0, stream), CUDA_SUCCESS)
            << "Failed to Prefetch Weights";
    }
}

size_t Model::GetWeightsMemorySize() const
{
    size_t total = 0;
    for (auto weights : m_Weights)
    {
        total += weights.size;
    }
    return total;
}

const size_t Model::GetMaxBufferSize() const
{
    size_t bytes = 0;
    for (auto &binding : m_Bindings)
    {
        bytes += binding.bytesPerBatchItem;
    }
    bytes *= GetMaxBatchSize();
    return bytes;
}

/**
 * @brief Construct a new Buffers object
 * 
 * In most cases, Buffers will be created with equal sized host and device stacks;
 * however, for very custom cases, you may choose to configure them to your problem.
 * 
 * @param host_size 
 * @param device_size 
 */
auto Buffers::Create(size_t host_size, size_t device_size) -> std::shared_ptr<Buffers>
{
    return std::shared_ptr<Buffers>(new Buffers(host_size, device_size));
}

Buffers::Buffers(size_t host_size, size_t device_size)
    : m_HostStack(MemoryStack<CudaHostAllocator>::make_shared(host_size)),
      m_DeviceStack(MemoryStack<CudaDeviceAllocator>::make_shared(device_size))
{
    //CHECK(cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking) == cudaSuccess); <-- breaks
    CHECK_EQ(cudaStreamCreate(&m_Stream), cudaSuccess) << "Failed to create cudaStream";
}

Buffers::~Buffers()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS) << "Failed to sync on stream while destroying Buffer";
    CHECK_EQ(cudaStreamDestroy(m_Stream), CUDA_SUCCESS) << "Failed to destroy stream";
}

auto Buffers::CreateBindings(const std::shared_ptr<Model> &model, uint32_t batch_size) -> std::shared_ptr<Bindings>
{
    return std::shared_ptr<Bindings>(new Bindings(model, shared_from_this(), batch_size));
}

/**
 * @brief Pushes both Host and Device Stack Pointers for each Binding in a Model
 * 
 * For each binding in the model, a stack pointer will be pushed on both host and device
 * memory stacks.  Buffers used a MemoryStackWithTracking object, so every Push is 
 * recorded and the Pointer and the Size of each stack allocation can be recalled by
 * passing the index of the binding.
 * 
 * @param model 
 * @param batch_size 
 * @return bindings
 */
auto Buffers::CreateAndConfigureBindings(const std::shared_ptr<Model> &model, uint32_t batch_size) -> std::shared_ptr<Bindings>
{
    auto bindings = CreateBindings(model, batch_size);
    for (uint32_t i = 0; i < model->GetBindingsCount(); i++)
    {
        auto binding_size = model->GetBinding(i).bytesPerBatchItem * batch_size;
        DLOG(INFO) << "Configuring Binding " << i << ": pushing " << binding_size << " to host/device stacks";
        bindings->SetHostAddress(i, m_HostStack->Allocate(binding_size));
        bindings->SetDeviceAddress(i, m_DeviceStack->Allocate(binding_size));
    }

    DLOG(INFO) << "Reserving Memory for Activations: " << model->GetActivationsMemorySize();
    m_DeviceStack->Allocate(128 * 1024); // push a cacheline
    bindings->SetActivationsAddress(
        m_DeviceStack->Allocate(model->GetActivationsMemorySize()));
    return bindings;
}

/**
 * @brief Resets the Host and Device Stack Pointers to their origins
 * 
 * @param writeZeros 
 */
void Buffers::Reset(bool writeZeros)
{
    m_HostStack->Reset(writeZeros);
    m_DeviceStack->Reset(writeZeros);
}

void Buffers::Synchronize()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS) << "Stream Sync failed";
}




Bindings::Bindings(const std::shared_ptr<Model> model, const std::shared_ptr<Buffers> buffers, uint32_t batch_size)
    : m_Model(model), m_Buffers(buffers), m_BatchSize(batch_size)
{
    auto count = model->GetBindingsCount();
    m_HostAddresses.resize(count);
    m_DeviceAddresses.resize(count);
    for (auto i = 0; i < count; i++)
    {
        m_HostAddresses[i] = m_DeviceAddresses[i] = nullptr;
    }
}

Bindings::~Bindings() {}

void Bindings::SetHostAddress(int binding_id, void *addr)
{
    CHECK_LT(binding_id, m_HostAddresses.size())
        << "Invalid binding_id (" << binding_id << ") must be < " << m_HostAddresses.size();
    m_HostAddresses[binding_id] = addr;
}

void Bindings::SetDeviceAddress(int binding_id, void *addr)
{
    CHECK_LT(binding_id, m_DeviceAddresses.size())
        << "Invalid binding_id (" << binding_id << ") must be < " << m_DeviceAddresses.size();
    m_DeviceAddresses[binding_id] = addr;
}

void *Bindings::HostAddress(uint32_t binding_id)
{
    CHECK_LT(binding_id, m_HostAddresses.size())
        << "Invalid binding_id (" << binding_id << ") must be < " << m_HostAddresses.size();
    return m_HostAddresses[binding_id];
}

void *Bindings::DeviceAddress(uint32_t binding_id)
{
    CHECK_LT(binding_id, m_DeviceAddresses.size())
        << "Invalid binding_id (" << binding_id << ") must be < " << m_DeviceAddresses.size();
    return m_DeviceAddresses[binding_id];
}

void **Bindings::DeviceAddresses()
{
    return (void **)m_DeviceAddresses.data();
}

void Bindings::CopyToDevice(uint32_t device_binding_id)
{
    auto host_src = HostAddress(device_binding_id);
    auto bytes = BindingSize(device_binding_id);
    CopyToDevice(device_binding_id, host_src, bytes);
}

void Bindings::CopyToDevice(const std::vector<uint32_t> &ids)
{
    for (auto id : ids)
    {
        CopyToDevice(id);
    }
}

void Bindings::CopyToDevice(uint32_t device_binding_id, void *src, size_t bytes)
{
    auto dst = DeviceAddress(device_binding_id);
    DLOG(INFO) << "CopyToDevice binding_id: " << device_binding_id << "; size: " << bytes;
    CHECK_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, Stream()), CUDA_SUCCESS)
        << "CopyToDevice for Binding " << device_binding_id << " failed - (dst, src, bytes) = "
        << "(" << dst << ", " << src << ", " << bytes << ")";
}

void Bindings::CopyFromDevice(uint32_t device_binding_id)
{
    auto host_dst = HostAddress(device_binding_id);
    auto bytes = BindingSize(device_binding_id);
    CopyFromDevice(device_binding_id, host_dst, bytes);
}

void Bindings::CopyFromDevice(const std::vector<uint32_t> &ids)
{
    for (auto id : ids)
    {
        CopyFromDevice(id);
    }
}

void Bindings::CopyFromDevice(uint32_t device_binding_id, void *dst, size_t bytes)
{
    auto src = DeviceAddress(device_binding_id);
    DLOG(INFO) << "CopyFromDevice binding_id: " << device_binding_id << "; size: " << bytes;
    CHECK_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, Stream()), CUDA_SUCCESS)
        << "CopyFromDevice for Binding " << device_binding_id << " failed - (dst, src, bytes) = "
        << "(" << dst << ", " << src << ", " << bytes << ")";
}

size_t Bindings::BindingSize(uint32_t binding_id) const
{
    return m_Model->GetBinding(binding_id).bytesPerBatchItem * m_BatchSize;
}

/**
 * @brief TensorRT convenience class for wrapping IExecutionContext
 * 
 * Designed to launch, time and sychronized the execution of an async gpu calculation.
 *
 * Note: It is critically important that Reset() is called if the ExecutionContext is not going to
 * be immediately deleted after use.  Most high-performance implementation will maintain a queue/pool
 * of ExecutionContexts.  You mush call Reset() before returning the ExecutionContext to the pool or
 * risk deadlock / resource starvation.
 *  
 * Note: Timing is not implemented yet.  When implemented, do so a the host level and not the GPU
 * level to maximize GPU performance by using cudaEventDisabledTiming.  We don't need super accurate
 * timings, they are simply a nice-to-have, so a reasonable approximation on the host is sufficient.
 */
ExecutionContext::ExecutionContext() : m_Context{nullptr}
{
    CHECK_EQ(cudaEventCreateWithFlags(&m_ExecutionContextFinished, cudaEventDisableTiming), CUDA_SUCCESS)
        << "Failed to Create Execution Context Finished Event";
}

ExecutionContext::~ExecutionContext()
{
    CHECK_EQ(cudaEventDestroy(m_ExecutionContextFinished), CUDA_SUCCESS) << "Failed to Destroy Enqueue Event";
}

/**
 * @brief Set the ExectionContext
 * @param context 
 */
void ExecutionContext::SetContext(std::shared_ptr<IExecutionContext> context)
{
    m_Context = context;
}

/**
 * @brief Enqueue an Inference calculation
 * 
 * Initiates a forward pass through a TensorRT optimized graph and registers an event on the stream
 * which is trigged when the compute has finished and the ExecutionContext can be reused by competing threads.
 * Use the Synchronize method to sync on this event.
 * 
 * @param bindings 
 */
void ExecutionContext::Infer(const std::shared_ptr<Bindings> &bindings)
{
    DLOG(INFO) << "Launching Inference Execution";
    m_Context->setDeviceMemory(bindings->ActivationsAddress());
    m_Context->enqueue(bindings->BatchSize(), bindings->DeviceAddresses(), bindings->Stream(), nullptr);
    CHECK_EQ(cudaEventRecord(m_ExecutionContextFinished, bindings->Stream()), CUDA_SUCCESS) << "ExeCtx Event Record Failed";
}

/**
 * @brief Synchronized on the Completion of the Inference Calculation
 */
void ExecutionContext::Synchronize()
{
    CHECK_EQ(cudaEventSynchronize(m_ExecutionContextFinished), CUDA_SUCCESS) << "ExeCtx Event Sync Failed";
}

/**
 * @brief Resets the Context
 * 
 * Note: It is very important to call Reset when you are finished using this object.  This is because
 * this object own a reference to the TensorRT IExecutionContext.  Failure to call Reset will maintain
 * a reference to the IExecutionContext, which could eventually starve resources and cause a deadlock.
 * 
 * See the implementation of TensorRT::Resources::GetExecutionContext for an example on how to use
 * with a yais::Pool<ExecutionContext>.
 */
void ExecutionContext::Reset()
{
    m_Context->setDeviceMemory(nullptr);
    m_Context.reset();
}

// Resources

/**
 * @brief General TensorRT Resource class
 * 
 * Derived from yais::Resources, this Resources object provides the basic memory and compute resources
 * needed for using with a TensorRT Context.  Limited quanity resources such as Buffers and ExecutionContexts
 * are managed by thead-safe Pools.  In general, the compute is always limited by the number of resources.
 * For example, limiting the number of ExecutionContexts to 1 will ensure only 1 Inference calcuation is
 * using the GPU.  This will ensure best possible latency.  However, if you wish to improve throughput at the
 * cost of increased latency, you can increase the number of in-flight ExecutionContexts.  This will cause
 * competition between the multiple forward passes; however, it will also allow the GPU to better utilize the
 * compute resources of the GPU.
 * 
 * Note: the number of Buffers should alway be nExec+1 or larger to ensure you are not resource bound on the
 * Buffers used for the Input and Output Tensors of the DNN.
 * 
 * @see Pool for more details on how limited quantity Resources are managed.
 */
Resources::Resources(int max_executions, int max_buffers)
    : m_MaxExecutions(max_executions), m_MaxBuffers(max_buffers),
      m_MinHostStack(0), m_MinDeviceStack(0),
      m_Buffers{nullptr}
{
    LOG(INFO) << "Initialzing TensorRT Resource Manager";
    LOG(INFO) << "Maximum Execution Concurrency: " << m_MaxExecutions;
    LOG(INFO) << "Maximum Copy Concurrency: " << m_MaxBuffers;

    m_ExecutionContexts = Pool<ExecutionContext>::Create();
    for (int i = 0; i < m_MaxExecutions; i++)
    {
        m_ExecutionContexts->EmplacePush(new ExecutionContext);
    }
}

Resources::~Resources() {}

/**
 * @brief Register a Model with the Resources object
 * 
 * @param name 
 * @param model 
 */
void Resources::RegisterModel(std::string name, std::shared_ptr<Model> model)
{
    RegisterModel(name, model, m_MaxExecutions);
}

/**
 * @brief Register a Model with the Resources object
 * 
 * This variant allows you to specify an alternate maximum concurrency for this model.  The value
 * must be 1 <= concurrency <= MaxConcurrency.  Larger values will be capped to the maximum
 * concurrency allowed by the Resources object.
 * 
 * @param name 
 * @param model 
 * @param max_concurrency 
 */
void Resources::RegisterModel(std::string name, std::shared_ptr<Model> model, uint32_t max_concurrency)
{
    auto item = m_Models.find(name);
    if (item != m_Models.end())
    {
        LOG(ERROR) << "Model naming collsion; Model with name=" << name << " is already registered.";
        return;
    }

    if (max_concurrency > m_MaxExecutions)
    {
        LOG(WARNING) << "Requested concurrency (" << max_concurrency << ") exceeds max concurrency. "
                     << "Value will be capped to " << m_MaxExecutions;
        max_concurrency = m_MaxExecutions;
    }

    // Size according to largest padding - hardcoded to 256
    size_t bindings = model->GetMaxBufferSize() + model->GetBindingsCount() * 256;
    size_t activations = model->GetActivationsMemorySize() + 128 * 1024; // add a cacheline

    size_t host = Align(bindings, 32 * 1024);
    size_t device = Align(bindings + activations, 128 * 1024);

    // TODO: Check to see if m_Buffers has been allocated.  If so, we should thown an exception
    // if the registered model requirements are larger than our allocated buffers.
    if (m_Buffers)
        if (host > m_MinHostStack || device > m_MinDeviceStack)
            throw std::runtime_error("Model requires more resources than currently allocated");

    m_MinHostStack = std::max(m_MinHostStack, host);
    m_MinDeviceStack = std::max(m_MinDeviceStack, device);

    LOG(INFO) << "-- Registering Model: " << name << " --";
    LOG(INFO) << "Input/Output Tensors require " << BytesToString(model->GetMaxBufferSize());
    LOG(INFO) << "Execution Activations require " << BytesToString(model->GetActivationsMemorySize());
    auto weights = model->GetWeightsMemorySize();
    if (weights)
        LOG(INFO) << "Weights require " << BytesToString(weights);
    LOG(INFO) << "-- End Model Details: " << name << " --";

    model->SetName(name);
    m_Models[name] = model;
    m_ModelExecutionContexts[model.get()] = Pool<IExecutionContext>::Create();
    for (int i = 0; i < max_concurrency; i++)
    {
                m_ModelExecutionContexts[model.get()]->Push(model->CreateExecutionContext());
    }
}

/**
 * @brief Allocates Host and Device Resources for Inference
 * 
 * Buffers are sized according to the registered models.  Models registered after AllocateResources
 * has been call that require larger buffers should throw an exception (TODO).
 */
void Resources::AllocateResources()
{
    LOG(INFO) << "-- TensorRT Resource Manager --";
    LOG(INFO) << "Creating " << m_MaxExecutions << " TensorRT execution tokens.";
    LOG(INFO) << "Creating a Pool of " << m_MaxBuffers << " Host/Device Memory Stacks";
    LOG(INFO) << "Each Host Stack contains " << BytesToString(m_MinHostStack);
    LOG(INFO) << "Each Device Stack contains " << BytesToString(m_MinDeviceStack);
    LOG(INFO) << "Total GPU Memory: " << BytesToString(m_MaxBuffers * m_MinDeviceStack);
    LOG(INFO) << "-- TensorRT Resource Manager --";

    m_Buffers = Pool<Buffers>::Create();
    for (int i = 0; i < m_MaxBuffers; i++)
    {
        DLOG(INFO) << "Allocating Host/Device Buffers #" << i;
        m_Buffers->Push(Buffers::Create(m_MinHostStack, m_MinDeviceStack));
    }
}

/**
 * @brief Get a registered Model by name
 * 
 * @param model_name 
 * @return std::shared_ptr<Model> 
 */
auto Resources::GetModel(std::string model_name) -> std::shared_ptr<Model>
{
    auto item = m_Models.find(model_name);
    if (item == m_Models.end())
        LOG(FATAL) << "Unable to find entry for model: " << model_name;
    return item->second;
}

/**
 * @brief Get a Buffers from the Resource Pool (May Block!)
 * 
 * This method aquires a limited quantity Buffers object from the Pool of Buffers.  This call may
 * block foward execution of the thread if no resources are available.
 * 
 * Note: The resource will be returned to the resource Pool when the reference count of the shared_ptr
 * goes to zero.  No action on the user is required, unless they want to release the object earlier by
 * using the reset() function on all instances of the shared_ptr.
 * 
 * @return std::shared_ptr<Buffers> 
 */
auto Resources::GetBuffers() -> std::shared_ptr<Buffers>
{
    return m_Buffers->Pop([](Buffers *ptr) {
        ptr->Reset();
        DLOG(INFO) << "Releasing Buffers";
    });
}

/**
 * @brief Get an Exeuction Context object from the Resource Pool (May Block!)
 * 
 * This method aquires a limited quantity ExecutionContext object from the Pool of ExecutionContexts.
 * This call may block foward execution of the thread if no resources are available.
 * 
 * Note: The resource will be returned to the resource Pool when the reference count of the shared_ptr
 * goes to zero.  No action on the user is required, unless they want to release the object earlier by
 * using the reset() function on all instances of the shared_ptr.
 * 
 * @return std::shared_ptr<ExecutionContext> 
 */
auto Resources::GetExecutionContext(const Model *model) -> std::shared_ptr<ExecutionContext>
{
    auto item = m_ModelExecutionContexts.find(model);
    if (item == m_ModelExecutionContexts.end())
        LOG(FATAL) << "No ExectionContext found for unregistered model name: "; // << model_name;
    auto ctx = m_ExecutionContexts->Pop([](ExecutionContext *ptr) {
        ptr->Reset();
        DLOG(INFO) << "Releasing Concurrency Limiter";
    });
    ctx->SetContext(item->second->Pop([](IExecutionContext *ptr) {
        DLOG(INFO) << "Releasing IExecutionContext";
    }));
    DLOG(INFO) << "Acquired Concurrency Limiting Execution Context";
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
auto Resources::GetExecutionContext(const std::shared_ptr<Model> &model) -> std::shared_ptr<ExecutionContext>
{
    return GetExecutionContext(model.get());
}

size_t Resources::Align(size_t size, size_t alignment)
{
    size_t remainder = size % alignment;
    size = (remainder == 0) ? size : size + alignment - remainder;
    return size;
}

} // end namespace TensorRT
} // end namespace yais
