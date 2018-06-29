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
        CHECK_EQ(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, 0), CUDA_SUCCESS) << "Bad advise";
        CHECK_EQ(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, 0), CUDA_SUCCESS) << "Bad advise";
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
    LOG(INFO) << "Initializing Bindings from Engine";
    m_Bindings.resize(m_Engine->getNbBindings());
    for (int i = 0; i < m_Bindings.size(); i++)
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
    LOG(INFO) << "Binding: " << name << "; isInput: " << (binding.isInput ? "true" : "false")
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

Buffers::Buffers(size_t host_size, size_t device_size)
    : m_HostStack(MemoryStack<CudaHostAllocator>::make_shared(host_size)),
      m_DeviceStack(MemoryStack<CudaDeviceAllocator>::make_shared(device_size)),
      m_Host(new MemoryStackTracker(m_HostStack)),
      m_Device(new MemoryStackTracker(m_DeviceStack))
{
    //CHECK(cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking) == cudaSuccess); <-- breaks
    CHECK_EQ(cudaStreamCreate(&m_Stream), cudaSuccess) << "Failed to create cudaStream";
}

Buffers::~Buffers()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS) << "Failed to sync on stream while destroying Buffer";
    CHECK_EQ(cudaStreamDestroy(m_Stream), CUDA_SUCCESS) << "Failed to destroy stream";
}

void Buffers::Configure(const Model *model, uint32_t batch_size)
{
    auto nBindings = model->GetBindingsCount();
    for (int i = 0; i < nBindings; i++)
    {
        auto binding_size = model->GetBinding(i).bytesPerBatchItem * batch_size;
        m_Host->Allocate(binding_size);
        m_Device->Allocate(binding_size);
    }
}

/*
void Buffers::SwapHostStack(std::unique_ptr<MemoryStackWithTracking<CudaHostAllocator>> other)
{
    if (m_Host->Size() == other->Size())
    {
        m_Host.swap(other);
    }
    else
    {
        LOG(ERROR) << "Error swapping Buffer Host Memory Stack - size does not match.";
        throw std::runtime_error("Invalid CudaHost MemoryStack swap");
    }
}
*/

void Buffers::Reset(bool writeZeros)
{
    m_Host.reset(new MemoryStackTracker(m_HostStack));
    m_Device.reset(new MemoryStackTracker(m_DeviceStack));
    m_HostStack->Reset(writeZeros);
    m_DeviceStack->Reset(writeZeros);
}

void Buffers::SynchronizeStream()
{
    CHECK_EQ(cudaStreamSynchronize(m_Stream), CUDA_SUCCESS) << "Stream Sync failed";
}

void Buffers::AsyncH2D(uint32_t device_binding_id)
{
    auto host_src = m_Host->GetPointer(device_binding_id);
    auto bytes = m_Host->GetSize(device_binding_id);
    AsyncH2D(device_binding_id, host_src, bytes);
}

void Buffers::AsyncH2D(uint32_t device_binding_id, void *src, size_t bytes)
{
    auto dst = m_Device->GetPointer(device_binding_id);
    //DLOG_FIRST_N(INFO, 10) << "async h2d of binding " << device_binding_id << " with size " << bytes;
    CHECK_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, m_Stream), CUDA_SUCCESS)
        << "AsyncH2D for Binding " << device_binding_id << " failed - (dst, src, bytes) = "
        << "(" << dst << ", " << src << ", " << bytes << ")";
}

void Buffers::AsyncD2H(uint32_t device_binding_id)
{
    auto host_dst = m_Host->GetPointer(device_binding_id);
    auto bytes = m_Host->GetSize(device_binding_id);
    AsyncD2H(device_binding_id, host_dst, bytes);
}

void Buffers::AsyncD2H(uint32_t device_binding_id, void *dst, size_t bytes)
{
    auto src = m_Device->GetPointer(device_binding_id);
    CHECK_LE(bytes, m_Device->GetSize(device_binding_id)) << "Requested copy larger than binding size";
    CHECK_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, m_Stream), CUDA_SUCCESS)
        << "AsyncD2H for Binding " << device_binding_id << " failed - (dst, src, bytes) = "
        << "(" << dst << ", " << src << ", " << bytes << ")";
}

// ExecutionContext

ExecutionContext::ExecutionContext() : m_Context{nullptr}
{
    CHECK_EQ(cudaEventCreateWithFlags(&m_ExecutionContextFinished, cudaEventDisableTiming), CUDA_SUCCESS)
        << "Failed to Create Execution Context Finished Event";
}

ExecutionContext::~ExecutionContext()
{
    CHECK_EQ(cudaEventDestroy(m_ExecutionContextFinished), CUDA_SUCCESS) << "Failed to Destroy Enqueue Event";
}

void ExecutionContext::SetContext(IExecutionContext *context)
{
    m_Context = context;
}

void ExecutionContext::Enqueue(int batch_size, void **device_bindings, void* activations, cudaStream_t stream)
{
    m_Context->setDeviceMemory(activations);
    m_Context->enqueue(batch_size, device_bindings, stream, nullptr);
    CHECK_EQ(cudaEventRecord(m_ExecutionContextFinished, stream), CUDA_SUCCESS) << "ExeCtx Event Record Failed";
}

void ExecutionContext::Synchronize()
{
    CHECK_EQ(cudaEventSynchronize(m_ExecutionContextFinished), CUDA_SUCCESS) << "ExeCtx Event Sync Failed";
}

void ExecutionContext::Reset()
{
    m_Context = nullptr;
}

// Resources

/*
Resources::Resources(std::shared_ptr<Model> model, int nBuffers, int nExecs)
    : m_Model(model),
      m_Buffers(Pool<Buffers>::Create()),
      m_ExecutionContexts(Pool<ExecutionContext>::Create())
{
    size_t bufferSize = m_Model->GetMaxBufferSize() + m_Model->GetBindingsCount() * 128;
    bufferSize += 128*1024 + model->GetDeviceMemorySize();
    LOG(INFO) << "Configuring TensorRT Resources";
    LOG(INFO) << "Creating " << nBuffers << " Host/Device Buffers each of size "
              << bufferSize << " on both host and device";
    // # if TRT4
    LOG(INFO) << "Creating " << nExecs << " TensorRT execution contexts ("
              << model->GetDeviceMemorySize() << " bytes/ctx)";

    for (int i = 0; i < nBuffers; i++)
    {
        DLOG(INFO) << "Allocating Host/Device Buffers #" << i;
        m_Buffers->EmplacePush(new Buffers(bufferSize, bufferSize));
    }

    for (int i = 0; i < nExecs; i++)
    {
        DLOG(INFO) << "Allocating Execution Context #" << i;
        m_ExecutionContexts->EmplacePush();
    }
}

Resources::~Resources()
{
    // Order is important
    m_Buffers.reset();
    m_ExecutionContexts.reset();
}
*/

// ResourceBuilder

Resources::Resources(int max_executions, int max_buffers)
    : m_MaxExecutions(max_executions), m_MaxBuffers(max_buffers),
      m_MinHostStack(0), m_MinDeviceStack(0),
      m_Buffers{nullptr}
{
    m_ExecutionContexts = Pool<ExecutionContext>::Create();
    for (int i = 0; i < m_MaxExecutions; i++)
    {
        m_ExecutionContexts->EmplacePush(new ExecutionContext);
    }
}

Resources::~Resources() {}

void Resources::RegisterModel(std::string name, std::shared_ptr<Model> model)
{
    RegisterModel(name, model, m_MaxExecutions);
}

void Resources::RegisterModel(std::string name, std::shared_ptr<Model> model, int max_concurrency)
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

    m_Models[name] = model;
    m_ModelExecutionContexts[name] = Pool<IExecutionContext>::Create();
    for (int i = 0; i < max_concurrency; i++)
    {
        m_ModelExecutionContexts[name]->Push(model->GetExecutionContext());
    }

    // Size according to largest padding - hardcoded to 256
    size_t bindings = model->GetMaxBufferSize() + model->GetBindingsCount() * 256;
    size_t activations = model->GetDeviceMemorySize() + 128 * 1024; // add a cacheline

    size_t host = Align(bindings, 32 * 1024);
    size_t device = Align(bindings + activations, 128 * 1024);

    m_MinHostStack = std::max(m_MinHostStack, host);
    m_MinDeviceStack = std::max(m_MinDeviceStack, device);

    LOG(INFO) << "-- Registering Model: " << name << " --";
    LOG(INFO) << "Input/Output Tensors require " << model->GetMaxBufferSize() << " bytes";
    LOG(INFO) << "Execution Activations require " << model->GetDeviceMemorySize() << " bytes";
    DLOG(INFO) << "With padding/cacheline rounding; host_size=" << host << "device_size: " << device;
}

void Resources::AllocateResources()
{
    LOG(INFO) << "Allocating TensorRT Resources";
    LOG(INFO) << "Creating a Pool of " << m_MaxBuffers << " Host/Device Memory Stacks";
    LOG(INFO) << "Each Host Stack contains " << m_MinHostStack << " bytes";
    LOG(INFO) << "Each Device Stack contains " << m_MinDeviceStack << " bytes";
    LOG(INFO) << "Creating " << m_MaxExecutions << " TensorRT execution tokens.";

    m_Buffers = Pool<Buffers>::Create();
    for (int i = 0; i < m_MaxBuffers; i++)
    {
        DLOG(INFO) << "Allocating Host/Device Buffers #" << i;
        m_Buffers->EmplacePush(new Buffers(m_MinHostStack, m_MinDeviceStack));
    }
}

auto Resources::GetModel(std::string model_name) -> std::shared_ptr<Model>
{
    auto item = m_Models.find(model_name);
    if (item == m_Models.end())
    {
        LOG(FATAL) << "Unable to find entry for model: " << model_name;
    }
    return item->second;
}

auto Resources::GetBuffers() -> std::shared_ptr<Buffers>
{
    auto from_pool = m_Buffers->Pop();
    return std::shared_ptr<Buffers>(from_pool.get(), [from_pool](void *ptr) {
        from_pool->Reset();
    });
}

auto Resources::GetExecutionContext(std::string model_name) -> std::shared_ptr<ExecutionContext>
{
    auto item = m_ModelExecutionContexts.find(model_name);
    if (item == m_ModelExecutionContexts.end())
    {
        LOG(FATAL) << "No ExectionContext found for unregistered model name: " << model_name;
    }
    auto ictx = item->second->Pop();
    auto ctx = m_ExecutionContexts->Pop();
    ctx->SetContext(ictx.get());
    // both ctx and token will be returned to their respective pools when their refcount -> 0
    return std::shared_ptr<ExecutionContext>(ctx.get(), [ictx, ctx](void *ptr) {
        ctx->Reset();
    });
}

size_t Resources::Align(size_t size, size_t alignment)
{
    size_t remainder = size % alignment;
    size = (remainder == 0) ? size : size + alignment - remainder;
    return size;
}

} // end namespace TensorRT
} // end namespace yais
