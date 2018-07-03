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
#ifndef NVIS_TENSORRT_H_
#define NVIS_TENSORRT_H_
#pragma once

#include "YAIS/Context.h"
#include "YAIS/Memory.h"
#include "YAIS/MemoryStack.h"
#include "YAIS/Pool.h"

#include "NvInfer.h"

#include <string>

namespace yais
{
namespace TensorRT
{

using ::nvinfer1::ICudaEngine;
using ::nvinfer1::IExecutionContext;
using ::nvinfer1::IRuntime;

class Model;
class Buffers;
class Bindings;
class ExecutionContext;
class Resources;

/**
 * @brief Deleter for nvinfer interface objects.
 */
struct NvInferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

/**
 * @brief Create a std::shared_ptr for an nvinfer interface object
 * 
 * @tparam T 
 * @param obj 
 * @return std::shared_ptr<T> 
 */
template <typename T>
std::shared_ptr<T> make_shared(T *obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, NvInferDeleter());
};

/**
 * @brief Create a std::unique_ptr for an nvinfer interface object
 * 
 * @tparam T 
 * @param obj 
 * @return std::unique_ptr<T, NvInferDeleter> 
 */
template <typename T>
std::unique_ptr<T, NvInferDeleter> make_unique(T *obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::unique_ptr<T, NvInferDeleter>(obj);
}

/**
 * @brief Convenience class wrapping nvinfer1::IRuntime.
 * 
 * Provide static method for deserializing a saved TensorRT engine/plan
 * and implements a Logger that conforms to YAIS logging.
 */
class Runtime
{
  public:
    static std::shared_ptr<Model> DeserializeEngine(std::string plan_file);
    virtual ~Runtime() {}

  protected:
    Runtime();
    std::vector<char> ReadEngineFile(std::string);
    IRuntime *GetRuntime() { return m_Runtime.get(); }

  private:
    static Runtime *GetSingleton();

    class Logger : public ::nvinfer1::ILogger
    {
      public:
        void log(::nvinfer1::ILogger::Severity severity, const char *msg) final override;
    };

    // Order is important.  In C++ variables in the member initializer list are instantiated in the
    // order they are declared, not in the order they appear in the initializer list.  Inverting these
    // causes m_Runtime to be initialized with a NULL m_Logger and was the source of much head banging.
    std::unique_ptr<Logger> m_Logger;
    std::unique_ptr<IRuntime, NvInferDeleter> m_Runtime;
};

/**
 * @brief Convenience class wrapping nvinfer1::IRuntime and allowing DNN weights to be oversubscribed.
 * 
 * Extends the default Runtime to use CUDA Unified Memory as the memory type for the weight tensors
 * in a TensorRT ICudaEngine.  This allows the weights to be paged in/out of device memory as needed.
 * This is currently the only way to oversubscribe GPU memory so we can load more models than we have 
 * GPU memory available.
 */
class ManagedRuntime : public Runtime
{
  public:
    static std::shared_ptr<Model> DeserializeEngine(std::string plan_file);
    virtual ~ManagedRuntime() override {}

  protected:
    struct Pointer
    {
        void *addr;
        size_t size;
    };

    class ManagedAllocator final : public ::nvinfer1::IGpuAllocator
    {
      public:
        // IGpuAllocator virtual overrides
        void *allocate(uint64_t size, uint64_t alignment, uint32_t flags) final override;
        void free(void *ptr) final override;

        ManagedAllocator() : m_UseManagedMemory(false) {}
        const std::vector<Pointer> &GetPointers() { return m_Pointers; }

      private:
        std::vector<Pointer> m_Pointers;
        std::recursive_mutex m_Mutex;
        bool m_UseManagedMemory;

        friend class ManagedRuntime;
    };

    ManagedAllocator *GetAllocator() { return m_Allocator.get(); }

    template <class F, class... Args>
    auto UseManagedMemory(F &&f, Args &&... args) -> typename std::result_of<F(Args...)>::type;

  private:
    ManagedRuntime();
    static ManagedRuntime *GetSingleton();
    std::unique_ptr<ManagedAllocator> m_Allocator;
};

template <class F, class... Args>
auto ManagedRuntime::UseManagedMemory(F &&f, Args &&... args) -> typename std::result_of<F(Args...)>::type
{
    std::lock_guard<std::recursive_mutex> lock(m_Allocator->m_Mutex);
    m_Allocator->m_Pointers.clear();
    m_Allocator->m_UseManagedMemory = true;
    auto retval = f(std::forward<Args>(args)...);
    m_Allocator->m_UseManagedMemory = false;
    m_Allocator->m_Pointers.clear();
    return retval;
}

/**
 * @brief Wrapper class for nvinfer1::ICudaEngine.
 * 
 * A Model object holds an instance of ICudaEngine and extracts some basic meta data
 * from the engine to simplify pushing the input/output bindings to a memory stack.
 */
class Model
{
  public:
    /**
     * @brief Construct a new Model object
     * 
     * @param engine 
     */
    Model(std::shared_ptr<ICudaEngine> engine);
    virtual ~Model() {}

    auto Name() const -> const std::string { return m_Name; }
    void SetName(std::string name) { m_Name = name; }

    void AddWeights(void *, size_t);
    void PrefetchWeights(cudaStream_t) const;

    auto CreateExecutionContext() const -> std::shared_ptr<IExecutionContext>;

    auto GetMaxBatchSize() const { return m_Engine->getMaxBatchSize(); }
    auto GetActivationsMemorySize() const { return m_Engine->getDeviceMemorySize(); }
    auto GetBindingMemorySize() const -> const size_t;
    auto GetWeightsMemorySize() const -> const size_t;

    struct Binding;

    auto GetBinding(uint32_t) const -> const Binding &;
    auto GetBindingsCount() const { return m_Bindings.size(); }
    auto GetInputBindingCount() const { return m_InputBindings.size(); }
    auto GetOutputBindingCount() const { return m_OutputBindings.size(); }
    auto GetInputBindingIds() const -> const std::vector<uint32_t> { return m_InputBindings; }
    auto GetOutputBindingIds() const -> const std::vector<uint32_t> { return m_OutputBindings; }

    struct Binding
    {
        bool isInput;
        int dtypeSize;
        size_t bytesPerBatchItem;
        size_t elementsPerBatchItem;
        std::vector<size_t> dims;
    };

  protected:
    void ConfigureBinding(Binding &, uint32_t);

  private:
    struct Weights
    {
        void *addr;
        size_t size;
    };

    std::shared_ptr<ICudaEngine> m_Engine;
    std::vector<Binding> m_Bindings;
    std::vector<uint32_t> m_InputBindings;
    std::vector<uint32_t> m_OutputBindings;
    std::vector<Weights> m_Weights;
    std::string m_Name;
};

/**
 * @brief Manages input/output buffers and CudaStream
 * 
 * Primary TensorRT resource class used to manage both a host and a device memory stacks
 * and owns the cudaStream_t that should be used for transfers or compute on these
 * resources.
 */
class Buffers : public std::enable_shared_from_this<Buffers>
{
    static auto Create(size_t host_size, size_t device_size) -> std::shared_ptr<Buffers>;

  public:
    virtual ~Buffers();

    void *AllocateHost(size_t size) { return m_HostStack->Allocate(size); }
    void *AllocateDevice(size_t size) { return m_DeviceStack->Allocate(size); } 

    auto CreateBindings(const std::shared_ptr<Model> &model, uint32_t batch_size) -> std::shared_ptr<Bindings>;
    auto CreateAndConfigureBindings(const std::shared_ptr<Model> &model, uint32_t batch_size) -> std::shared_ptr<Bindings>;

    inline cudaStream_t Stream() { return m_Stream; }
    void Synchronize();

  private:
    Buffers(size_t host_size, size_t device_size);
    void Reset(bool writeZeros = false);

    std::shared_ptr<MemoryStack<CudaHostAllocator>> m_HostStack;
    std::shared_ptr<MemoryStack<CudaDeviceAllocator>> m_DeviceStack;
    cudaStream_t m_Stream;

    friend class Resources;
};

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
    virtual ~Bindings();

    void *HostAddress(uint32_t binding_id);
    void *DeviceAddress(uint32_t binding_id);
    void **DeviceAddresses();
    void SetHostAddress(int binding_id, void *addr);
    void SetDeviceAddress(int binding_id, void *addr);

    void *ActivationsAddress() { return m_ActivationsAddress; }
    void SetActivationsAddress(void *addr) { m_ActivationsAddress = addr; }

    void CopyToDevice(uint32_t);
    void CopyToDevice(const std::vector<uint32_t>&);
    void CopyToDevice(uint32_t, void *, size_t);

    void CopyFromDevice(uint32_t);
    void CopyFromDevice(const std::vector<uint32_t>&);
    void CopyFromDevice(uint32_t, void *, size_t);

    auto InputBindings() const { return m_Model->GetInputBindingIds(); }
    auto OutputBindings() const { return m_Model->GetOutputBindingIds(); }

    auto GetModel() -> const std::shared_ptr<Model>& { return m_Model; }
    auto BatchSize() const { return m_BatchSize; }

    inline cudaStream_t Stream() const { return m_Buffers->Stream(); }
    void Synchronize() const { m_Buffers->Synchronize(); };

  private:
    Bindings(const std::shared_ptr<Model>, const std::shared_ptr<Buffers>, uint32_t batch_size);
    size_t BindingSize(uint32_t binding_id) const;

    const std::shared_ptr<Model> m_Model;
    const std::shared_ptr<Buffers> m_Buffers;
    uint32_t m_BatchSize;

    std::vector<void*> m_HostAddresses;
    std::vector<void*> m_DeviceAddresses;
    void *m_ActivationsAddress;

    friend class Buffers;
};

/**
 * @brief Manages the execution of an inference calculation
 * 
 * The ExecutionContext is a limited quanity resource used to control the number
 * of simultaneous calculations allowed on the device at any given time.
 * 
 * A properly configured Bindings object is required to initiate the TensorRT
 * inference calculation.
 */
class ExecutionContext
{
  public:
    virtual ~ExecutionContext();

    void SetContext(std::shared_ptr<IExecutionContext> context);
    void Infer(const std::shared_ptr<Bindings> &);
    void Synchronize();

  private:
    ExecutionContext();
    void Reset();

    cudaEvent_t m_ExecutionContextFinished;
    std::shared_ptr<IExecutionContext> m_Context;

    friend class Resources;
};


/**
 * @brief TensorRT Resource Manager
 */
class Resources : public ::yais::Resources
{
  public:
    Resources(int max_executions, int max_buffers);
    virtual ~Resources();

    void RegisterModel(std::string name, std::shared_ptr<Model> model);
    void RegisterModel(std::string name, std::shared_ptr<Model> model, uint32_t max_concurrency);

    void AllocateResources();

    auto GetBuffers() -> std::shared_ptr<Buffers>;
    auto GetModel(std::string model_name) -> std::shared_ptr<Model>;
    auto GetExecutionContext(const Model *model) -> std::shared_ptr<ExecutionContext>;
    auto GetExecutionContext(const std::shared_ptr<Model>& model) -> std::shared_ptr<ExecutionContext>;

  private:
    size_t Align(size_t size, size_t alignment);

    int m_MaxExecutions;
    int m_MaxBuffers;
    size_t m_MinHostStack;
    size_t m_MinDeviceStack;
    std::shared_ptr<Pool<Buffers>> m_Buffers;
    std::shared_ptr<Pool<ExecutionContext>> m_ExecutionContexts;
    std::map<std::string, std::shared_ptr<Model>> m_Models;
    std::map<const Model*, std::shared_ptr<Pool<IExecutionContext>>> m_ModelExecutionContexts;
};

} // end namespace TensorRT
} // end namespace yais

#endif // NVIS_TENSORRT_H_
