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

template <typename T>
std::shared_ptr<T> make_shared(T *obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, NvInferDeleter());
};

template <typename T>
std::unique_ptr<T, NvInferDeleter> make_unique(T *obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::unique_ptr<T, NvInferDeleter>(obj);
}

class Model;

/**
 * @brief Convenience class wrapping nvinfer1::IRuntime.
 * 
 * Provide static method for deserializing a saved TensorRT engine/plan
 * and implements a Logger that conforms to YAIS logging.
 */
class Runtime
{
  public:
    /**
     * @brief Deserialize a TensorRT Engine/Plan
     *
     * @param filename Path to the engine/plan file 
     * @return std::shared_ptr<Model>
     */
    static std::shared_ptr<Model> DeserializeEngine(std::string plan_file);

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

#if NV_TENSORRT_MAJOR >= 4

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

  protected:
    struct Pointer
    {
        void *addr;
        size_t size;
    };

    class ManagedAllocator : public ::nvinfer1::IGpuAllocator
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

#endif

class ExecutionContext;

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

    struct Binding;

    /**
     * @brief Get the Max Batch Size for the compiled plan
     * 
     * @return auto Integer value
     */
    auto GetMaxBatchSize() const { return m_Engine->getMaxBatchSize(); }

    /**
     * @brief Get the size of the Device Memory required for inference
     * 
     * @return auto 
     */
    auto GetDeviceMemorySize() const { return m_Engine->getDeviceMemorySize(); }

    /**
     * @brief Get the number of Bindings for the compiled plan
     * 
     * @return auto Integer value
     */
    auto GetBindingsCount() const { return m_Bindings.size(); }

    /**
     * @brief Get required memory storage for binding i.
     * 
     * @param i Binding ID
     * @return auto Integer value
     */
    const Binding &GetBinding(uint32_t id) const
    {
        CHECK_LT(id, m_Bindings.size()) << "Invalid BindingId; given: " << id << "; max: " << m_Bindings.size();
        return m_Bindings[id];
    }

    /**
     * @brief Get the number of input Bindings
     * 
     * @return auto 
     */
    auto GetInputBindingCount() const { return m_InputBindings.size(); }

    /**
     * @brief Get the vector<int> of Input Bindings
     * 
     * @return auto 
     */
    const auto GetInputBindingIds() const { return m_InputBindings; }

    /**
     * @brief Get the number of output Bindings
     * 
     * @return auto 
     */
    auto GetOutputBindingCount() const { return m_OutputBindings.size(); }

    /**
     * @brief Get the vector<int> of Output Bindings
     * 
     * @return auto 
     */
    auto GetOutputBindingIds() const { return m_OutputBindings; }

    size_t GetMaxBufferSize();
    size_t GetMaxWorkspaceSize();

    struct Binding
    {
        bool isInput;
        int dtypeSize;
        size_t bytesPerBatchItem;
        size_t elementsPerBatchItem;
        std::vector<size_t> dims;
    };

    void AddWeights(void *, size_t);
    void PrefetchWeights(cudaStream_t) const;

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
    std::vector<int> m_InputBindings;
    std::vector<int> m_OutputBindings;
    std::vector<Weights> m_Weights;

    friend class ExecutionContext;
};

/**
 * @brief TensorRT managed input/output buffers and Cuda Stream
 * 
 * Primary TensorRT resource class used to manage both a host and a device memory stack,
 * and any assocated async transfers and/or compute on the memory resources.  A Buffers
 * object owns the cudaStream_t that should be used for transfers or compute on these
 * resources.
 */
class Buffers
{
  public:
    /**
     * @brief Construct a new Buffers object
     * 
     * In most cases, Buffers will be created with equal sized host and device stacks;
     * however, for very custom cases, you may choose to configure them to your problem.
     * 
     * @param host_size 
     * @param device_size 
     */
    Buffers(size_t host_size, size_t device_size);
    ~Buffers();

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
     */
    void Configure(const Model *model, uint32_t batch_size);

    /**
     * @brief Swap the Host MemoryStack with an equivalently sized MemoryStack.
     * 
     * This function allows you to pull the swap the Buffer's Host MemoryStack with another
     * stack.  This is useful in pre/post-processing step, where if you have thread-local
     * or function local stack which you setup/use without needing to pull a full Buffers
     * object from the resource pool.  Example, after the D2H copy resulting in the results
     * of an inference calculation in the host memory stack, swap it with a local copy, then
     * release the Buffers object back to the pull, the operate on the host memorystack with
     * the inference results.
     */
    // void SwapHostStack(std::unique_ptr<MemoryStackWithTracking<CudaHostAllocator>>);

    /**
     * @brief Resets the Host and Device Stack Pointers to their origins
     * 
     * @param writeZeros 
     */
    void Reset(bool writeZeros = false);

    /**
     * @brief Get the Host Binding Pointer for a given Binding
     * 
     * @param id 
     * @return void* 
     */
    void *GetHostBinding(uint32_t id) { return m_Host->GetPointer(id); }

    /**
     * @brief Get the Device Binding Pointer for a given Binding
     * 
     * @param id 
     * @return void* 
     */
    void *GetDeviceBinding(uint32_t id) { return m_Device->GetPointer(id); }

    /**
     * @brief Get the all the Device Pointers in an ordered list.
     * 
     * This call provides access to the pointer to the beginning of an array of void* for
     * all device bindings.  This method provides the `buffers` input argument for a TensorRT
     * IExecutionContext `execute` or `enqueue` method.
     * 
     * @return void** 
     */
    void **GetDeviceBindings() { return m_Device->GetPointers(); }

    /**
     * @brief Get the Stream that shoud be used to manage all transfers and compute associated
     * with the data stored in this Buffers object.
     * 
     * @return cudaStream_t 
     */
    inline cudaStream_t GetStream() { return m_Stream; }

    /**
     * @brief Syncrhonize on Cuda Stream
     * 
     * Calls cudaStreamSynchronize on the internal stream.
     */
    void SynchronizeStream();

    /**
     * @brief Performs a H2D copy for a given Binding ID
     * 
     * Performs an H2D copy of the memory associcated with Binding ID.  The copy is performed
     * from the Host MemoryStack -> Device MemoryStack for a given Binding.  The size was
     * recorded when the Buffers object was Configured.
     * 
     * @param binding_id
     */
    void AsyncH2D(uint32_t); // specify the binding idx for to copy

    /**
     * @brief Performs a D2H copy for a given Binding ID
     * 
     * Performs an D2H copy of the memory associcated with Binding ID.  The copy is performed
     * from the Device MemoryStack -> Host MemoryStack for a given Binding.  The size was
     * recorded when the Buffers object was Configured.
     * 
     * @param binding_id
     */
    void AsyncD2H(uint32_t); // specify the binding idx for to copy

    /**
     * @brief Performs a H2D copy from a Source Host Pointer to a Device Binding ID
     * 
     * Copies from given host location and size to the Device allocation asssocated with Binding ID.
     * The copy size is check to be less than or equal to the Binding Size.
     */
    void AsyncH2D(uint32_t, void *, size_t); // binding idx, host src H2D ptr and custom size

    /**
     * @brief Performs a D2H copy from the Device Binding ID to a custom Host Pointer
     * 
     * Copies from the Device memory assocated with Binding ID to a passed destination pointer on
     * the host.  This allows you to skip the Host MemoryStack if you choose to do so.
     */
    void AsyncD2H(uint32_t, void *, size_t); // binding idx, host dst D2H ptr and custom size

  private:
    std::shared_ptr<MemoryStack<CudaHostAllocator>> m_HostStack;
    std::shared_ptr<MemoryStack<CudaDeviceAllocator>> m_DeviceStack;
    std::shared_ptr<MemoryStackTracker> m_Host;
    std::shared_ptr<MemoryStackTracker> m_Device;
    cudaStream_t m_Stream;
};

/**
 * @brief TensorRT convenience class for wrapping IExecutionContext
 * 
 * Designed to launch, time and sychronized the execution of an async inference calculation.
 * 
 * A TensorRT IExecutionContext allows you to pass an event which will be trigged when the input bindings
 * have been fully consumed and can be reused.  However, our input and output bindings are in the same
 * memory stack, so we are most interested in when the IExecutionContext has finished it's forward pass,
 * so we can immediately release the ExecutionContext to threads blocking on access.
 * 
 * Note: Timing is not implemented yet.  When implemented, do so a the host level and not the GPU
 * level to maximize GPU performance by using cudaEventDisabledTiming.  We don't need super accurate
 * timings, they are simply a nice-to-have, so a reasonable approximation on the host is sufficient.
 */
class ExecutionContext
{
  public:
    /**
     * @brief Construct a new Execution Context object
     * 
     * @param ctx 
     */
    ExecutionContext(std::shared_ptr<Model> model);
    virtual ~ExecutionContext();

    /**
     * @brief Enqueue an Inference calculation on a Stream
     * 
     * Initiates a forward pass through a TensorRT optimized graph and registers an event on the stream
     * which is trigged when the compute has finished and the ExecutionContext can be reused by competing threads.
     * Use the Synchronize method to sync on this event.
     * 
     * @param batch_size 
     * @param buffers 
     * @param stream 
     */
    void Enqueue(int batch_size, void **buffers, cudaStream_t stream);

    /**
     * @brief Synchronized on the Completion of the Inference Calculation
     */
    void Synchronize();

  private:
    std::shared_ptr<Model> m_Model;
    std::shared_ptr<IExecutionContext> m_Context;
    cudaEvent_t m_ExecutionContextFinished;
};

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
class Resources : public ::yais::Resources
{
  public:
    /**
     * @brief Construct a new TensorRT Resources object
     * 
     * Instantiated with the ICudaEngine and the size of the Resource Pools for memory Buffers and compute
     * ExecutionContexts.  The constructor will appropriately size the memory Buffers by inspecting the
     * requirements of the ICudaEngine.
     * 
     * @param engine 
     * @param nBuffers 
     * @param nExec 
     */
    explicit Resources(std::shared_ptr<Model> model, int nBuffers = 4, int nExec = 3);
    ~Resources();

  public:
    /**
     * @brief Get the Model object
     * 
     * @return const Model* 
     */
    const Model *GetModel() { return m_Model.get(); }

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
    std::shared_ptr<Buffers> GetBuffers()
    {
        auto from_pool = m_Buffers->Pop(); // Deleter returns Buffers to Pool
        // Chaining Custom Deleters to Reset the Buffer object prior to returning it to the Pool
        return std::shared_ptr<Buffers>(from_pool.get(), [from_pool](void *ptr) {
            from_pool->Reset();
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
    std::shared_ptr<ExecutionContext> GetExeuctionContext() { return m_ExecutionContexts->Pop(); }

  private:
    std::shared_ptr<Model> m_Model;
    std::shared_ptr<Pool<Buffers>> m_Buffers;
    std::shared_ptr<Pool<ExecutionContext>> m_ExecutionContexts;
};

} // end namespace TensorRT
} // end namespace yais

#endif // NVIS_TENSORRT_H_
