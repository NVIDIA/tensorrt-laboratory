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
#ifndef _YAIS_TENSORRT_RUNTIME_H_
#define _YAIS_TENSORRT_RUNTIME_H_

#include <fstream>
#include <mutex>

#include "YAIS/TensorRT/Common.h"
#include "YAIS/TensorRT/Model.h"
#include "YAIS/TensorRT/Utils.h"

namespace yais
{
namespace TensorRT
{

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
    ::nvinfer1::IRuntime *GetRuntime() { return m_Runtime.get(); }

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
    std::unique_ptr<::nvinfer1::IRuntime, NvInferDeleter> m_Runtime;
};

/**
 * @brief Convenience class wrapping nvinfer1::IRuntime and allowing DNN weights to be stored in
 * unified memory, i.e. memory that can be paged between the host and the device.
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

} // namespace TensorRT
} // namespace yais

#endif // _YAIS_TENSORRT_RUNTIME_H_
