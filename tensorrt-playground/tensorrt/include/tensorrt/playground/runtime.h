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
#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "tensorrt/playground/common.h"
#include "tensorrt/playground/allocator.h"
#include "tensorrt/playground/model.h"

namespace yais {
namespace TensorRT {
/**
 * @brief Convenience class wrapping nvinfer1::IRuntime.
 *
 * Provide static method for deserializing a saved TensorRT engine/plan
 * and implements a Logger that conforms to YAIS logging.
 */
class Runtime : public std::enable_shared_from_this<Runtime>
{
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    Runtime(Runtime&&) noexcept = delete;
    Runtime& operator=(Runtime&&) = delete;

  public:
    virtual ~Runtime();

    std::shared_ptr<Model> DeserializeEngine(const std::string&);
    std::shared_ptr<Model> DeserializeEngine(const std::string&, ::nvinfer1::IPluginFactory*);
    std::shared_ptr<Model> DeserializeEngine(const void*, size_t);
    virtual std::shared_ptr<Model> DeserializeEngine(const void*, size_t, ::nvinfer1::IPluginFactory*) = 0;

  protected:
    Runtime();

    ::nvinfer1::IRuntime& NvRuntime() const;
    std::vector<char> ReadEngineFile(const std::string&) const;

  private:
    class Logger : public ::nvinfer1::ILogger
    {
      public:
        virtual ~Logger() override;
        void log(::nvinfer1::ILogger::Severity severity, const char* msg) final override;
    };

    // Order is important.  In C++ variables in the member initializer list are instantiated in
    // the order they are declared, not in the order they appear in the initializer list.
    // Inverting these causes m_Runtime to be initialized with a NULL m_Logger and was the
    // source of much head banging.
    std::unique_ptr<::nvinfer1::ILogger> m_Logger;
    std::unique_ptr<::nvinfer1::IRuntime, NvInferDeleter> m_NvRuntime;
};

class RuntimeWithAllocator : public Runtime 
{
  public:
    using Runtime::Runtime;
    virtual ~RuntimeWithAllocator() override;

  protected:
    RuntimeWithAllocator(std::unique_ptr<NvAllocator> allocator);
    std::shared_ptr<Model> DeserializeEngine(const void*, size_t, ::nvinfer1::IPluginFactory*) final override;
    NvAllocator& Allocator()
    {
        return *m_Allocator;
    }

  private:
    std::unique_ptr<NvAllocator> m_Allocator;
};

template<typename AllocatorType>
struct CustomRuntime : public RuntimeWithAllocator
{
    CustomRuntime() : RuntimeWithAllocator(std::make_unique<AllocatorType>()) {}
    virtual ~CustomRuntime() override {}
};

using StandardRuntime = CustomRuntime<StandardAllocator>;
using ManagedRuntime = CustomRuntime<ManagedAllocator>;

} // namespace TensorRT
} // namespace yais
