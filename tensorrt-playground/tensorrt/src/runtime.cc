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
#include "tensorrt/playground/runtime.h"

#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

namespace yais {
namespace TensorRT {

Runtime::Runtime()
    : m_Logger(std::make_unique<Logger>()),
      m_NvRuntime(nv_unique(::nvinfer1::createInferRuntime(*(m_Logger.get()))))
{
    m_Logger->log(::nvinfer1::ILogger::Severity::kINFO, "IRuntime Logger Initialized");
}

Runtime::~Runtime()
{
    DLOG(INFO) << "Destorying Runtime " << this;
}

::nvinfer1::IRuntime& Runtime::NvRuntime() const
{
    return *m_NvRuntime;
}

std::shared_ptr<Model> Runtime::DeserializeEngine(const std::string& plan_file)
{
    DLOG(INFO) << "Deserializing TensorRT ICudaEngine from file: " << plan_file;
    const auto& buffer = ReadEngineFile(plan_file);
    return DeserializeEngine(buffer.data(), buffer.size(), nullptr);
}

std::shared_ptr<Model> Runtime::DeserializeEngine(const std::string& plan_file,
                                                  ::nvinfer1::IPluginFactory* plugin_factory)
{
    DLOG(INFO) << "Deserializing TensorRT ICudaEngine from file: " << plan_file;
    const auto& buffer = ReadEngineFile(plan_file);
    return DeserializeEngine(buffer.data(), buffer.size(), plugin_factory);
}

std::shared_ptr<Model> Runtime::DeserializeEngine(const void* data, size_t size)
{
    return DeserializeEngine(data, size, nullptr);
}

std::vector<char> Runtime::ReadEngineFile(const std::string& plan_file) const
{
    DLOG(INFO) << "Reading Engine: " << plan_file;
    std::ifstream file(plan_file, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0, std::ios::beg);
    CHECK(file.read(buffer.data(), size)) << "Unable to read engine file: " << plan_file;
    return buffer;
}

Runtime::Logger::~Logger()
{
    DLOG(INFO) << "Destroying Logger";
}

void Runtime::Logger::log(::nvinfer1::ILogger::Severity severity, const char* msg)
{
    switch(severity)
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

RuntimeWithAllocator::RuntimeWithAllocator(std::unique_ptr<NvAllocator> allocator)
    : Runtime(), m_Allocator(std::move(allocator))
{
    NvRuntime().setGpuAllocator(m_Allocator.get());
}

RuntimeWithAllocator::~RuntimeWithAllocator()
{
    // DLOG(INFO) << "Destroying CustomRuntime";
    NvRuntime().setGpuAllocator(nullptr);
}

std::shared_ptr<Model>
    RuntimeWithAllocator::DeserializeEngine(const void* data, size_t size,
                                                  ::nvinfer1::IPluginFactory* plugin_factory)
{
    DLOG(INFO) << "Deserializing Custom TensorRT ICudaEngine";
    return Allocator().UseWeightAllocator(
        [this, data, size, plugin_factory]() mutable -> std::shared_ptr<Model> {
            auto runtime = this->shared_from_this();
            auto engine = nv_shared(NvRuntime().deserializeCudaEngine(data, size, plugin_factory),
                                    [runtime] { DLOG(INFO) << "Destroying ICudaEngine"; });
            CHECK(engine) << "Unable to create ICudaEngine";
            auto model = std::make_shared<Model>(engine);
            for(const auto& ptr : Allocator().GetPointers())
            {
                model->AddWeights(ptr.addr, ptr.size);
                DLOG(INFO) << "TensorRT weights: " << ptr.addr << "; size=" << ptr.size;
            }
            return model;
        });
}

} // end namespace TensorRT
} // end namespace yais
