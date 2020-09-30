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
#include "trtlab/tensorrt/runtime.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <glog/logging.h>

using namespace trtlab;
using namespace TensorRT;

namespace
{
    bool file_exists(const std::string& name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
} // namespace

Runtime::Runtime() : m_Logger(std::make_unique<Logger>()), m_Runtime(nv_unique(::nvinfer1::createInferRuntime(*(m_Logger.get()))))
{
    m_Logger->log(::nvinfer1::ILogger::Severity::kINFO, "IRuntime Logger Initialized");
}

Runtime::~Runtime()
{
    VLOG(2) << "Destorying Runtime " << this;
}

::nvinfer1::IRuntime& Runtime::get_runtime() const
{
    return *m_Runtime;
}

std::shared_ptr<Model> Runtime::deserialize_engine(const std::string& plan_file)
{
    VLOG(2) << "Deserializing TensorRT ICudaEngine from file: " << plan_file;
    const auto& buffer = read_engine_file(plan_file);
    return deserialize_engine(buffer.data(), buffer.size(), nullptr);
}

std::shared_ptr<Model> Runtime::deserialize_engine(const std::string& plan_file, ::nvinfer1::IPluginFactory* plugin_factory)
{
    VLOG(2) << "Deserializing TensorRT ICudaEngine from file: " << plan_file;
    const auto& buffer = read_engine_file(plan_file);
    return deserialize_engine(buffer.data(), buffer.size(), plugin_factory);
}

std::shared_ptr<Model> Runtime::deserialize_engine(const void* data, size_t size)
{
    return deserialize_engine(data, size, nullptr);
}

std::vector<char> Runtime::read_engine_file(const std::string& plan_file) const
{
    if(!file_exists(plan_file))
    {
        throw std::runtime_error("tensorrt engine file does not exist: " + plan_file);
    }
    VLOG(2) << "Reading Engine: " << plan_file;
    std::ifstream   file(plan_file, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    CHECK_GT(size, 0);
    std::vector<char> buffer(size);
    file.seekg(0, std::ios::beg);
    CHECK(file.read(buffer.data(), size)) << "Unable to read engine file: " << plan_file;
    return buffer;
}

Runtime::Logger::~Logger()
{
    VLOG(2) << "Destroying Logger " << this;
}

void Runtime::Logger::log(::nvinfer1::ILogger::Severity severity, const char* msg)
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
        VLOG(2) << "[TensorRT.INFO]: " << msg;
        break;
    default:
        VLOG(2) << "[TensorRT.DEBUG]: " << msg;
        break;
    }
}

RuntimeWithAllocator::RuntimeWithAllocator(std::unique_ptr<NvAllocator> allocator) : Runtime(), m_Allocator(std::move(allocator))
{
    get_runtime().setGpuAllocator(m_Allocator.get());
}

RuntimeWithAllocator::~RuntimeWithAllocator()
{
    get_runtime().setGpuAllocator(nullptr);
}

std::shared_ptr<Model> RuntimeWithAllocator::deserialize_engine(const void* data, size_t size, ::nvinfer1::IPluginFactory* plugin_factory)
{
    VLOG(2) << "Deserializing Custom TensorRT ICudaEngine";
    return get_allocator().use_weights_allocator([this, data, size, plugin_factory]() mutable -> std::shared_ptr<Model> {
        auto runtime = this->shared_from_this();
        auto engine  = nv_shared(get_runtime().deserializeCudaEngine(data, size, plugin_factory), [runtime]() mutable { runtime.reset(); });
        CHECK(engine) << "Unable to create ICudaEngine";
        return std::make_shared<Model>(engine, get_allocator().get_pointers());
    });
}
