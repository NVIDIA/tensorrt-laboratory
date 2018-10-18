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
#include "YAIS/TensorRT/Runtime.h"

#include <sys/stat.h>
#include <unistd.h>

#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

namespace yais
{
namespace TensorRT
{

/**
 * @brief Construct a new Runtime object
 */
Runtime::Runtime()
    : m_Logger(std::make_unique<Logger>()),
      m_Runtime(make_unique(::nvinfer1::createInferRuntime(*(m_Logger.get()))))
{
    m_Logger->log(::nvinfer1::ILogger::Severity::kINFO, "IRuntime Logger Initialized");
}

/**
 * @brief Deserialize a TensorRT Engine/Plan
 *
 * @param filename Path to the engine/plan file 
 * @return std::shared_ptr<Model>
 */
std::shared_ptr<Model> Runtime::DeserializeEngine(std::string plan_file)
{
    auto singleton = Runtime::GetSingleton();
    auto buffer = singleton->ReadEngineFile(plan_file);
    // Create Engine / Deserialize Plan - need this step to be broken up plz!!
    DLOG(INFO) << "Deserializing TensorRT ICudaEngine";
    return singleton->DeserializeEngine(buffer.data(), buffer.size());
}

std::shared_ptr<Model> Runtime::DeserializeEngine(void *data, size_t size)
{
    auto singleton = Runtime::GetSingleton();
    auto engine = make_shared(
        singleton->GetRuntime()->deserializeCudaEngine(data, size, nullptr));
    CHECK(engine) << "Unable to create ICudaEngine";
    return std::make_shared<Model>(engine);
}

Runtime *Runtime::GetSingleton()
{
    static Runtime singleton;
    return &singleton;
}

std::vector<char> Runtime::ReadEngineFile(std::string plan_file)
{
    struct stat stat_buffer;
    CHECK_EQ(stat(plan_file.c_str(), &stat_buffer), 0) << "File not found: " << plan_file;
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

/**
 * @brief Construct a new ManagedRuntime object
 * 
 * ManagedRuntime overrides the default TensorRT memory allocator by providing a custom
 * allocator, conforming to nvinfer1::IGpuAllocator, to enable the allocation of memory
 * for storing the weights of a TensorRT engine using CUDA Unified Memory.  Specifically,
 * cudaMallocManaged is used and tracked for any TensorRT operation wrapped by a lambda
 * passed to UseManagedMemory.
 */
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
    return singleton->DeserializeEngine(buffer.data(), buffer.size());
}

std::shared_ptr<Model> ManagedRuntime::DeserializeEngine(void *data, size_t size)
{
    auto singleton = ManagedRuntime::GetSingleton();
    return singleton->UseManagedMemory([singleton, data, size]() -> std::shared_ptr<Model> {
        DLOG(INFO) << "Deserializing TensorRT ICudaEngine";
        auto engine = make_shared(
            singleton->GetRuntime()->deserializeCudaEngine(data, size, nullptr));

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
        DLOG(INFO) << "TensoRT cudaMallocManaged size = " << size << "; " << ptr;
        m_Pointers.push_back(Pointer{ptr, size});
    }
    else
    {
        CHECK_EQ(cudaMalloc(&ptr, size), CUDA_SUCCESS)
            << "Failed to allocate TensorRT device memory (not managed)";
        DLOG(INFO) << "TensoRT cudaMalloc size = " << size;
    }
    return ptr;
}

void ManagedRuntime::ManagedAllocator::free(void *ptr)
{
    CHECK_EQ(cudaFree(ptr), CUDA_SUCCESS) << "Failed to free TensorRT device memory";
}

} // end namespace TensorRT
} // end namespace yais
