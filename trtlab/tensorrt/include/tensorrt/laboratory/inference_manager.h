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
#pragma once

#include <map>
#include <memory>
#include <mutex>
// #include <shared_mutex> /* C++17 - not found in g++ 5.4 */

#include <NvInfer.h>

#include "tensorrt/laboratory/buffers.h"
#include "tensorrt/laboratory/common.h"
#include "tensorrt/laboratory/core/pool.h"
#include "tensorrt/laboratory/core/resources.h"
#include "tensorrt/laboratory/core/thread_pool.h"
#include "tensorrt/laboratory/execution_context.h"
#include "tensorrt/laboratory/model.h"
#include "tensorrt/laboratory/runtime.h"

namespace trtlab {
namespace TensorRT {

class InferenceManager : public ::trtlab::Resources
{
  public:
    InferenceManager(int max_executions, int max_buffers);
    virtual ~InferenceManager();

    void RegisterModel(const std::string& name, std::shared_ptr<Model> model);
    void RegisterModel(const std::string& name, std::shared_ptr<Model> model,
                       uint32_t max_concurrency);

    // void RegisterModel(const std::string& name, const std::string& model_path, uint32_t
    // max_concurrency); void RegisterModel(const std::string& name, const std::string& model_path,
    // uint32_t max_concurrency);

    void AllocateResources();

    auto GetBuffers() -> std::shared_ptr<Buffers>;
    auto GetModel(std::string model_name) -> std::shared_ptr<Model>;
    auto GetExecutionContext(const Model* model) -> std::shared_ptr<ExecutionContext>;
    auto GetExecutionContext(const std::shared_ptr<Model>& model)
        -> std::shared_ptr<ExecutionContext>;

    auto AcquireThreadPool(const std::string&) -> ThreadPool&;
    void RegisterThreadPool(const std::string&, std::unique_ptr<ThreadPool> threads);
    bool HasThreadPool(const std::string&) const;
    void JoinAllThreads();

    Runtime& ActiveRuntime();
    void RegisterRuntime(const std::string&, std::shared_ptr<Runtime>);
    void SetActiveRuntime(const std::string&);
    void SetActiveRuntimeToDefault();

    int MaxExecConcurrency() const;
    int MaxCopyConcurrency() const;

    void ForEachModel(std::function<void(const Model&)>);

  private:
    int m_MaxExecutions;
    int m_MaxBuffers;
    size_t m_HostStackSize;
    size_t m_DeviceStackSize;
    size_t m_ActivationsSize;
    Runtime* m_ActiveRuntime;

    std::map<std::string, std::unique_ptr<ThreadPool>> m_ThreadPools;
    std::map<std::string, std::shared_ptr<Runtime>> m_Runtimes;
    std::map<std::string, std::shared_ptr<Model>> m_Models;
    std::map<const Model*, std::shared_ptr<Pool<::nvinfer1::IExecutionContext>>>
        m_ModelExecutionContexts;

    std::shared_ptr<Pool<Buffers>> m_Buffers;
    std::shared_ptr<Pool<ExecutionContext>> m_ExecutionContexts;

    std::size_t Align(std::size_t size, std::size_t alignment)
    {
        std::size_t remainder = size % alignment;
        size = (remainder == 0) ? size : size + alignment - remainder;
        return size;
    }
};

} // namespace TensorRT
} // namespace trtlab
