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

#include "trtlab/tensorrt/bindings.h"
#include "trtlab/core/async_compute.h"
#include "trtlab/tensorrt/inference_manager.h"
#include "trtlab/tensorrt/model.h"

namespace trtlab {
namespace TensorRT {

struct InferRunner : public AsyncComputeWrapper<void(std::shared_ptr<Bindings>&)>
{
    InferRunner(std::shared_ptr<Model> model, std::shared_ptr<InferenceManager> resources)
        : m_Model{model}, m_Resources{resources}
    {
    }

    InferRunner(InferRunner&&) = delete;
    InferRunner& operator=(InferRunner&&) = delete;

    InferRunner(const InferRunner&) = delete;
    InferRunner& operator=(const InferRunner&) = delete;

    virtual ~InferRunner() {}

    using BindingsHandle = std::shared_ptr<Bindings>;
    using PreFn = std::function<void(Bindings&)>;

    template<typename Post>
    auto Infer(PreFn pre, Post post)
    {
        auto compute = Wrap(post);
        auto future = compute->Future();
        Enqueue(pre, compute);
        return future.share();
    }

    template<typename Post>
    auto Infer(std::shared_ptr<Bindings> bindings, Post post)
    {
        auto compute = Wrap(post);
        auto future = compute->Future();
        Enqueue(bindings, compute);
        return future.share();
    }

  protected:
    template<typename T>
    void Enqueue(PreFn Pre, std::shared_ptr<AsyncCompute<T>> Post)
    {
        Workers("pre").enqueue([this, Pre, Post]() mutable {
            auto bindings = InitializeBindings();
            Pre(*bindings);
            Enqueue(bindings, Post);
        });
    }

    template<typename T>
    void Enqueue(std::shared_ptr<Bindings> bindings, std::shared_ptr<AsyncCompute<T>> Post)
    {
        Workers("cuda").enqueue([this, bindings, Post]() mutable {
            DLOG(INFO) << "H2D";
            bindings->CopyToDevice(bindings->InputBindings());
            DLOG(INFO) << "Compute";
            auto trt_ctx = Compute(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            Workers("post").enqueue([this, bindings, trt_ctx, Post]() mutable {
                trt_ctx->Synchronize();
                trt_ctx.reset();
                DLOG(INFO) << "Sync TRT";
                bindings->Synchronize();
                DLOG(INFO) << "Sync D2H";
                (*Post)(bindings);
                bindings.reset();
                DLOG(INFO) << "Execute Finished";
            });
        });
    }

    BindingsHandle InitializeBindings()
    {
        auto buffers = m_Resources->GetBuffers();
        return buffers->CreateBindings(m_Model);
    }

    auto Compute(BindingsHandle& bindings) -> std::shared_ptr<ExecutionContext>
    {
        auto trt_ctx = m_Resources->GetExecutionContext(bindings->GetModel());
        trt_ctx->Infer(bindings);
        return trt_ctx;
    }

    inline ThreadPool& Workers(std::string name) { return m_Resources->AcquireThreadPool(name); }
    /*
        void CopyInputsToInputBindings(const HostMap& inputs, BindingsHandle& bindings)
        {
            for (const auto& id : bindings->InputBindings())
            {
                const auto& b = bindings->GetBindings(id);
                auto search = inputs.find(b.name);
                CHECK(search != inputs.end());
                Copy(bindings->HostMemoryDescriptor(id), *inputs[b.name],
       bindings->BindingSize(id));
            }
        }
    */
    /*j
        const Model& Model() const
        {
            return *m_Model;
        }

        const InferenceManager& Resources() const
        {
            return *m_Resources;
        }
    */

  public:
    const int MaxBatchSize() const { return m_Model->GetMaxBatchSize(); }

    const Model& GetModel() const { return *m_Model; }

    const std::shared_ptr<Model> GetModelSmartPtr() const { return m_Model; }

    InferenceManager& Resources() { return *m_Resources; }

  private:
    std::shared_ptr<Model> m_Model;
    std::shared_ptr<InferenceManager> m_Resources;
};

} // namespace TensorRT
} // namespace trtlab