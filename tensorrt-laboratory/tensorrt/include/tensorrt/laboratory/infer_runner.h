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

#include "tensorrt/laboratory/core/async_compute.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/model.h"
#include "tensorrt/laboratory/bindings.h"

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

    using Clock = std::chrono::high_resolution_clock;
    using Deadline = typename Clock::time_point;

    template<typename Post>
    auto InferWithDeadline(std::shared_ptr<Bindings> bindings, Post post,
            Deadline deadline, std::function<void(std::function<void()>)> on_timeout)
    {
        auto compute = Wrap(post);
        auto future = compute->Future();
        EnqueueWithDeadline(bindings, compute, deadline, on_timeout);
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
            CHECK_EQ(cudaSetDevice(Resources().DeviceID()), CUDA_SUCCESS);
            bindings->CopyToDevice(bindings->InputBindings());
            auto trt_ctx = Compute(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            Workers("post").enqueue([this, bindings, trt_ctx, Post]() mutable {
                CHECK_EQ(cudaSetDevice(Resources().DeviceID()), CUDA_SUCCESS);
                DVLOG(1) << this << " - " << trt_ctx.get() << ": sync trt_ctx";
                trt_ctx->Synchronize();
                DVLOG(1) << this << " - " << trt_ctx.get() << ": sync trt_ctx - completed";
                trt_ctx.reset();
                DVLOG(1) << this << " - " << bindings.get() << ": sync bindings";
                bindings->Synchronize();
                DVLOG(1) << this << " - " << bindings.get() << ": sync bindings - completed";
                (*Post)(bindings);
                bindings.reset();
                DVLOG(2) << this << ": completed inference task";
            });
        });
    }

    template<typename T>
    void EnqueueWithDeadline(std::shared_ptr<Bindings> bindings, std::shared_ptr<AsyncCompute<T>> Post,
            Deadline deadline, std::function<void(std::function<void()>)> on_timeout)
    {
        CHECK(false) << "infer with deadline disabled";
        Workers("cuda").enqueue([this, bindings, Post, deadline, on_timeout]() mutable {
            if (Clock::now() > deadline)
            {
                // deadline requirement failed - cancel task
		size_t val = 1000 * bindings->BatchSize();
                on_timeout([Post, val] {
                    Post->Override(val);
                });
                return;
            }
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
        const auto& model = bindings->GetModel();
        auto trt_ctx = m_Resources->GetExecutionContext(model);
        trt_ctx->SetGraphWorkspace(bindings->GetGraphWorkspace());
        trt_ctx->Infer(bindings);
        return trt_ctx;
    }

    inline ThreadPool& Workers(std::string name)
    {
        return m_Resources->AcquireThreadPool(name);
    }
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
    const Model& GetModel() const
    {
        return *m_Model;
    }

    const std::shared_ptr<Model> GetModelSmartPtr() const { return m_Model; }

    InferenceManager& Resources()
    {
        return *m_Resources;
    }

  private:
    std::shared_ptr<Model> m_Model;
    std::shared_ptr<InferenceManager> m_Resources;
};

}
}
