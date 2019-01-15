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
#include "tensorrt/playground/infer_bench.h"
#include "tensorrt/playground/bindings.h"
#include "tensorrt/playground/infer_runner.h"

#include <glog/logging.h>

namespace playground {
namespace TensorRT {

InferBench::InferBench(std::shared_ptr<InferenceManager> resources) : m_Resources(resources) {}
InferBench::~InferBench() {}

InferBench::Results InferBench::Run(std::shared_ptr<Model> model, uint32_t batch_size, double seconds)
{
    std::vector<std::shared_ptr<Model>> models = { model };
    Run(models, batch_size, seconds);
}

InferBench::Results InferBench::Run(const ModelsList& models, uint32_t batch_size, double seconds)
{
    size_t batch_count = 0;
    std::vector<std::shared_future<void>> futures;
    futures.reserve(1024 * 1024);

    // Check ModelsList to ensure the requested batch_size is appropriate
    for(const auto& model : models)
    {
        CHECK_LE(batch_size, model->GetMaxBatchSize());
    }

    // Setup std::chrono deadline - no more elapsed lambda
    auto start = std::chrono::high_resolution_clock::now();
    auto last = start + std::chrono::milliseconds(static_cast<long>(seconds * 1000));

    // Benchmark loop over Models modulo size of ModelsList
    while(std::chrono::high_resolution_clock::now() < last && ++batch_count)
    {
        size_t model_idx = batch_count % models.size();
        const auto& model = models[model_idx];

        auto buffers = InferResources().GetBuffers(); // <=== Limited Resource; May Block !!!
        auto bindings = buffers->CreateBindings(model);
        bindings->SetBatchSize(batch_size);

        InferRunner runner(model, m_Resources);
        futures.push_back(runner.Infer(
            bindings, [](std::shared_ptr<Bindings>& bindings) mutable { bindings.reset(); }));
    }

    // Join worker threads
    for(const auto& f : futures)
    {
        f.wait();
    }

    auto total_time =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    auto inferences = batch_count * batch_size;

    Results results;
    results["batch_size"] = batch_size;
    results["max_exec_concurrency"] = m_Resources->MaxExecConcurrency();
    results["max_copy_concurrency"] = m_Resources->MaxCopyConcurrency();
    results["batch_count"] = batch_count;
    results["total_time"] = total_time;
    results["secs-per-batch-per-stream"] =
        total_time / (batch_count / m_Resources->MaxExecConcurrency());
    results["secs-per-batch"] = total_time / batch_count;
    results["batches-per-sec"] = batch_count / total_time;
    results["inferences-per-sec"] = inferences / total_time;

    DLOG(INFO) << "Benchmark Run Complete";

    return std::move(results);
}

} // namespace TensorRT
} // namespace playground