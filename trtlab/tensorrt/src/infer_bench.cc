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
#include "trtlab/tensorrt/infer_bench.h"
#include "trtlab/tensorrt/bindings.h"
#include "trtlab/tensorrt/infer_runner.h"

#include <glog/logging.h>

namespace trtlab {
namespace TensorRT {

InferBench::InferBench(std::shared_ptr<InferenceManager> resources) : m_Resources(resources) {}
InferBench::~InferBench() {}

std::unique_ptr<InferBench::Results> InferBench::Run(std::shared_ptr<Model> model,
                                                     uint32_t batch_size, double seconds)
{
    std::vector<std::shared_ptr<Model>> models = {model};
    return std::move(Run(models, batch_size, seconds));
}

std::unique_ptr<InferBench::Results> InferBench::Run(const ModelsList& models, uint32_t batch_size,
                                                     double seconds)
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

    auto results_ptr = std::make_unique<InferBench::Results>();
    Results& results = *results_ptr;
    results[kBatchSize] = batch_size;
    results[kMaxExecConcurrency] = m_Resources->MaxExecConcurrency();
    results[kMaxCopyConcurrency] = m_Resources->MaxCopyConcurrency();
    results[kBatchesComputed] = batch_count;
    results[kWalltime] = total_time;
    results[kBatchesPerSecond] = batch_count / total_time;
    results[kInferencesPerSecond] = inferences / total_time;
    results[kExecutionTimePerBatch] =
        total_time / (batch_count / m_Resources->MaxExecConcurrency());

    DLOG(INFO) << "Benchmark Run Complete";

    DLOG(INFO) << "Inference Results: " << results[kBatchesComputed] << " batches computed in "
               << results[kWalltime] << " seconds on " << results[kMaxExecConcurrency]
               << " compute streams using batch_size: " << results[kBatchSize]
               << "; inf/sec: " << results[kInferencesPerSecond]
               << "; batches/sec: " << results[kBatchesPerSecond]
               << "; execution time per batch: " << results[kExecutionTimePerBatch];

    return std::move(results_ptr);
}

} // namespace TensorRT
} // namespace trtlab