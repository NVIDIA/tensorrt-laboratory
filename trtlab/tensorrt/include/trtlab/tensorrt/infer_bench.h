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

#include "trtlab/tensorrt/inference_manager.h"
#include "trtlab/tensorrt/model.h"

namespace trtlab {
namespace TensorRT {

enum InferBenchKey
{
    kMaxExecConcurrency = 0,
    kMaxCopyConcurrency,
    kBatchSize,
    kWalltime,
    kBatchesComputed,
    kBatchesPerSecond,
    kInferencesPerSecond,
    kSecondsPerBatch,
    kExecutionTimePerBatch
};

class InferBench
{
  public:
    InferBench(std::shared_ptr<InferenceManager>);
    virtual ~InferBench();

    using ModelsList = std::vector<std::shared_ptr<Model>>;
    using Results = std::map<InferBenchKey, double>;

    std::unique_ptr<Results> Run(const std::shared_ptr<Model> model, uint32_t batch_size,
                                 double seconds = 5.0);
    std::unique_ptr<Results> Run(const ModelsList& models, uint32_t batch_size,
                                 double seconds = 5.0);

  protected:
    InferenceManager& InferResources() { return *m_Resources; }

  private:
    std::shared_ptr<InferenceManager> m_Resources;
};

} // namespace TensorRT
} // namespace trtlab