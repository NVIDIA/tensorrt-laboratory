/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"

#include "tensorrt/playground/core/affinity.h"
#include "tensorrt/playground/core/thread_pool.h"
#include "tensorrt/playground/inference_manager.h"
#include "tensorrt/playground/infer_runner.h"
#include "tensorrt/playground/infer_bench.h"

using playground::Affinity;
using playground::ThreadPool;
using playground::TensorRT::Runtime;
using playground::TensorRT::StandardRuntime;
using playground::TensorRT::InferenceManager;
using playground::TensorRT::Bindings;
using playground::TensorRT::InferRunner;
using playground::TensorRT::InferBench;

#include "tensorrt/playground/core/hybrid_mutex.h"
#include "tensorrt/playground/core/hybrid_condition.h"
using playground::BaseThreadPool;
using HybridThreadPool = BaseThreadPool<hybrid_mutex, hybrid_condition>;

#include "moodycamel/blockingconcurrentqueue.h"

using moodycamel::BlockingConcurrentQueue;
using moodycamel::ConsumerToken;
using moodycamel::ProducerToken;

#include "trace_generator.h"

#include <iostream>

DEFINE_int32(concurrency, 4, "Number of concurrency execution streams");
DEFINE_double(alpha, 1.5, "Scaling Parameter to account for overheads and queue depth");
DEFINE_int32(latency, 12000, "Latency Threshold in microseconds");
DEFINE_int32(timeout,  3000, "Batching Timeout in microseconds");

int main(int argc, char*argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    std::vector<int> query_library(20);
    std::iota(query_library.begin(), query_library.end(), 0);

    // The target latency bound.
    std::chrono::microseconds latency_bound(FLAGS_latency);

    // Batching timeout
    volatile bool stop_batcher = false;
    constexpr uint64_t quanta = 100; // microseconds
    const double timeout = static_cast<double>(FLAGS_timeout - quanta) / 1000000.0; // microseconds

    auto resources = std::make_shared<InferenceManager>(FLAGS_concurrency, FLAGS_concurrency + 4);

    const auto& cuda_thread_affinity = Affinity::GetCpusFromString("0");
    const auto& post_thread_affinity = Affinity::GetCpusFromString("1,2,3,4");
    const auto& batcher_thread_affinity = Affinity::GetCpusFromString("5");

    auto cuda_thread_pool = std::make_unique<ThreadPool>(cuda_thread_affinity);
    auto post_thread_pool = std::make_unique<ThreadPool>(post_thread_affinity);
    auto batcher_thread_pool = std::make_unique<ThreadPool>(batcher_thread_affinity);

    resources->RegisterThreadPool("cuda", std::move(cuda_thread_pool));
    resources->RegisterThreadPool("post", std::move(post_thread_pool));

    std::vector<std::string> engine_files = {
        "/work/models/ResNet-50-b24-fp16.engine",
        "/work/models/ResNet-50-b22-fp16.engine",
        "/work/models/ResNet-50-b20-fp16.engine",
        "/work/models/ResNet-50-b18-fp16.engine",
        "/work/models/ResNet-50-b16-fp16.engine",
        "/work/models/ResNet-50-b14-fp16.engine",
        "/work/models/ResNet-50-b12-fp16.engine",
        "/work/models/ResNet-50-b10-fp16.engine",
        "/work/models/ResNet-50-b8-fp16.engine",
        "/work/models/ResNet-50-b6-fp16.engine"
    };

    std::shared_ptr<Runtime> runtime = std::make_shared<StandardRuntime>();
    std::map<int, std::shared_ptr<InferRunner>> runners_by_batch_size;
    std::map<double, std::shared_ptr<InferRunner>> runners_by_batching_window;
    int max_batch_size = 0;

    for (const auto& file : engine_files)
    {
        auto model = runtime->DeserializeEngine(file);
        resources->RegisterModel(file, model);
        runners_by_batch_size[model->GetMaxBatchSize()] = std::make_shared<InferRunner>(model, resources);
    }

    resources->AllocateResources();

    for (auto& item : runners_by_batch_size)
    {
        InferBench benchmark(resources);
        auto runner = item.second;
        auto model = runner->GetModelSmartPtr();
        int batch_size = model->GetMaxBatchSize();

        auto warmup = benchmark.Run(model, batch_size, 0.2);
        auto result = benchmark.Run(model, batch_size, 5.0);

        using namespace playground::TensorRT;
        auto time_per_batch = result->at(kExecutionTimePerBatch);

        auto batching_window = std::chrono::duration<double>(latency_bound).count() - time_per_batch * FLAGS_alpha;
        if (batching_window <= 0.0) {
            LOG(INFO) << "Batch Sizes >= " << batch_size << " exceed latency threshold";
            break;
        }
        runners_by_batching_window[batching_window] = runner;
        max_batch_size = std::max(max_batch_size, batch_size);

        LOG(INFO) << runner->GetModel().Name() << " exec per batch:  " << result->at(kExecutionTimePerBatch);
        LOG(INFO) << runner->GetModel().Name() << " batching window: " << batching_window;
    }

    struct WorkPacket
    {
        std::chrono::nanoseconds start;
        int query;
        std::function<void()> completion_callback;
    };

    BlockingConcurrentQueue<WorkPacket> work_queue;

    std::vector<std::shared_future<size_t>> futures;
    futures.reserve(1048576);

    batcher_thread_pool->enqueue([&work_queue, &runners_by_batch_size, &runners_by_batching_window, &futures, 
                                  resources, max_batch_size, timeout]() mutable {
        size_t total_count;
        size_t max_deque;
        size_t adjustable_max_batch_size;
        std::vector<WorkPacket> work_packets(max_batch_size);
        thread_local ConsumerToken token(work_queue);
        double elapsed;

        // Batching Loop
        for (;;)
        {
            if (unlikely(stop_batcher)) { return; }
            total_count = 0;
            max_deque = max_batch_size;
            adjustable_max_batch_size = max_batch_size;
            auto start = std::chrono::high_resolution_clock::now();
            auto elapsed_time = [start]() -> double {
                return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
            };

            // if we have a work packet, we stay in this loop and continue to batch until the timeout is reached
            do
            {
                auto count = work_queue.wait_dequeue_bulk_timed(token, &work_packets[total_count], max_deque, quanta);
                total_count += count;
                elapsed = elapsed_time();
                auto runner = runners_by_batching_window.lower_bound(elapsed)->second;
                adjustable_max_batch_size = runner->GetModel().GetMaxBatchSize();
                max_deque = adjustable_max_batch_size - total_count;
                // TODO: update timeout with load-penalty
            } while (total_count && total_count < adjustable_max_batch_size && elapsed < timeout);

            // batching complete, now queue the execution
            if(total_count) {
                // TODO: Move to independent thread
                work_packets.resize(total_count);
                auto buffers = resources->GetBuffers(); // <=== Limited Resource; May Block !!!
                auto runner = runners_by_batch_size.lower_bound(total_count)->second;
                auto bindings = buffers->CreateBindings(runner->GetModelSmartPtr());
                bindings->SetBatchSize(total_count);
                futures.push_back(
                    runner->Infer(
                        bindings, 
                        [wps = std::move(work_packets)](std::shared_ptr<Bindings>& bindings) -> size_t {
                            for(const auto& wp : wps) { wp.completion_callback(); }
                            bindings.reset();
                            return wps.size();
                    })
                );

                // reset work_packets
                CHECK_EQ(work_packets.size(), 0);
                work_packets.resize(max_batch_size);
            }
        }
    });

    auto sync = [&futures]() mutable {
        LOG(INFO) << "Syncing " << futures.size() << " futures.";
        std::map<size_t, size_t> histogram;
        size_t count = 0;
        for (const auto& f : futures) { 
            auto batch_size = f.get();
            count += batch_size;
            histogram[batch_size]++;
        }
        LOG(INFO) << "Histogram of batched work:";
        for (const auto& item : histogram )
        {
            LOG(INFO) << "bs " << item.first << ": " << item.second;
        }
        futures.clear();
        CHECK(futures.size() == 0);
        LOG(INFO) << "Sync Complete - " << count << " work packets completed";
    };

    // The enqueue function takes a start time, a query, and a
    // completion_callback. After infering the query, the completion_callback must
    // be invoked. The enqueue function can execute the inference locally, but the
    // most sensible implementation puts the query and the completion callback in
    // a queue to be executed by another thread. The start time can be ignored or
    // used by some policy mechanism to help determine on batch size.

    // Immmediately enqueue work packet on the TensorRT batching work_queue
    TraceGenerator::EnqueueFn<int> enqueue =
        [&work_queue](std::chrono::nanoseconds start, int query, std::function<void(void)> completion_callback) {
            work_queue.enqueue(WorkPacket{start, query, completion_callback});
        };

    // TODO(tjablin): These parameters should all be read from the command-line or
    // maybe a configuration file?

    // TODO(tjablin): The latency bound of 400 us is unphysically low.

    // The minimum number of queries.
    // uint64_t min_queries = 4096;
    uint64_t min_queries = 4096;

    // The minimum duration of the trace.
    std::chrono::seconds min_duration(5);

    // The minimum percent of queries meeting the latency bound.
    double latency_bound_percentile = 0.95;

    // The pseudo-random number generator's seed.
    uint64_t seed = 0;

    // Given an enqueue implementation and all other parameters, conduct a series
    // of experiments to determine the maximum QPS.
    double max_qps =
        TraceGenerator::FindMaxQPS(query_library, enqueue, sync, seed, latency_bound, min_queries,
                                   min_duration, latency_bound_percentile);

    // Print the results.
    std::cout << "Max QPS subject to " << latency_bound.count() << " us "
              << std::roundl(100 * latency_bound_percentile) << "% latency bound: " << max_qps
              << "\n";

    stop_batcher = true;
    
    return 0;
}
