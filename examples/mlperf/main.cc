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

#include "tensorrt/laboratory/core/affinity.h"
#include "tensorrt/laboratory/core/thread_pool.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/infer_runner.h"
#include "tensorrt/laboratory/infer_bench.h"

using trtlab::Affinity;
using trtlab::ThreadPool;
using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::StandardRuntime;
using trtlab::TensorRT::InferenceManager;
using trtlab::TensorRT::Bindings;
using trtlab::TensorRT::InferRunner;
using trtlab::TensorRT::InferBench;

#include "tensorrt/laboratory/core/hybrid_mutex.h"
#include "tensorrt/laboratory/core/hybrid_condition.h"
using trtlab::BaseThreadPool;
using HybridThreadPool = BaseThreadPool<hybrid_mutex, hybrid_condition>;

#include "moodycamel/blockingconcurrentqueue.h"

using moodycamel::BlockingConcurrentQueue;
using moodycamel::ConsumerToken;
using moodycamel::ProducerToken;

#include "trace_generator.h"

#include <iostream>

#define likely(x) __builtin_expect((x),1)

DEFINE_int32(concurrency, 4, "Number of concurrency execution streams");
DEFINE_double(alpha, 1.5, "Parameter to account for overheads and queue depth - scalar");
DEFINE_double(beta, 0.0, "Parameter to account for overheads and queue depth - constant in microseconds");
DEFINE_double(gamma, 1.00, "Parameter to account for overheads and queue depth - scalar");
DEFINE_int32(latency, 10000, "Latency Threshold in microseconds");
DEFINE_int32(timeout,  3000, "Batching Timeout in microseconds");
DEFINE_int64(seed, 0, "Random Seed");
DEFINE_int32(percentile, 90, "Percentile");
DEFINE_int32(duration, 5, "Min Trace Duration");

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
    volatile bool running = true;
    constexpr uint64_t quanta = 20; // microseconds
    const double timeout = static_cast<double>(FLAGS_timeout - quanta) / 1000000.0; // microseconds

    auto resources = std::make_shared<InferenceManager>(FLAGS_concurrency, FLAGS_concurrency + 4);

    const auto& cuda_thread_affinity = Affinity::GetCpusFromString("0");
    const auto& post_thread_affinity = Affinity::GetCpusFromString("1,2,3,4");
    const auto& batcher_thread_affinity = Affinity::GetCpusFromString("5");
    const auto& death_thread_affinity = Affinity::GetCpusFromString("6,7,8");

    auto cuda_thread_pool = std::make_unique<ThreadPool>(cuda_thread_affinity);
    auto post_thread_pool = std::make_unique<ThreadPool>(post_thread_affinity);
    auto batcher_thread_pool = std::make_unique<ThreadPool>(batcher_thread_affinity);
    auto death_thread_pool = std::make_unique<ThreadPool>(death_thread_affinity);

    resources->RegisterThreadPool("cuda", std::move(cuda_thread_pool));
    resources->RegisterThreadPool("post", std::move(post_thread_pool));

    std::vector<std::string> engine_files = {

        "/work/models/ResNet-50-b20-fp16.engine",
        "/work/models/ResNet-50-b18-fp16.engine",
        "/work/models/ResNet-50-b16-fp16.engine",
        "/work/models/ResNet-50-b14-fp16.engine",
        "/work/models/ResNet-50-b12-fp16.engine",
        "/work/models/ResNet-50-b10-fp16.engine",
        "/work/models/ResNet-50-b8-fp16.engine",
        "/work/models/ResNet-50-b6-fp16.engine",
        "/work/models/ResNet-50-b4-fp16.engine",
        "/work/models/ResNet-50-b2-fp16.engine",
    };

    std::shared_ptr<Runtime> runtime = std::make_shared<StandardRuntime>();
    std::map<int, std::shared_ptr<InferRunner>> runners_by_batch_size;
    std::map<double, std::shared_ptr<InferRunner>> runners_by_batching_window;
    std::map<std::shared_ptr<InferRunner>, double> time_per_batch_by_runner;
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

        using namespace trtlab::TensorRT;
        auto time_per_batch = result->at(kExecutionTimePerBatch);
        time_per_batch_by_runner[runner] = time_per_batch;

        auto batching_window = std::chrono::duration<double>(latency_bound).count() - (time_per_batch * FLAGS_alpha + 0.000001 * FLAGS_beta);
        if (batching_window <= 0.0) {
            LOG(INFO) << "Batch Sizes >= " << batch_size << " exceed latency threshold";
            break;
        }
        runners_by_batching_window[batching_window] = runner;
        max_batch_size = std::max(max_batch_size, batch_size);

        LOG(INFO) << runner->GetModel().Name() << " exec per batch:  " << result->at(kExecutionTimePerBatch);
        LOG(INFO) << runner->GetModel().Name() << " batching window: " << batching_window;
    }

    auto smallest_runner = runners_by_batch_size.upper_bound(0)->second;
    runners_by_batching_window[std::numeric_limits<double>::max()] = smallest_runner;

    struct WorkPacket
    {
        std::chrono::nanoseconds start;
        int query;
        std::function<void()> completion_callback;
        std::chrono::high_resolution_clock::time_point start_timestamp;
    };

    BlockingConcurrentQueue<WorkPacket> work_queue;

    size_t global_counter = 0;
    std::vector<std::shared_future<size_t>> futures;
    futures.reserve(1048576);

    batcher_thread_pool->enqueue([&work_queue, &runners_by_batch_size, &runners_by_batching_window, &futures, &running,
                                  resources, max_batch_size, timeout, &death_thread_pool, &latency_bound,
                                  &time_per_batch_by_runner, gamma = FLAGS_gamma]() mutable {
        size_t total_count;
        size_t max_deque;
        size_t adjustable_max_batch_size;
        std::vector<WorkPacket> work_packets(max_batch_size);
        thread_local ConsumerToken token(work_queue);
        double elapsed;
        double quanta_in_secs = quanta / 1000000.0; // convert microsecs to 
        const double latency_budget = std::chrono::duration<double>(latency_bound).count();

        std::deque<std::pair<size_t, double>> load_tracker;

        auto elapsed_time = [](std::chrono::high_resolution_clock::time_point start) -> double {
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        };

        // Batching Loop
        for (;likely(running);)
        {
            total_count = 0;
            max_deque = max_batch_size;
            adjustable_max_batch_size = max_batch_size;
            auto start = std::chrono::high_resolution_clock::now();

            // if we have a work packet, we stay in this loop and continue to batch until the timeout is reached
            do
            {
                auto count = work_queue.wait_dequeue_bulk_timed(token, &work_packets[total_count], max_deque, quanta);
                total_count += count;
                elapsed = elapsed_time(start);
                auto runner = runners_by_batching_window.lower_bound(elapsed + quanta_in_secs)->second;
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
                auto time_per_batch = time_per_batch_by_runner.at(runner);
                //LOG_FIRST_N(INFO, 1000) << "tpb: " << time_per_batch << "; bs: " << total_count;
                //LOG_FIRST_N(INFO, 1000) << "elapsed: " << elapsed_time(work_packets[0].start_timestamp);
                const unsigned long tpb_us = static_cast<unsigned long>( time_per_batch * gamma * 1000000 );
                auto deadline = (work_packets[0].start_timestamp + latency_bound) - std::chrono::microseconds(tpb_us);
                auto time_to_deadline = std::chrono::duration<double>(deadline - std::chrono::high_resolution_clock::now()).count();
                // LOG_FIRST_N(INFO, 1000) << "time to deadline: " << time_to_deadline;
                auto bindings = buffers->CreateBindings(runner->GetModelSmartPtr());
                bindings->SetBatchSize(total_count);
                auto shared_wps = std::make_shared<std::vector<WorkPacket>>(std::move(work_packets));
                futures.push_back(
                    runner->InferWithDeadline(
                        bindings, 
                        [wps = shared_wps](std::shared_ptr<Bindings>& bindings) -> size_t {
                            for(const auto& wp : *wps) { wp.completion_callback(); }
                            bindings.reset();
                            return wps->size();
                        },
                        deadline,
                        [wps = shared_wps, &death_thread_pool](std::function<void()> completer) {
                            // this function is called if the deadline check fails
                            death_thread_pool->enqueue([wps, completer]() {
                                // LOG_FIRST_N(INFO, 1000) << "skipping workpacket of size: " << wps->size();
                                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                                for(const auto& wp : *wps) { wp.completion_callback(); }
                                completer();
                            });
                        }
                    )
                );
                //load_tracker.emplace_back(total_count, )
                // reset work_packets
                CHECK_EQ(work_packets.size(), 0);
                work_packets.resize(max_batch_size);
            }
        }
    });

    auto sync = [&futures, &global_counter]() mutable {
        LOG(INFO) << "Syncing futures.";
        std::map<size_t, size_t> histogram;
        size_t count = 0;
        size_t skipped_count = 0;
        auto iterator = futures.begin();
        while (count < global_counter)
        {
            auto batch_size = iterator->get();
            if(batch_size < 1000)
            {
                count += batch_size;
                histogram[batch_size]++;
            }
            else
            {
                auto bs = batch_size/1000;
                count += bs;
                skipped_count += bs;
            }
            if (count == global_counter) { break; }
            while (iterator+1 == futures.end()) { std::this_thread::sleep_for(std::chrono::microseconds(200)); }
            iterator++;
        }
        LOG(INFO) << "Histogram of batched work:";
        for (const auto& item : histogram )
        {
            LOG(INFO) << "bs " << item.first << ": " << item.second;
        }
        auto batches = futures.size();
        global_counter = 0;
        futures.clear();
        CHECK(futures.size() == 0);
        LOG(INFO) << "Sync Complete - " << batches << " batches; " << count << " inferences (" << skipped_count << " skipped)";
    };

    // The enqueue function takes a start time, a query, and a
    // completion_callback. After infering the query, the completion_callback must
    // be invoked. The enqueue function can execute the inference locally, but the
    // most sensible implementation puts the query and the completion callback in
    // a queue to be executed by another thread. The start time can be ignored or
    // used by some policy mechanism to help determine on batch size.

    // Immmediately enqueue work packet on the TensorRT batching work_queue
    TraceGenerator::EnqueueFn<int> enqueue =
        [&work_queue, &global_counter](std::chrono::nanoseconds start, int query, std::function<void(void)> completion_callback) {
            work_queue.enqueue(WorkPacket{start, query, completion_callback, std::chrono::high_resolution_clock::now()});
            ++global_counter;
        };

    // TODO(tjablin): These parameters should all be read from the command-line or
    // maybe a configuration file?

    // TODO(tjablin): The latency bound of 400 us is unphysically low.

    // The minimum number of queries.
    // uint64_t min_queries = 4096;
    uint64_t min_queries = 4096;

    // The minimum duration of the trace.
    std::chrono::seconds min_duration(FLAGS_duration);

    // The minimum percent of queries meeting the latency bound.
    double latency_bound_percentile = FLAGS_percentile * 0.01;

    // The pseudo-random number generator's seed.
    uint64_t seed = FLAGS_seed;

    // Given an enqueue implementation and all other parameters, conduct a series
    // of experiments to determine the maximum QPS.
    double max_qps =
        TraceGenerator::FindMaxQPS(query_library, enqueue, sync, seed, latency_bound, min_queries,
                                   min_duration, latency_bound_percentile);

    // Print the results.
    std::cout << "Max QPS subject to " << latency_bound.count() << " us "
              << std::roundl(100 * latency_bound_percentile) << "% latency bound: " << max_qps
              << "\n";

    std::this_thread::sleep_for(std::chrono::seconds(5));
    running = false;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return 0;
}
