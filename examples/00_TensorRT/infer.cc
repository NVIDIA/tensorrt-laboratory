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
#include <sys/stat.h>
#include <unistd.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tensorrt/playground/core/thread_pool.h"
#include "tensorrt/playground/infer_bench.h"
#include "tensorrt/playground/inference_manager.h"
#include "tensorrt/playground/model.h"
#include "tensorrt/playground/runtime.h"

#ifdef PLAYGROUND_USE_MPI
#include "mpi.h"
#define MPI_CHECK(mpicall) mpicall
#else
#define MPI_CHECK(mpicall)
#endif

using playground::ThreadPool;
using playground::TensorRT::InferBench;
using playground::TensorRT::InferenceManager;
using playground::TensorRT::ManagedRuntime;
using playground::TensorRT::Model;
using playground::TensorRT::Runtime;
using playground::TensorRT::StandardRuntime;

static std::string ModelName(int model_id)
{
    std::ostringstream stream;
    stream << model_id;
    return stream.str();
}

static bool ValidateEngine(const char* flagname, const std::string& value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_string(runtime, "default", "TensorRT Runtime");
DEFINE_int32(seconds, 5, "Approximate number of seconds for the timing loop");
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers, 0, "Number of Buffers (default: 2x contexts)");
DEFINE_int32(cudathreads, 1, "Number Cuda Launcher Threads");
DEFINE_int32(respthreads, 1, "Number Response Sync Threads");
DEFINE_int32(replicas, 1, "Number of Replicas of the Model to load");
DEFINE_int32(batch_size, 0, "Overrides the max batch_size of the provided engine");

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT Inference");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    MPI_CHECK(MPI_Init(&argc, &argv));

    auto contexts = FLAGS_contexts;
    auto buffers = FLAGS_buffers ? FLAGS_buffers : 2 * FLAGS_contexts;

    auto resources = std::make_shared<InferenceManager>(contexts, buffers);
    resources->RegisterThreadPool("pre", std::make_unique<ThreadPool>(1));
    resources->RegisterThreadPool("cuda", std::make_unique<ThreadPool>(1));
    resources->RegisterThreadPool("post", std::make_unique<ThreadPool>(3));

    //, FLAGS_cudathreads, FLAGS_respthreads);
    std::shared_ptr<Runtime> runtime;
    if(FLAGS_runtime == "default")
    {
        runtime = std::make_shared<StandardRuntime>();
    }
    else if(FLAGS_runtime == "unified")
    {
        runtime = std::make_shared<ManagedRuntime>();
    }
    else
    {
        LOG(FATAL) << "Invalid TensorRT Runtime";
    }

    std::vector<std::shared_ptr<Model>> models;

    models.push_back(runtime->DeserializeEngine(FLAGS_engine));
    resources->RegisterModel("0", models.back());
    resources->AllocateResources();

    auto batch_size = FLAGS_batch_size ? FLAGS_batch_size : models.back()->GetMaxBatchSize();

    for(int i = 1; i < FLAGS_replicas; i++)
    {
        models.push_back(runtime->DeserializeEngine(FLAGS_engine));
        resources->RegisterModel(ModelName(i), models.back());
    }

    {
        InferBench benchmark(resources);
        benchmark.Run(models, batch_size, 0.1);

        // if testing mps - sync all processes before executing timed loop
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        auto results = benchmark.Run(models, batch_size, FLAGS_seconds);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        // todo: perform an mpi_allreduce to collect the per process timings
        //       for a simplified report
        MPI_CHECK(MPI_Finalize());

        LOG(INFO) << "Inference Results: " << results["batch_count"] << " batches in "
                  << results["total_time"] << " seconds; batch_size: " << results["batch_size"]
                  << "; inf/sec: " << results["inferences-per-sec"];
        LOG(INFO) << "; sec/batch/stream: " << results["secs-per-batch-per-stream"]
                  << "; batches/sec: " << results["secs-per-batch"];
    }

    return 0;
}
