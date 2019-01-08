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
#include "tensorrt/playground/infer_runner.h"
#include "tensorrt/playground/inference_manager.h"
#include "tensorrt/playground/runtime.h"

#ifdef YAIS_USE_MPI
#include "mpi.h"
#define MPI_CHECK(mpicall) mpicall
#else
#define MPI_CHECK(mpicall)
#endif

using yais::ThreadPool;
using yais::TensorRT::Bindings;
using yais::TensorRT::InferenceManager;
using yais::TensorRT::InferRunner;
using yais::TensorRT::Runtime;
using yais::TensorRT::StandardRuntime;
using yais::TensorRT::ManagedRuntime;

static std::string ModelName(int model_id)
{
    std::ostringstream stream;
    stream << model_id;
    return stream.str();
}

class InferenceResources : public InferenceManager
{
  public:
    InferenceResources(int max_executions, int max_buffers)
        : InferenceManager(max_executions, max_buffers)
    {
        RegisterThreadPool("pre", std::make_unique<ThreadPool>(1));
        RegisterThreadPool("cuda", std::make_unique<ThreadPool>(1));
        RegisterThreadPool("post", std::make_unique<ThreadPool>(3));
    }

    ~InferenceResources() override {}
};

class Inference final
{
  public:
    Inference(std::shared_ptr<InferenceResources> resources) : m_Resources(resources) {}

    void Run(float seconds, bool warmup, int replicas, uint32_t requested_batch_size)
    {
        int replica = 0;
        uint64_t inf_count = 0;
        std::vector<std::shared_future<void>> futures;

        auto start = std::chrono::steady_clock::now();
        auto elapsed = [start]() -> float {
            return std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        };

        auto model = GetResources()->GetModel(ModelName(replica++));
        auto batch_size = requested_batch_size ? requested_batch_size : model->GetMaxBatchSize();
        LOG_IF(FATAL, batch_size > model->GetMaxBatchSize())
            << "Requested batch_size greater than allowed by the compiled TensorRT Engine";

        // Inference Loop - Main thread copies, cuda thread launches, response thread completes
        if(!warmup)
        {
            LOG(INFO) << "-- Inference: Running for ~" << (int)seconds
                      << " seconds with batch_size " << batch_size << " --";
        }

        while(elapsed() < seconds && ++inf_count)
        {
            if(replica >= replicas)
                replica = 0;

            // This thread only async copies buffers H2D
            auto model = GetResources()->GetModel(ModelName(replica++));
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            auto bindings = buffers->CreateBindings(model);
            bindings->SetBatchSize(batch_size);

            InferRunner runner(model, GetResources());
            futures.push_back(
                runner.Infer(bindings, [](std::shared_ptr<Bindings>& bindings) mutable {
                    bindings.reset();
                }));
        }

        // Join worker threads
        for(auto& f : futures)
        {
            f.wait();
        }

        // End timing and report
        auto total_time = std::chrono::duration<float>(elapsed()).count();
        auto inferences = inf_count * batch_size;
        if(!warmup)
            LOG(INFO) << "Inference Results: " << inf_count << "; batches in " << total_time
                      << " seconds"
                      << "; sec/batch/stream: " << total_time / (inf_count / m_Resources->MaxExecConcurrency())
                      << "; batches/sec: " << inf_count / total_time
                      << "; inf/sec: " << inferences / total_time;
    }

  protected:
    inline std::shared_ptr<InferenceResources> GetResources()
    {
        return m_Resources;
    }

  private:
    std::shared_ptr<InferenceResources> m_Resources;
};

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

    auto resources = std::make_shared<InferenceResources>(contexts, buffers);

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

    resources->RegisterModel("0", runtime->DeserializeEngine(FLAGS_engine));
    resources->AllocateResources();

    for(int i = 1; i < FLAGS_replicas; i++)
    {
        resources->RegisterModel(ModelName(i), runtime->DeserializeEngine(FLAGS_engine));
    }

    {
        Inference inference(resources);
        inference.Run(0.1, true, 1, 0); // warmup

        // if testing mps - sync all processes before executing timed loop
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        inference.Run(FLAGS_seconds, false, FLAGS_replicas, FLAGS_batch_size);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        // todo: perform an mpi_allreduce to collect the per process timings
        //       for a simplified report
        MPI_CHECK(MPI_Finalize());
    }

    return 0;
}
