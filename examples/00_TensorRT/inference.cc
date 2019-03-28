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
#include <sys/stat.h>
#include <unistd.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "trtlab/core/thread_pool.h"
#include "trtlab/tensorrt/inference_manager.h"
#include "trtlab/tensorrt/runtime.h"

#ifdef PLAYGROUND_USE_MPI
#include "mpi.h"
#define MPI_CHECK(mpicall) mpicall
#else
#define MPI_CHECK(mpicall)
#endif

using trtlab::ThreadPool;
using trtlab::TensorRT::CustomRuntime;
using trtlab::TensorRT::InferenceManager;
using trtlab::TensorRT::ManagedAllocator;
using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::StandardAllocator;

static int g_Concurrency = 0;

static std::string ModelName(int model_id)
{
    std::ostringstream stream;
    stream << model_id;
    return stream.str();
}

class InferenceResources : public InferenceManager
{
  public:
    InferenceResources(int max_executions, int max_buffers, size_t nCuda, size_t nResp)
        : InferenceManager(max_executions, max_buffers),
          m_CudaThreadPool(std::make_unique<ThreadPool>(nCuda)),
          m_ResponseThreadPool(std::make_unique<ThreadPool>(nResp))
    {
    }

    ~InferenceResources() override {}

    std::unique_ptr<ThreadPool>& GetCudaThreadPool() { return m_CudaThreadPool; }
    std::unique_ptr<ThreadPool>& GetResponseThreadPool() { return m_ResponseThreadPool; }

  private:
    std::unique_ptr<ThreadPool> m_CudaThreadPool;
    std::unique_ptr<ThreadPool> m_ResponseThreadPool;
};

class Inference final
{
  public:
    Inference(std::shared_ptr<InferenceResources> resources) : m_Resources(resources) {}

    void Run(float seconds, bool warmup, int replicas, uint32_t requested_batch_size)
    {
        int replica = 0;
        uint64_t inf_count = 0;

        auto start = std::chrono::steady_clock::now();
        auto elapsed = [start]() -> float {
            return std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        };

        auto model = GetResources()->GetModel(ModelName(replica++));
        auto batch_size = requested_batch_size ? requested_batch_size : model->GetMaxBatchSize();
        if(batch_size > model->GetMaxBatchSize())
        {
            LOG(FATAL)
                << "Requested batch_size greater than allowed by the compiled TensorRT Engine";
        }

        // Inference Loop - Main thread copies, cuda thread launches, response thread completes
        if(!warmup)
        {
            LOG(INFO) << "-- Inference: Running for ~" << (int)seconds
                      << " seconds with batch_size " << batch_size << " --";
        }

        std::vector<std::future<void>> futures;

        while(elapsed() < seconds && ++inf_count)
        {
            if(replica >= replicas) replica = 0;

            // This thread only async copies buffers H2D
            auto model = GetResources()->GetModel(ModelName(replica++));
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            auto bindings = buffers->CreateBindings(model);
            auto promise = std::make_shared<std::promise<void>>();
            futures.push_back(promise->get_future());

            bindings->SetBatchSize(batch_size);
            bindings->CopyToDevice(bindings->InputBindings());

            GetResources()->GetCudaThreadPool()->enqueue([this, bindings, promise]() mutable {
                // This thread enqueues two async kernels:
                //  1) TensorRT execution
                //  2) D2H of output tensors
                auto trt = GetResources()->GetExecutionContext(
                    bindings->GetModel()); // <=== Limited Resource; May Block !!!
                trt->Infer(bindings);
                bindings->CopyFromDevice(bindings->OutputBindings());

                GetResources()->GetResponseThreadPool()->enqueue(
                    [bindings, trt, promise]() mutable {
                        // This thread waits on the completion of the async compute and the async
                        // copy
                        trt->Synchronize();
                        trt.reset(); // Finished with the Execution Context - Release it to
                                     // competing threads
                        bindings->Synchronize();
                        bindings.reset(); // Finished with Buffers - Release it to competing threads
                        promise->set_value();
                    });
            });
        }

        for(const auto& f : futures)
        {
            f.wait();
        }

        /*
                // Join worker threads
                if (!warmup)
                    GetResources()->GetCudaThreadPool().reset();
                if (!warmup)
                    GetResources()->GetResponseThreadPool().reset();
        */
        // End timing and report
        auto total_time = std::chrono::duration<float>(elapsed()).count();
        auto inferences = inf_count * batch_size;
        if(!warmup)
            LOG(INFO) << "Inference Results: " << inf_count << "; batches in " << total_time
                      << " seconds"
                      << "; sec/batch/stream: " << total_time / (inf_count / g_Concurrency)
                      << "; batches/sec: " << inf_count / total_time
                      << "; inf/sec: " << inferences / total_time;
    }

  protected:
    inline std::shared_ptr<InferenceResources> GetResources() { return m_Resources; }

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

    auto contexts = g_Concurrency = FLAGS_contexts;
    auto buffers = FLAGS_buffers ? FLAGS_buffers : 2 * FLAGS_contexts;

    auto resources = std::make_shared<InferenceResources>(contexts, buffers, FLAGS_cudathreads,
                                                          FLAGS_respthreads);

    std::shared_ptr<Runtime> runtime;
    if(FLAGS_runtime == "default")
    {
        runtime = std::make_shared<CustomRuntime<StandardAllocator>>();
    }
    else if(FLAGS_runtime == "unified")
    {
        runtime = std::make_shared<CustomRuntime<ManagedAllocator>>();
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

    Inference inference(resources);
    inference.Run(0.1, true, 1, 0); // warmup

    // if testing mps - sync all processes before executing timed loop
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    inference.Run(FLAGS_seconds, false, FLAGS_replicas, FLAGS_batch_size);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // todo: perform an mpi_allreduce to collect the per process timings
    //       for a simplified report
    MPI_CHECK(MPI_Finalize());
    return 0;
}
