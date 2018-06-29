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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>

#include "YAIS/YAIS.h"
#include "YAIS/TensorRT.h"

#ifdef YAIS_USE_MPI
#include "mpi.h"
#define MPI_CHECK(mpicall) mpicall
#else
#define MPI_CHECK(mpicall)
#endif

using yais::ThreadPool;
using yais::TensorRT::Runtime;
using yais::TensorRT::ManagedRuntime;
using yais::TensorRT::Model;
using yais::TensorRT::Resources;

using ResourcesTensorRT = yais::TensorRT::Resources;


class BenchmarkResources : public Resources
{
  public:
    BenchmarkResources(int max_executions, int max_buffers, int nCuda, int nResp)
        : Resources(max_executions, max_buffers), 
          m_CudaThreadPool(std::make_unique<ThreadPool>(nCuda)),
          m_ResponseThreadPool(std::make_unique<ThreadPool>(nResp)) {}

    ~BenchmarkResources() override {}

    std::unique_ptr<ThreadPool>& GetCudaThreadPool() { return m_CudaThreadPool; }
    std::unique_ptr<ThreadPool>& GetResponseThreadPool() { return m_ResponseThreadPool; }

  private:
    std::unique_ptr<ThreadPool> m_CudaThreadPool;
    std::unique_ptr<ThreadPool> m_ResponseThreadPool;
};


class Benchmark final
{
  public:
    Benchmark(std::shared_ptr<BenchmarkResources> resources) : m_Resources(resources) {}

    void Run(float seconds, bool warmup)
    {
        auto name = "default";
        auto model = GetResources()->GetModel(name);
        auto batch_size = model->GetMaxBatchSize();
        // auto exe = model->GetExecutionContext();

        auto start = std::chrono::system_clock::now();
        auto elapsed = [start]()->float
        {
            return std::chrono::duration<float>(std::chrono::system_clock::now() - start).count();
        };

        uint64_t inf_count = 0;

        // Benchmark Loop - Main thread copies, cuda thread launches, response thread completes
        if(!warmup) LOG(INFO) << "Benchmark: Running for ~" << (int)seconds << " seconds with batch_size " << batch_size;
        while(elapsed() < seconds && ++inf_count) {
            // This thread only async copies buffers from H2D
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            buffers->Configure(model.get(), batch_size);
            for (auto binding_id : model->GetInputBindingIds()) {
              buffers->AsyncH2D(binding_id);
            }

            auto bytes = model->GetDeviceMemorySize();
            buffers->PushDeviceStack(128*1024); // reserve a cacheline between input/output tensors and scratch space
            GetResources()->GetCudaThreadPool()->enqueue([this, buffers, batch_size, name, bytes]() mutable{
                // This thread enqueues two async kernels: TensorRT execution and D2H of output tensors
                auto ctx = GetResources()->GetExecutionContext(name); // <=== Limited Resource; May Block !!!
                ctx->Enqueue(batch_size, buffers->GetDeviceBindings(), buffers->PushDeviceStack(bytes), buffers->GetStream());
                for (auto binding_id : GetResources()->GetModel(name)->GetOutputBindingIds()) {
                    buffers->AsyncD2H(binding_id);
                }
                GetResources()->GetResponseThreadPool()->enqueue([this, buffers, ctx]() mutable {
                    // This thread waits on the completion of the async compute and the async copy
                    ctx->Synchronize(); ctx.reset(); // Finished with the Execution Context - Release it to competing threads
                    buffers->SynchronizeStream(); buffers.reset(); // Finished with Buffers - Release it to competing threads
                });
            });
        }

        // Join worker threads
        if(!warmup) GetResources()->GetCudaThreadPool().reset();
        if(!warmup) GetResources()->GetResponseThreadPool().reset();
        auto total_time = std::chrono::duration<float>(elapsed()).count();
        auto inferences = inf_count * batch_size;
        if(!warmup) LOG(INFO) << "Benchmark Results: " << inf_count << " batches in " << total_time << " seconds; "
                              << "sec/batch: " << total_time / inf_count << "; inf/sec: " << inferences / total_time; 
    }

  protected:
    inline std::shared_ptr<BenchmarkResources> GetResources() { return m_Resources; } 

  private:
    std::shared_ptr<BenchmarkResources> m_Resources;
};

static bool ValidateEngine (const char* flagname, const std::string& value) {
  struct stat buffer;
  return (stat (value.c_str(), &buffer) == 0);
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_int32(seconds, 5, "Approximate number of seconds for the timing loop");
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers,  0, "Number of Buffers (default: 2x contexts)");
DEFINE_int32(cudathreads, 1, "Number Cuda Launcher Threads");
DEFINE_int32(respthreads, 1, "Number Response Sync Threads");

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT Benchmark");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    MPI_CHECK(MPI_Init(&argc, &argv));

    auto contexts   = FLAGS_contexts;
    auto buffers    = FLAGS_buffers ? FLAGS_buffers : 2*FLAGS_contexts;

    auto resources = std::make_shared<BenchmarkResources>(
        contexts,
        buffers,
        FLAGS_cudathreads,
        FLAGS_respthreads
    );

    resources->RegisterModel("default", Runtime::DeserializeEngine(FLAGS_engine));
    resources->AllocateResources();

    Benchmark benchmark(resources);
    benchmark.Run(0.1, true); // warmup

    // if testing mps - sync all processes before executing timed loop
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    benchmark.Run(FLAGS_seconds, false);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // todo: perform an mpi_allreduce to collect the per process timings
    //       for a simplified report
    MPI_CHECK(MPI_Finalize());
    return 0;
}
