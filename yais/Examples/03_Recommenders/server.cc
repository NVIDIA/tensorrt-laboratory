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
#include <ostream>
#include <string>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>

#include "YAIS/YAIS.h"
#include "YAIS/TensorRT.h"

using yais::AsyncRPC;
using yais::AsyncService;
using yais::Context;
using yais::Executor;
using yais::Server;
using yais::ThreadPool;
using yais::TensorRT::Runtime;
using yais::TensorRT::Model;

using ResourcesTensorRT = yais::TensorRT::Resources;

// Flowers Protos
#include "api.pb.h"
#include "api.grpc.pb.h"

using DeepRecommender::BatchInput;
using DeepRecommender::BatchPredictions;

class RecommenderResources : public ResourcesTensorRT
{
  public:
    explicit RecommenderResources(std::shared_ptr<Model> engine, int nBuffers, int nExeCtxs, int nCuda, int nResp)
        : ResourcesTensorRT(engine, nBuffers, nExeCtxs),
          m_CudaThreadPool(nCuda),
          m_ResponseThreadPool(nResp) {}

    ThreadPool &GetCudaThreadPool() { return m_CudaThreadPool; }
    ThreadPool &GetResponseThreadPool() { return m_ResponseThreadPool; }

  private:
    ThreadPool m_CudaThreadPool;
    ThreadPool m_ResponseThreadPool;
};

class RecommenderContext final : public Context<BatchInput, BatchPredictions, RecommenderResources>
{
    void ExecuteRPC(RequestType_t &input, ResponseType_t &output) final override
    {
        // Executing on a Executor threads - we don't want to block message handling, so we offload
        GetResources()->GetCudaThreadPool().enqueue([this, &input, &output]() {
            // Executed on a thread from CudaThreadPool
            auto model = GetResources()->GetModel();
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            auto batch_size = input.user_preferences_size();
            buffers->Configure(model, batch_size);
            DecodeInputEncodings(input, model->GetBinding(0).elementsPerBatchItem, (float *)buffers->GetHostBinding(0));
            buffers->AsyncH2D(0);
            auto ctx = GetResources()->GetExeuctionContext(); // <=== Limited Resource; May Block !!!
            auto t_start = std::chrono::high_resolution_clock::now();
            ctx->Enqueue(batch_size, buffers->GetDeviceBindings(), buffers->GetStream());
            buffers->AsyncD2H(1); // GPU -> Host Buffers Output Binding
            // All Async CUDA work has been queued - this thread's work is done.
            GetResources()->GetResponseThreadPool().enqueue([this, &input, &output, buffers, ctx, t_start]() mutable {
                // Executed on a thread from ResponseThreadPool
                auto model = GetResources()->GetModel();
                ctx->Synchronize();
                ctx.reset();                  // Finished with the Execution Context - Release it to competing threads
                buffers->SynchronizeStream(); // Blocks on H2D, Compute, D2H Pipeline
                EncodeRecommendations(input, output, model->GetBinding(0).elementsPerBatchItem,
                                      (float *)buffers->GetHostBinding(0),
                                      (float *)buffers->GetHostBinding(1));
                buffers.reset(); // Finished with Buffers - Release it to competing threads
                auto t_end = std::chrono::high_resolution_clock::now();
                output.set_compute_time(std::chrono::duration<float>(t_end - t_start).count());
                LOG_EVERY_N(INFO, 200) << output.batch_id() << " " << output.compute_time();
                this->FinishResponse();
            });
        });
    }

    void DecodeInputEncodings(const RequestType_t &input, size_t n, float *in)
    {
        auto batch_size = input.user_preferences_size();
        std::memset(in, 0, n * batch_size);
        for (auto b = 0; b < batch_size; b++)
        {
            auto preferences = input.user_preferences(b);
            for (int i = 0; i < preferences.sparse_input_size(); i++)
            {
                auto u = preferences.sparse_input(i);
                in[n * b + u.offset()] = u.value();
            }
        }
    }

    void EncodeRecommendations(const RequestType_t &input, ResponseType_t &output, size_t n, float *in, float *out)
    {
        auto batch_size = input.user_preferences_size();
        for (auto b = 0; b < batch_size; b++)
        {
            auto r = output.add_top_n();
            auto top1 = r->add_sparse_output();
            float max = -1.0;
            size_t offset = b * n;
            for (uint32_t i = 0; i < n; i++)
            {
                if (in[offset + i] == 0.0 && out[offset + i] > max)
                {
                    max = out[offset + i];
                    top1->set_offset(i);
                    top1->set_value(max);
                }
            }
        }
        output.set_batch_id(input.batch_id());
    }
};

static bool ValidateEngine(const char *flagname, const std::string &value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_int32(port, 50051, "Port to listen on");
DEFINE_int32(nctx, 1, "Number of Execution Contexts");

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("flowers");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // A server will bind an IP:PORT to listen on
    std::ostringstream ip_port;
    ip_port << "0.0.0.0:" << FLAGS_port;
    Server server(ip_port.str());

    // A server can host multiple services
    LOG(INFO) << "Register Service (flowers::Inference) with Server";
    auto inferenceService = server.RegisterAsyncService<DeepRecommender::Inference>();

    // An RPC has two components that need to be specified when registering with the service:
    //  1) Type of Execution Context (FlowersContext).  The execution context defines the behavor
    //     of the RPC, i.e. it contains the control logic for the execution of the RPC.
    //  2) The Request function (RequestCompute) which was generated by gRPC when compiling the
    //     protobuf which defined the service.  This function is responsible for queuing the
    //     RPC's execution context to the
    LOG(INFO) << "Register RPC (flowers::Inference::Compute) with Service (flowers::Inference)";
    auto rpcCompute = inferenceService->RegisterRPC<RecommenderContext>(
        &DeepRecommender::Inference::AsyncService::RequestCompute);

    // Initialize Resources
    LOG(INFO) << "Initializing Resources for RPC (flowers::Inference::Compute)";
    auto rpcResources = std::make_shared<RecommenderResources>(
        Runtime::DeserializeEngine(FLAGS_engine),
        FLAGS_nctx * 2, // number of host/device buffers for input/output tensors
        FLAGS_nctx,     // number of IExecutionContexts - scratch space for DNN activations
        1,              // number of threads used to execute cuda kernel launches
        2               // number of threads used to write and complete responses
    );

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    LOG(INFO) << "Initializing Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    LOG(INFO) << "Registering Execution Contexts for RPC (flowers::Inference::Compute) with Executor";
    executor->RegisterContexts(rpcCompute, rpcResources, 10);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {
    });
}
