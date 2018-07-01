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
#include "YAIS/Affinity.h"

using yais::Affinity;
using yais::AsyncRPC;
using yais::AsyncService;
using yais::Context;
using yais::Executor;
using yais::Server;
using yais::ThreadPool;
using yais::TensorRT::ManagedRuntime;
using yais::TensorRT::Model;
using yais::TensorRT::Resources;
using yais::TensorRT::Runtime;

// Flowers Protos
#include "flowers.pb.h"
#include "flowers.grpc.pb.h"

// Dataset Protos
float *GetSharedMemory(const std::string &address);

class FlowersResources : public Resources
{
  public:
    explicit FlowersResources(int max_executions, int max_buffers, int nCuda, int nResp, float *sysv_data)
        : Resources(max_executions, max_buffers),
          m_CudaThreadPool(nCuda),
          m_ResponseThreadPool(nResp),
          m_SharedMemory(sysv_data) {}

    ThreadPool &GetCudaThreadPool() { return m_CudaThreadPool; }
    ThreadPool &GetResponseThreadPool() { return m_ResponseThreadPool; }

    float *GetSysvOffset(size_t offset_in_bytes) { return &m_SharedMemory[offset_in_bytes / sizeof(float)]; }

  private:
    ThreadPool m_CudaThreadPool;
    ThreadPool m_ResponseThreadPool;
    float *m_SharedMemory;
};

class FlowersContext final : public Context<ssd::BatchInput, ssd::BatchPredictions, FlowersResources>
{
    void ExecuteRPC(RequestType_t &input, ResponseType_t &output) final override
    {
        // Executing on a Executor threads - we don't want to block message handling, so we offload
        GetResources()->GetCudaThreadPool().enqueue([this, &input, &output]() {
            // Executed on a thread from CudaThreadPool
            auto model = GetResources()->GetModel("flowers");
            auto buffers = GetResources()->GetBuffers();            // <=== Limited Resource; May Block !!!
            auto bindings = buffers->CreateAndConfigureBindings(model, input.batch_size());
            bindings->SetHostAddress(0, GetResources()->GetSysvOffset(input.sysv_offset()));
            bindings->CopyToDevice(bindings->InputBindings());
            auto ctx = GetResources()->GetExecutionContext(model);  // <=== Limited Resource; May Block !!!
            auto t_start = std::chrono::high_resolution_clock::now();
            ctx->Infer(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            // All Async CUDA work has been queued - this thread's work is done.
            GetResources()->GetResponseThreadPool().enqueue([this, &input, &output, bindings, ctx, t_start]() mutable {
                // Executed on a thread from ResponseThreadPool
                ctx->Synchronize(); ctx.reset();    // Finished with the Execution Context - Release it to competing threads
                bindings->Synchronize();            // Blocks on H2D, Compute, D2H Pipeline
                WriteBatchPredictions(input, output, (float *)bindings->HostAddress(1));
                bindings.reset();                   // Finished with Buffers - Release it to competing threads
                auto t_end = std::chrono::high_resolution_clock::now();
                output.set_compute_time(std::chrono::duration<float>(t_end - t_start).count());
                output.set_total_time(std::chrono::duration<float>(t_end - t_start).count());
                LOG_EVERY_N(INFO, 200) << output.batch_id() << " " << output.compute_time();
                this->FinishResponse();
            });
        });
    }

    void WriteBatchPredictions(RequestType_t &input, ResponseType_t &output, float *scores)
    {
        int N = input.batch_size();
        auto nClasses = GetResources()->GetModel("flowers")->GetBinding(1).elementsPerBatchItem;
        size_t cnt = 0;
        for (int p = 0; p < N; p++)
        {
            auto element = output.add_elements();
            /*
            float max_val = -1.0;
            int max_idx = -1;
            for (int i=0; i < nClasses; i++) {
                if(max_val < scores[cnt]) {
                    max_val = scores[cnt];
                    max_idx = i;
                }
                cnt++;
            }
	    auto top1 = element->add_predictions();
            top1->set_class_id(max_idx);
            top1->set_score(max_val);
*/
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
DEFINE_string(dataset, "127.0.0.1:4444", "GRPC Dataset/SharedMemory Service Address");
DEFINE_int32(nctx, 1, "Number of Execution Contexts");
DEFINE_int32(port, 50051, "Port to listen on");

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("flowers");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // set affinity to device
    auto cpus = Affinity::GetDeviceAffinity(0);
    Affinity::SetAffinity(cpus);

    // A server will bind an IP:PORT to listen on
    std::ostringstream ip_port;
    ip_port << "0.0.0.0:" << FLAGS_port;
    Server server(ip_port.str());

    // A server can host multiple services
    LOG(INFO) << "Register Service (flowers::Inference) with Server";
    auto inferenceService = server.RegisterAsyncService<ssd::Inference>();

    // An RPC has two components that need to be specified when registering with the service:
    //  1) Type of Execution Context (FlowersContext).  The execution context defines the behavor
    //     of the RPC, i.e. it contains the control logic for the execution of the RPC.
    //  2) The Request function (RequestCompute) which was generated by gRPC when compiling the
    //     protobuf which defined the service.  This function is responsible for queuing the
    //     RPC's execution context to the
    LOG(INFO) << "Register RPC (flowers::Inference::Compute) with Service (flowers::Inference)";
    auto rpcCompute = inferenceService->RegisterRPC<FlowersContext>(
        &ssd::Inference::AsyncService::RequestCompute);

    // Initialize Resources
    LOG(INFO) << "Initializing Resources for RPC (flowers::Inference::Compute)";
    auto rpcResources = std::make_shared<FlowersResources>(
        FLAGS_nctx,                    // number of IExecutionContexts - scratch space for DNN activations
        FLAGS_nctx * 2,                // number of host/device buffers for input/output tensors
        1,                             // number of threads used to execute cuda kernel launches
        2,                             // number of threads used to write and complete responses
        GetSharedMemory(FLAGS_dataset) // pointer to data in shared memory
    );
    rpcResources->RegisterModel("flowers", Runtime::DeserializeEngine(FLAGS_engine));
    rpcResources->AllocateResources();

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    LOG(INFO) << "Initializing Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    LOG(INFO) << "Registering Execution Contexts for RPC (flowers::Inference::Compute) with Executor";
    executor->RegisterContexts(rpcCompute, rpcResources, 100);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {
    });
}

static auto pinned_memory = yais::CudaHostAllocator::make_unique(1024 * 1024 * 1024);

float *GetSharedMemory(const std::string &address)
{
    /* data in shared memory should go here - for the sake of quick examples just use and emptry array */
    pinned_memory->WriteZeros();
    return (float *)pinned_memory->Data();
    // the following code connects to a shared memory service to allow for non-serialized transfers
    // between microservices
    /*
    InfoRequest request;
    Info reply;
    grpc::ClientContext context;
    auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    auto stub = SharedMemoryDataSet::NewStub(channel);
    auto status = stub->GetInfo(&context, request, &reply);
    CHECK(status.ok()) << "Dataset shared memory request failed";
    DLOG(INFO) << "SysV ShmKey: " << reply.sysv_key();
    int shmid = shmget(reply.sysv_key(), 0, 0);
    DLOG(INFO) << "SysV ShmID: " << shmid;
    float* data = (float*) shmat(shmid, 0, 0);
    CHECK(data) << "SysV Attached failed";
    return data;
    */
}
