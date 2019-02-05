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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#include "nvml.h"

#include "tensorrt/laboratory/core/affinity.h"
#include "tensorrt/laboratory/core/memory/allocator.h"
#include "tensorrt/laboratory/cuda/device_info.h"
#include "tensorrt/laboratory/cuda/memory/cuda_pinned_host.h"
#include "tensorrt/laboratory/runtime.h"
#include "tensorrt/laboratory/inference_manager.h"

#include "nvrpc/context.h"
#include "nvrpc/executor.h"
#include "nvrpc/service.h"
#include "nvrpc/server.h"

#include "metrics.h"

using trtlab::Affinity;
using nvrpc::AsyncRPC;
using nvrpc::AsyncService;
using nvrpc::Context;
using trtlab::DeviceInfo;
using nvrpc::Executor;
using trtlab::Metrics;
using nvrpc::Server;
using trtlab::ThreadPool;
using trtlab::Allocator;
using trtlab::CudaPinnedHostMemory;
using trtlab::TensorRT::Model;
using trtlab::TensorRT::InferenceManager;
using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::StandardRuntime;
using trtlab::TensorRT::ManagedRuntime;

// Flowers Protos
#include "inference.pb.h"
#include "inference.grpc.pb.h"

using ssd::BatchInput;
using ssd::BatchPredictions;
using ssd::Inference;

/*
 * Prometheus Metrics
 * 
 * It is important to make collect measurements to find bottlenecks, performance issues,
 * and to trigger auto-scaling.
 */
static auto &registry = Metrics::GetRegistry();

// Summaries - Request and Compute duration on a per service basis
static auto &inf_compute = prometheus::BuildSummary()
                                .Name("yais_inference_compute_duration_ms")
                                .Register(registry);
static auto &inf_request = prometheus::BuildSummary()
                                .Name("yais_inference_request_duration_ms")
                                .Register(registry);
static const auto &quantiles = prometheus::Summary::Quantiles{{0.5, 0.05}, {0.90, 0.01}, {0.99, 0.001}};

// Histogram - Load Ratio = Request/Compute duration - should just above one for a service
//             that can keep up with its current load.  This metrics provides more 
//             detailed information on the impact of the queue depth because it accounts
//             for request time.
static const std::vector<double> buckets = {1.25, 1.50, 2.0, 10.0, 100.0}; // unitless
static auto &inf_load_ratio_fam = prometheus::BuildHistogram()
                                .Name("yais_inference_load_ratio")
                                .Register(registry);
static auto &inf_load_ratio = inf_load_ratio_fam.Add({}, buckets);

// Gauge - Periodically measure and report GPU power utilization.  As the load increases
//         on the service, the power should increase proprotionally, until the power is capped
//         either by device limits or compute resources.  At this level, the inf_load_ratio
//         will begin to increase under futher increases in traffic
static auto &power_gauge_fam = prometheus::BuildGauge()
                                .Name("yais_gpus_power_usage")
                                .Register(registry);
static auto &power_gauge = power_gauge_fam.Add({{"gpu", "0"}});

/*
 * External Data Source
 * 
 * Attaches to a System V shared memory segment owned by an external resources.
 * Example: the results of an image decode service could use this mechanism to transfer
 *          large tensors to an inference service by simply passing an offset.
 */
float *GetSharedMemory(const std::string &address);

/*
 * YAIS Resources - TensorRT InferenceManager + ThreadPools + External Datasource
 */
class FlowersResources : public InferenceManager
{
  public:
    explicit FlowersResources(int max_executions, int max_buffers, int nCuda, int nResp, float *sysv_data)
        : InferenceManager(max_executions, max_buffers),
          m_CudaThreadPool(nCuda),
          m_ResponseThreadPool(nResp),
          m_SharedMemory(sysv_data)
    {
    }

    ThreadPool &GetCudaThreadPool() { return m_CudaThreadPool; }
    ThreadPool &GetResponseThreadPool() { return m_ResponseThreadPool; }

    float *GetSysvOffset(size_t offset_in_bytes) { return &m_SharedMemory[offset_in_bytes / sizeof(float)]; }

  private:
    ThreadPool m_CudaThreadPool;
    ThreadPool m_ResponseThreadPool;
    float *m_SharedMemory;
};

/*
 * nvRPC Context - Defines the logic of the RPC. 
 */
class FlowersContext final : public Context<BatchInput, BatchPredictions, FlowersResources>
{
    void ExecuteRPC(RequestType &input, ResponseType &output) final override
    {
        // Executing on a Executor threads - we don't want to block message handling, so we offload
        GetResources()->GetCudaThreadPool().enqueue([this, &input, &output]() {
            // Executed on a thread from CudaThreadPool
            auto model = GetResources()->GetModel("flowers");
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            auto bindings = buffers->CreateBindings(model);
            bindings->SetBatchSize(input.batch_size());
            bindings->SetHostAddress(0, GetResources()->GetSysvOffset(input.sysv_offset()));
            bindings->CopyToDevice(bindings->InputBindings());
            auto ctx = GetResources()->GetExecutionContext(model); // <=== Limited Resource; May Block !!!
            ctx->Infer(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            // All Async CUDA work has been queued - this thread's work is done.
            GetResources()->GetResponseThreadPool().enqueue([this, &input, &output, model, bindings, ctx]() mutable {
                // Executed on a thread from ResponseThreadPool
                auto compute_time = ctx->Synchronize(); 
                ctx.reset(); // Finished with the Execution Context - Release it to competing threads
                bindings->Synchronize(); // Blocks on H2D, Compute, D2H Pipeline
                WriteBatchPredictions(input, output, (float *)bindings->HostAddress(1));
                bindings.reset(); // Finished with Buffers - Release it to competing threads
                auto request_time = Walltime();
                output.set_compute_time(static_cast<float>(compute_time));
                output.set_total_time(static_cast<float>(request_time));
                this->FinishResponse();
                // The Response is now sending; Record some metrics and be done
                inf_compute.Add({{"model", model->Name()}}, quantiles).Observe(compute_time * 1000);
                inf_request.Add({{"model", model->Name()}}, quantiles).Observe(request_time * 1000);
                inf_load_ratio.Observe(request_time / compute_time);
            });
        });
    }

    void WriteBatchPredictions(RequestType &input, ResponseType &output, float *scores)
    {
        int N = input.batch_size();
        auto nClasses = GetResources()->GetModel("flowers")->GetBinding(1).elementsPerBatchItem;
        size_t cntr = 0;
        for (int p = 0; p < N; p++)
        {
            auto element = output.add_elements();
            /* Customize the post-processing of the output tensor *\
            float max_val = -1.0;
            int max_idx = -1;
            for (int i = 0; i < nClasses; i++)
            {
                if (max_val < scores[cntr])
                {
                    max_val = scores[cntr];
                    max_idx = i;
                }
                cntr++;
            }
            auto top1 = element->add_predictions();
            top1->set_class_id(max_idx);
            top1->set_score(max_val);
            \* Customize the post-processing of the output tensor */
        }
        output.set_batch_id(input.batch_id());
    }
};

static bool ValidateEngine(const char *flagname, const std::string &value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

static bool ValidateBytes(const char *flagname, const std::string &value)
{
    trtlab::StringToBytes(value);
    return true;
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_string(dataset, "127.0.0.1:4444", "GRPC Dataset/SharedMemory Service Address");
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers,  0, "Number of Input/Output Buffers");
DEFINE_string(runtime, "default", "TensorRT Runtime");
DEFINE_int32(execution_threads, 1, "Number of RPC execution threads");
DEFINE_int32(preprocessing_threads, 0, "Number of preprocessing threads");
DEFINE_int32(kernel_launching_threads, 1, "Number of threads to launch CUDA kernels");
DEFINE_int32(postprocessing_threads, 2, "Number of postprocessing threads");
DEFINE_string(max_recv_bytes, "10MiB", "Maximum number of bytes for incoming messages");
DEFINE_validator(max_recv_bytes, &ValidateBytes);
DEFINE_int32(port, 50051, "Port to listen for gRPC requests");
DEFINE_int32(metrics, 50078, "Port to expose metrics for scraping");

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("flowers");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // Set CPU Affinity to be near the GPU
    auto cpus = DeviceInfo::Affinity(0);
    Affinity::SetAffinity(cpus);

    // Enable metrics on port
    Metrics::Initialize(FLAGS_metrics);

    // Create a gRPC server bound to IP:PORT
    std::ostringstream ip_port;
    ip_port << "0.0.0.0:" << FLAGS_port;
    Server server(ip_port.str());

    // Modify MaxReceiveMessageSize
    auto bytes = trtlab::StringToBytes(FLAGS_max_recv_bytes);
    server.Builder().SetMaxReceiveMessageSize(bytes);
    LOG(INFO) << "gRPC MaxReceiveMessageSize = " << trtlab::BytesToString(bytes);

    // A server can host multiple services
    LOG(INFO) << "Register Service (flowers::Inference) with Server";
    auto inferenceService = server.RegisterAsyncService<Inference>();

    // An RPC has two components that need to be specified when registering with the service:
    //  1) Type of Execution Context (FlowersContext).  The execution context defines the behavor
    //     of the RPC, i.e. it contains the control logic for the execution of the RPC.
    //  2) The Request function (RequestCompute) which was generated by gRPC when compiling the
    //     protobuf which defined the service.  This function is responsible for queuing the
    //     RPC's execution context to the
    LOG(INFO) << "Register RPC (flowers::Inference::Compute) with Service (flowers::Inference)";
    auto rpcCompute = inferenceService->RegisterRPC<FlowersContext>(
        &Inference::AsyncService::RequestCompute);

    // Buffers default to execution contexts + 2
    // Allows for 1 H2D, N TensorRT Executions, 1 D2H to be inflight
    auto buffers = FLAGS_buffers;
    if (buffers == 0)
        buffers = FLAGS_contexts + 2;

    // Initialize Resources
    LOG(INFO) << "Initializing Resources for RPC (flowers::Inference::Compute)";
    auto rpcResources = std::make_shared<FlowersResources>(
        FLAGS_contexts,                 // number of IExecutionContexts - scratch space for DNN activations
        buffers,                        // number of host/device buffers for input/output tensors
        FLAGS_kernel_launching_threads, // number of threads used to execute cuda kernel launches
        FLAGS_postprocessing_threads,   // number of threads used to write and complete responses
        GetSharedMemory(FLAGS_dataset)  // pointer to data in shared memory
    );

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

    rpcResources->RegisterModel("flowers", runtime->DeserializeEngine(FLAGS_engine));
    rpcResources->AllocateResources();

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    LOG(INFO) << "Initializing Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    LOG(INFO) << "Registering Execution Contexts for RPC (flowers::Inference::Compute) with Executor";
    executor->RegisterContexts(rpcCompute, rpcResources, 100);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {
        // Query GPU Power
        nvmlDevice_t gpu;
        unsigned int power;
        CHECK_EQ(nvmlDeviceGetHandleByIndex(0, &gpu), NVML_SUCCESS)
            << "Failed to get Device for index=" << 0;
        CHECK_EQ(nvmlDeviceGetPowerUsage(gpu, &power), NVML_SUCCESS)
            << "Failed to get Power Usage for GPU=" << 0;
        power_gauge.Set((double)power * 0.001);
    });
}

static auto pinned_memory = std::make_unique<Allocator<CudaPinnedHostMemory>>(1024 * 1024 * 1024);

float *GetSharedMemory(const std::string &address)
{
    /* data in shared memory should go here - for the sake of quick examples just use and emptry array */
    pinned_memory->Fill((char)0);
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
