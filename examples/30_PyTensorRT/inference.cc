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
#include "YAIS/Metrics.h"
#include "YAIS/TensorRT/TensorRT.h"
#include "YAIS/YAIS.h"
#include "tensorrt/playground/cuda/device_info.h"

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using yais::Affinity;
using yais::Context;
using yais::Executor;
using yais::Metrics;
using yais::Server;
using yais::ThreadPool;
using yais::TensorRT::Bindings;
using yais::TensorRT::InferenceManager;
using yais::TensorRT::ManagedRuntime;
using yais::TensorRT::Runtime;

class BaseInfer
{
    using BindingsHandle = std::shared_ptr<Bindings>;
    using HostMap = std::map<std::string, DescriptorHandle<HostMemory>>;
    using DeviceMap = std::map<std::string, DescriptorHandle<DeviceMemory>>;

    template<typename T>
    class Result
    {
      public:
        Result() {}
        Result(Result&& other) : m_Promise{std::move(other.m_Promise)} {}
        virtual Result() {}

        using ResultHandle = std::unqiue_ptr<T>;
        using Future = std::future<ResultHandle>;
        using PostFn = std::function<ResultHandle(std : shared_ptr<Bindings>)>;

        void operator()(ResultHandle result)
        {
            m_Promise.set_value(std::move(result));
        }

        Future GetFuture()
        {
            return m_Promise.get_future();
        }

      private:
        std::promise<ResultHandle> m_Promise;
    };

    template<typename T>
    typename Result<T>::Future operator(PreFn pre, Result<T>::PostFn post)
    {
        auto result = std::make_shared<Result<T>>();
        auto wrapped_post = [post, result](BindingsHandle bindings) mutable {
            auto val = post(bindings);
            results(std::move(val));
        } Execute(pre, post);
        return results->GetFuture();
    }

    auto operator(HostMap& inputs)
    {
        auto Preprocess = [this, &inputs](BindingsHandle bindings) mutable {
            PreprocessInputs(inputs, bindings);
            PreprocessOutputs(outputs, bindings);
            for(const auto& id : bindings->GetOutputBindingIds())
            {
                const auto& b = bindings->GetBinding(id);
                auto search = outputs.find(b.name);
                if(search == outputs.end())
                {
                    DLOG(INFO) << "Skipping d2h xfer for output binding: " << b.name;
                    bindings->SetHostAddress(id, nullptr);
                }
            }
        } auto Postprocess = [this, result, outputs](std::shared_ptr<Bindings> bindings) mutable {
            for(const auto& kw : outputs)
            {
                bindings->CopyFromHost(kw->first, kw->second);
            }
            results(std::move(outputs));
        };
        Compute(Preprocess, Postprocess);
        return result->GetFuture();
    }

    virtual void PreprocessInputs(Kwargs& inputs, Bindings& bindings)
    {
        for(const auto& id : bindings->InputBindings())
        {
            const auto& b = bindings->GetBinding(id);
            auto search = inputs.find(b.name);
            CHECK_NE(search, inputs.end()) << "Binding " << b.name << " not provided";
            CopyOfSetBindingAddress(id, search->second);
        }
    }

    void CopyToOrSetBindingAddress(uint32_t binding_id, DescriptorHandle<CoreMemory>) {}
}

class InferenceManagerImpl : public InferenceManager
{
  public:
    template<typename ResultsType>
    class RawTensorInfer
    {
      public:
        using BindingsHandle = std::shared_ptr<Bindings>;
        using ResultsHandle = std::unique_ptr<ResultsType>;
        using FutureResult = std::future<ResultsHandle>;

        RawTensorInfer();
        RawTensorInfer(RawTensorInfer&&);
        virtual ~Runner();

        template<typename ResultsType>
        FutureResult operator()(BindingsHandle bindings)

            FutureResult GetFuture();

      private:
        std::promise<ResultsType> m_Promise;
    };

    InferenceManager(int max_executions, int max_buffers, int nCuda, int nResp)
        : InferenceManager(max_executions, max_buffers)
    {
        RegisterThreadPool("CudaLauncher", std::make_unique<ThreadPool>(1));
        RegisterThreadPool("Preprocess", std::make_unique<ThreadPool>(1));
        RegisterThreadPool("Postprocess", std::make_unique<ThreadPool>(3));
    }

    ~InferenceManager() override {}

    std::shared_ptr<Inference> RegisterModelByPath(const std::string& name, const std::string& path)
    {
        auto model = Runtime::DeserializeEngine(path);
        RegisterModel(name, model);
        return GetHandler(name);
    }

    std::shared_ptr<Inference> GetHandler(std::string name)
    {
        CHECK(GetModel(name));
        return std::make_shared<Inference>(name, casted_shared_from_this<InferenceManager>());
    }

    std::shared_ptr<InferenceManager> cuda()
    {
        AllocateResources();
        return casted_shared_from_this<InferenceManager>();
    }

    void serve()
    {
        Serve(casted_shared_from_this<InferenceManager>());
    }

    std::unique_ptr<ThreadPool>& GetCudaThreadPool()
    {
        return m_CudaThreadPool;
    }
    std::unique_ptr<ThreadPool>& GetResponseThreadPool()
    {
        return m_ResponseThreadPool;
    }
    std::unique_ptr<ThreadPool>& GetPreprocessThreadPool()
    {
        return m_PreprocessThreadPool;
    }

  private:
    std::unique_ptr<ThreadPool> m_CudaThreadPool;
    std::unique_ptr<ThreadPool> m_ResponseThreadPool;
    std::unique_ptr<ThreadPool> m_PreprocessThreadPool;
};

class Inference : public std::enable_shared_from_this<Inference>
{
  public:
    Inference(std::string model_name, std::shared_ptr<InferenceManager> resources)
        : m_Resources(resources), m_ModelName(model_name)
    {
    }

    using FutureResult = std::future<std::shared_ptr<Inference>>;

    FutureResult PyData(py::array_t<float>& data)
    {
        auto context = std::make_shared<Inference>(m_ModelName, m_Resources);
        // GetResources()->GetPreprocessThreadPool()->enqueue([this, context, &data] {
        auto bindings = GetBindings();
        CHECK_EQ(data.shape(0), bindings->GetModel()->GetMaxBatchSize());
        LOG(INFO) << "data from python is " << data.nbytes() << " bytes";
        // bindings->SetHostAddress(0, (void *)data.data());
        auto pinned = bindings->HostAddress(0);
        std::memcpy(pinned, data.data(), data.nbytes());
        Infer(context, bindings);
        //});
        return context->GetFuture();
    }

    FutureResult Compute()
    {
        auto context = std::make_shared<Inference>(m_ModelName, m_Resources);
        auto bindings = GetBindings();
        Infer(context, bindings);
        return context->GetFuture();
    }

    std::string Test(py::kwargs kwargs)
    {
        for(auto item : kwargs)
        {
            LOG(INFO) << "key: " << item.first;
            LOG(INFO) << "batch_size: " << item.second;
            // auto data = static_cast<py::array_t<float>>(item.second);
            LOG(INFO) << item.second.get_type();
            auto data = py::cast<py::array_t<float>>(kwargs[item.first]);
            LOG(INFO) << data.shape(0);
            LOG(INFO) << data.nbytes();
        }
        return "wow";
    }

  protected:
    std::shared_ptr<Bindings> GetBindings()
    {
        auto model = GetResources()->GetModel(m_ModelName);
        auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
        auto bindings = buffers->CreateAndConfigureBindings(model);
        bindings->SetBatchSize(model->GetMaxBatchSize());
        return bindings;
    }

    FutureResult GetFuture()
    {
        return m_Promise.get_future();
    }

    static void Infer(std::shared_ptr<Inference> context, std::shared_ptr<Bindings> bindings)
    {
        context->GetResources()->GetCudaThreadPool()->enqueue([context, bindings]() mutable {
            DLOG(INFO) << "cuda launch thread";
            bindings->CopyToDevice(bindings->InputBindings());
            auto trt = context->GetResources()->GetExecutionContext(
                bindings->GetModel()); // <=== Limited Resource; May Block !!!
            trt->Infer(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            DLOG(INFO) << "cuda launch thread finished";

            context->GetResources()->GetResponseThreadPool()->enqueue(
                [context, bindings, trt]() mutable {
                    DLOG(INFO) << "response thread starting";
                    // This thread waits on the completion of the async compute and the async copy
                    trt->Synchronize();
                    trt.reset(); // Finished with the Execution Context - Release it to competing
                                 // threads
                    bindings->Synchronize();
                    bindings.reset(); // Finished with Buffers - Release it to competing threads
                    context->m_Promise.set_value(context);
                    DLOG(INFO) << "response thread finished - inference done";
                });
        });
    }

  protected:
    inline std::shared_ptr<InferenceManager> GetResources()
    {
        return m_Resources;
    }

  private:
    std::string m_ModelName;
    std::shared_ptr<InferenceManager> m_Resources;
    std::promise<std::shared_ptr<Inference>> m_Promise;
};

// NVIDIA Inference Server Protos
#include "nvidia_inference.grpc.pb.h"
#include "nvidia_inference.pb.h"

using nvidia::inferenceserver::GRPCService;
using nvidia::inferenceserver::InferRequest;
using nvidia::inferenceserver::InferResponse;
using nvidia::inferenceserver::ServerStatus;
using nvidia::inferenceserver::StatusRequest;
using nvidia::inferenceserver::StatusResponse;

static auto& registry = Metrics::GetRegistry();

// Counters
static auto& inf_counter = prometheus::BuildCounter().Name("nv_inference_count").Register(registry);

static auto& inf_compute_time =
    prometheus::BuildCounter().Name("nv_inference_compute_duration_us").Register(registry);

static auto& inf_request_time =
    prometheus::BuildCounter().Name("nv_inference_request_duration_us").Register(registry);

// Gauge - Periodically measure and report GPU power utilization.  As the load increases
//         on the service, the power should increase proprotionally, until the power is capped
//         either by device limits or compute resources.  At this level, the inf_load_ratio
//         will begin to increase under futher increases in traffic
static auto& power_usage_gauge =
    prometheus::BuildGauge().Name("nv_gpu_power_usage").Register(registry);

static auto& power_limit_gauge =
    prometheus::BuildGauge().Name("nv_gpu_power_limit").Register(registry);

// Histogram - Load Ratio = Request/Compute duration - should just above one for a service
//             that can keep up with its current load.  This metrics provides more
//             detailed information on the impact of the queue depth because it accounts
//             for request time.
static auto& inf_load_ratio =
    prometheus::BuildHistogram().Name("nv_inference_load_ratio").Register(registry);

static const std::vector<double> buckets = {1.25, 1.50, 2.0, 10.0, 100.0}; // unitless

class StatusContext final : public Context<StatusRequest, StatusResponse, InferenceManager>
{
    void ExecuteRPC(StatusRequest& request, StatusResponse& response) final override
    {
        GetResources()->GetResponseThreadPool()->enqueue([this, &request, &response] {
            auto model = GetResources()->GetModel(request.model_name());
            auto server_status = response.mutable_server_status();
            server_status->set_ready_state(
                ::nvidia::inferenceserver::ServerReadyState::SERVER_READY);
            auto model_status = server_status->mutable_model_status();
            auto config = (*model_status)[model->Name()].mutable_config();
            config->set_name(model->Name());
            config->set_max_batch_size(model->GetMaxBatchSize());
            for(auto i : model->GetInputBindingIds())
            {
                const auto& binding = model->GetBinding(i);
                auto input = config->add_input();
                input->set_name(binding.name);
                input->set_data_type(::nvidia::inferenceserver::DataType::TYPE_INT8);
                for(auto d : binding.dims)
                {
                    input->add_dims(d);
                }
            }
            for(auto i : model->GetOutputBindingIds())
            {
                const auto& binding = model->GetBinding(i);
                auto output = config->add_output();
                output->set_name(binding.name);
                output->set_data_type(::nvidia::inferenceserver::DataType::TYPE_FP32);
                for(auto d : binding.dims)
                {
                    output->add_dims(d);
                }
            }
            auto request_status = response.mutable_request_status();
            request_status->set_code(::nvidia::inferenceserver::RequestStatusCode::SUCCESS);
            LOG(INFO) << response.DebugString();
            this->FinishResponse();
        });
    }
};

class FlowersContext final : public Context<InferRequest, InferResponse, InferenceManager>
{
    void ExecuteRPC(RequestType& input, ResponseType& output) final override
    {
        // Executing on a Executor threads - we don't want to block message handling, so we offload
        GetResources()->GetCudaThreadPool()->enqueue([this, &input, &output]() {
            // Executed on a thread from CudaThreadPool
            auto model = GetResources()->GetModel(input.model_name());
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            auto bindings = buffers->CreateAndConfigureBindings(model);
            bindings->SetBatchSize(input.batch_size());
            // bindings->SetHostAddress(0, GetResources()->GetSysvOffset(input.sysv_offset()));
            bindings->CopyToDevice(bindings->InputBindings());
            auto ctx =
                GetResources()->GetExecutionContext(model); // <=== Limited Resource; May Block !!!
            auto t_start = Walltime();
            ctx->Infer(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            // All Async CUDA work has been queued - this thread's work is done.
            GetResources()->GetResponseThreadPool()->enqueue([this, &input, &output, model,
                                                              bindings, ctx, t_start]() mutable {
                // Executed on a thread from ResponseThreadPool
                ctx->Synchronize();
                ctx.reset(); // Finished with the Execution Context - Release it to competing
                             // threads
                auto compute_time = Walltime() - t_start;
                bindings->Synchronize(); // Blocks on H2D, Compute, D2H Pipeline
                WriteBatchPredictions(input, output, (float*)bindings->HostAddress(1));
                bindings.reset(); // Finished with Buffers - Release it to competing threads
                auto request_time = Walltime();
                output.set_compute_time(static_cast<float>(compute_time));
                output.set_request_time(static_cast<float>(request_time));
                auto request_status = output.mutable_request_status();
                request_status->set_code(::nvidia::inferenceserver::RequestStatusCode::SUCCESS);
                auto batch_size = input.batch_size();
                this->FinishResponse();
                // The Response is now sending; Record some metrics and be done
                std::map<std::string, std::string> labels = {{"model", model->Name()},
                                                             {"gpu_uuid", yais::GetDeviceUUID(0)}};
                inf_counter.Add(labels).Increment(batch_size);
                inf_compute_time.Add(labels).Increment(compute_time * 1000);
                inf_request_time.Add(labels).Increment(request_time * 1000);
                inf_load_ratio.Add(labels, buckets).Observe(request_time / compute_time);
            });
        });
    }

    void WriteBatchPredictions(RequestType& input, ResponseType& output, float* scores)
    {
        int N = input.batch_size();
        auto nClasses =
            GetResources()->GetModel(input.model_name())->GetBinding(1).elementsPerBatchItem;
        size_t cntr = 0;
        auto meta_data = output.mutable_meta_data();
        auto meta_data_output = meta_data->add_output();
        for(int p = 0; p < N; p++)
        {
            auto bcls = meta_data_output->add_batch_classes();
            bcls->add_cls();
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

/*
void Serve(std::shared_ptr<InferenceManager> resources)
{
    // Set CPU Affinity to be near the GPU
    auto cpus = yais::GetDeviceAffinity(0);
    Affinity::SetAffinity(cpus);

    // Enable metrics on port
    Metrics::Initialize(50078);

    // registerAllTensorRTPlugins();

    // Create a gRPC server bound to IP:PORT
    std::ostringstream ip_port;
    ip_port << "0.0.0.0:" << 50051;
    Server server(ip_port.str());

    // Modify MaxReceiveMessageSize
    auto bytes = yais::StringToBytes("10MiB");
    server.Builder().SetMaxReceiveMessageSize(bytes);
    LOG(INFO) << "gRPC MaxReceiveMessageSize = " << yais::BytesToString(bytes);

    // A server can host multiple services
    auto inferenceService = server.RegisterAsyncService<GRPCService>();

    auto rpcCompute = inferenceService->RegisterRPC<FlowersContext>(
        &GRPCService::AsyncService::RequestInfer);

    auto rpcStatus = inferenceService->RegisterRPC<StatusContext>(
        &GRPCService::AsyncService::RequestStatus);

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    LOG(INFO) << "Initializing Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    executor->RegisterContexts(rpcCompute, resources, 100);
    executor->RegisterContexts(rpcStatus, resources, 1);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {
        power_usage_gauge.Add({{"gpu_uuid",
yais::GetDeviceUUID(0)}}).Set(yais::GetDevicePowerUsage(0)); power_limit_gauge.Add({{"gpu_uuid",
yais::GetDeviceUUID(0)}}).Set(yais::GetDevicePowerLimit(0));
    });
}
*/

PYBIND11_MODULE(py_yais, m)
{
    py::class_<InferenceManager, std::shared_ptr<InferenceManager>>(m, "InferenceManager")
        .def(py::init([](int concurrency) {
            return std::make_shared<InferenceManager>(concurrency, concurrency + 4, 1, 3);
        }))
        .def("register_tensorrt_engine", &InferenceManager::RegisterModelByPath)
        .def("get_model", &InferenceManager::GetHandler)
        .def("cuda", &InferenceManager::cuda)
        .def("serve", &InferenceManager::serve, py::call_guard<py::gil_scoped_release>());

    py::class_<Inference, std::shared_ptr<Inference>>(m, "Inference")
        .def("pyinfer", &Inference::PyData, py::call_guard<py::gil_scoped_release>())
        .def("compute", &Inference::Compute, py::call_guard<py::gil_scoped_release>())
        .def("__call__", &Inference::Test);

    py::class_<Inference::FutureResult>(m, "InferenceFutureResult")
        .def("wait", &Inference::FutureResult::wait, py::call_guard<py::gil_scoped_release>())
        .def("get", &Inference::FutureResult::get, py::call_guard<py::gil_scoped_release>());
}

/*


/**

static bool ValidateEngine(const char *flagname, const std::string &value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_int32(seconds, 5, "Approximate number of seconds for the timing loop");
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers, 0, "Number of Buffers (default: 2x contexts)");
DEFINE_int32(cudathreads, 1, "Number Cuda Launcher Threads");
DEFINE_int32(respthreads, 1, "Number Response Sync Threads");
DEFINE_int32(replicas, 1, "Number of Replicas of the Model to load");
DEFINE_int32(batch_size, 0, "Overrides the max batch_size of the provided engine");

int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT Inference");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    MPI_CHECK(MPI_Init(&argc, &argv));

    auto contexts = g_Concurrency = FLAGS_contexts;
    auto buffers = FLAGS_buffers ? FLAGS_buffers : 2 * FLAGS_contexts;

    auto resources = std::make_shared<InferenceResources>(
        contexts,
        buffers,
        FLAGS_cudathreads,
        FLAGS_respthreads);

    resources->RegisterModel("0", ManagedRuntime::DeserializeEngine(FLAGS_engine));
    resources->AllocateResources();

    for (int i = 1; i < FLAGS_replicas; i++)
    {
        resources->RegisterModel(ModelName(i), ManagedRuntime::DeserializeEngine(FLAGS_engine));
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
*/
