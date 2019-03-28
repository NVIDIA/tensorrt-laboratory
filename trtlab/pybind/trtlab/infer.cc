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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <future>
#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "trtlab/tensorrt/bindings.h"
#include "trtlab/core/async_compute.h"
#include "trtlab/core/thread_pool.h"
#include "trtlab/tensorrt/infer_bench.h"
#include "trtlab/tensorrt/infer_runner.h"
#include "trtlab/tensorrt/inference_manager.h"
#include "trtlab/tensorrt/model.h"
#include "trtlab/tensorrt/runtime.h"
#include "trtlab/tensorrt/utils.h"

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/descriptor.h"
#include "trtlab/core/memory/malloc.h"

#include "utils.h"

using namespace trtlab;
using namespace trtlab;
using namespace trtlab::TensorRT;

#include "nvrpc/context.h"
#include "nvrpc/executor.h"
#include "nvrpc/server.h"
#include "nvrpc/service.h"

#include "nvrpc/client/client_unary.h"
#include "nvrpc/client/executor.h"

using nvrpc::AsyncService;
using nvrpc::Context;
using nvrpc::Executor;
using nvrpc::Server;

// NVIDIA Inference Server Protos
#include "trtlab/trtis/protos/grpc_service.grpc.pb.h"
#include "trtlab/trtis/protos/grpc_service.pb.h"

namespace trtis = ::nvidia::inferenceserver;

#include "dlpack.h"
#include "numpy.h"
using namespace trtlab::python;

ThreadPool& GetThreadPool()
{
    static ThreadPool threads(Affinity::GetCpusFromString("0"));
    return threads;
}

void BasicInferService(std::shared_ptr<InferenceManager> resources, int port = 50052,
                       const std::string& max_recv_msg_size = "100MiB");

class TrtisModel;
class PyInferRunner;
class PyInferRemoteRunner;

class PyInferenceManager final : public InferenceManager
{
  public:
    PyInferenceManager(int max_executions, int max_buffers, int pre_threads, int cuda_threads,
                       int post_threads)
        : InferenceManager(max_executions, max_buffers)
    {
        RegisterThreadPool("pre", std::make_unique<ThreadPool>(pre_threads));
        RegisterThreadPool("cuda", std::make_unique<ThreadPool>(cuda_threads));
        RegisterThreadPool("post", std::make_unique<ThreadPool>(post_threads));
        RegisterRuntime("default", std::make_shared<StandardRuntime>());
        RegisterRuntime("unified", std::make_shared<ManagedRuntime>());
        SetActiveRuntime("default");
    }

    ~PyInferenceManager() override {}

    std::shared_ptr<PyInferRunner> RegisterModelByPath(const std::string& name,
                                                       const std::string& path)
    {
        auto model = ActiveRuntime().DeserializeEngine(path);
        RegisterModel(name, model);
        return this->InferRunner(name);
    }

    std::shared_ptr<PyInferRunner> InferRunner(std::string name)
    {
        return std::make_shared<PyInferRunner>(GetModel(name),
                                               casted_shared_from_this<PyInferenceManager>());
    }

    void Serve(int port) { BasicInferService(casted_shared_from_this<PyInferenceManager>(), port); }

    std::vector<std::string> Models()
    {
        std::vector<std::string> model_names;
        ForEachModel([&model_names](const Model& model) { model_names.push_back(model.Name()); });
        return model_names;
    }
};

class PyRemoteInferenceManager
{
  public:
    PyRemoteInferenceManager(py::kwargs kwargs)
    {
        m_Hostname = "localhost:50052";
        int client_threads = 1;
        for(const auto& item : kwargs)
        {
            auto key = py::cast<std::string>(item.first);
            if(key == "hostname")
            {
                m_Hostname = py::cast<std::string>(item.second);
            }
        }

        ::grpc::ChannelArguments ch_args;
        ch_args.SetMaxReceiveMessageSize(-1);
        m_Channel =
            grpc::CreateCustomChannel(m_Hostname, grpc::InsecureChannelCredentials(), ch_args);
        m_Stub = ::trtis::GRPCService::NewStub(m_Channel);
        m_Executor = std::make_shared<::nvrpc::client::Executor>(client_threads);
    }

    std::vector<std::string> Models()
    {
        const auto& status = TrtisStatus();
        auto model_status = status.server_status().model_status();
        DLOG(INFO) << status.DebugString();
        const ::trtis::ModelConfig* model_config;
        std::vector<std::string> models;
        m_Models.clear();
        for(auto it = model_status.begin(); it != model_status.end(); it++)
        {
            DLOG(INFO) << it->first;
            models.push_back(it->first);
            m_Models[it->first] = std::make_shared<TrtisModel>(it->second.config());
            /*
            if(FLAGS_model == it->first)
            {
                LOG(INFO) << "found model_config for " << FLAGS_model;
                model_config = &(it->second.config());
            }
            */
        }
        return models;
    }

    std::shared_ptr<PyInferRemoteRunner> InferRunner(const std::string& model_name)
    {
        auto infer_prepare_fn = [this](::grpc::ClientContext * context,
                                       const ::trtis::InferRequest& request,
                                       ::grpc::CompletionQueue* cq) -> auto
        {
            return std::move(m_Stub->PrepareAsyncInfer(context, request, cq));
        };

        auto runner = std::make_unique<
            ::nvrpc::client::ClientUnary<::trtis::InferRequest, ::trtis::InferResponse>>(
            infer_prepare_fn, m_Executor);

        return std::make_shared<PyInferRemoteRunner>(GetModel(model_name), std::move(runner));
    }

    std::shared_ptr<TrtisModel> GetModel(const std::string& name) const
    {
        auto search = m_Models.find(name);
        LOG_IF(FATAL, search == m_Models.end()) << "Model: " << name << " not found";
        return search->second;
    }

  protected:
    ::trtis::StatusResponse TrtisStatus()
    {
        ::grpc::ClientContext context;
        ::trtis::StatusRequest request;
        ::trtis::StatusResponse response;
        auto status = m_Stub->Status(&context, request, &response);
        CHECK(status.ok());
        return response;
    }

  private:
    std::string m_Hostname;
    std::map<std::string, std::shared_ptr<TrtisModel>> m_Models;
    std::shared_ptr<::grpc::Channel> m_Channel;
    std::unique_ptr<::trtis::GRPCService::Stub> m_Stub;
    std::shared_ptr<::nvrpc::client::Executor> m_Executor;
};

struct TrtisModel : BaseModel
{
    TrtisModel(const ::trtis::ModelConfig& model)
    {
        SetName(model.name());
        m_MaxBatchSize = model.max_batch_size();
        for(int i = 0; i < model.input_size(); i++)
        {
            const auto& b = model.input(i);
            TensorBindingInfo binding;
            binding.name = b.name();
            binding.isInput = true;
            binding.dtype = nvinfer1::DataType::kFLOAT;
            binding.dtypeSize =
                SizeofDataType(binding.dtype); // TODO: map trtis DataType enum; model.data_type()
            size_t count = 1;
            for(int j = 0; j < b.dims_size(); j++)
            {
                auto val = b.dims(j);
                binding.dims.push_back(val);
                count *= val;
            }

            binding.elementsPerBatchItem = count;
            binding.bytesPerBatchItem = count * binding.dtypeSize;
            AddBinding(std::move(binding));
        }
        for(int i = 0; i < model.output_size(); i++)
        {
            const auto& b = model.output(i);
            TensorBindingInfo binding;
            binding.name = b.name();
            binding.isInput = false;
            binding.dtype = nvinfer1::DataType::kFLOAT;
            binding.dtypeSize =
                SizeofDataType(binding.dtype); // TODO: map trtis DataType enum; model.data_type()
            size_t count = 1;
            for(int j = 0; j < b.dims_size(); j++)
            {
                auto val = b.dims(j);
                binding.dims.push_back(val);
                count *= val;
            }

            binding.elementsPerBatchItem = count;
            binding.bytesPerBatchItem = count * binding.dtypeSize;
            AddBinding(std::move(binding));
        }
    }
    ~TrtisModel() override {}

    int GetMaxBatchSize() const final override { return m_MaxBatchSize; }

  private:
    int m_MaxBatchSize;
};

struct PyInferRemoteRunner
{
    PyInferRemoteRunner(
        std::shared_ptr<TrtisModel> model,
        std::unique_ptr<::nvrpc::client::ClientUnary<::trtis::InferRequest, ::trtis::InferResponse>>
            runner)
        : m_Model(model), m_Runner(std::move(runner))
    {
    }

    using InferResults = py::dict;
    using InferFuture = std::shared_future<InferResults>;

    const BaseModel& GetModel() const { return *m_Model; }

    InferFuture Infer(py::kwargs kwargs)
    {
        const auto& model = GetModel();
        int batch_size = -1; // will be infered from the input tensors

        // Build InferRequest
        ::trtis::InferRequest request;
        request.set_model_name(model.Name());
        // request.set_version("");
        auto meta_data = request.mutable_meta_data();

        DLOG(INFO) << "Processing Python kwargs - holding the GIL";
        for(auto item : kwargs)
        {
            auto key = py::cast<std::string>(item.first);
            DLOG(INFO) << "Processing Python Keyword: " << key;
            const auto& binding = model.GetBinding(key);
            // TODO: throw a python exception
            LOG_IF(FATAL, !binding.isInput) << item.first << " is not an InputBinding";
            if(binding.isInput)
            {
                CHECK(py::isinstance<py::array>(item.second));
                auto data = py::cast<py::array_t<float>>(item.second);
                CHECK_LE(data.shape(0), model.GetMaxBatchSize());
                if(batch_size == -1)
                {
                    DLOG(INFO) << "Inferred batch_size=" << batch_size << " from dimensions ";
                    batch_size = data.shape(0);
                }
                else
                {
                    CHECK_EQ(data.shape(0), batch_size);
                }
                CHECK_EQ(data.nbytes(), binding.bytesPerBatchItem * batch_size);
                request.add_raw_input(data.data(), data.nbytes());
                auto meta = meta_data->add_input();
                meta->set_name(binding.name);
                meta->set_batch_byte_size(binding.elementsPerBatchItem);
            }
        }

        for(auto id : model.GetOutputBindingIds())
        {
            const auto& binding = model.GetBinding(id);
            auto meta = meta_data->add_output();
            meta->set_name(binding.name);
        }
        meta_data->set_batch_size(batch_size);

        // Submit to TRTIS
        py::gil_scoped_release release;
        return m_Runner->Enqueue(std::move(request),
                                 [this](::trtis::InferRequest& request,
                                        ::trtis::InferResponse& response,
                                        ::grpc::Status& status) -> py::dict {
                                     // Convert InferResponse to py::dict
                                     LOG(INFO) << response.DebugString();
                                     return ConvertResponseToNumpy(response);
                                 });
    }

    py::dict InputBindings() const
    {
        auto dict = py::dict();
        for(const auto& id : GetModel().GetInputBindingIds())
        {
            AddBindingInfo(dict, id);
        }
        return dict;
    }

    py::dict OutputBindings() const
    {
        auto dict = py::dict();
        for(const auto& id : GetModel().GetOutputBindingIds())
        {
            AddBindingInfo(dict, id);
        }
        return dict;
    }

  protected:
    py::dict ConvertResponseToNumpy(const ::trtis::InferResponse& response)
    {
        py::gil_scoped_acquire acquire;
        py::dict results;
        const auto& meta_data = response.meta_data();
        for(int i = 0; i < meta_data.output_size(); i++)
        {
            const auto& out = meta_data.output(i);
            const auto& binding = GetModel().GetBinding(out.name());
            LOG(INFO) << "Processing binding: " << out.name();
            auto value = py::array(DataTypeToNumpy(binding.dtype), binding.dims);
            py::buffer_info buffer = value.request();
            CHECK_EQ(value.nbytes(), binding.bytesPerBatchItem * meta_data.batch_size());
            const auto& raw = response.raw_output(i);
            std::memcpy(buffer.ptr, raw.c_str(), value.nbytes());
            py::str key = binding.name;
            results[key] = value;
        }
        return results;
    }

    void AddBindingInfo(py::dict& dict, int id) const
    {
        const auto& binding = GetModel().GetBinding(id);
        py::str key = binding.name;
        py::dict value;
        value["shape"] = binding.dims;
        value["dtype"] = DataTypeToNumpy(binding.dtype);
        dict[key] = value;
    }

  private:
    std::shared_ptr<TrtisModel> m_Model;
    std::unique_ptr<::nvrpc::client::ClientUnary<::trtis::InferRequest, ::trtis::InferResponse>>
        m_Runner;
};

struct PyInferRunner : public InferRunner
{
    using InferRunner::InferRunner;
    using InferResults = py::dict;
    using InferFuture = std::shared_future<InferResults>;

/*
    auto Infer(py::dict inputs, py::dict output_objs, py::dict output_locs)
    {
        int batch_size = -1;
        const auto& model = GetModel();
        auto bindings = InitializeBindings();

        py::gil_scoped_acquire acquire;
        ProcessInputs(inputs, bindings, batch_size);
    }

    void ProcessInputs(const Model& model, py::dict inputs)
    {
        int batch_size = -1;
        for(auto item : inputs)
        {
            auto name = item.first.cast<std::string>();
            auto type = model.GetBindingType(name);
            std::unique_ptr<CoreMemory> memory;
            if(type != Model::BindingType::Input)
            {
                throw std::runtime_error("Invalid input binding: " + name);
            }
            if(py::isinstance<py::array>(item.second))
            {
                memory = std::move(Numpy::ImportUnique(item.second));
            }
            else if(py::isinstance<py::capsule>(item.second))
            {
                memory = std::move(DLPack::ImportUnique(item.second));
            }
            else
            {
                throw std::runtime_error("Invalid value for input: " + name +
                                         "; expected a numpy or dlpack");
            }
            auto shape = memory->Shape();
            // if bindings.shape + 1 == shape, then infer batch size if -1
            // check that total bytes are consistent between input and expected
            if(batch_size == -1) { batch_size = bs; }
            else if(batch_size !=)
        }
    }
*/
    auto Infer(py::kwargs kwargs)
    {
        const auto& model = GetModel();
        auto bindings = InitializeBindings();
        int batch_size = -1;
        DLOG(INFO) << "Processing Python kwargs - holding the GIL";
        {
            py::gil_scoped_acquire acquire;
            for(auto item : kwargs)
            {
                auto key = py::cast<std::string>(item.first);
                DLOG(INFO) << "Processing Python Keyword: " << key;
                const auto& binding = model.GetBinding(key);
                // TODO: throw a python exception
                LOG_IF(FATAL, !binding.isInput) << item.first << " is not an InputBinding";
                if(binding.isInput)
                {
                    const void* ptr;
                    size_t size;
                    size_t batch;
                    CHECK(py::isinstance<py::array>(item.second));
                    switch(binding.dtype)
                    {
                        case nvinfer1::DataType::kFLOAT:
                        {
                            auto data = py::cast<py::array_t<float>>(item.second);
                            ptr = data.data();
                            size = data.nbytes();
                            batch = data.shape(0);
                            break;
                        }
                        case nvinfer1::DataType::kINT8:
                        {
                            auto data = py::cast<py::array_t<std::int8_t>>(item.second);
                            ptr = data.data();
                            size = data.nbytes();
                            batch = data.shape(0);
                            break;
                        }
                        case nvinfer1::DataType::kINT32:
                        {
                            auto data = py::cast<py::array_t<std::int32_t>>(item.second);
                            ptr = data.data();
                            size = data.nbytes();
                            batch = data.shape(0);
                            break;
                        }
                        default:
                            LOG(FATAL) << "Unknown dtype";
                    }
                    CHECK_LE(batch, model.GetMaxBatchSize());
                    if(batch_size == -1)
                    {
                        batch_size = batch;
                        DLOG(INFO) << "Inferred batch_size=" << batch_size << " from dimensions";
                    }
                    else
                    {
                        CHECK_EQ(batch, batch_size);
                    }
                    CHECK_EQ(size, binding.bytesPerBatchItem * batch_size);
                    auto id = model.BindingId(key);
                    auto host = bindings->HostAddress(id);
                    // TODO: enhance the Copy method for py::buffer_info objects
                    std::memcpy(host, ptr, size);
                }
            }
        }
        py::gil_scoped_release release;
        bindings->SetBatchSize(batch_size);
        return InferRunner::Infer(
            bindings, [](std::shared_ptr<Bindings>& bindings) -> InferResults {
                py::gil_scoped_acquire acquire;
                auto results = InferResults();
                DLOG(INFO) << "Copying Output Bindings to Numpy arrays";
                for(const auto& id : bindings->OutputBindings())
                {
                    const auto& binding = bindings->GetModel()->GetBinding(id);
                    DLOG(INFO) << "Processing binding: " << binding.name << "with index " << id;
                    std::vector<int> dims;
                    dims.push_back(bindings->BatchSize());
                    for(const auto& d : binding.dims)
                    {
                        dims.push_back(d);
                    }
                    auto value = py::array(DataTypeToNumpy(binding.dtype), dims);
                    // auto value = py::array_t<float>(binding.dims);
                    // auto value = py::array_t<float>(binding.elementsPerBatchItem *
                    // bindings->BatchSize());
                    py::buffer_info buffer = value.request();
                    CHECK_EQ(value.nbytes(), bindings->BindingSize(id));
                    std::memcpy(buffer.ptr, bindings->HostAddress(id), value.nbytes());
                    py::str key = binding.name;
                    results[key] = value;
                }
                DLOG(INFO) << "Finished Postprocessing Numpy arrays - setting future/promise value";
                return results;
            });
    }

    py::dict InputBindings() const
    {
        auto dict = py::dict();
        for(const auto& id : GetModel().GetInputBindingIds())
        {
            AddBindingInfo(dict, id);
        }
        return dict;
    }

    py::dict OutputBindings() const
    {
        auto dict = py::dict();
        for(const auto& id : GetModel().GetOutputBindingIds())
        {
            AddBindingInfo(dict, id);
        }
        return dict;
    }

  protected:
    void AddBindingInfo(py::dict& dict, int id) const
    {
        const auto& binding = GetModel().GetBinding(id);
        py::str key = binding.name;
        py::dict value;
        value["shape"] = binding.dims;
        value["dtype"] = DataTypeToNumpy(binding.dtype);
        dict[key] = value;
    }
};

class StatusContext final
    : public Context<::trtis::StatusRequest, ::trtis::StatusResponse, InferenceManager>
{
    void ExecuteRPC(::trtis::StatusRequest& request,
                    ::trtis::StatusResponse& response) final override
    {
        GetResources()->AcquireThreadPool("post").enqueue([this, &request, &response] {
            // create a status response
            auto server_status = response.mutable_server_status();
            server_status->set_ready_state(::trtis::ServerReadyState::SERVER_READY);
            auto model_status = server_status->mutable_model_status();

            // populate each model
            GetResources()->ForEachModel([model_status](const Model& model) {
                auto config = (*model_status)[model.Name()].mutable_config();
                config->set_name(model.Name());
                config->set_max_batch_size(model.GetMaxBatchSize());
                for(auto i : model.GetInputBindingIds())
                {
                    const auto& binding = model.GetBinding(i);
                    auto input = config->add_input();
                    input->set_name(binding.name);
                    input->set_data_type(::trtis::DataType::TYPE_FP32);
                    for(auto d : binding.dims)
                    {
                        input->add_dims(d);
                    }
                }
                for(auto i : model.GetOutputBindingIds())
                {
                    const auto& binding = model.GetBinding(i);
                    auto output = config->add_output();
                    output->set_name(binding.name);
                    output->set_data_type(::trtis::DataType::TYPE_FP32);
                    for(auto d : binding.dims)
                    {
                        output->add_dims(d);
                    }
                }
            });

            auto request_status = response.mutable_request_status();
            request_status->set_code(::trtis::RequestStatusCode::SUCCESS);
            LOG(INFO) << response.DebugString();
            this->FinishResponse();
        });
    }
};

class InferContext final
    : public Context<::trtis::InferRequest, ::trtis::InferResponse, InferenceManager>
{
    void ExecuteRPC(RequestType& input, ResponseType& output) final override
    {
        // Executing on a Executor threads - we don't want to block message handling, so we offload
        GetResources()->AcquireThreadPool("pre").enqueue([this, &input, &output]() {
            // Executed on a thread from CudaThreadPool
            auto model = GetResources()->GetModel(input.model_name());
            auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
            auto bindings = buffers->CreateBindings(model);

            // prepare input bindings - copy data from input
            const auto& meta_data = input.meta_data();
            bindings->SetBatchSize(meta_data.batch_size());
            for(int input_idx = 0; input_idx < input.raw_input_size(); input_idx++)
            {
                const auto& in = meta_data.input(input_idx);
                auto binding_idx = model->BindingId(in.name());
                const std::string& raw = input.raw_input(input_idx);
                CHECK_EQ(raw.size(), bindings->BindingSize(binding_idx));
                DLOG(INFO) << "Copying binding " << in.name() << " from raw_input " << input_idx
                           << " to binding " << binding_idx;
                std::memcpy(bindings->HostAddress(binding_idx), raw.c_str(), raw.size());
            }
            InferRunner runner(model, GetResources());
            runner.Infer(bindings, [this, &input, &output](std::shared_ptr<Bindings>& bindings) {
                // post processing function - write response
                // for each output binding - write populate the response from data in bindings
                const auto& input_meta_data = input.meta_data();
                auto output_meta_data = output.mutable_meta_data();
                output_meta_data->set_model_name(bindings->GetModel()->Name());
                output_meta_data->set_batch_size(bindings->BatchSize());
                for(int idx = 0; idx < input_meta_data.output_size(); idx++)
                {
                    const auto& out = input_meta_data.output(idx);
                    auto binding_idx = bindings->GetModel()->BindingId(out.name());
                    auto meta = output_meta_data->add_output();
                    meta->set_name(out.name());
                    output.add_raw_output(bindings->HostAddress(binding_idx),
                                          bindings->BindingSize(binding_idx));
                }
                this->FinishResponse();
            });
        });
    }
};

void BasicInferService(std::shared_ptr<InferenceManager> resources, int port,
                       const std::string& max_recv_msg_size)
{
    // registerAllTensorRTPlugins();

    // Create a gRPC server bound to IP:PORT
    std::ostringstream ip_port;
    ip_port << "0.0.0.0:" << port;
    Server server(ip_port.str());

    // Modify MaxReceiveMessageSize
    auto bytes = trtlab::StringToBytes(max_recv_msg_size);
    server.Builder().SetMaxReceiveMessageSize(bytes);
    LOG(INFO) << "gRPC MaxReceiveMessageSize = " << trtlab::BytesToString(bytes);

    // A server can host multiple services
    auto inferenceService = server.RegisterAsyncService<::trtis::GRPCService>();

    auto rpcCompute = inferenceService->RegisterRPC<InferContext>(
        &::trtis::GRPCService::AsyncService::RequestInfer);

    auto rpcStatus = inferenceService->RegisterRPC<StatusContext>(
        &::trtis::GRPCService::AsyncService::RequestStatus);

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    LOG(INFO) << "Initializing Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    executor->RegisterContexts(rpcCompute, resources, 100);
    executor->RegisterContexts(rpcStatus, resources, 10);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(1000), [] {});
}

/*
// moved to: dlpack.h and dlpack.cc in current path
class DLPack final
{
    template<typename MemoryType>
    class DLPackDescriptor;

  public:
    static py::capsule Export(std::shared_ptr<CoreMemory> memory)
    {
        auto self = new DLPack(memory);
        return self->to_dlpack();
    }

    static std::shared_ptr<CoreMemory> Import(py::capsule obj)
    {
        DLManagedTensor* dlm_tensor = (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
        if(dlm_tensor == nullptr)
        {
            throw std::runtime_error("dltensor not found in capsule");
        }
        const auto& dltensor = dlm_tensor->dl_tensor;
        // take ownership of the capsule by incrementing the reference count on the handle
        // note: we use a py::handle object instead of a py::object becasue we need to manually
        // control the reference counting in the deleter after re-acquiring the GIL
        auto count = obj.ref_count();
        auto handle = py::cast<py::handle>(obj);
        handle.inc_ref();
        auto deleter = [handle] {
            // releasing ownership of the capsule
            // if the capsule still exists after we decrement the reference count,
            // then, reset the name so ownership can be optionally reacquired.
            DLOG(INFO) << "DLPack Wrapper Releasing Ownership of Capsule " << handle.ptr();
            py::gil_scoped_acquire acquire;
            PyCapsule_SetName(handle.ptr(), "dltensor");
            handle.dec_ref();
        };
        DCHECK_EQ(obj.ref_count(), count + 1);
        PyCapsule_SetName(obj.ptr(), "used_dltensor");

        if(dltensor.ctx.device_type == kDLCPU)
        {
            return std::make_shared<DLPackDescriptor<Malloc>>(dltensor, deleter);
        }
        else if(dltensor.ctx.device_type == kDLCPUPinned)
        {
            return std::make_shared<DLPackDescriptor<CudaPinnedHostMemory>>(dltensor, deleter);
        }
        else if(dltensor.ctx.device_type == kDLGPU)
        {
            return std::make_shared<DLPackDescriptor<CudaDeviceMemory>>(dltensor, deleter);
        }
        else
        {
            throw std::runtime_error("Invalid DLPack device_type");
        }
        return nullptr;
    }

  private:
    DLPack(std::shared_ptr<CoreMemory> shared) : m_ManagedMemory(shared)
    {
        m_ManagedTensor.dl_tensor = shared->TensorInfo();
        m_ManagedTensor.manager_ctx = static_cast<void*>(this);
        m_ManagedTensor.deleter = [](DLManagedTensor* ptr) mutable {
            if(ptr)
            {
                DLPack* self = (DLPack*)ptr->manager_ctx;
                if(self)
                {
                    DLOG(INFO) << "Deleting DLPack Wrapper via DLManagedTensor::deleter";
                    delete self;
                }
            }
        };
    }

    ~DLPack() {}

    py::capsule to_dlpack()
    {
        return py::capsule((void*)&m_ManagedTensor, "dltensor", [](PyObject* ptr) {
            auto obj = PyCapsule_GetPointer(ptr, "dltensor");
            if(obj == nullptr)
            {
                throw std::logic_error(
                    "dltensor not found in capsule; there must be a race condition on ownership");
            }
            DLOG(INFO) << "Deallocating Capsule " << ptr;
            DLManagedTensor* managed_tensor = (DLManagedTensor*)obj;
            managed_tensor->deleter(managed_tensor);
        });
    }

    template<typename MemoryType>
    class DLPackDescriptor : public Descriptor<MemoryType>
    {
      public:
        DLPackDescriptor(const DLTensor& dltensor, std::function<void()> deleter)
            : Descriptor<MemoryType>(dltensor, deleter, "DLPack")
        {
        }

        ~DLPackDescriptor() override {}
    };

    std::shared_ptr<CoreMemory> m_ManagedMemory;
    DLManagedTensor m_ManagedTensor;
};
*/

using PyInferFuture = std::shared_future<typename PyInferRunner::InferResults>;
PYBIND11_MAKE_OPAQUE(PyInferFuture);

PYBIND11_MODULE(_cpp_trtlab, m)
{
    py::class_<PyInferenceManager, std::shared_ptr<PyInferenceManager>>(m, "InferenceManager")
        .def(py::init<int, int, int, int, int>(), py::arg("max_exec_concurrency") = 1,
             py::arg("max_copy_concurrency") = 0, py::arg("pre_threads") = 1,
             py::arg("cuda_threads") = 1, py::arg("post_threads") = 3)
        .def("register_tensorrt_engine", &PyInferenceManager::RegisterModelByPath)
        .def("update_resources", &PyInferenceManager::AllocateResources)
        .def("infer_runner", &PyInferenceManager::InferRunner)
        .def("get_model", &PyInferenceManager::GetModel)
        .def("get_models", &PyInferenceManager::Models)
        .def("serve", &PyInferenceManager::Serve, py::arg("port") = 50052);
    // py::call_guard<py::gil_scoped_release>());

    py::class_<PyRemoteInferenceManager, std::shared_ptr<PyRemoteInferenceManager>>(
        m, "RemoteInferenceManager")
        .def(py::init(
            [](py::kwargs kwargs) { return std::make_shared<PyRemoteInferenceManager>(kwargs); }))
        .def("get_models", &PyRemoteInferenceManager::Models)
        .def("infer_runner", &PyRemoteInferenceManager::InferRunner);

    py::class_<PyInferRunner, std::shared_ptr<PyInferRunner>>(m, "InferRunner")
        .def("infer", &PyInferRunner::Infer, py::call_guard<py::gil_scoped_release>())
        .def("input_bindings", &PyInferRunner::InputBindings)
        .def("output_bindings", &PyInferRunner::OutputBindings)
        .def("max_batch_size", &PyInferRunner::MaxBatchSize);
    //      .def("__repr__", [](const PyInferRunner& obj) {
    //          return obj.Description();
    //      });

    py::class_<PyInferRemoteRunner, std::shared_ptr<PyInferRemoteRunner>>(m, "InferRemoteRunner")
        .def("infer", &PyInferRemoteRunner::Infer)
        .def("input_bindings", &PyInferRemoteRunner::InputBindings)
        .def("output_bindings", &PyInferRemoteRunner::OutputBindings);

    py::class_<PyInferFuture, std::shared_ptr<PyInferFuture>>(m, "InferFuture")
        .def("wait", &std::shared_future<typename PyInferRunner::InferResults>::wait,
             py::call_guard<py::gil_scoped_release>())
        .def("get", &std::shared_future<typename PyInferRunner::InferResults>::get,
             py::call_guard<py::gil_scoped_release>());

    py::class_<CoreMemory, std::shared_ptr<CoreMemory>>(m, "CoreMemory")
        .def("to_dlpack", [](py::object self) { return DLPack::Export(self); })
        .def("__repr__",
             [](const CoreMemory& mem) { return "<trtlab.Memory: " + mem.Description() + ">"; });

    py::class_<HostMemory, std::shared_ptr<HostMemory>, CoreMemory>(m, "HostMemory")
        .def("to_numpy", [](py::object self) { return NumPy::Export(self); })
        .def("__repr__", [](const HostMemory& mem) {
            return "<trtlab.HostMemory: " + mem.Description() + ">";
        });

    py::class_<DeviceMemory, std::shared_ptr<DeviceMemory>, CoreMemory>(m, "DeviceMemory")
        .def("__repr__", [](const DeviceMemory& mem) {
            return "<trtlab.DeviceMemory: " + mem.Description() + ">";
        });

    m.def("test_from_dlpack", [](py::capsule obj) {
        auto core = DLPack::Import(obj);
        // release gil for testing
        DLOG(INFO) << "CoreMemory Descriptor: " << *core;
        DLOG(INFO) << "Dropping the GIL";
        py::gil_scoped_release release;
        GetThreadPool().enqueue([core] {
            DLOG(INFO) << "Holding the descriptor on another thread for 5 seconds";
            DLOG(INFO)
                << "You can deference the dlpack object you gave me; I own a reference to it";
            std::this_thread::sleep_for(std::chrono::seconds(5));
            DLOG(INFO) << "Thread that could have been doing work on the data complete";
        });
    });

    m.def("from_dlpack", [](py::capsule obj) {
        auto core = DLPack::Import(obj);
        return core;
    });

    m.def("malloc", [](int64_t size) {
        std::shared_ptr<HostMemory> mem = std::make_shared<Allocator<Malloc>>(size);
        return mem;
    });

    m.def("cuda_malloc_host", [](int64_t size) {
        std::shared_ptr<HostMemory> mem = std::make_shared<Allocator<CudaPinnedHostMemory>>(size);
        return mem;
    });

    m.def("cuda_malloc", [](int64_t size) {
        std::shared_ptr<DeviceMemory> mem = std::make_shared<Allocator<CudaDeviceMemory>>(size);
        return mem;
    });

    m.def("dlpack_from_malloc",
          [](int64_t size) { return DLPack::Export(std::make_shared<Allocator<Malloc>>(size)); });

    m.def("dlpack_from_cuda_malloc", [](int64_t size) {
        return DLPack::Export(std::move(std::make_unique<Allocator<CudaDeviceMemory>>(size)));
    });

    m.def("tt", []() {
        py::print("hi");
    });

    /*
        py::class_<InferBench, std::shared_ptr<InferBench>>(m, "InferBench")
            .def(py::init([](std::shared_ptr<PyInferenceManager> man) {
                return std::make_shared<InferBench>(man);
            }))
            .def("run", py::overload_cast<std::shared_ptr<Model>, uint32_t,
       double>(&InferBench::Run));

        py::class_<Model, std::shared_ptr<Model>>(m, "Model");

        py::class_<typename InferBench::Results>(m, "InferBenchResults");
    */
    // py::bind_map<std::map<std::string, double>>(m, "InferBenchResults");
}
