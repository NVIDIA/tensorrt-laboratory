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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <future>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include <sys/stat.h>
#include <unistd.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tensorrt/playground/bindings.h"
#include "tensorrt/playground/core/async_compute.h"
#include "tensorrt/playground/core/memory/descriptor.h"
#include "tensorrt/playground/core/memory/host_memory.h"
#include "tensorrt/playground/core/thread_pool.h"
#include "tensorrt/playground/cuda/memory/device_memory.h"
#include "tensorrt/playground/inference_manager.h"
#include "tensorrt/playground/model.h"
#include "tensorrt/playground/runtime.h"

using namespace yais;
using namespace yais::Memory;
using namespace yais::TensorRT;

class InferModel;
class PyInferModel;

class InferenceManagerImpl : public InferenceManager
{
  public:
    InferenceManagerImpl(int max_executions, int max_buffers)
        : InferenceManager(max_executions, max_buffers)
    {
        SetThreadPool("pre", std::make_unique<ThreadPool>(1));
        SetThreadPool("cuda", std::make_unique<ThreadPool>(1));
        SetThreadPool("post", std::make_unique<ThreadPool>(3));
    }

    ~InferenceManagerImpl() override {}

    std::shared_ptr<PyInferModel> RegisterModelByPath(const std::string& name,
                                                      const std::string& path)
    {
        auto model = Runtime::DeserializeEngine(path);
        RegisterModel(name, model);
        return GetHandler(name);
    }

    std::shared_ptr<PyInferModel> GetHandler(std::string name)
    {
        return std::make_shared<PyInferModel>(GetModel(name),
                                              casted_shared_from_this<InferenceManagerImpl>());
    }

    std::shared_ptr<InferenceManager> cuda()
    {
        AllocateResources();
        return casted_shared_from_this<InferenceManager>();
    }
};

struct InferModel : public AsyncCompute<void(std::shared_ptr<Bindings>&)>
{
    InferModel(std::shared_ptr<Model> model, std::shared_ptr<InferenceManager> resources)
        : m_Model{model}, m_Resources{resources}
    {
    }

    InferModel(InferModel&&) = delete;
    InferModel& operator=(InferModel&&) = delete;

    InferModel(const InferModel&) = delete;
    InferModel& operator=(const InferModel&) = delete;

    virtual ~InferModel() {}

    using BindingsHandle = std::shared_ptr<Bindings>;
    using PreFn = std::function<void(Bindings&)>;

    template<typename Post>
    auto Infer(PreFn pre, Post post)
    {
        auto compute = Wrap(post);
        auto future = compute->Future();
        Enqueue(pre, compute);
        return future.share();
    }

    template<typename Post>
    auto Infer(std::shared_ptr<Bindings> bindings, Post post)
    {
        auto compute = Wrap(post);
        auto future = compute->Future();
        Enqueue(bindings, compute);
        return future.share();
    }

  protected:
    template<typename T>
    void Enqueue(PreFn Pre, std::shared_ptr<AsyncCompute<T>> Post)
    {
        Workers("pre").enqueue([this, Pre, Post]() mutable {
            auto bindings = InitializeBindings();
            Pre(*bindings);
            Enqueue(bindings, Post);
        });
    }

    template<typename T>
    void Enqueue(std::shared_ptr<Bindings> bindings, std::shared_ptr<AsyncCompute<T>> Post)
    {
        Workers("cuda").enqueue([this, bindings, Post]() mutable {
            bindings->CopyToDevice(bindings->InputBindings());
            auto trt_ctx = Compute(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            Workers("post").enqueue([this, bindings, trt_ctx, Post]() mutable {
                trt_ctx->Synchronize();
                trt_ctx.reset();
                bindings->Synchronize();
                (*Post)(bindings);
                bindings.reset();
                LOG(INFO) << "Execute Finished";
            });
        });
    }

    BindingsHandle InitializeBindings()
    {
        auto buffers = m_Resources->GetBuffers();
        return buffers->CreateBindings(m_Model);
    }

    auto Compute(BindingsHandle& bindings) -> std::shared_ptr<ExecutionContext>
    {
        auto trt_ctx = m_Resources->GetExecutionContext(bindings->GetModel());
        trt_ctx->Infer(bindings);
        return trt_ctx;
    }

    inline ThreadPool& Workers(std::string name)
    {
        return m_Resources->GetThreadPool(name);
    }
    /*
        void CopyInputsToInputBindings(const HostMap& inputs, BindingsHandle& bindings)
        {
            for (const auto& id : bindings->InputBindings())
            {
                const auto& b = bindings->GetBindings(id);
                auto search = inputs.find(b.name);
                CHECK(search != inputs.end());
                Copy(bindings->HostMemoryDescriptor(id), *inputs[b.name],
       bindings->BindingSize(id));
            }
        }
    */
/*j
    const Model& Model() const
    {
        return *m_Model;
    }

    const InferenceManager& Resources() const
    {
        return *m_Resources;
    }
*/
    std::shared_ptr<Model> m_Model;
    std::shared_ptr<InferenceManager> m_Resources;
};

struct PyInferModel : public InferModel
{
    using InferModel::InferModel;
    using InferResults = py::dict;

    auto InferData(py::array_t<float> data)
    {
        return InferModel::Infer(
            [data](Bindings& bindings) {
                CHECK_LE(data.shape(0), bindings.GetModel()->GetMaxBatchSize());
                bindings.SetBatchSize(data.shape(0));
                CHECK_EQ(data.nbytes(), bindings.BindingSize(0));
                auto pinned = bindings.HostAddress(0);
                std::memcpy(pinned, data.data(), data.nbytes());
            },
            [](std::shared_ptr<Bindings>& bindings) -> py::array_t<float> {
                const auto& binding = bindings->GetModel()->GetBinding(1);
                auto result =
                    py::array_t<float>(binding.elementsPerBatchItem * bindings->BatchSize());
                py::buffer_info buffer = result.request();
                CHECK_EQ(result.nbytes(), bindings->BindingSize(1));
                std::memcpy(buffer.ptr, bindings->HostAddress(1), result.nbytes());
                LOG(INFO) << "Finshed Post - setting future/promise value";
                return result;
            });
    }

    auto Infer(py::kwargs kwargs)
    {
        auto model = m_Model;
        auto bindings = InitializeBindings();
        int batch_size = -1;
        for(auto item : kwargs)
        {
            auto key = py::cast<std::string>(item.first);
            DLOG(INFO) << "Processing Python Keyword: " << key;
            const auto& binding = model->GetBinding(key);
            LOG_IF(FATAL, !binding.isInput) << item.first << " is not an InputBinding";
            if(binding.isInput)
            {
                CHECK(py::isinstance<py::array>(item.second));
                auto data = py::cast<py::array_t<float>>(item.second);
                CHECK_LE(data.shape(0), model->GetMaxBatchSize());
                if(batch_size == -1)
                {
                    batch_size = data.shape(0);
                }
                else
                {
                    CHECK_EQ(data.shape(0), batch_size);
                }
                CHECK_EQ(data.nbytes(), binding.bytesPerBatchItem * batch_size);
                auto id = model->BindingId(key);
                auto host = bindings->HostAddress(id);
                std::memcpy(host, data.data(), data.nbytes());
            }
        }
        bindings->SetBatchSize(batch_size);
        // return InferModel::Infer(bindings, [](std::shared_ptr<Bindings>& bindings) -> py::dict {
        // return InferModel::Infer(bindings, [](std::shared_ptr<Bindings>& bindings) -> py::array_t<float> {
        // return InferModel::Infer(bindings, [](std::shared_ptr<Bindings>& bindings) -> std::vector<py::array_t<float>> {
        // return InferModel::Infer(bindings, [](std::shared_ptr<Bindings>& bindings) -> py::array_t<float> {
        return InferModel::Infer(bindings, [](std::shared_ptr<Bindings>& bindings) -> InferResults {
            // py::gil_scoped_acquire acquire;
            auto results = InferResults();
            // std::vector<py::array_t<float>> results;
            // py::array_t<float> results;
            for (const auto& id : bindings->OutputBindings())
            {
                const auto& binding = bindings->GetModel()->GetBinding(id);
                auto value = py::array_t<float>(binding.elementsPerBatchItem * bindings->BatchSize());
                py::buffer_info buffer = value.request();
                CHECK_EQ(value.nbytes(), bindings->BindingSize(id));
                std::memcpy(buffer.ptr, bindings->HostAddress(id), value.nbytes());
                py::str key = binding.name;
                results[key] = value;
            }
            //const auto& binding = bindings->GetModel()->GetBinding(1);
            //auto result = py::array_t<float>(binding.elementsPerBatchItem * bindings->BatchSize());
            //py::buffer_info buffer = result.request();
            //CHECK_EQ(result.nbytes(), bindings->BindingSize(1));
            //std::memcpy(buffer.ptr, bindings->HostAddress(1), result.nbytes());
            LOG(INFO) << "Finshed Post - setting future/promise value";
            return results;
        });
    }
};

PYBIND11_MODULE(infer, m)
{
    py::class_<InferenceManagerImpl, std::shared_ptr<InferenceManagerImpl>>(m, "InferenceManager")
        .def(py::init([](int concurrency) {
            return std::make_shared<InferenceManagerImpl>(concurrency, concurrency + 4);
        }))
        .def("register_tensorrt_engine", &InferenceManagerImpl::RegisterModelByPath)
        .def("get_model", &InferenceManagerImpl::GetHandler)
        .def("cuda", &InferenceManagerImpl::cuda);

    py::class_<PyInferModel, std::shared_ptr<PyInferModel>>(m, "Inference")
        .def("infer_data", &PyInferModel::InferData) // , py::call_guard<py::gil_scoped_release>())
        .def("infer", &PyInferModel::Infer); //, py::call_guard<py::gil_scoped_release>());

    py::class_<std::shared_future<typename PyInferModel::InferResults>>(m, "InferenceFutureResult")
        .def("get", &std::shared_future<typename PyInferModel::InferResults>::get, py::call_guard<py::gil_scoped_release>());
        //.def("wait", &std::shared_future<py::array_t<float>>::wait,
        //     py::call_guard<py::gil_scoped_release>())
        // .def("get", &std::shared_future<py::array_t<float>>::get,
             // py::call_guard<py::gil_scoped_release>());
}

static bool ValidateEngine(const char* flagname, const std::string& value)
{
    struct stat buffer;
    return (stat(value.c_str(), &buffer) == 0);
}

DEFINE_string(engine, "/path/to/tensorrt.engine", "TensorRT serialized engine");
DEFINE_validator(engine, &ValidateEngine);
DEFINE_int32(contexts, 1, "Number of Execution Contexts");
DEFINE_int32(buffers, 0, "Number of Buffers (default: 2x contexts)");
DEFINE_int32(prethreads, 1, "Number of preproessing threads");
DEFINE_int32(cudathreads, 1, "Number of cuda kernel launching threads");
DEFINE_int32(postthreads, 3, "Number of postprocessing threads");

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT Inference");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    auto contexts = FLAGS_contexts;
    auto buffers = FLAGS_buffers ? FLAGS_buffers : 2 * FLAGS_contexts;

    auto resources = std::make_shared<InferenceManager>(contexts, buffers);

    resources->SetThreadPool("pre", std::make_unique<ThreadPool>(FLAGS_prethreads));
    resources->SetThreadPool("cuda", std::make_unique<ThreadPool>(FLAGS_cudathreads));
    resources->SetThreadPool("post", std::make_unique<ThreadPool>(FLAGS_postthreads));

    auto model = Runtime::DeserializeEngine(FLAGS_engine);
    resources->RegisterModel("flowers", model);
    resources->AllocateResources();
    LOG(INFO) << "Resources Allocated";

    InferModel flowers(model, resources);

    {
        auto future = flowers.Infer(
            [](Bindings& bindings) {
                // TODO: Copy Input Data to Host Input Bindings
                DLOG(INFO) << "Pre";
                bindings.SetBatchSize(bindings.GetModel()->GetMaxBatchSize());
            },
            [](std::shared_ptr<Bindings>& bindings) -> std::shared_ptr<bool> {
                DLOG(INFO) << "Post";
                return std::make_shared<bool>(false);
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        //  auto result = std::move(future.get());
        auto result = future.get();
        CHECK(result);
        LOG(INFO) << "Waited 1 second to ensure compute cleaned up";
        LOG(INFO) << "Result: " << (*result ? "True" : "False") << " " << result.get();
        return 0;
    }

    return 0;
}