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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <future>
#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tensorrt/playground/bindings.h"
#include "tensorrt/playground/core/async_compute.h"
#include "tensorrt/playground/core/thread_pool.h"
#include "tensorrt/playground/infer_runner.h"
#include "tensorrt/playground/inference_manager.h"
#include "tensorrt/playground/model.h"
#include "tensorrt/playground/runtime.h"

#include "tensorrt/playground/python/tensorrt/utils.h"

using namespace yais;
using namespace yais::Memory;
using namespace yais::TensorRT;

class PyInferRunner;

class PyInferenceManager final : public InferenceManager
{
  public:
    PyInferenceManager(int max_executions, int max_buffers)
        : InferenceManager(max_executions, max_buffers)
    {
    }

    static std::shared_ptr<PyInferenceManager> Init(py::kwargs kwargs)
    {
        int max_executions = 1;
        int max_buffers = 0;
        size_t max_allocation_size = 0;

        int pre_thread_count = 1;
        int cuda_thread_count = 1;
        int post_thread_count = 3;

        for(const auto& item : kwargs)
        {
            auto key = py::cast<std::string>(item.first);
            if(key == "max_executions")
            {
                max_executions = py::cast<int>(item.second);
            }
            else if(key == "max_buffers")
            {
                max_buffers = py::cast<int>(item.second);
            }
            else if(key == "max_allocation_size")
            {
                max_allocation_size = py::cast<size_t>(item.second);
            }
            else if(key == "pre_thread_count")
            {
                pre_thread_count = py::cast<int>(item.second);
            }
            else if(key == "cuda_thread_count")
            {
                cuda_thread_count = py::cast<int>(item.second);
            }
            else if(key == "post_thread_count")
            {
                post_thread_count = py::cast<int>(item.second);
            }
            else
            {
                throw std::runtime_error("Unknown keyword: " + key);
            }
        }

        if(max_buffers == 0)
        {
            max_buffers = max_executions + 3;
        }

        auto manager = std::make_unique<PyInferenceManager>(max_executions, max_buffers);
        manager->RegisterThreadPool("pre", std::make_unique<ThreadPool>(pre_thread_count));
        manager->RegisterThreadPool("cuda", std::make_unique<ThreadPool>(cuda_thread_count));
        manager->RegisterThreadPool("post", std::make_unique<ThreadPool>(post_thread_count));
        return manager;
    }

    ~PyInferenceManager() override {}

    std::shared_ptr<PyInferRunner> RegisterModelByPath(const std::string& name,
                                                       const std::string& path)
    {
        auto model = Runtime::DeserializeEngine(path);
        RegisterModel(name, model);
        return this->InferRunner(name);
    }

    std::shared_ptr<PyInferRunner> InferRunner(std::string name)
    {
        return std::make_shared<PyInferRunner>(GetModel(name),
                                               casted_shared_from_this<PyInferenceManager>());
    }
};

struct PyInferRunner : public InferRunner
{
    using InferRunner::InferRunner;
    using InferResults = py::dict;

    auto Infer(py::kwargs kwargs)
    {
        const auto& model = GetModel();
        auto bindings = InitializeBindings();
        int batch_size = -1;
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
                    DLOG(INFO) << "Inferred batch_size=" << batch_size << " from dimensions";
                    batch_size = data.shape(0);
                }
                else
                {
                    CHECK_EQ(data.shape(0), batch_size);
                }
                CHECK_EQ(data.nbytes(), binding.bytesPerBatchItem * batch_size);
                auto id = model.BindingId(key);
                auto host = bindings->HostAddress(id);
                // TODO: enhance the Copy method for py::buffer_info objects
                std::memcpy(host, data.data(), data.nbytes());
            }
        }
        // py::gil_scoped_release release;
        bindings->SetBatchSize(batch_size);
        return InferRunner::Infer(
            bindings, [](std::shared_ptr<Bindings>& bindings) -> InferResults {
                // py::gil_scoped_acquire acquire;
                auto results = InferResults();
                DLOG(INFO) << "Copying Output Bindings to Numpy arrays";
                for(const auto& id : bindings->OutputBindings())
                {
                    const auto& binding = bindings->GetModel()->GetBinding(id);
                    DLOG(INFO) << "Processing binding: " << binding.name << "with index " << id;
                    auto value =
                        py::array_t<float>(binding.dims);
                        // py::array_t<float>(binding.elementsPerBatchItem * bindings->BatchSize());
                    // TODO: set shape dimensions on the numpy array
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
        for (const auto& id : GetModel().GetInputBindingIds())
        {
            AddBindingInfo(dict, id);
        }
        return dict;
    }

    py::dict OutputBindings() const
    {
        auto dict = py::dict();
        for (const auto& id : GetModel().GetOutputBindingIds())
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

PYBIND11_MODULE(infer, m)
{
    py::class_<PyInferenceManager, std::shared_ptr<PyInferenceManager>>(m, "InferenceManager")
        .def(py::init([](py::kwargs kwargs) { return PyInferenceManager::Init(kwargs); }))
        .def("register_tensorrt_engine", &PyInferenceManager::RegisterModelByPath)
        .def("update_resources", &PyInferenceManager::AllocateResources)
        .def("infer_runner", &PyInferenceManager::InferRunner);

    py::class_<PyInferRunner, std::shared_ptr<PyInferRunner>>(m, "InferRunner")
        .def("infer", &PyInferRunner::Infer)
        .def("input_bindings", &PyInferRunner::InputBindings)
        .def("output_bindings", &PyInferRunner::OutputBindings); //, py::call_guard<py::gil_scoped_release>());

    py::class_<std::shared_future<typename PyInferRunner::InferResults>>(m, "InferenceFutureResult")
        .def("wait", &std::shared_future<typename PyInferRunner::InferResults>::wait) // py::call_guard<py::gil_scoped_release>())
        .def("get", &std::shared_future<typename PyInferRunner::InferResults>::get); // py::call_guard<py::gil_scoped_release>());

}
