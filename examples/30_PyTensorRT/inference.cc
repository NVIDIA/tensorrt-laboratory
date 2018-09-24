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
#include "YAIS/YAIS.h"
#include "YAIS/TensorRT/TensorRT.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using yais::InheritableResources;
using yais::ThreadPool;
using yais::TensorRT::Bindings;
using yais::TensorRT::ManagedRuntime;
using yais::TensorRT::ResourceManager;
using yais::TensorRT::Runtime;

class Inference;

class InferenceManager : public ResourceManager
{
  public:
    InferenceManager(int max_executions, int max_buffers, int nCuda, int nResp)
        : ResourceManager(max_executions, max_buffers),
          m_CudaThreadPool(std::make_unique<ThreadPool>(nCuda)),
          m_ResponseThreadPool(std::make_unique<ThreadPool>(nResp)),
          m_PreprocessThreadPool(std::make_unique<ThreadPool>(3)) {}

    ~InferenceManager() override {}

    std::shared_ptr<Inference> RegisterModelByPath(std::string path, std::string name)
    {
        auto model = Runtime::DeserializeEngine(path);
        RegisterModel(name, model);
        return std::make_shared<Inference>(name, casted_shared_from_this<InferenceManager>());
    }

    std::unique_ptr<ThreadPool> &GetCudaThreadPool() { return m_CudaThreadPool; }
    std::unique_ptr<ThreadPool> &GetResponseThreadPool() { return m_ResponseThreadPool; }
    std::unique_ptr<ThreadPool> &GetPreprocessThreadPool() { return m_PreprocessThreadPool; }

  private:
    std::unique_ptr<ThreadPool> m_CudaThreadPool;
    std::unique_ptr<ThreadPool> m_ResponseThreadPool;
    std::unique_ptr<ThreadPool> m_PreprocessThreadPool;
};

class Inference : public std::enable_shared_from_this<Inference>
{

  public:
    Inference(std::string model_name, std::shared_ptr<InferenceManager> resources)
        : m_Resources(resources), m_ModelName(model_name) {}

    class Result
    {
    };
    using FutureResult = std::future<std::shared_ptr<Result>>;

    FutureResult PyData(py::array_t<float> &data)
    {
        // auto buffer = data.request();
        auto infer = std::make_shared<Inference>(m_ModelName, m_Resources);
        // GetResources()->GetPreprocessThreadPool()->enqueue([this, infer, &data] {
            auto bindings = GetBindings();
            CHECK_EQ(data.shape(0), bindings->GetModel()->GetMaxBatchSize());
            LOG(INFO) << "data from python is " << data.nbytes() << " bytes";
            // bindings->SetHostAddress(0, (void *)data.data());
            auto pinned = bindings->HostAddress(0);
            std::memcpy(pinned, data.data(), data.nbytes());
            Infer(infer, bindings);
        //});
        return infer->GetFuture();
    }

    std::shared_ptr<Bindings> GetBindings()
    {
        auto model = GetResources()->GetModel(m_ModelName);
        auto buffers = GetResources()->GetBuffers(); // <=== Limited Resource; May Block !!!
        return buffers->CreateAndConfigureBindings(model, model->GetMaxBatchSize());
    }

    FutureResult Compute()
    {
        auto infer = std::make_shared<Inference>(m_ModelName, m_Resources);
        auto bindings = GetBindings();
        Infer(infer, bindings);
        return infer->GetFuture();
    }

    FutureResult GetFuture()
    {
        return m_Promise.get_future();
    }

    static void Infer(std::shared_ptr<Inference> infer, std::shared_ptr<Bindings> bindings)
    {
        infer->GetResources()->GetCudaThreadPool()->enqueue([infer, bindings]() mutable {
            DLOG(INFO) << "cuda launch thread";
            bindings->CopyToDevice(bindings->InputBindings());
            auto trt = infer->GetResources()->GetExecutionContext(bindings->GetModel()); // <=== Limited Resource; May Block !!!
            trt->Infer(bindings);
            bindings->CopyFromDevice(bindings->OutputBindings());
            DLOG(INFO) << "cuda launch thread finished";

            infer->GetResources()->GetResponseThreadPool()->enqueue([infer, bindings, trt]() mutable {
                DLOG(INFO) << "response thread starting";
                // This thread waits on the completion of the async compute and the async copy
                trt->Synchronize();
                trt.reset(); // Finished with the Execution Context - Release it to competing threads
                bindings->Synchronize();
                bindings.reset(); // Finished with Buffers - Release it to competing threads
                infer->m_Promise.set_value(std::make_shared<Result>());
                DLOG(INFO) << "response thread finished - inference done";
            });
        });
    }

  protected:
    inline std::shared_ptr<InferenceManager> GetResources() { return m_Resources; }

  private:
    std::string m_ModelName;
    std::shared_ptr<InferenceManager> m_Resources;
    std::promise<std::shared_ptr<Result>> m_Promise;
};

PYBIND11_MODULE(py_yais, m)
{
    py::class_<InferenceManager, std::shared_ptr<InferenceManager>>(m, "InferenceManager")
        .def(py::init([](int concurrency) {
            return std::make_shared<InferenceManager>(concurrency, concurrency + 6, 1, 3);
        }))
        .def("register_tensorrt_engine", &InferenceManager::RegisterModelByPath)
        .def("allocate_resources", &InferenceManager::AllocateResources);

    py::class_<Inference, std::shared_ptr<Inference>>(m, "Inference")
        .def("pyinfer", &Inference::PyData, py::call_guard<py::gil_scoped_release>())
        .def("compute", &Inference::Compute, py::call_guard<py::gil_scoped_release>());

    py::class_<Inference::FutureResult>(m, "InferenceFutureResult")
        .def("wait", &Inference::FutureResult::wait, py::call_guard<py::gil_scoped_release>())
        .def("get", &Inference::FutureResult::get, py::call_guard<py::gil_scoped_release>());

    py::class_<Inference::Result, std::shared_ptr<Inference::Result>>(m, "InferenceResult");
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
