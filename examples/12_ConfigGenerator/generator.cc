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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "YAIS/YAIS.h"
#include "YAIS/TensorRT.h"

// NVIDIA Inference Server Protos
#include "nvidia_inference.pb.h"
#include "nvidia_inference.grpc.pb.h"

using nvidia::inferenceserver::ModelConfig;

using yais::TensorRT::Runtime;

/*
    struct Binding
    {
        bool isInput;
        int dtypeSize;
        size_t bytesPerBatchItem;
        size_t elementsPerBatchItem;
        std::vector<size_t> dims;
    };
*/

static size_t DataTypeToBytes(nvidia::inferenceserver::DataType dataType)
{
    switch (dataType) {
    case nvidia::inferenceserver::TYPE_INVALID:
        CHECK(false) << "Invalid DataType used";
        return 0;
    case nvidia::inferenceserver::TYPE_BOOL:
    case nvidia::inferenceserver::TYPE_UINT8:
    case nvidia::inferenceserver::TYPE_INT8:
        return 1;
    case nvidia::inferenceserver::TYPE_UINT16:
    case nvidia::inferenceserver::TYPE_INT16:
    case nvidia::inferenceserver::TYPE_FP16:
        return 2;
    case nvidia::inferenceserver::TYPE_UINT32:
    case nvidia::inferenceserver::TYPE_INT32:
    case nvidia::inferenceserver::TYPE_FP32:
        return 4;
    case nvidia::inferenceserver::TYPE_UINT64:
    case nvidia::inferenceserver::TYPE_INT64:
    case nvidia::inferenceserver::TYPE_FP64:
        return 8;
    default:
        CHECK(false) << "Invalid DataType used";
        return 0;
    }
}

/*
    kFLOAT = 0, //!< FP32 format.
    kHALF = 1,  //!< FP16 format.
    kINT8 = 2,  //!< quantized INT8 format.
    kINT32 = 3  //!< INT32 format.
*/
static nvidia::inferenceserver::DataType ConvertTensorRTDataType(nvinfer1::DataType trt_datatype)
{
    switch(trt_datatype) {
    case nvinfer1::DataType::kFLOAT:
        return nvidia::inferenceserver::TYPE_FP32;
    case nvinfer1::DataType::kHALF:
        return nvidia::inferenceserver::TYPE_FP16;
    case nvinfer1::DataType::kINT8:
        return nvidia::inferenceserver::TYPE_INT8;
    case nvinfer1::DataType::kINT32:
        return nvidia::inferenceserver::TYPE_INT32;
    default:
        LOG(FATAL) << "Unknown TensorRT DataType";
    }
}

std::string tensorrt_engine(std::string model_name, std::string engine, int concurrency)
{
    ModelConfig config;
    auto model = yais::TensorRT::Runtime::DeserializeEngine(engine);
    config.set_name(model_name);
    config.set_platform("tensorrt_plan");
    config.set_max_batch_size(model->GetMaxBatchSize());

    for(auto i : model->GetInputBindingIds()) {
        const auto& binding = model->GetBinding(i);
        auto input = config.add_input();
        input->set_name(binding.name);
        input->set_data_type(ConvertTensorRTDataType(binding.dtype));
        for(auto d : binding.dims) { input->add_dims(d); }
    }

    for(auto i : model->GetOutputBindingIds()) {
        const auto& binding = model->GetBinding(i);
        auto output = config.add_output();
        output->set_name(binding.name);
        output->set_data_type(ConvertTensorRTDataType(binding.dtype));
        for(auto d : binding.dims) { output->add_dims(d); }
    }

    auto instance_group = config.add_instance_group();
    CHECK(concurrency > 0) << "Concurrency must be >= 0";
    instance_group->set_count(concurrency);
    instance_group->add_gpus(0);

    return config.DebugString();
}

namespace py = pybind11;

PYBIND11_MODULE(config_generator, m) {
    m.doc() = R"pbdoc(
        Pybind11 Yais plugin
        --------------------
        .. currentmodule:: config_generator
        .. autosummary::
           :toctree: _generate
           tensorrt_engine
    )pbdoc";

    m.def("tensorrt_engine", &tensorrt_engine, R"pbdoc(
        Generate a TensorRT Inference Server ModelConfig from a serialized engine file
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

/*
int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT Inference Server Config Generator");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    auto model = Runtime::DeserializeEngine(FLAGS_engine);
    auto model_config = trtis::ModelConfig(model);
    

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
