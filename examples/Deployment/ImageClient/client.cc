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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "client.h"

using namespace trtlab;
using namespace nvrpc;

namespace py = pybind11;

using deploy::image_client::Classifications;
using deploy::image_client::Detections;
using deploy::image_client::ImageInfo;
using deploy::image_client::Inference;

ImageClient::ImageClient(std::string hostname)
{
    auto executor = std::make_shared<client::Executor>(1);

    auto channel = grpc::CreateChannel(hostname, grpc::InsecureChannelCredentials());
    std::shared_ptr<Inference::Stub> stub = Inference::NewStub(channel);

    auto classify_prepare_fn = [stub](::grpc::ClientContext * context, const ImageInfo& request,
                                      ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(stub->PrepareAsyncClassify(context, request, cq));
    };

    auto detection_prepare_fn = [stub](::grpc::ClientContext * context, const ImageInfo& request,
                                       ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(stub->PrepareAsyncDetection(context, request, cq));
    };

    m_ClassifyClient = std::make_unique<client::ClientUnary<ImageInfo, Classifications>>(
        classify_prepare_fn, executor);
    m_DetectionClient = std::make_unique<client::ClientUnary<ImageInfo, Detections>>(
        detection_prepare_fn, executor);
}

std::shared_future<ClassifyResult> ImageClient::Classify(const std::string& model_name,
                                                         const std::string& image_uuid)
{
    ImageInfo image_info;
    image_info.set_model_name(model_name);
    image_info.set_image_uuid(image_uuid);
    std::map<std::string, std::string> headers = {{"custom-metadata-model-name", model_name}};
    auto post = [](ImageInfo& input, Classifications& output,
                   ::grpc::Status& status) -> ClassifyResult {
        ClassifyResult results(output);
        return std::move(results);
    };
    return m_ClassifyClient->Enqueue(std::move(image_info), post, headers);
}

std::shared_future<DetectionResult> ImageClient::Detection(const std::string& model_name,
                                                           const std::string& image_uuid)
{
    ImageInfo image_info;
    image_info.set_model_name(model_name);
    image_info.set_image_uuid(image_uuid);
    std::map<std::string, std::string> headers = {{"custom-metadata-model-name", model_name}};
    auto post = [](ImageInfo& input, Detections& output,
                   ::grpc::Status& status) -> DetectionResult {
        DetectionResult results(output);
        return std::move(results);
    };
    return m_DetectionClient->Enqueue(std::move(image_info), post, headers);
}

ClassifyResult::ClassifyResult(const ::trtlab::deploy::image_client::Classifications& pb)
    : m_UUID(pb.image_uuid())
{
}

DetectionResult::DetectionResult(const ::trtlab::deploy::image_client::Detections& pb)
    : m_UUID(pb.image_uuid())
{
}

using PyClassifyFuture = std::shared_future<ClassifyResult>;
using PyDetectionFuture = std::shared_future<DetectionResult>;

PYBIND11_MAKE_OPAQUE(PyClassifyFuture);
PYBIND11_MAKE_OPAQUE(PyDetectionFuture);

PYBIND11_MODULE(deploy_image_client, m)
{
    py::class_<ImageClient, std::shared_ptr<ImageClient>>(m, "ImageClient")
        .def(py::init<std::string>(), py::arg("hostname") = "trt.lab")
        .def("classify", &ImageClient::Classify)
        .def("detection", &ImageClient::Detection);

    py::class_<PyClassifyFuture, std::shared_ptr<PyClassifyFuture>>(m, "ClassifyFuture")
        .def("wait", &PyClassifyFuture::wait, py::call_guard<py::gil_scoped_release>())
        .def("get", &PyClassifyFuture::get, py::call_guard<py::gil_scoped_release>());

    py::class_<PyDetectionFuture, std::shared_ptr<PyDetectionFuture>>(m, "DetectionFuture")
        .def("wait", &PyDetectionFuture::wait, py::call_guard<py::gil_scoped_release>())
        .def("get", &PyDetectionFuture::get, py::call_guard<py::gil_scoped_release>());

    py::class_<ClassifyResult, std::shared_ptr<ClassifyResult>>(m, "ClassifyResult")
        .def_property_readonly("uuid", &ClassifyResult::UUID);

    py::class_<DetectionResult, std::shared_ptr<DetectionResult>>(m, "DetectionResult")
        .def_property_readonly("uuid", &DetectionResult::UUID);
}