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

#include "trtlab/core/pool.h"
#include "trtlab/core/resources.h"
#include "trtlab/core/thread_pool.h"

using trtlab::Resources;
using trtlab::ThreadPool;

#include "nvrpc/executor.h"
#include "nvrpc/server.h"
#include "nvrpc/service.h"

using nvrpc::AsyncRPC;
using nvrpc::AsyncService;
using nvrpc::Context;
using nvrpc::Executor;
using nvrpc::Server;

#include "api.grpc.pb.h"
#include "api.pb.h"

using trtlab::deploy::image_client::Classifications;
using trtlab::deploy::image_client::Detections;
using trtlab::deploy::image_client::ImageInfo;
using trtlab::deploy::image_client::Inference;

// CLI Options
DEFINE_string(hostname, "localhost", "Hostname");
DEFINE_string(ip_port, "0.0.0.0:50051", "IP/Port on which to listen");

class TestResources : public Resources
{
  public:
    TestResources(const std::string& hostname) : m_Hostname(hostname) {}
    const std::string& Hostname() const { return m_Hostname; }

  private:
    std::string m_Hostname;
};

template<typename Output>
class TestContext final : public Context<ImageInfo, Output, TestResources>
{
    void ExecuteRPC(ImageInfo& input, Output& output) final override
    {
        LOG(INFO) << input.model_name() << " served by " << this->GetResources()->Hostname();
        output.set_image_uuid(this->GetResources()->Hostname());
        this->FinishResponse();
    }
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("test_deploy_client");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    Server server(FLAGS_ip_port);
    auto service = server.RegisterAsyncService<Inference>();
    auto rpc_classify = service->RegisterRPC<TestContext<Classifications>>(
        &Inference::AsyncService::RequestClassify);
    auto rpc_detection =
        service->RegisterRPC<TestContext<Detections>>(&Inference::AsyncService::RequestDetection);

    auto resources = std::make_shared<TestResources>(FLAGS_hostname);
    auto executor = server.RegisterExecutor(new Executor(1));

    executor->RegisterContexts(rpc_classify, resources, 1);
    executor->RegisterContexts(rpc_detection, resources, 1);

    server.Run(std::chrono::milliseconds(2000), [] {});
}
