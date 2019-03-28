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
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

#include "nvrpc/context.h"
#include "nvrpc/executor.h"
#include "nvrpc/rpc.h"
#include "nvrpc/server.h"
#include "nvrpc/service.h"
#include "trtlab/core/resources.h"
#include "trtlab/core/thread_pool.h"

using nvrpc::AsyncRPC;
using nvrpc::AsyncService;
using nvrpc::BatchingContext;
using nvrpc::Executor;
using nvrpc::Server;
using trtlab::Resources;
using trtlab::ThreadPool;

#include "echo.grpc.pb.h"
#include "echo.pb.h"

class SimpleContext final : public BatchingContext<simple::Input, simple::Output, Resources>
{
    void ExecuteRPC(std::vector<RequestType>& inputs,
                    std::vector<ResponseType>& outputs) final override
    {
        for(auto input = inputs.cbegin(); input != inputs.cend(); input++)
        {
            auto output = outputs.emplace(outputs.end());
            output->set_batch_id(input->batch_id());
            LOG(INFO) << "Response with batch_id=" << output->batch_id();
        }
        this->FinishResponse();
    }

    void OnRequestReceived(const RequestType& request) final override
    {
        LOG(INFO) << "Recieved request with batch_id=" << request.batch_id();
    }
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console

    ::google::InitGoogleLogging("simpleServer");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    Server server("0.0.0.0:50051");

    LOG(INFO) << "Register Service (simple::Inference)";
    auto simpleInference = server.RegisterAsyncService<simple::Inference>();

    LOG(INFO)
        << "Register RPC (simple::Inference::BatchedCompute) with Service (simple::Inference)";
    auto rpcCompute = simpleInference->RegisterRPC<SimpleContext>(
        &simple::Inference::AsyncService::RequestBatchedCompute);

    LOG(INFO) << "Initializing Resources for RPC (simple::Inference::BatchedCompute)";
    auto rpcResources = std::make_shared<Resources>();

    LOG(INFO) << "Creating Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    LOG(INFO) << "Creating Execution Contexts for RPC (simple::Inference::Compute) with Executor";
    executor->RegisterContexts(rpcCompute, rpcResources, 10);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(2000), [] {
        // This is a timeout loop executed every 2seconds
        // Run() with no arguments will run an empty timeout loop every 5 seconds.
        // RunAsync() will return immediately, its your responsibility to ensure the
        // server doesn't go out of scope or a Shutdown will be triggered on your services.
    });
}
