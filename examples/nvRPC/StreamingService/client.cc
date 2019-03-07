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
#include <iostream>
#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "nvrpc/client/client_streaming.h"
#include "nvrpc/client/executor.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using nvrpc::client::ClientStreaming;
using nvrpc::client::Executor;

#include "echo.grpc.pb.h"

using simple::Inference;
using simple::Input;
using simple::Output;

static bool ValidateEven(const char* flagname, int value)
{
    LOG_IF(ERROR, value % 2) << "Examples require an even number of messages";
    return (value % 2 == 0);
}

DEFINE_int32(count, 100, "number of grpc messages to send");
DEFINE_validator(count, &ValidateEven);
DEFINE_int32(thread_count, 1, "Size of thread pool");
DEFINE_string(hostname, "127.0.0.1:50051", "hostname and port");

int main(int argc, char** argv)
{
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint (in this case,
    // localhost at port 50051). We indicate that the channel isn't authenticated
    // (use of InsecureChannelCredentials()).
    FLAGS_alsologtostderr = 1; // It will dump to console
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    std::mutex mutex;
    std::size_t count = 0;

    auto executor = std::make_shared<Executor>(FLAGS_thread_count);

    auto channel = grpc::CreateChannel(FLAGS_hostname, grpc::InsecureChannelCredentials());
    auto stub = Inference::NewStub(channel);

    auto infer_prepare_fn = [&stub](::grpc::ClientContext * context,
                                    ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(stub->PrepareAsyncBidirectional(context, cq));
    };

    auto stream = std::make_unique<ClientStreaming<Input, Output>>(
        infer_prepare_fn, executor,
        [](Input&& request) {
            LOG_FIRST_N(INFO, 10) << "Sent Request with BatchID: " << request.batch_id();
        },
        [&mutex, &count](Output&& response) {
            static size_t last = 0;
            LOG_FIRST_N(INFO, 10) << "Received Response with BatchID: " << response.batch_id();
            CHECK_EQ(++last, response.batch_id());
            std::lock_guard<std::mutex> lock(mutex);
            --count;
        });

    auto start = std::chrono::steady_clock::now();
    auto elapsed = [start]() -> float {
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    };

    for(int i = 1; i < FLAGS_count + 1; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        stream->Write(std::move(input));
    }
    std::cout << FLAGS_count << " queued in " << elapsed() << "seconds" << std::endl;
    auto future = stream->Done();
    // auto future = stream->Status();
    auto status = future.get();
    std::cout << FLAGS_count << " completed in " << elapsed() << "seconds" << std::endl;
    std::cout << "gRPC Status: " << (status.ok() ? "OK" : "NOT OK") << std::endl;
    executor->ShutdownAndJoin();
    CHECK_EQ(count, 0UL);
    return 0;
}