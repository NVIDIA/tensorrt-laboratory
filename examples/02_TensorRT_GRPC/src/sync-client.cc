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
 *
 * Original Copyright proivded below.
 * This work extends the original gRPC client examples to work with the
 * implemented server.
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "inference.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using ssd::BatchInput;
using ssd::BatchPredictions;
using ssd::Inference;

class SimpleClient
{
  public:
    SimpleClient(std::shared_ptr<Channel> channel) : stub_(Inference::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    int Compute(const int batch_id, const int batch_size)
    {
        // Data we are sending to the server.
        BatchInput request;
        request.set_batch_id(batch_id);
        request.set_batch_size(batch_size);

        // Container for the data we expect from the server.
        BatchPredictions reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->Compute(&context, request, &reply);

        // Act upon its status.
        if(status.ok())
        {
            return reply.batch_id();
        }
        else
        {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return -1;
        }
    }

  private:
    std::unique_ptr<Inference::Stub> stub_;
};

DEFINE_int32(count, 1000, "number of grpc messages to send");
DEFINE_int32(port, 50051, "server_port");
DEFINE_int32(batch, 1, "batch size");

int main(int argc, char** argv)
{
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint (in this case,
    // localhost at port 50051). We indicate that the channel isn't authenticated
    // (use of InsecureChannelCredentials()).
    FLAGS_alsologtostderr = 1; // It will dump to console
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    std::ostringstream ip_port;
    ip_port << "localhost:" << FLAGS_port;
    SimpleClient client(grpc::CreateChannel(ip_port.str(), grpc::InsecureChannelCredentials()));
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < FLAGS_count; i++)
    {
        auto reply = client.Compute(i, FLAGS_batch);
        if(reply == -1 || reply != i) std::cout << "BatchId received: " << reply << std::endl;
    }
    auto end = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(end - start).count();
    std::cout << FLAGS_count << " requests in " << elapsed
              << " seconds; inf/sec: " << FLAGS_count * FLAGS_batch / elapsed << std::endl;
    return 0;
}
