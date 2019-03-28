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

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <thread>

#include "inference.grpc.pb.h"

#include "trtlab/core/utils.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using ssd::BatchInput;
using ssd::BatchPredictions;
using ssd::Inference;

static int g_BatchSize = 1;

class GreeterClient
{
  public:
    explicit GreeterClient(std::shared_ptr<Channel> channel, int max_outstanding)
        : stub_(Inference::NewStub(channel)), m_OutstandingMessageCount(0),
          m_MaxOutstandingMessageCount(max_outstanding)
    {
    }

    // Assembles the client's payload and sends it to the server.
    void SayHello(const size_t batch_id, const int batch_size, char* bytes, uint64_t total)
    {
        // Data we are sending to the server.
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            m_OutstandingMessageCount++;
            while(m_OutstandingMessageCount >= m_MaxOutstandingMessageCount)
            {
                LOG_FIRST_N(WARNING, 10) << "Initiated Backoff - (Siege Rate > Server Compute "
                                            "Rate) - Server Queues are full.";
                m_Condition.wait(lock);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        BatchInput request;
        request.set_batch_id(batch_id);
        request.set_batch_size(batch_size);
        if(total)
        {
            request.set_data(bytes, total);
        }

        // Call object to store rpc data
        AsyncClientCall* call = new AsyncClientCall;

        // stub_->PrepareAsyncSayHello() creates an RPC object, returning
        // an instance to store in "call" but does not actually start the RPC
        // Because we are using the asynchronous API, we need to hold on to
        // the "call" instance in order to get updates on the ongoing RPC.
        call->response_reader = stub_->PrepareAsyncCompute(&call->context, request, &cq_);

        // StartCall initiates the RPC call
        call->response_reader->StartCall();

        // Request that, upon completion of the RPC, "reply" be updated with the
        // server's response; "status" with the indication of whether the operation
        // was successful. Tag the request with the memory address of the call object.
        call->response_reader->Finish(&call->reply, &call->status, (void*)call);

        float elapsed =
            std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
        m_RequestCalls++;
        m_TotalRequestTime += elapsed;
        // LOG_EVERY_N(INFO, 200) << "Request overhead: " << m_TotalRequestTime/m_RequestCalls;
    }

    // Loop while listening for completed responses.
    // Prints out the response from the server.
    void AsyncCompleteRpc()
    {
        void* got_tag;
        bool ok = false;
        size_t cntr = 0;
        auto start = std::chrono::steady_clock::now();
        float last = 0.0;

        // Block until the next result is available in the completion queue "cq".
        while(cq_.Next(&got_tag, &ok))
        {
            // The tag in this example is the memory location of the call object
            AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);

            // Verify that the request was completed successfully. Note that "ok"
            // corresponds solely to the request for updates introduced by Finish().
            GPR_ASSERT(ok);

            if(call->status.ok())
            {
                // std::cout << "Greeter received: " << call->reply.batch_id() << std::endl;
            }
            else
            {
                std::cout << "RPC failed" << std::endl;
            }
            // Once we're complete, deallocate the call object.
            delete call;

            cntr++;
            float elapsed =
                std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
            if(elapsed - last > 0.5)
            {
                LOG(INFO) << "avg. rate: " << (float)cntr / (elapsed - last) << "( "
                          << (float)(cntr * g_BatchSize) / (elapsed - last) << " inf/sec)";
                last = elapsed;
                cntr = 0;
            }

            {
                std::unique_lock<std::mutex> lock(m_Mutex);
                m_OutstandingMessageCount--;
            }
            m_Condition.notify_one();
        }
    }

    void Shutdown() { cq_.Shutdown(); }

  private:
    // struct for keeping state and data information
    struct AsyncClientCall
    {
        // Container for the data we expect from the server.
        BatchPredictions reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // Storage for the status of the RPC upon completion.
        Status status;

        std::unique_ptr<ClientAsyncResponseReader<BatchPredictions>> response_reader;
    };

    // Out of the passed in Channel comes the stub, stored here, our view of the
    // server's exposed services.
    std::unique_ptr<Inference::Stub> stub_;

    // The producer-consumer queue we use to communicate asynchronously with the
    // gRPC runtime.
    CompletionQueue cq_;

    // mutex to help control rate
    std::mutex m_Mutex;
    std::condition_variable m_Condition;
    int m_OutstandingMessageCount;
    int m_MaxOutstandingMessageCount;
    float m_TotalRequestTime;
    size_t m_RequestCalls;
};

static bool ValidateBytes(const char* flagname, const std::string& value)
{
    trtlab::StringToBytes(value);
    return true;
}

DEFINE_int32(count, 1000000, "number of grpc messages to send");
DEFINE_int32(batch_size, 1, "batch_size");
DEFINE_int32(max_outstanding, 950, "maximum outstanding requests");
DEFINE_int32(port, 50051, "server_port");
DEFINE_double(rate, 1.0, "messages per second");
DEFINE_double(max_rate, 100000, "maximum number of messages per second when func is applied");
DEFINE_double(alpha, 0, "alpha");
DEFINE_double(beta, 1, "beta");
DEFINE_string(func, "constant", "constant, linear or cyclic");
DEFINE_string(bytes, "0B", "add extra bytes to the request payload");
DEFINE_validator(bytes, &ValidateBytes);

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1; // It will dump to console
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    g_BatchSize = FLAGS_batch_size;

    auto bytes = trtlab::StringToBytes(FLAGS_bytes);
    char extra_bytes[bytes];
    if(bytes)
        LOG(INFO) << "Sending an addition " << trtlab::BytesToString(bytes)
                  << " bytes in request payload";

    // using a fixed rate of 15us per rpc call.  i could adjust dynamically as i'm tracking
    // the call overhead, but it's close enough.
    auto start = std::chrono::system_clock::now();
    auto walltime = [start]() -> double {
        return std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
    };
    std::map<std::string, std::function<double()>> rates_by_name;
    rates_by_name["constant"] = []() -> double { return std::min(FLAGS_rate, FLAGS_max_rate); };
    rates_by_name["linear"] = [start, walltime]() -> double {
        return std::min(FLAGS_rate + (FLAGS_alpha / 60.0) * walltime(), FLAGS_max_rate);
    };
    rates_by_name["cyclic"] = [start, walltime]() -> double {
        return std::min(FLAGS_rate + FLAGS_alpha *
                                         std::sin(2.0 * 3.14159 * (FLAGS_beta / 60.0) * walltime()),
                        FLAGS_max_rate);
    };
    auto search = rates_by_name.find(FLAGS_func);
    if(search == rates_by_name.end())
    {
        LOG(FATAL) << "--func must be constant, linear or cyclic; your value = " << FLAGS_func;
    }
    auto sleepy = [search]() -> double {
        auto sleep_time = ((std::chrono::seconds(1) / std::max((search->second)(), 2.0))) -
                          std::chrono::microseconds(15);
        return std::chrono::duration<double>(sleep_time).count();
    };

    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint (in this case,
    // localhost at port 50051). We indicate that the channel isn't authenticated
    // (use of InsecureChannelCredentials()).
    std::ostringstream ip_port;
    ip_port << "localhost:" << FLAGS_port;

    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    GreeterClient greeter(
        grpc::CreateCustomChannel(ip_port.str(), grpc::InsecureChannelCredentials(), ch_args),
        FLAGS_max_outstanding);

    // Spawn reader thread that loops indefinitely
    std::thread thread_ = std::thread(&GreeterClient::AsyncCompleteRpc, &greeter);

    for(size_t i = 0; i < FLAGS_count; i++)
    {
        greeter.SayHello(i, FLAGS_batch_size, extra_bytes, bytes); // The actual RPC call!
        auto start = std::chrono::high_resolution_clock::now();
        while(std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start)
                  .count() < sleepy())
        {
            std::this_thread::yield();
        }
    }

    greeter.Shutdown();
    thread_.join(); // blocks forever
    auto elapsed = walltime();
    std::cout << FLAGS_count << " requests in " << elapsed
              << "seconds; inf/sec: " << FLAGS_count * FLAGS_batch_size / elapsed << std::endl;

    return 0;
}
