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
using nvrpc::BidirectionalContext;
using nvrpc::Executor;
using nvrpc::Server;

#include "echo.grpc.pb.h"
#include "echo.pb.h"

// CLI Options
DEFINE_int32(thread_count, 1, "Size of thread pool");

// Define the resources your RPC will need to execute
// ==================================================
// In this case, all simple::Inference::Compute RPCs share a threadpool in which they will
// queue up some work on.  This essentially means, after the message as been received and
// processed, the actual work for the RPC is pushed to a worker pool outside the scope of
// the transaction processing system (TPS).  This is essentially async computing, we have
// decoupled the transaction from the workers executing the implementation.  The TPS can
// continue to queue work, while the workers process the load.
struct SimpleResources : public Resources
{
    SimpleResources(int numThreadsInPool = 3) : m_ThreadPool(numThreadsInPool)
    {
        LOG(INFO) << "Server ThreadCount: " << numThreadsInPool;
    }

    ThreadPool& AcquireThreadPool() { return m_ThreadPool; }

  private:
    ThreadPool m_ThreadPool;
};

// Contexts hold the state and provide the definition of the work to be performed by the RPC.
// This is where you define what gets executed for a given RPC.
// Incoming Message = simple::Input (RequestType)
// Outgoing Message = simple::Output (ResponseType)
class SimpleContext final
    : public BidirectionalContext<simple::Input, simple::Output, SimpleResources>
{
    void ExecuteRPC(RequestType& input, ResponseType& output) final override
    {
        // We could do work here, but we'd block the TPS, i.e. the threads pulling messages
        // off the incoming recieve queue.  Very quick responses are best done here; however,
        // longer running workload should be offloaded so the TPS can avoid being blocked.
        // GetResources()->AcquireThreadPool().enqueue([this, &input, &output]{
        // Now running on a worker thread of the ThreadPool defined in SimpleResources.
        // Here we are just echoing back the incoming // batch_id; however, in later
        // examples, we'll show how to run an async cuda pipline.
        LOG_FIRST_N(INFO, 10) << "BatchID: " << input.batch_id() << " Tag = " << Tag()
                              << " Thread = " << std::this_thread::get_id();
        output.set_batch_id(input.batch_id());
        this->FinishResponse();
        // });
        // The TPS thread is now free to continue processing message - async ftw!
    }
};

DEFINE_string(ip_port, "0.0.0.0:50051", "IP/Port");

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console

    ::google::InitGoogleLogging("simpleServer");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    // A server will bind an IP:PORT to listen on
    Server server(FLAGS_ip_port);

    // A server can host multiple services
    LOG(INFO) << "Register Service (simple::Inference) with Server";
    auto simpleInference = server.RegisterAsyncService<simple::Inference>();

    // An RPC has two components that need to be specified when registering with the service:
    //  1) Type of Execution Context (SimpleContext).  The execution context defines the behavor
    //     of the RPC, i.e. it contains the control logic for the execution of the RPC.
    //  2) The Request function (RequestCompute) which was generated by gRPC when compiling the
    //     protobuf which defined the service.  This function is responsible for queuing the
    //     RPC's execution context to the
    LOG(INFO) << "Register RPC (simple::Inference::Compute) with Service (simple::Inference)";
    auto rpcCompute = simpleInference->RegisterRPC<SimpleContext>(
        &simple::Inference::AsyncService::RequestBidirectional);

    LOG(INFO) << "Initializing Resources for RPC (simple::Inference::Compute)";
    auto rpcResources = std::make_shared<SimpleResources>(FLAGS_thread_count);

    // Create Executors - Executors provide the messaging processing resources for the RPCs
    // Multiple Executors can be registered with a Server.  The executor is responsible
    // for pulling incoming message off the receive queue and executing the associated
    // context.  By default, an executor only uses a single thread.  A typical usecase is
    // an Executor executes a context, which immediate pushes the work to a thread pool.
    // However, for very low-latency messaging, you might want to use a multi-threaded
    // Executor and a Blocking Context - meaning the Context performs the entire RPC function
    // on the Executor's thread.
    LOG(INFO) << "Creating Executor";
    auto executor = server.RegisterExecutor(new Executor(1));

    // You can register RPC execution contexts from any registered RPC on any executor.
    // The power of that will become clear in later examples. For now, we will register
    // 10 instances of the simple::Inference::Compute RPC's SimpleContext execution context
    // with the Executor.
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
