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
#include "common.h"

// clang-format off
struct SimpleResources : public Resources
{
    SimpleResources(int numThreadsInPool = 3) : m_ThreadPool(numThreadsInPool) {}
    ThreadPool& AcquireThreadPool() { return m_ThreadPool; }
  private:
    ThreadPool m_ThreadPool;
};
// clang-format on

class SimpleContext final : public StreamingContext<simple::Input, simple::Output, SimpleResources>
{
    void RequestReceived(RequestType&& input, std::shared_ptr<ServerStream> stream) final override
    {
        LOG_FIRST_N(INFO, 10) << "BatchID: " << input.batch_id() << " Tag = " << Tag()
                              << " Thread = " << std::this_thread::get_id();

        // If even, send back two responses.
        // If odd, do nothing
        if(input.batch_id() % 2 == 0)
        {
            LOG_FIRST_N(INFO, 5) << "Received Even an BatchID: Sending back two responses";
            for(int i = input.batch_id() - 1; i <= input.batch_id(); i++)
            {
                ResponseType output;
                output.set_batch_id(i);
                stream->WriteResponse(std::move(output));
            }
        }
        else
        {
            LOG_FIRST_N(INFO, 5) << "Received an Odd BatchID: No Response will be sent";
        }
    }
};

// CLI Options
DEFINE_int32(thread_count, 1, "Size of thread pool");
DEFINE_string(ip_port, "0.0.0.0:50051", "IP/Port");

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("simpleServer");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    Server server(FLAGS_ip_port);
    auto simpleInference = server.RegisterAsyncService<simple::Inference>();
    auto rpcCompute = simpleInference->RegisterRPC<SimpleContext>(
        &simple::Inference::AsyncService::RequestBidirectional);
    auto rpcResources = std::make_shared<SimpleResources>(FLAGS_thread_count);
    auto executor = server.RegisterExecutor(new Executor(1));
    executor->RegisterContexts(rpcCompute, rpcResources, 10);
    server.Run(std::chrono::milliseconds(2000), [] {});
}
