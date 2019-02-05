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
#include "nvrpc/server.h"

#include <thread>
#include <csignal>

#include <glog/logging.h>

namespace {
std::function<void(int)> shutdown_handler;
void signal_handler(int signal) { shutdown_handler(signal); }
} // namespace

namespace nvrpc {

Server::Server(std::string server_address) : m_ServerAddress(server_address), m_Running(false)
{
    LOG(INFO) << "gRPC listening on: " << m_ServerAddress;
    m_Builder.AddListeningPort(m_ServerAddress, ::grpc::InsecureServerCredentials());
}

::grpc::ServerBuilder& Server::Builder()
{
    LOG_IF(FATAL, m_Running) << "Unable to access Builder after the Server is running.";
    return m_Builder;
}

void Server::Run()
{
    Run(std::chrono::milliseconds(1000), [] {});
}

void Server::Run(std::chrono::milliseconds timeout, std::function<void()> control_fn)
{
    m_Running = true;
    auto server = m_Builder.BuildAndStart();
    volatile bool running = true;

    shutdown_handler = [this, &running](int signal) {
        LOG(INFO) << "Trapped Signal: " << signal;
        running = false;
        for(int i = 0; i < m_Executors.size(); i++)
        {
            m_Executors[i]->Shutdown();
        }
        //exit(911);
    };
    std::signal(SIGINT, signal_handler);

    for(int i = 0; i < m_Executors.size(); i++)
    {
        m_Executors[i]->Run();
    }
    while(running)
    {
        control_fn();
        std::this_thread::sleep_for(timeout);
    }
}

} // namespace nvrpc
