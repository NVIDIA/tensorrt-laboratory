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
#pragma once

#include "nvrpc/executor.h"
#include "nvrpc/server.h"

#include "test_resources.h"

#include "testing.grpc.pb.h"
#include "testing.pb.h"

namespace nvrpc {
namespace testing {

template<typename Context>
std::unique_ptr<Server> BuildServer();

template<typename T, typename ExecutorType = Executor>
std::unique_ptr<Server> BuildStreamingServer()
{
    auto server = std::make_unique<Server>("0.0.0.0:13377");
    auto resources = std::make_shared<TestResources>(3);
    auto executor = server->RegisterExecutor(new ExecutorType(1));
    auto service = server->RegisterAsyncService<TestService>();
    auto rpc_streaming = service->RegisterRPC<T>(&TestService::AsyncService::RequestStreaming);
    executor->RegisterContexts(rpc_streaming, resources, 10);
    return std::move(server);
}

template<typename UnaryContext, typename StreamingContext, typename ExecutorType = Executor>
std::unique_ptr<Server> BuildServer()
{
    auto server = std::make_unique<Server>("0.0.0.0:13377");
    auto resources = std::make_shared<TestResources>(3);
    auto executor = server->RegisterExecutor(new ExecutorType(1));
    auto service = server->RegisterAsyncService<TestService>();
    auto rpc_unary = service->RegisterRPC<UnaryContext>(&TestService::AsyncService::RequestUnary);
    auto rpc_streaming =
        service->RegisterRPC<StreamingContext>(&TestService::AsyncService::RequestStreaming);
    executor->RegisterContexts(rpc_unary, resources, 10);
    executor->RegisterContexts(rpc_streaming, resources, 10);
    return std::move(server);
}

} // namespace testing
} // namespace nvrpc