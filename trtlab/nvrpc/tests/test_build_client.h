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

#include "nvrpc/client/client_streaming.h"
#include "nvrpc/client/client_unary.h"

#include "test_resources.h"

#include "testing.grpc.pb.h"
#include "testing.pb.h"

namespace nvrpc {
namespace testing {

std::unique_ptr<client::ClientUnary<Input, Output>> BuildUnaryClient()
{
    auto executor = std::make_shared<client::Executor>(1);

    auto channel = grpc::CreateChannel("localhost:13377", grpc::InsecureChannelCredentials());
    std::shared_ptr<TestService::Stub> stub = TestService::NewStub(channel);

    auto infer_prepare_fn = [stub](::grpc::ClientContext * context, const Input& request,
                                   ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(stub->PrepareAsyncUnary(context, request, cq));
    };

    return std::make_unique<client::ClientUnary<Input, Output>>(infer_prepare_fn, executor);
}

std::unique_ptr<client::ClientStreaming<Input, Output>>
    BuildStreamingClient(std::function<void(Input&&)> on_sent,
                         std::function<void(Output&&)> on_recv)
{
    auto executor = std::make_shared<client::Executor>(1);

    auto channel = grpc::CreateChannel("localhost:13377", grpc::InsecureChannelCredentials());
    std::shared_ptr<TestService::Stub> stub = TestService::NewStub(channel);

    auto infer_prepare_fn = [stub](::grpc::ClientContext * context,
                                   ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(stub->PrepareAsyncStreaming(context, cq));
    };

    return std::make_unique<client::ClientStreaming<Input, Output>>(infer_prepare_fn, executor,
                                                                    on_sent, on_recv);
}

} // namespace testing
} // namespace nvrpc