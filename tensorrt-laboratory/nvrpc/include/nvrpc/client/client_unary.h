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
#include <functional>
#include <memory>

#include <glog/logging.h>
#include <grpc++/grpc++.h>

#include "nvrpc/client/base_context.h"
#include "nvrpc/client/executor.h"
#include "tensorrt/laboratory/core/async_compute.h"

namespace nvrpc {
namespace client {

template<typename Request, typename Response>
struct ClientUnary
    : public ::trtlab::AsyncComputeWrapper<void(Request&, Response&, ::grpc::Status&)>
{
  public:
    using PrepareFn = std::function<std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>>(
        ::grpc::ClientContext*, const Request&, ::grpc::CompletionQueue*)>;

    ClientUnary(PrepareFn prepare_fn, std::shared_ptr<Executor> executor)
        : m_PrepareFn(prepare_fn), m_Executor(executor)
    {
    }

    ~ClientUnary() {}

    template<typename OnReturnFn>
    auto Enqueue(Request* request, Response* response, OnReturnFn on_return,
                 std::map<std::string, std::string>& headers)
    {
        auto wrapped = this->Wrap(on_return);
        auto future = wrapped->Future();

        Context* ctx = new Context;
        ctx->m_Request = request;
        ctx->m_Response = response;
        ctx->m_Callback = [ctx, wrapped]() mutable {
            (*wrapped)(*ctx->m_Request, *ctx->m_Response, ctx->m_Status);
        };

        for(auto& header : headers)
        {
            ctx->m_Context.AddMetadata(header.first, header.second);
        }

        ctx->m_Reader = m_PrepareFn(&ctx->m_Context, *ctx->m_Request, m_Executor->GetNextCQ());
        ctx->m_Reader->StartCall();
        ctx->m_Reader->Finish(ctx->m_Response, &ctx->m_Status, ctx->Tag());

        return future.share();
    }

    template<typename OnReturnFn>
    auto Enqueue(Request&& request, OnReturnFn on_return)
    {
        std::map<std::string, std::string> empty_headers;
        return Enqueue(std::move(request), on_return, empty_headers);
    }

    template<typename OnReturnFn>
    auto Enqueue(Request&& request, OnReturnFn on_return,
                 std::map<std::string, std::string>& headers)
    {
        auto req = std::make_shared<Request>(std::move(request));
        auto resp = std::make_shared<Response>();

        auto extended_on_return = [req, resp, on_return](Request & request, Response & response,
                                                         ::grpc::Status & status) mutable -> auto
        {
            return on_return(request, response, status);
        };

        return Enqueue(req.get(), resp.get(), extended_on_return, headers);
    }

  private:
    PrepareFn m_PrepareFn;
    std::shared_ptr<Executor> m_Executor;

    class Context : public BaseContext
    {
        Context() : m_NextState(&Context::StateFinishedDone) {}
        ~Context() override {}

        bool RunNextState(bool ok) final override
        {
            bool ret = (this->*m_NextState)(ok);
            // DLOG_IF(INFO, !ret) << "RunNextState returning false";
            return ret;
        }

        bool ExecutorShouldDeleteContext() const override { return true; }

      protected:
        bool StateFinishedDone(bool ok)
        {
            DLOG(INFO) << "ClientContext: " << Tag() << " finished with "
                       << (m_Status.ok() ? "OK" : "CANCELLED");
            m_Callback();
            DLOG(INFO) << "ClientContext: " << Tag() << " callback completed";
            return false;
        }

      private:
        Request* m_Request;
        Response* m_Response;
        std::function<void()> m_Callback;
        ::grpc::Status m_Status;
        ::grpc::ClientContext m_Context;
        std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>> m_Reader;
        bool (Context::*m_NextState)(bool);

        friend class ClientUnary;
    };
};

} // namespace client
} // namespace nvrpc