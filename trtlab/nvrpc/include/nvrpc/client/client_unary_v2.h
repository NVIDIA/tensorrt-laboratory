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
#include <future>

#include <glog/logging.h>
#include <grpc++/grpc++.h>

#include "nvrpc/client/base_context.h"
#include "nvrpc/client/executor.h"
#include "trtlab/core/async_compute.h"

namespace nvrpc
{
    namespace client
    {
        namespace v2
        {
            template <typename Request, typename Response>
            struct ClientUnary : public BaseContext
            {
                using Client = ClientUnary<Request, Response>;
                using Reader = std::unique_ptr<::grpc_impl::ClientAsyncResponseReader<Response>>;

            public:
                using PrepareFn = std::function<Reader(::grpc::ClientContext*, const Request&, ::grpc::CompletionQueue*)>;

                ClientUnary(PrepareFn prepare_fn, std::shared_ptr<Executor> executor) : m_PrepareFn(prepare_fn), m_Executor(executor)
                {
                    m_NextState = &Client::StateInvalid;
                }

                ~ClientUnary() {}

                void Write(Request&&);

                virtual void CallbackOnRequestSent(Request&&) {}
                virtual void CallbackOnResponseReceived(Response&&)    = 0;
                virtual void CallbackOnComplete(const ::grpc::Status&) = 0;

                bool ExecutorShouldDeleteContext() const override
                {
                    return false;
                }

            protected:
                ::grpc::ClientContext& GetClientContext()
                {
                    return m_Context;
                }

            private:
                PrepareFn                 m_PrepareFn;
                std::shared_ptr<Executor> m_Executor;

                ::grpc::Status        m_Status;
                ::grpc::ClientContext m_Context;
                Reader                m_Stream;

                Request  m_Request;
                Response m_Response;

                bool RunNextState(bool ok) final override
                {
                    return (this->*m_NextState)(ok);
                }

                bool StateFinishDone(bool);
                bool StateInvalid(bool);

                bool (Client::*m_NextState)(bool);
            };

            template <typename Request, typename Response>
            void ClientUnary<Request, Response>::Write(Request&& request)
            {
                CHECK(m_Stream == nullptr);

                m_Request   = std::move(request);
                m_NextState = &Client::StateFinishDone;

                m_Stream = m_PrepareFn(&m_Context, m_Request, m_Executor->GetNextCQ());
                m_Stream->StartCall();
                m_Stream->Finish(&m_Response, &m_Status, this->Tag());
            }

            template <typename Request, typename Response>
            bool ClientUnary<Request, Response>::StateFinishDone(bool ok)
            {
                m_NextState = &Client::StateInvalid;

                if (!ok)
                {
                    DVLOG(1) << "FinishDone handler called with NOT OK";
                }

                DVLOG(1) << "calling on complete callback";
                if (m_Status.ok())
                {
                    CallbackOnResponseReceived(std::move(m_Response));
                }
                CallbackOnComplete(m_Status);

                return false;
            }

            template <typename Request, typename Response>
            bool ClientUnary<Request, Response>::StateInvalid(bool ok)
            {
                LOG(FATAL) << "logic error in ClientUnary state management";
                return false;
            }

        } // namespace v2
    }     // namespace client
} // namespace nvrpc