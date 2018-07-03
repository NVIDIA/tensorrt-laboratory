/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#ifndef NVIS_RPC_H_
#define NVIS_RPC_H_

#include "YAIS/Context.h"

namespace yais
{

template <class ContextType, class ServiceType>
class AsyncRPC : public IRPC
{
  public:
    using ContextType_t = ContextType;
    using RequestType_t = typename ContextType::RequestType_t;
    using ResponseType_t = typename ContextType::ResponseType_t;
    using RequestFunc_t = std::function<void(
        ServiceType *, ::grpc::ServerContext *, RequestType_t *,
        ::grpc::ServerAsyncResponseWriter<ResponseType_t> *,
        ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>;
    using QueuingFunc_t = std::function<void(
        ::grpc::ServerContext *, RequestType_t *,
        ::grpc::ServerAsyncResponseWriter<ResponseType_t> *, void *)>;

    AsyncRPC(std::shared_ptr<ServiceType> service, RequestFunc_t req_fn);
    ~AsyncRPC() override {}

  protected:
    std::unique_ptr<IContext> CreateContext(::grpc::ServerCompletionQueue*, std::shared_ptr<Resources>) final override;

  private:
    RequestFunc_t m_RequestFunc;
    std::shared_ptr<ServiceType> m_Service;
    std::vector<std::unique_ptr<IContext>> m_Contexts;
};


template <class ContextType, class ServiceType>
AsyncRPC<ContextType, ServiceType>::AsyncRPC(std::shared_ptr<ServiceType> service, RequestFunc_t req_fn)
    : IRPC(), m_Service(service), m_RequestFunc(req_fn) 
{

}


template <class ContextType, class ServiceType>
std::unique_ptr<IContext> AsyncRPC<ContextType, ServiceType>::CreateContext(
    ::grpc::ServerCompletionQueue *cq, std::shared_ptr<Resources> r)
{
    auto ctx_resources = std::dynamic_pointer_cast<typename ContextType::ResourceType_t>(r);
    if (!ctx_resources) { throw std::runtime_error("Incompatible Resource object"); }
    auto q_fn = std::bind(
        m_RequestFunc,
        m_Service.get(),
        std::placeholders::_1, // ServerContext*
        std::placeholders::_2, // InputType
        std::placeholders::_3, // AsyncResponseWriter<OutputType>
        cq,
        cq,
        std::placeholders::_4 // Tag
    );
    std::unique_ptr<IContext> ctx = ContextFactory<ContextType>(q_fn, ctx_resources);
    return ctx;
}

} // end namespace yais

#endif // NVIS_RPC_H_