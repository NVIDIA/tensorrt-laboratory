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
#ifndef NVIS_SERVICE_H_
#define NVIS_SERVICE_H_
#pragma once

#include "YAIS/Interfaces.h"
#include "YAIS/RPC.h"

namespace yais
{

template <class ServiceType>
class AsyncService : public IService
{
  public:
    using ServiceType_t = ServiceType;

    AsyncService() : IService(), m_Service(std::make_shared<ServiceType>()) {}
    ~AsyncService() override {}

    void Initialize(::grpc::ServerBuilder& builder) final override
    {
        builder.RegisterService(m_Service.get());
    }

    template <typename ContextType, typename RequestFunction>
    IRPC* RegisterRPC(RequestFunction req_fn)
    {
        auto rpc = new AsyncRPC<ContextType, ServiceType>(m_Service, req_fn);
        auto base = static_cast<IRPC*>(rpc);
        m_RPCs.emplace_back(base);
        return base;
    }

  protected:
/*
    template <class RPCType>
    IRPC *RegisterRPC(typename RPCType::RequestFunc_t req_fn)
    {
        auto rpc = new RPCType(m_Service, req_fn);
        m_RPCs.emplace_back(rpc);
        return rpc
    }
*/
  private:
    std::shared_ptr<ServiceType> m_Service;
    std::vector<std::unique_ptr<IRPC>> m_RPCs;
};

} // end namespace yais

#endif // NVIS_SERVICE_H_