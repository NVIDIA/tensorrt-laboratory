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
#ifndef NVIS_CONTEXT_H_
#define NVIS_CONTEXT_H_
#pragma once

#include "YAIS/Interfaces.h"
#include "YAIS/ThreadPool.h"

#include "glog/logging.h"

namespace yais
{

template <class RequestType, class ResponseType, class ResourceType>
class Context : public IContext
{
  public:
    Context() : QueuingFunc(nullptr), m_Resources(nullptr), m_NextState(nullptr) {}
    ~Context() override {}

    using RequestType_t = RequestType;
    using ResponseType_t = ResponseType;
    using ResourceType_t = ResourceType;

    // gRPC Service function used add the context object to the receive queue
    using QueuingFunc_t = std::function<void(
        ::grpc::ServerContext *, RequestType *,
        ::grpc::ServerAsyncResponseWriter<ResponseType> *, void *)>;

  protected:

    /**
     * Derived classes use this to access the Resources object
     */
    const std::shared_ptr<ResourceType> GetResources() { return m_Resources; }

    /**
     * Allows derived classes a hook into FactoryInitializer and Reset
     */
    virtual void OnInitialize() {}
    virtual void OnReset() {}

    /**
     * Triggers the response message to be sent; this will re-queue the context as the response messgage
     * is being sent.  If a context fails to call one of the following two functions, that context will be
     * in permanent limbo.  This is the equivalent of a memory leak.  The service will eventually exhaust
     * all available contexts and deadlock.
     */
    void FinishResponse() { m_ResponseWriter->Finish(m_Response, ::grpc::Status::OK, IContext::Tag()); }
    void CancelResponse() { m_ResponseWriter->Finish(m_Response, ::grpc::Status::CANCELLED, IContext::Tag()); }


  private:
    /**
     * Called by the CreateContext factory to initialize the context
     */
    void FactoryInitializer(QueuingFunc_t q_fn, std::shared_ptr<ResourceType> resources)
    {
        QueuingFunc = q_fn;
        m_Resources = resources;
        OnInitialize();
    }

    /**
     * Implemetnation of the IContext virtual functions
     */
    bool RunNextState(bool ok) final override { (this->*m_NextState)(ok); }

    void Reset() final override
    {
        DLOG(INFO) << "Context::Reset " << IContext::Tag();
        m_Request.Clear();
        m_Response.Clear();
        m_Context.reset(new ::grpc::ServerContext);
        m_ResponseWriter.reset(new ::grpc::ServerAsyncResponseWriter<ResponseType>(m_Context.get()));
        m_NextState = &Context<RequestType, ResponseType, ResourceType>::InitState;
        OnReset(); // Allows a derived object to perform an action prior to re-queuing
        QueuingFunc(m_Context.get(), &m_Request, m_ResponseWriter.get(), IContext::Tag());
    }

    /**
     * Manage the Execution state of the Context. Derived classes must implement ExecuteRPC to either
     * directly perform the RPC action, or expose a new structure for solving the task, e.g. see
     * ContextWithThreadPool for a two-stage workflow.  This Unary Context has only two states:
     *  1) InitState - context has just come off the queue and will run ExecuteRPC
     *  2) FiniState - the context has completed the ExecuteRPC and the response message was sent;
     *                 the context will be reset and the NextState set to InitState
     */
    virtual void ExecuteRPC(RequestType &request, ResponseType &response) = 0;

    bool InitState(bool ok)
    {
        DLOG(INFO) << "Context::Init " << IContext::Tag();
        if (!ok) { return false; }
        ExecuteRPC(m_Request, m_Response);
        DLOG(INFO) << "Context::ExecuteRPC finished " << IContext::Tag();
        m_NextState = &Context<RequestType, ResponseType, ResourceType>::FiniState;
        return true;
    }

    bool FiniState(bool ok)
    { 
        DLOG(INFO) << "Context::Fini " << IContext::Tag();
        return false; 
    }


    /**
     * Function pointers
     */
    QueuingFunc_t QueuingFunc;
    bool (Context<RequestType, ResponseType, ResourceType>::*m_NextState)(bool);

    /**
     * Variables
     */
    RequestType m_Request;
    ResponseType m_Response;
    std::shared_ptr<ResourceType> m_Resources;
    std::unique_ptr<::grpc::ServerContext> m_Context;
    std::unique_ptr<::grpc::ServerAsyncResponseWriter<ResponseType>> m_ResponseWriter;

    /**
     * Allows the CreateContext factory to access private members, e.g. FactoryInitializer
     * of the Context base class.  This is the only way to properly initialize a Context
     */
    template <class ContextType>
    friend std::unique_ptr<ContextType> ContextFactory(
        typename ContextType::QueuingFunc_t q_fn,
        std::shared_ptr<typename ContextType::ResourceType_t> resources);

  public:
    // Convenience method to acquire the Context base pointer from a derived class
    Context<RequestType, ResponseType, ResourceType>* GetBase() 
    { 
        return dynamic_cast<Context<RequestType, ResponseType, ResourceType> *>(this); 
    }
};


/**
 * ContextFactory is the only function in the library allowed to create an IContext object.
 */
template <class ContextType>
std::unique_ptr<ContextType> ContextFactory(
    typename ContextType::QueuingFunc_t q_fn, 
    std::shared_ptr<typename ContextType::ResourceType_t> resources)
{
    auto ctx = std::make_unique<ContextType>();
    auto base = ctx->GetBase();
    base->FactoryInitializer(q_fn, resources);
    return ctx;
}

} // end namespace yais

#endif // NVIS_CONTEXT_H_