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

#include "nvrpc/interfaces.h"
#include "nvrpc/life_cycle_streaming.h"

namespace nvrpc {

template<typename Request, typename Response>
class LifeCycleBatchingNew : public LifeCycleStreaming<Request, Response>
{
  public:
    using LifeCycleStreaming<Request, Response>::LifeCycleStreaming;
    using Stream = typename LifeCycleStreaming<Request, Response>::ServerStream;

  protected:
    virtual void ExecuteRPC(std::vector<Request>&, std::shared_ptr<Stream>) = 0;

  private:
    void RequestReceived(Request&&, std::shared_ptr<Stream>) final override;
    void RequestsFinished(std::shared_ptr<Stream>) final override;

    std::vector<Request> m_Requests;
};

template<class Request, class Response>
void LifeCycleBatchingNew<Request, Response>::RequestReceived(Request&& request,
                                                              std::shared_ptr<Stream> stream)
{
    m_Requests.push_back(std::move(request));
}

template<class Request, class Response>
void LifeCycleBatchingNew<Request, Response>::RequestsFinished(std::shared_ptr<Stream> stream)
{
    ExecuteRPC(m_Requests, stream);
}

/**
 * @brief LifeCycle State Machine All-In, then All-Out BATCHING
 *
 * Client sends message until it says it is done, then execute the RPC
 * on all received items, then for each item, return a response on the
 * stream in the same order the requests were received.
 *
 * @tparam Request
 * @tparam Response
 */
template<class Request, class Response>
class LifeCycleBatching : public IContextLifeCycle
{
  public:
    using RequestType = Request;
    using ResponseType = Response;
    using ServiceQueueFuncType = std::function<void(
        ::grpc::ServerContext*, ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*,
        ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*)>;
    using ExecutorQueueFuncType =
        std::function<void(::grpc::ServerContext*,
                           ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*, void*)>;

    virtual ~LifeCycleBatching() override {}

  protected:
    LifeCycleBatching() = default;
    void SetQueueFunc(ExecutorQueueFuncType q_fn) { m_QueuingFunc = q_fn; }

    virtual void ExecuteRPC(std::vector<RequestType>&, std::vector<ResponseType>&) = 0;
    virtual void OnRequestReceived(const RequestType&) {}

    void FinishResponse() final override;
    void CancelResponse() final override;

  private:
    // IContext Methods
    bool RunNextState(bool ok) final override;
    void Reset() final override;

    bool StateRequestDone(bool);
    bool StateReadDone(bool);
    bool StateWriteDone(bool);
    bool StateFinishedDone(bool);

    // Function pointers
    ExecutorQueueFuncType m_QueuingFunc;
    bool (LifeCycleBatching<RequestType, ResponseType>::*m_NextState)(bool);

    std::vector<RequestType> m_Requests;
    std::vector<ResponseType> m_Responses;
    std::unique_ptr<::grpc::ServerContext> m_Context;
    std::unique_ptr<::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>> m_Stream;
    typename std::vector<ResponseType>::const_iterator m_ResponseIterator;

  public:
    template<class RequestFuncType, class ServiceType>
    static ServiceQueueFuncType BindServiceQueueFunc(
        /*
        std::function<void(
            ServiceType*, ServerContextType*,
            ServerAsyncReaderWriter<ResponseType, RequestType>*,
            CompletionQueue*, ServerCompletionQueue*, void*)>
        */
        RequestFuncType request_fn, ServiceType* service_type)
    {
        return std::bind(request_fn, service_type,
                         std::placeholders::_1, // ServerContext*
                         std::placeholders::_2, // AsyncReaderWriter<ResponseType, RequestType>*
                         std::placeholders::_3, // CQ*
                         std::placeholders::_4, // ServerCQ*
                         std::placeholders::_5 // Tag
        );
    }

    static ExecutorQueueFuncType BindExecutorQueueFunc(ServiceQueueFuncType service_q_fn,
                                                       ::grpc::ServerCompletionQueue* cq)
    {
        return std::bind(service_q_fn,
                         std::placeholders::_1, // ServerContext*
                         std::placeholders::_2, // AsyncReaderWriter<ResponseType, RequestType>*
                         cq, cq,
                         std::placeholders::_3 // Tag
        );
    }
};

// Implementations

template<class Request, class Response>
bool LifeCycleBatching<Request, Response>::RunNextState(bool ok)
{
    return (this->*m_NextState)(ok);
}

template<class Request, class Response>
void LifeCycleBatching<Request, Response>::Reset()
{
    OnLifeCycleReset();
    m_Requests.clear();
    m_Responses.clear();
    m_Context.reset(new ::grpc::ServerContext);
    m_Stream.reset(new ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>(m_Context.get()));
    m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateRequestDone;
    m_QueuingFunc(m_Context.get(), m_Stream.get(), IContext::Tag());
}

template<class Request, class Response>
bool LifeCycleBatching<Request, Response>::StateRequestDone(bool ok)
{
    if(!ok) return false;
    OnLifeCycleStart();
    m_Requests.emplace(m_Requests.end());
    m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateReadDone;
    m_Stream->Read(&m_Requests.back(), IContext::Tag());
    return true;
}

template<class Request, class Response>
bool LifeCycleBatching<Request, Response>::StateReadDone(bool ok)
{
    if(ok)
    {
        // Execute Callback for the Request item received
        OnRequestReceived(m_Requests.back());
        // Stream is still open, so pull off another request
        m_Requests.emplace(m_Requests.end());
        m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateReadDone;
        m_Stream->Read(&m_Requests.back(), IContext::Tag());
    }
    else
    {
        // Client has signaled it will not send any more requests
        // Remove the Request we didn't actually use
        m_Requests.pop_back();
        // Execute Batched RPC on all Requests
        ExecuteRPC(m_Requests, m_Responses);
    }
    return true;
}

template<class Request, class Response>
bool LifeCycleBatching<Request, Response>::StateWriteDone(bool ok)
{
    if(!ok) return false;
    if(m_ResponseIterator != m_Responses.cend())
    {
        if(m_ResponseIterator + 1 != m_Responses.cend())
        {
            m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateWriteDone;
            m_Stream->Write(*m_ResponseIterator, IContext::Tag());
            // The following hangs even though the client is guaranteed to have sent WritesDone();
            // thus, all the response writes from the server should not rely on any of the previous
            // messages https://grpc.io/grpc/cpp/md_doc_cpp_perf_notes.html
            // m_Stream->Write(*m_ResponseIterator, ::grpc::WriteOptions().set_buffer_hint(),
            // IContext::Tag());
            m_ResponseIterator++;
        }
        else
        {
            m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateFinishedDone;
            m_Stream->WriteAndFinish(*m_ResponseIterator, ::grpc::WriteOptions(),
                                     ::grpc::Status::OK, IContext::Tag());
        }
    }
    else
    {
        m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateFinishedDone;
        m_Stream->Finish(::grpc::Status::OK, IContext::Tag());
    }
    return true;
}

template<class Request, class Response>
bool LifeCycleBatching<Request, Response>::StateFinishedDone(bool ok)
{
    return false;
}

template<class Request, class Response>
void LifeCycleBatching<Request, Response>::FinishResponse()
{
    m_ResponseIterator = m_Responses.cbegin();
    StateWriteDone(true);
}

template<class Request, class Response>
void LifeCycleBatching<Request, Response>::CancelResponse()
{
    m_NextState = &LifeCycleBatching<RequestType, ResponseType>::StateFinishedDone;
    m_Stream->Finish(::grpc::Status::CANCELLED, IContext::Tag());
}

} // namespace nvrpc
