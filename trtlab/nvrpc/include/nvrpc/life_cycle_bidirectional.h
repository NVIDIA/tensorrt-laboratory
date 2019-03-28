// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <queue>

#include "nvrpc/interfaces.h"

#include <glog/logging.h>

namespace nvrpc {

// A bidirectional streaming version of LifeCycleUnary class
// Note that the bidirectional streaming feature in gRPC supports
// arbitrary call order of ServerReaderWriter::Read() and
// ServerReaderWriter::Write(), so we are able to handle
// reading request and writing response seperately.
template<class Request, class Response>
class BidirectionalLifeCycleStreaming : public IContextLifeCycle
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

    ~BidirectionalLifeCycleStreaming() override {}

  protected:
    // Class to wrap over the State function pointers to allow the use of
    // different tags while referencing to the same Context.
    // Executor Detag() the tag, which points to the StateContext, which contains
    // a pointer to the actual context (master context).
    template<class RequestType, class ResponseType>
    class StateContext : public IContext
    {
      public:
        StateContext(IContext* master) : IContext(master) {}

      private:
        // IContext Methods
        bool RunNextState(bool ok) final override
        {
            return static_cast<BidirectionalLifeCycleStreaming*>(m_MasterContext)
                ->RunNextState(m_NextState, ok);
        }
        void Reset() final override { LOG(FATAL) << "ooops; call reset on master"; }

        bool (BidirectionalLifeCycleStreaming<RequestType, ResponseType>::*m_NextState)(bool);

        friend class BidirectionalLifeCycleStreaming<RequestType, ResponseType>;
    };

    BidirectionalLifeCycleStreaming();
    void SetQueueFunc(ExecutorQueueFuncType);

    // Function to actually process the request
    virtual void ExecuteRPC(RequestType& request, ResponseType& response) = 0;

    void FinishResponse() final override;
    void CancelResponse() final override;

  private:
    std::tuple<bool, bool, bool> EvaluateState();
    void ProgressState(bool should_write, bool should_execute, bool should_finish);

    // IContext Methods
    bool RunNextState(bool ok) final override;
    bool RunNextState(bool (BidirectionalLifeCycleStreaming<Request, Response>::*state_fn)(bool),
                      bool ok);
    void Reset() final override;

    // BidirectionalLifeCycleStreaming Specific Methods
    bool StateInitializedDone(bool ok);
    bool StateRequestDone(bool ok);
    bool StateResponseDone(bool ok);
    bool StateFinishDone(bool ok);
    bool StateInvalid(bool ok);

    // Function pointers
    ExecutorQueueFuncType m_QueuingFunc;
    bool (BidirectionalLifeCycleStreaming<RequestType, ResponseType>::*m_NextState)(bool);

    // Variables
    // The mutex will be more useful once we can keep reading requests
    // without waiting for response to be sent
    std::mutex m_QueueMutex;
    std::queue<RequestType> m_RequestQueue;
    std::queue<ResponseType> m_ResponseQueue;
    std::queue<ResponseType> m_ResponseWriteBackQueue;

    StateContext<RequestType, ResponseType> m_ReadStateContext;
    StateContext<RequestType, ResponseType> m_WriteStateContext;

    bool m_Writing;
    bool m_Executing;
    bool m_WritesDone;
    bool m_Finishing;

    std::unique_ptr<::grpc::ServerContext> m_Context;
    std::unique_ptr<::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>> m_ReaderWriter;

    friend class StateContext<RequestType, ResponseType>;

  public:
    template<class RequestFuncType, class ServiceType>
    static ServiceQueueFuncType BindServiceQueueFunc(
        /*
        std::function<void(
            ServiceType *, ::grpc::ServerContext *,
            ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*,
            ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
        */
        RequestFuncType request_fn, ServiceType* service_type)
    {
        return std::bind(request_fn, service_type,
                         std::placeholders::_1, // ServerContext*
                         std::placeholders::_2, // AsyncReaderWriter<OutputType, InputType>
                         std::placeholders::_3, // CQ
                         std::placeholders::_4, // ServerCQ
                         std::placeholders::_5 // Tag
        );
    }

    static ExecutorQueueFuncType BindExecutorQueueFunc(ServiceQueueFuncType service_q_fn,
                                                       ::grpc::ServerCompletionQueue* cq)
    {
        return std::bind(service_q_fn,
                         std::placeholders::_1, // ServerContext*
                         std::placeholders::_2, // AsyncReaderWriter<Response, Request> *
                         cq, cq,
                         std::placeholders::_3 // Tag
        );
    }
};

// Implementation
template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::RunNextState(bool ok)
{
    return (this->*m_NextState)(ok);
}

template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::RunNextState(
    bool (BidirectionalLifeCycleStreaming<Request, Response>::*state_fn)(bool), bool ok)
{
    return (this->*state_fn)(ok);
}

template<class Request, class Response>
BidirectionalLifeCycleStreaming<Request, Response>::BidirectionalLifeCycleStreaming()
    : m_ReadStateContext(static_cast<IContext*>(this)),
      m_WriteStateContext(static_cast<IContext*>(this)), m_WritesDone(false), m_Writing(false),
      m_Executing(false), m_Finishing(false)
{
    m_ReadStateContext.m_NextState =
        &BidirectionalLifeCycleStreaming<RequestType, ResponseType>::StateRequestDone;
    m_WriteStateContext.m_NextState =
        &BidirectionalLifeCycleStreaming<RequestType, ResponseType>::StateResponseDone;
}

template<class Request, class Response>
void BidirectionalLifeCycleStreaming<Request, Response>::Reset()
{
    std::queue<RequestType> empty_request_queue;
    std::queue<ResponseType> empty_response_queue;
    std::queue<ResponseType> empty_response_write_back_queue;
    OnLifeCycleReset();
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        m_Writing = false;
        m_Executing = false;
        m_WritesDone = false;
        m_Finishing = false;
        m_RequestQueue.swap(empty_request_queue);
        m_ResponseQueue.swap(empty_response_queue);
        m_ResponseWriteBackQueue.swap(empty_response_write_back_queue);
        m_Context.reset(new ::grpc::ServerContext);
        m_ReaderWriter.reset(
            new ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>(m_Context.get()));

        m_NextState =
            &BidirectionalLifeCycleStreaming<RequestType, ResponseType>::StateInitializedDone;
    }
    m_QueuingFunc(m_Context.get(), m_ReaderWriter.get(), IContext::Tag());
}

template<class Request, class Response>
std::tuple<bool, bool, bool> BidirectionalLifeCycleStreaming<Request, Response>::EvaluateState()
{
    bool should_write = false;
    bool should_execute = false;
    bool should_finish = false;

    if(!m_Executing && !m_ResponseQueue.empty())
    {
        should_execute = true;
        m_Executing = true;
    }

    if(!m_Writing && !m_ResponseWriteBackQueue.empty())
    {
        should_write = true;
        m_Writing = true;
    }

    if(!should_write && !should_execute && !m_Writing && !m_Executing && !m_Finishing &&
       m_WritesDone)
    {
        should_finish = true;
        m_Finishing = true;
        m_NextState = &BidirectionalLifeCycleStreaming<RequestType, ResponseType>::StateFinishDone;
    }

    DLOG(INFO) << should_write << "; " << should_execute << "; " << should_finish << " -- "
               << m_Writing << "; " << m_Executing << "; " << m_WritesDone;

    return std::make_tuple(should_write, should_execute, should_finish);
}

template<class Request, class Response>
void BidirectionalLifeCycleStreaming<Request, Response>::ProgressState(bool should_write,
                                                                       bool should_execute,
                                                                       bool should_finish)
{
    if(should_write)
    {
        DLOG(INFO) << "Writing response";
        m_ReaderWriter->Write(m_ResponseWriteBackQueue.front(),
                              m_WriteStateContext.IContext::Tag());
    }
    if(should_execute)
    {
        DLOG(INFO) << "Executing";
        ExecuteRPC(m_RequestQueue.front(), m_ResponseQueue.front());
    }
    if(should_finish)
    {
        DLOG(INFO) << "Triggering Finish";
        m_ReaderWriter->Finish(::grpc::Status::OK, IContext::Tag());
    }
}

// The following are a set of functions used as function pointers
// to keep track of the state of the context.
template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::StateInitializedDone(bool ok)
{
    if(!ok)
    {
        return false;
    }

    OnLifeCycleStart();
    // Start reading once connection is created
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        m_RequestQueue.emplace();
        m_NextState = &BidirectionalLifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
    }
    m_ReaderWriter->Read(&m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
    return true;
}

// If the Context.m_NextState is at this state or at StateResponseDone state,
// it will keep reading requests from the stream until no more requests will
// be read (Read() brings back status ok==false)
template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::StateRequestDone(bool ok)
{
    // No more message to be read from this stream, however, if it is executing
    // a request, then a ServerReaderWriter::Write() will be called. In that case,
    // let WriteStateContext handle the reset procedure.

    bool should_write, should_execute, should_finish, should_read = true;

    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        DLOG(INFO) << "RequestDone Triggered";

        if(!ok)
        {
            {
                // Client called WritesDone
                DLOG(INFO) << "WritesDone received from Client; closing Server Reads";
                m_WritesDone = true;
            }
        }

        // Successfully receive request
        should_read = !m_WritesDone;
        if(should_read)
        {
            m_ResponseQueue.emplace(); // add a response object which will be written on execution
            m_RequestQueue.emplace(); // post a read/recv on a new request object
        }

        auto should_wef = EvaluateState();
        should_write = std::get<0>(should_wef);
        should_execute = std::get<1>(should_wef);
        should_finish = std::get<2>(should_wef);
    }
    if(should_read)
    {
        // Post Read/Receive
        m_ReaderWriter->Read(&m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
    }
    ProgressState(should_write, should_execute, should_finish);
    return true;
}

// If the Context.m_NextState is at this state or at StateResponseDone state,
// it will keep writing completed response to the stream until it is closed
// (Write() brings back status ok==false)
template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::StateResponseDone(bool ok)
{
    // If write didn't go through, then the call is dead. Start reseting
    if(!ok)
    {
        // this is likely an unrecoverable error on the client
        // i think we should return a false without trying to cancel;
        DLOG(ERROR) << "not ok in ResponseDone";
        CancelResponse();
        return true;
    }

    // Done writing back one response
    bool should_write, should_execute, should_finish;
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        DLOG(ERROR) << "Finished Writing a Response - ResponseDone";

        m_Writing = false;
        m_ResponseWriteBackQueue.pop();

        auto should_wef = EvaluateState();
        should_write = std::get<0>(should_wef);
        should_execute = std::get<1>(should_wef);
        should_finish = std::get<2>(should_wef);
    }
    ProgressState(should_write, should_execute, should_finish);
    return true;
}

template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::StateFinishDone(bool ok)
{
    DLOG(INFO) << "Server closed Write stream - FinishedDone";
    return false;
}

template<class Request, class Response>
bool BidirectionalLifeCycleStreaming<Request, Response>::StateInvalid(bool ok)
{
    throw std::runtime_error("invalid state");
    return false;
}

template<class Request, class Response>
void BidirectionalLifeCycleStreaming<Request, Response>::FinishResponse()
{
    bool should_write, should_execute, should_finish;
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        DLOG(INFO) << "InFinishResponse";

        m_Executing = false;
        m_ResponseWriteBackQueue.push(std::move(m_ResponseQueue.front()));
        m_RequestQueue.pop();
        m_ResponseQueue.pop();

        auto should_wef = EvaluateState();
        should_write = std::get<0>(should_wef);
        should_execute = std::get<1>(should_wef);
        should_finish = std::get<2>(should_wef);
    }
    ProgressState(should_write, should_execute, should_finish);
}

template<class Request, class Response>
void BidirectionalLifeCycleStreaming<Request, Response>::CancelResponse()
{
    bool reset_ready = false;
    {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        m_NextState = &BidirectionalLifeCycleStreaming<RequestType, ResponseType>::StateFinishDone;
        reset_ready = !m_Executing;
    }
    // Only call Finish() when no RPC is being executed to avoid clearing
    // request and response while they are being referenced in the RPC
    if(reset_ready)
    {
        DLOG(INFO) << "Closing Server Writes";
        m_ReaderWriter->Finish(::grpc::Status::CANCELLED, IContext::Tag());
    }
}

template<class Request, class Response>
void BidirectionalLifeCycleStreaming<Request, Response>::SetQueueFunc(
    ExecutorQueueFuncType queue_fn)
{
    m_QueuingFunc = queue_fn;
}

} // namespace nvrpc