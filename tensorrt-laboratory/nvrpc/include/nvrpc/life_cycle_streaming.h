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

/**
 * @brief Base Steaming Life Cycle
 *
 * For every incoming request, `ReceivedRequest` is executed with a shared pointer to a
 * `ServerStream`.  The `ServerStream` is used to write responses back on the stream or
 * optionally cancel the stream.  The life cycle maintains a weak pointer to the `ServerStream`
 * which allows it detect when all external `ServerStream` objects have been dereferenced.
 *
 * The `ServerStream` can be handed off to an external Resource.  If this happens, the external
 * resource can write back responses on the stream until either it decides to release the
 * `ServerStream` object or the stream becomes invalidated.  All `ServerStream` object become
 * invalid when either `CloseStream` or `FinishStream` is called on either a `ServerStream`
 * or by the LifeCycle.
 *
 * The stream is closed when:
 *  1) the client has closed its half of stream, meaning no more Requests will be received,
 *  2) either,
 *     a) all `ServerStream` object have been dereferenced, or
 *     b) `CancelStream` or `FinishStream` is called on either the lifecycle or an external
 *        `ServerStream`.
 *
 * @tparam Request
 * @tparam Response
 */
template<class Request, class Response>
class LifeCycleStreaming : public IContextLifeCycle
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

    ~LifeCycleStreaming() override {}

  protected:
    LifeCycleStreaming();
    void SetQueueFunc(ExecutorQueueFuncType);

    // template<typename RequestType, typename ResponseType>
    class ServerStream;

    // Called immediately on receiving a Request
    // The ServerStream object provides the response interface
    // virtual void RequestReceived(Request&&, std::shared_ptr<ServerStream<RequestType,
    // ResponseType>>) = 0;
    virtual void RequestReceived(Request&&, std::shared_ptr<ServerStream>) = 0;
    virtual void RequestsFinished(std::shared_ptr<ServerStream>) {}

    // template<typename RequestType, typename ResponseType>
    class ServerStream
    {
      public:
        // ServerStream(LifeCycleStreaming<RequestType, ResponseType>* master) : m_Master(master) {}
        ServerStream(LifeCycleStreaming<Request, Response>* master) : m_Master(master) {}
        ~ServerStream() {}

        bool IsConnected()
        {
            return m_Master;
        }

        std::uint64_t StreamID()
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            if(!IsConnected())
            {
                DLOG(WARNING) << "Attempted to get ID of a disconnected stream";
                return 0UL;
            }
            return static_cast<std::uint64_t>(m_Master->Tag());
        }

        bool WriteResponse(ResponseType&& response)
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            if(!IsConnected())
            {
                DLOG(WARNING) << "Attempted to write to a disconnected stream";
                return false;
            }
            m_Master->WriteResponse(std::move(response));
            return true;
        }

        bool CancelStream()
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            if(!IsConnected())
            {
                DLOG(WARNING) << "Attempted to cancel to a disconnected stream";
                return false;
            }
            m_Master->CancelResponse();
            return false;
        }

        bool FinishStream()
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            if(!IsConnected())
            {
                DLOG(WARNING) << "Attempted to finish to a disconnected stream";
                return false;
            }
            m_Master->FinishResponse();
            return false;
        }

      protected:
        void Invalidate()
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            m_Master = nullptr;
        }

      private:
        std::recursive_mutex m_Mutex;
        LifeCycleStreaming<Request, Response>* m_Master;

        friend class LifeCycleStreaming<Request, Response>;
    };

    template<class RequestType, class ResponseType>
    class StateContext : public IContext
    {
      public:
        StateContext(IContext* master) : IContext(master) {}

      private:
        // IContext Methods
        bool RunNextState(bool ok) final override
        {
            return static_cast<LifeCycleStreaming*>(m_MasterContext)->RunNextState(m_NextState, ok);
        }
        void Reset() final override
        {
            static_cast<LifeCycleStreaming*>(m_MasterContext)->Reset();
        }

        bool (LifeCycleStreaming<RequestType, ResponseType>::*m_NextState)(bool);

        friend class LifeCycleStreaming<RequestType, ResponseType>;
    };

  private:
    using ReadHandle = bool;
    using WriteHandle = bool;
    using ExecuteHandle = std::function<void()>;
    using FinishHandle = bool;
    using Actions = std::tuple<ReadHandle, WriteHandle, ExecuteHandle, FinishHandle>;

    // IContext Methods
    void Reset() final override;
    bool RunNextState(bool ok) final override;
    bool RunNextState(bool (LifeCycleStreaming<Request, Response>::*state_fn)(bool), bool ok);

    // IContextLifeCycle Methods
    void FinishResponse() final override;
    void CancelResponse() final override;

    // LifeCycleStreaming Specific Methods
    bool StateInitializedDone(bool ok);
    bool StateReadDone(bool ok);
    bool StateWriteDone(bool ok);
    bool StateFinishedDone(bool ok);
    bool StateInvalid(bool ok);

    // Progress Engine
    Actions EvaluateState();
    void ForwardProgress(Actions&);

    // User Actions
    void WriteResponse(Response&& response);
    void CloseStream(::grpc::Status);

    // Function pointers
    ExecutorQueueFuncType m_QueuingFunc;
    bool (LifeCycleStreaming<RequestType, ResponseType>::*m_NextState)(bool);

    // Internal State
    std::recursive_mutex m_QueueMutex;
    std::queue<RequestType> m_RequestQueue;
    std::queue<ResponseType> m_ResponseQueue;

    std::shared_ptr<ServerStream> m_ServerStream;
    std::weak_ptr<ServerStream> m_ExternalStream;

    bool m_Reading, m_Writing, m_Finishing, m_ReadsDone, m_WritesDone, m_ReadsFinished;

    StateContext<RequestType, ResponseType> m_ReadStateContext;
    StateContext<RequestType, ResponseType> m_WriteStateContext;

    std::unique_ptr<::grpc::Status> m_Status;
    std::unique_ptr<::grpc::ServerContext> m_Context;
    std::unique_ptr<::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>> m_Stream;

    friend class StateContext<RequestType, ResponseType>;
    // friend class ServerStream<RequestType, ResponseType>;
    friend class ServerStream;

  public:
    template<class RequestFuncType, class ServiceType>
    static ServiceQueueFuncType BindServiceQueueFunc(RequestFuncType request_fn,
                                                     ServiceType* service_type)
    {
        /*
            RequestFuncType = std::function<void(
                ServiceType *, ::grpc::ServerContext *,
                ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*,
                ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
        */
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
LifeCycleStreaming<Request, Response>::LifeCycleStreaming()
    : m_ReadStateContext(static_cast<IContext*>(this)),
      m_WriteStateContext(static_cast<IContext*>(this)), m_Reading(false), m_Writing(false),
      m_Finishing(false), m_ReadsDone(false), m_WritesDone(false), m_ReadsFinished(false)
{
    m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
    m_ReadStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
    m_WriteStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::Reset()
{
    std::queue<RequestType> empty_request_queue;
    std::queue<ResponseType> empty_response_queue;
    OnLifeCycleReset();
    {
        std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
        m_Reading = false;
        m_Writing = false;
        m_Finishing = false;
        m_ReadsDone = false;
        m_WritesDone = false;
        m_ReadsFinished = false;
        m_RequestQueue.swap(empty_request_queue);
        m_ResponseQueue.swap(empty_response_queue);
        m_ServerStream.reset();
        m_ExternalStream.reset();

        m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInitializedDone;
        m_ReadStateContext.m_NextState =
            &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
        m_WriteStateContext.m_NextState =
            &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;

        m_Status.reset();
        m_Context.reset(new ::grpc::ServerContext);
        m_Stream.reset(
            new ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>(m_Context.get()));
    }
    m_QueuingFunc(m_Context.get(), m_Stream.get(), IContext::Tag());
}

template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::RunNextState(bool ok)
{
    return (this->*m_NextState)(ok);
}

template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::RunNextState(
    bool (LifeCycleStreaming<Request, Response>::*state_fn)(bool), bool ok)
{
    return (this->*state_fn)(ok);
}

template<class Request, class Response>
typename LifeCycleStreaming<Request, Response>::Actions
    LifeCycleStreaming<Request, Response>::EvaluateState()
{
    ReadHandle should_read = false;
    WriteHandle should_write = false;
    ExecuteHandle should_execute = nullptr;
    FinishHandle should_finish = false;

    if(!m_Reading && !m_ReadsDone)
    {
        should_read = true;
        m_Reading = true;
        m_RequestQueue.emplace();
        m_ReadStateContext.m_NextState =
            &LifeCycleStreaming<RequestType, ResponseType>::StateReadDone;

        should_execute = [this, request = std::move(m_RequestQueue.front()),
                          stream = m_ServerStream]() mutable {
            RequestReceived(std::move(request), stream);
        };
        m_RequestQueue.pop();
    }

    if(!m_Reading && m_ReadsDone && !m_ReadsFinished)
    {
        m_ReadsFinished = true;
        should_execute = [this, stream = m_ServerStream]() mutable {
            DLOG(INFO) << "Client sent WritesDone";
            RequestsFinished(stream);
        };
    }

    if(!m_Writing && !m_ResponseQueue.empty())
    {
        should_write = true;
        m_Writing = true;
        m_WriteStateContext.m_NextState =
            &LifeCycleStreaming<RequestType, ResponseType>::StateWriteDone;
    }

    if(!m_Reading && !m_Writing && !m_Finishing && (m_Status || m_ExternalStream.expired()))
    {
        should_finish = true;
        m_Finishing = true;
        m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateFinishedDone;
        if(!m_Status)
        {
            m_Status = std::make_unique<::grpc::Status>(::grpc::Status::OK);
        }
    }
    // clang-format off
    DLOG(INFO) << (should_read ? 1 : 0) << (should_write ? 1 : 0) << (should_execute ? 1 : 0)
               << (should_finish ? 1 : 0) 
               << " -- " << m_Reading << m_Writing 
               << " -- " << m_ReadsDone << m_ReadsFinished << (m_Status ? 1 : 0) << (m_ExternalStream.expired() ? 1 : 0)
               << " -- " << m_Finishing;
    // clang-format on

    return std::make_tuple(should_read, should_write, should_execute, should_finish);
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::ForwardProgress(Actions& actions)
{
    ReadHandle should_read = std::get<0>(actions);
    WriteHandle should_write = std::get<1>(actions);
    ExecuteHandle should_execute = std::get<2>(actions);
    FinishHandle should_finish = std::get<3>(actions);

    if(should_read)
    {
        DLOG(INFO) << "Posting Read/Recv";
        m_Stream->Read(&m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
    }
    if(should_write)
    {
        DLOG(INFO) << "Writing Response";
        m_Stream->Write(m_ResponseQueue.front(), m_WriteStateContext.IContext::Tag());
    }
    if(should_execute)
    {
        DLOG(INFO) << "Kicking off Execution";
        should_execute();
    }
    if(should_finish)
    {
        DLOG(INFO) << "Closing Stream - Finish";
        m_Stream->Finish(*m_Status, IContext::Tag());
    }
}

// The following are a set of functions used as function pointers
// to keep track of the state of the context.
template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::StateInitializedDone(bool ok)
{
    if(!ok)
    {
        // if initialization fails, then the server/cq are shutting down
        // return true so we don't reset the context
        LOG_FIRST_N(ERROR, 10) << "Stream Initialization Failed - Server Shutting Down";
        return true;
    }

    OnLifeCycleStart();
    {
        std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
        DLOG(INFO) << "Initialize Stream";

        m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;

        // Start reading once connection is created - State
        m_Reading = true;
        m_RequestQueue.emplace();
        m_ReadStateContext.m_NextState =
            &LifeCycleStreaming<RequestType, ResponseType>::StateReadDone;

        // Object that allows the server to response on the stream
        // new ServerStream<RequestType, ResponseType>(this), [this](auto ptr) mutable {
        m_ServerStream =
            std::shared_ptr<ServerStream>(new ServerStream(this), [this](auto ptr) mutable {
                // Custom Deleter - may trigger stream closing
                Actions actions;
                {
                    std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
                    DLOG(INFO) << "All ServerStream objects have been deleted";
                    actions = this->EvaluateState();
                };
                this->ForwardProgress(actions);
                delete ptr;
            });
        m_ExternalStream = m_ServerStream;
    }

    // Start reading once connection is created - Action
    m_Stream->Read(&m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
    return true;
}

template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::StateReadDone(bool ok)
{
    Actions actions;
    {
        std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
        DLOG(INFO) << "ReadDone Event: " << (ok ? "OK" : "NOT OK");

        m_Reading = false;

        if(!ok)
        {
            {
                // Client called WritesDone
                DLOG(INFO) << "WritesDone received from Client; closing Server Reads";
                m_ReadsDone = true;
                m_RequestQueue.pop();
                m_ReadStateContext.m_NextState =
                    &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;

                // Clear the m_ServerStream since we will not be launching any new tasks
                // m_ExternalStream holds a weak_ptr to the original m_ServerStream so we
                // can track when the last external reference was released.
                m_ServerStream.reset();
            }
        }

        actions = EvaluateState();
    }
    ForwardProgress(actions);
    return true;
}

template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::StateWriteDone(bool ok)
{
    // If write didn't go through, then the call is dead. Start reseting

    Actions actions;
    {
        std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
        DLOG(INFO) << "WriteDone Event: " << (ok ? "OK" : "NOT OK");

        m_Writing = false;

        if(!ok)
        {
            LOG(ERROR) << "not ok in ResponseDone";
            return false;
        }

        m_ResponseQueue.pop();
        m_WriteStateContext.m_NextState =
            &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;

        actions = EvaluateState();
    }
    ForwardProgress(actions);
    return true;
}

template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::StateFinishedDone(bool ok)
{
    DLOG(INFO) << "Server closed Write stream - FinishedDone";
    return false;
}

template<class Request, class Response>
bool LifeCycleStreaming<Request, Response>::StateInvalid(bool ok)
{
    throw std::runtime_error("invalid state");
    return false;
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::FinishResponse()
{
    CloseStream(::grpc::Status::OK);
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::CancelResponse()
{
    CloseStream(::grpc::Status::CANCELLED);
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::WriteResponse(Response&& response)
{
    Actions actions;
    {
        std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
        DLOG(INFO) << "Queuing Response";

        m_ResponseQueue.push(std::move(response));
        actions = EvaluateState();
    }
    ForwardProgress(actions);
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::CloseStream(::grpc::Status status)
{
    Actions actions;
    {
        std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
        DLOG(INFO) << "Queue Close Stream: " << (status.ok() ? "OK" : "NOT OK");

        m_WritesDone = true;
        m_Status = std::make_unique<::grpc::Status>(status);

        if(!m_ReadsDone)
        {
            DLOG(INFO) << "Server Closing before Client; issue TryCancel() to flush Read Tags";
            m_Context->TryCancel();
        }

        m_ServerStream.reset();
        if(!m_ExternalStream.expired())
        {
            auto sp = m_ExternalStream.lock();
            sp->Invalidate();
        }

        actions = EvaluateState();
    }
    ForwardProgress(actions);
}

template<class Request, class Response>
void LifeCycleStreaming<Request, Response>::SetQueueFunc(ExecutorQueueFuncType queue_fn)
{
    m_QueuingFunc = queue_fn;
}

} // namespace nvrpc