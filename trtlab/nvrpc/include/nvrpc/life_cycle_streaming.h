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

namespace nvrpc
{
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
    template <class Request, class Response>
    class LifeCycleStreaming : public IContextLifeCycle
    {
    public:
        using RequestType  = Request;
        using ResponseType = Response;
        using ServiceQueueFuncType =
            std::function<void(::grpc::ServerContext*, ::grpc_impl::ServerAsyncReaderWriter<ResponseType, RequestType>*,
                               ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*)>;
        using ExecutorQueueFuncType =
            std::function<void(::grpc::ServerContext*, ::grpc_impl::ServerAsyncReaderWriter<ResponseType, RequestType>*, void*)>;

        ~LifeCycleStreaming() override {}

        class ServerStream;

    protected:
        LifeCycleStreaming();
        void SetQueueFunc(ExecutorQueueFuncType);

        // developer must implement RequestsReceived
        virtual void RequestReceived(Request&&, std::shared_ptr<ServerStream>) = 0;

        // optional hooks to perform actions when the stream is openned/closed
        virtual void StreamInitialized(std::shared_ptr<ServerStream>) {}
        virtual void RequestsFinished(std::shared_ptr<ServerStream>) {}

        // TODO: Add an OnInitialized virtual method

    public:
        // template<typename RequestType, typename ResponseType>
        class ServerStream
        {
        public:
            // ServerStream(LifeCycleStreaming<RequestType, ResponseType>* master) : m_Stream(master) {}
            ServerStream(LifeCycleStreaming<Request, Response>* master) : m_Stream(master) {}
            ~ServerStream()
            {
                if (m_Stream)
                {
                    LOG(WARNING) << "ServerStream is still valid on deconstruction; "
                                 << "this means an external handler was the last thing to own the ServerStream, "
                                 << "but Cancel/FinishStream was never called";
                    m_Stream->UnblockFinish();
                    m_Stream->CancelResponse();
                }
            }

            bool IsConnected()
            {
                return m_Stream;
            }

            std::uint64_t StreamID()
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                if (!IsConnected())
                {
                    DLOG(WARNING) << "Attempted to get ID of a disconnected stream";
                    return 0UL;
                }
                return reinterpret_cast<std::uint64_t>(m_Stream->Tag());
            }

            bool WriteResponse(ResponseType&& response)
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                if (!IsConnected())
                {
                    DLOG(WARNING) << "Attempted to write to a disconnected stream";
                    return false;
                }
                m_Stream->WriteResponse(std::move(response));
                return true;
            }

            bool CancelStream()
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                if (!IsConnected())
                {
                    LOG(WARNING) << "Attempted to cancel to a disconnected stream";
                    return false;
                }
                m_Stream->CancelResponse();
                return false;
            }

            bool FinishStream()
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                if (!IsConnected())
                {
                    LOG(WARNING) << "Attempted to finish to a disconnected stream";
                    return false;
                }
                m_Stream->FinishResponse();
                return false;
            }

            bool BlockFinish()
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                if (!IsConnected())
                {
                    DLOG(WARNING) << "Attempted to block to a disconnected stream";
                    return false;
                }
                m_Stream->BlockFinish();
                return true;
            }

            bool UnblockFinish()
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                if (!IsConnected())
                {
                    DLOG(WARNING) << "Attempted to block to a disconnected stream";
                    return false;
                }
                m_Stream->UnblockFinish();
                return true;
            }

        protected:
            void Invalidate()
            {
                std::lock_guard<std::recursive_mutex> lock(m_Mutex);
                m_Stream = nullptr;
            }

        private:
            std::recursive_mutex                   m_Mutex;
            LifeCycleStreaming<Request, Response>* m_Stream;

            friend class LifeCycleStreaming<Request, Response>;
        };

    protected:
        template <class RequestType, class ResponseType>
        class StateContext : public IContext
        {
        public:
            StateContext(IContext* primary_context) : IContext(primary_context) {}

        private:
            // IContext Methods
            bool RunNextState(bool ok) final override
            {
                return static_cast<LifeCycleStreaming*>(m_PrimaryContext)->RunNextState(m_NextState, ok);
            }
            void Reset() final override
            {
                static_cast<LifeCycleStreaming*>(m_PrimaryContext)->Reset();
            }

            bool (LifeCycleStreaming<RequestType, ResponseType>::*m_NextState)(bool);

            friend class LifeCycleStreaming<RequestType, ResponseType>;
        };

    protected:
        void BlockFinish();
        void UnblockFinish();

    private:
        using ReadHandle    = bool;
        using WriteHandle   = bool;
        using ExecuteHandle = std::function<void()>;
        using FinishHandle  = bool;
        using Actions       = std::tuple<ReadHandle, WriteHandle, ExecuteHandle, FinishHandle>;

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
        void    ForwardProgress(Actions&);

        // User Actions
        void WriteResponse(Response&& response);
        void CloseStream(::grpc::Status);

        // Function pointers
        ExecutorQueueFuncType m_QueuingFunc;
        bool (LifeCycleStreaming<RequestType, ResponseType>::*m_NextState)(bool);

        // Internal State
        std::recursive_mutex     m_QueueMutex;
        std::queue<RequestType>  m_RequestQueue;
        std::queue<ResponseType> m_ResponseQueue;

        std::shared_ptr<ServerStream> m_ServerStream;
        std::weak_ptr<ServerStream>   m_ExternalStream;

        bool m_Reading, m_Writing, m_Finishing, m_ReadsDone, m_WritesDone, m_ReadsFinished, m_BlockFinish;

        StateContext<RequestType, ResponseType> m_ReadStateContext;
        StateContext<RequestType, ResponseType> m_WriteStateContext;

        std::unique_ptr<::grpc::Status>                                                  m_Status;
        std::unique_ptr<::grpc::ServerContext>                                           m_Context;
        std::unique_ptr<::grpc_impl::ServerAsyncReaderWriter<ResponseType, RequestType>> m_Stream;

        friend class StateContext<RequestType, ResponseType>;
        // friend class ServerStream<RequestType, ResponseType>;
        friend class ServerStream;

    public:
        template <class RequestFuncType, class ServiceType>
        static ServiceQueueFuncType BindServiceQueueFunc(RequestFuncType request_fn, ServiceType* service_type)
        {
            /*
            RequestFuncType = std::function<void(
                ServiceType *, ::grpc::ServerContext *,
                ::grpc_impl::ServerAsyncReaderWriter<ResponseType, RequestType>*,
                ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
        */
            return std::bind(request_fn, service_type,
                             std::placeholders::_1, // ServerContext*
                             std::placeholders::_2, // AsyncReaderWriter<OutputType, InputType>
                             std::placeholders::_3, // CQ
                             std::placeholders::_4, // ServerCQ
                             std::placeholders::_5  // Tag
            );
        }

        static ExecutorQueueFuncType BindExecutorQueueFunc(ServiceQueueFuncType service_q_fn, ::grpc::ServerCompletionQueue* cq)
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
    template <class Request, class Response>
    LifeCycleStreaming<Request, Response>::LifeCycleStreaming()
    : m_ReadStateContext(static_cast<IContext*>(this)),
      m_WriteStateContext(static_cast<IContext*>(this)),
      m_Reading(false),
      m_Writing(false),
      m_Finishing(false),
      m_ReadsDone(false),
      m_WritesDone(false),
      m_ReadsFinished(false),
      m_BlockFinish(false)
    {
        m_NextState                     = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
        m_ReadStateContext.m_NextState  = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
        m_WriteStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::Reset()
    {
        std::queue<RequestType>  empty_request_queue;
        std::queue<ResponseType> empty_response_queue;
        OnLifeCycleReset();
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            m_Reading       = false;
            m_Writing       = false;
            m_Finishing     = false;
            m_ReadsDone     = false;
            m_WritesDone    = false;
            m_ReadsFinished = false;
            m_RequestQueue.swap(empty_request_queue);
            m_ResponseQueue.swap(empty_response_queue);
            m_ServerStream.reset();
            m_ExternalStream.reset();

            m_NextState                     = &LifeCycleStreaming<RequestType, ResponseType>::StateInitializedDone;
            m_ReadStateContext.m_NextState  = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
            m_WriteStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;

            m_Status.reset();
            m_Context.reset(new ::grpc::ServerContext);
            m_Stream.reset(new ::grpc_impl::ServerAsyncReaderWriter<ResponseType, RequestType>(m_Context.get()));
        }
        m_QueuingFunc(m_Context.get(), m_Stream.get(), IContext::Tag());
    }

    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::RunNextState(bool ok)
    {
        return (this->*m_NextState)(ok);
    }

    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::RunNextState(bool (LifeCycleStreaming<Request, Response>::*state_fn)(bool), bool ok)
    {
        return (this->*state_fn)(ok);
    }

    template <class Request, class Response>
    typename LifeCycleStreaming<Request, Response>::Actions LifeCycleStreaming<Request, Response>::EvaluateState()
    {
        ReadHandle    should_read    = false;
        WriteHandle   should_write   = false;
        ExecuteHandle should_execute = nullptr;
        FinishHandle  should_finish  = false;

        if (!m_Reading && !m_ReadsDone)
        {
            should_read = true;
            m_Reading   = true;
            m_RequestQueue.emplace();
            m_ReadStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateReadDone;

            should_execute = [this, request = std::move(m_RequestQueue.front()), stream = m_ServerStream]() mutable {
                RequestReceived(std::move(request), stream);
            };
            m_RequestQueue.pop();
        }

        if (!m_Reading && m_ReadsDone && !m_ReadsFinished && !(m_Status || m_ExternalStream.expired()))
        {
            DCHECK_NOTNULL(m_ServerStream);
            m_ReadsFinished = true;
            should_execute  = [this, stream = m_ServerStream]() mutable {
                DVLOG(1) << "Client sent WritesDone";
                RequestsFinished(stream);
            };
            // we will never hand m_ServerStream to another handler
            // it is safe for the context to decrement the refcount
            // the ServerStream will Cancel the Stream if it is still
            // valid on deconstruction
            m_ServerStream.reset();
        }

        if (m_Status && !m_Status->ok())
        {
            DVLOG(1) << "Stream was CANCELLED by Server - Cancel All Callbacks";
            should_execute = nullptr;
        }

        if (!m_Writing && !m_ResponseQueue.empty())
        {
            should_write                    = true;
            m_Writing                       = true;
            m_WriteStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateWriteDone;
        }

        if (!m_Reading && !m_Writing && !m_Finishing && m_Status && !m_BlockFinish)
        {
            should_finish = true;
            m_Finishing   = true;
            m_NextState   = &LifeCycleStreaming<RequestType, ResponseType>::StateFinishedDone;
        }
        // clang-format off
    DVLOG(1) << (should_read ? 1 : 0) << (should_write ? 1 : 0) << (should_execute ? 1 : 0)
               << (should_finish ? 1 : 0) 
               << " -- " << m_Reading << m_Writing 
               << " -- " << m_ReadsDone << m_ReadsFinished << (m_Status ? 1 : 0) << (m_ExternalStream.expired() ? 1 : 0)
               << " -- " << m_Finishing;
        // clang-format on

        return std::make_tuple(should_read, should_write, should_execute, should_finish);
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::ForwardProgress(Actions& actions)
    {
        ReadHandle    should_read    = std::get<0>(actions);
        WriteHandle   should_write   = std::get<1>(actions);
        ExecuteHandle should_execute = std::get<2>(actions);
        FinishHandle  should_finish  = std::get<3>(actions);

        if (should_write)
        {
            DVLOG(1) << "Writing Response";
            m_Stream->Write(m_ResponseQueue.front(), m_WriteStateContext.IContext::Tag());
        }
        if (should_execute)
        {
            DVLOG(1) << "Kicking off Execution";
            should_execute();
        }
        // moved to after should_execute; this allows the m_ReadCallback method
        // to update the state of the context without contending with another
        // potential m_ReadCallback call should the Read have been posted
        if (should_read)
        {
            DVLOG(1) << "Posting Read/Recv";
            m_Stream->Read(&m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
        }
        if (should_finish)
        {
            DVLOG(1) << "Closing Stream - " << (m_Status->ok() ? "OK" : "CANCELLED");
            m_Stream->Finish(*m_Status, IContext::Tag());
        }
    }

    // The following are a set of functions used as function pointers
    // to keep track of the state of the context.
    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::StateInitializedDone(bool ok)
    {
        if (!ok)
        {
            DVLOG(2) << "Stream Initialization Failed - Server Shutting Down";
            return false;
        }

        OnLifeCycleStart();
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "Initialize Stream";

            m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;

            // Object that allows the server to response on the stream
            // new ServerStream<RequestType, ResponseType>(this), [this](auto ptr) mutable {
            m_ServerStream   = std::shared_ptr<ServerStream>(new ServerStream(this), [this](auto ptr) mutable {
                // Custom Deleter - may trigger stream closing
                Actions actions;
                {
                    std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
                    DVLOG(1) << "All ServerStream objects have been deleted";
                    actions = this->EvaluateState();
                };
                this->ForwardProgress(actions);
                delete ptr;
            });
            m_ExternalStream = m_ServerStream;
            StreamInitialized(m_ServerStream);

            // Start reading once connection is created - State
            m_Reading = true;
            m_RequestQueue.emplace();
            m_ReadStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateReadDone;
        }

        // Start reading once connection is created - Action
        m_Stream->Read(&m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
        return true;
    }

    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::StateReadDone(bool ok)
    {
        Actions actions;
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "ReadDone Event: " << (ok ? "OK" : "NOT OK");

            m_Reading = false;

            if (!ok)
            {
                {
                    // Client called WritesDone
                    DVLOG(1) << "WritesDone received from Client; closing Server Reads";
                    m_ReadsDone = true;
                    m_RequestQueue.pop();
                    m_ReadStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
                }
            }

            actions = EvaluateState();
        }
        ForwardProgress(actions);
        return true;
    }

    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::StateWriteDone(bool ok)
    {
        // If write didn't go through, then the call is dead. Start reseting
        DCHECK(m_ResponseQueue.size());

        Actions actions;
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "WriteDone Event: " << (ok ? "OK" : "NOT OK");

            m_Writing = false;

            if (!ok)
            {
                DVLOG(1) << "Write Response failed on Stream; Client may have Cancelled";
                CancelResponse();
            }
            else
            {
                // todo: add a CallbackOnResponseSent
                m_ResponseQueue.pop();
                m_WriteStateContext.m_NextState = &LifeCycleStreaming<RequestType, ResponseType>::StateInvalid;
            }

            actions = EvaluateState();
        }
        ForwardProgress(actions);
        return true;
    }

    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::StateFinishedDone(bool ok)
    {
        DVLOG(1) << "Server closed Write stream - FinishedDone - " << (ok ? "OK" : "CANCELLED");
        // Clear the m_ServerStream since we will not be launching any new tasks
        // m_ExternalStream holds a weak_ptr to the original m_ServerStream so we
        // can track when the last external reference was released.
        m_ServerStream.reset();
        if (!m_ExternalStream.expired())
        {
            auto sp = m_ExternalStream.lock();
            sp->Invalidate();
        }
        return false;
    }

    template <class Request, class Response>
    bool LifeCycleStreaming<Request, Response>::StateInvalid(bool ok)
    {
        throw std::runtime_error("invalid state");
        return false;
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::FinishResponse()
    {
        CloseStream(::grpc::Status::OK);
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::CancelResponse()
    {
        CloseStream(::grpc::Status::CANCELLED);
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::BlockFinish()
    {
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "Setting BlockFinish to true; UnblockFinish must be called for RPC to complete";
            m_BlockFinish = true;
        }
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::UnblockFinish()
    {
        Actions actions;
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "Setting BlockFinish to false - this will also trigger a step thru the progress engine";
            m_BlockFinish = false;
        }
        ForwardProgress(actions);
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::WriteResponse(Response&& response)
    {
        Actions actions;
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "Queuing Response";

            m_ResponseQueue.push(std::move(response));
            actions = EvaluateState();
        }
        ForwardProgress(actions);
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::CloseStream(::grpc::Status status)
    {
        Actions actions;
        {
            std::lock_guard<std::recursive_mutex> lock(m_QueueMutex);
            DVLOG(1) << "Queue Close Stream: " << (status.ok() ? "OK" : "NOT OK");

            m_WritesDone = true;
            m_Status     = std::make_unique<::grpc::Status>(status);

            if (!m_ReadsDone)
            {
                DVLOG(1) << "Server Canceling before Client WritesDone; issue TryCancel() to flush Read Tags";
                m_ResponseQueue = std::queue<Response>();
                m_Context->TryCancel();
            }

            actions = EvaluateState();
        }
        ForwardProgress(actions);
    }

    template <class Request, class Response>
    void LifeCycleStreaming<Request, Response>::SetQueueFunc(ExecutorQueueFuncType queue_fn)
    {
        m_QueuingFunc = queue_fn;
    }

} // namespace nvrpc