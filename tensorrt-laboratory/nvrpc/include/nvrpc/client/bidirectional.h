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

#include <grpc++/grpc++.h>

#include "nvrpc/client/base_context.h"
#include "nvrpc/client/executor.h"
#include "tensorrt/laboratory/core/async_compute.h"

#include <glog/logging.h>

namespace nvrpc {
namespace client {

template<typename Request, typename Response>
struct ClientBidirectional : public BaseContext
{
  public:
    using PrepareFn =
        std::function<std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>>(
            ::grpc::ClientContext*, ::grpc::CompletionQueue*)>;

    using ReadCallback = std::function<void(Response&&)>;
    using WriteCallback = std::function<void(Request&&)>;

    ClientBidirectional(PrepareFn, std::shared_ptr<Executor>, WriteCallback, ReadCallback);
    ~ClientBidirectional() { DLOG(INFO) << "ClientBidirectional dtor"; }

    void Send(Request*);
    void Send(Request&&);
    std::future<::grpc::Status> Done();

  private:
    bool RunNextState(bool ok) final override
    {
        LOG(FATAL) << "should never be called";
    }

    bool RunNextState(bool (ClientBidirectional<Request, Response>::*state_fn)(bool), bool ok)
    {
        return (this->*state_fn)(ok);
    }

    template<typename RequestType, typename ResponseType>
    class Context : public BaseContext
    {
      public:
        Context(BaseContext* master) : BaseContext(master) {}
        ~Context() override {}

      private:
        bool RunNextState(bool ok) final override
        {
            // DLOG(INFO) << "Event for Tag: " << Tag();
            return static_cast<ClientBidirectional*>(m_MasterContext)
                ->RunNextState(m_NextState, ok);
        }

        bool (ClientBidirectional<RequestType, ResponseType>::*m_NextState)(bool);

        friend class ClientBidirectional<RequestType, ResponseType>;
    };

    ::grpc::Status m_Status;
    ::grpc::ClientContext m_Context;
    std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>> m_Stream;
    std::promise<::grpc::Status> m_Promise;

    ReadCallback m_ReadCallback;
    WriteCallback m_WriteCallback;

    Context<Request, Response> m_ReadState;
    Context<Request, Response> m_WriteState;

    std::mutex m_Mutex;
    std::queue<Response> m_ReadQueue;
    std::queue<Request> m_WriteQueue;

    bool m_WritesDone;

    std::shared_ptr<Executor> m_Executor;

    bool StateStreamInitialized(bool);
    bool StateReadDone(bool);
    bool StateWriteDone(bool);
    bool StateWritesDoneDone(bool);
    bool StateFinishedDone(bool);
    bool StateInvalid(bool);
    bool StateIdle(bool);
};

template<typename Request, typename Response>
ClientBidirectional<Request, Response>::ClientBidirectional(PrepareFn prepare_fn,
                                                            std::shared_ptr<Executor> executor,
                                                            WriteCallback OnWrite,
                                                            ReadCallback OnRead)
    : m_Executor(executor), m_ReadState(this), m_WriteState(this), m_WritesDone(false),
      m_ReadCallback(OnRead), m_WriteCallback(OnWrite)
{
    m_ReadState.m_NextState = &ClientBidirectional<Request, Response>::StateInvalid;
    m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateStreamInitialized;

    // DLOG(INFO) << "Read Tag: " << m_ReadState.Tag();
    // DLOG(INFO) << "Write Tag: " << m_WriteState.Tag();
    // DLOG(INFO) << "Master Tag: " << Tag();

    m_Stream = prepare_fn(&m_Context, m_Executor->GetNextCQ());
    m_Stream->StartCall(m_WriteState.Tag());
}

template<typename Request, typename Response>
void ClientBidirectional<Request, Response>::Send(Request* request)
{
    Send(std::move(*request));
}

template<typename Request, typename Response>
void ClientBidirectional<Request, Response>::Send(Request&& request)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    if(m_WritesDone)
    {
        std::runtime_error("Cannot Send new requests after Done has been called");
    }
    m_WriteQueue.push(std::move(request));
    if(m_WriteState.m_NextState == &ClientBidirectional<Request, Response>::StateIdle)
    {
        m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateWriteDone;
        m_Stream->Write(m_WriteQueue.front(), m_WriteState.Tag());
    }
}

template<typename Request, typename Response>
std::future<::grpc::Status> ClientBidirectional<Request, Response>::Done()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_WritesDone = true;
    if(m_WriteState.m_NextState == &ClientBidirectional<Request, Response>::StateIdle)
    {
        m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateWritesDoneDone;
        m_Stream->WritesDone(m_WriteState.Tag());
    }
    return m_Promise.get_future();
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateStreamInitialized(bool ok)
{
    if(!ok)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_Mutex);

    DLOG(INFO) << "StateStreamInitialized";

    // WriteState was used to create the stream connection
    // Set to idle until a write is queued; otherwise, write next request

    if(m_WriteQueue.empty())
    {
        m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateIdle;
        if(m_WritesDone)
        {
            m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateWritesDoneDone;
            m_Stream->WritesDone(m_WriteState.Tag());
        }
    }
    else
    {
        m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateWriteDone;
        m_Stream->Write(m_WriteQueue.front(), m_WriteState.Tag());
    }

    // Independent of the Writes, queue up a Read
    m_ReadQueue.emplace();
    m_ReadState.m_NextState = &ClientBidirectional<Request, Response>::StateReadDone;
    m_Stream->Read(&m_ReadQueue.back(), m_ReadState.Tag());
    return true;
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateWriteDone(bool ok)
{
    if(!ok)
    {
        // Invalidate any outstanding reads on stream
        DLOG(ERROR) << "WriteDone got a false";
        m_Context.TryCancel();
        return false;
    }

    // First request in m_WriteQueue has completed sending
    // Process that request after we [possibly] queue the next write
    Request sent_request;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        sent_request = std::move(m_WriteQueue.front());
        m_WriteQueue.pop();

        if(m_WriteQueue.empty())
        {
            m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateIdle;
            if(m_WritesDone)
            {
                m_WriteState.m_NextState =
                    &ClientBidirectional<Request, Response>::StateWritesDoneDone;
                m_Stream->WritesDone(m_WriteState.Tag());
            }
        }
        else
        {
            m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateWriteDone;
            m_Stream->Write(m_WriteQueue.front(), m_WriteState.Tag());
        }
    }

    m_WriteCallback(std::move(sent_request));
    return true;
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateReadDone(bool ok)
{
    // Message received by reading from stream
    // No need to lock as the user has no access to the Read portion of the stream

    DLOG(INFO) << "StateReadDone triggered";

    if(!ok)
    {
        DLOG(INFO) << "Server is closing the read/download portion of the stream";
        m_ReadState.m_NextState = &ClientBidirectional<Request, Response>::StateFinishedDone;
        m_Stream->Finish(&m_Status, m_ReadState.Tag());
        m_ReadQueue.pop();
        return true;
    }

    // Before processing the current message, post a read
    m_ReadQueue.emplace();
    m_ReadState.m_NextState = &ClientBidirectional<Request, Response>::StateReadDone;
    m_Stream->Read(&m_ReadQueue.back(), m_ReadState.Tag());

    // Process the message received
    m_ReadCallback(std::move(m_ReadQueue.front()));
    m_ReadQueue.pop();

    return true;
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateWritesDoneDone(bool ok)
{
    if(!ok)
    {
        LOG(ERROR) << "Failed to close write/upload portion of stream";
        m_Context.TryCancel();
        return false;
    }

    DLOG(INFO) << "Write/Upload portion of the stream has closed";

    std::lock_guard<std::mutex> lock(m_Mutex);
    m_WriteState.m_NextState = &ClientBidirectional<Request, Response>::StateInvalid;

    if(m_ReadState.m_NextState == &ClientBidirectional<Request, Response>::StateInvalid)
    {
        DLOG(INFO) << "Read/Download has already finished - completing future";
        m_Promise.set_value(std::move(m_Status));
    }
    else
    {
        DLOG(INFO) << "Waiting for Read/Download half of the stream to close";
    }

    return true;
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateFinishedDone(bool ok)
{
    if(!ok)
    {
        LOG(ERROR) << "Failed to close read/download portion of the stream";
        m_Context.TryCancel();
        return false;
    }

    DLOG(INFO) << "Read/Download portion of the stream has closed";

    std::lock_guard<std::mutex> lock(m_Mutex);
    m_ReadState.m_NextState = &ClientBidirectional<Request, Response>::StateInvalid;

    if(m_WriteState.m_NextState == &ClientBidirectional<Request, Response>::StateInvalid)
    {
        DLOG(INFO) << "Write/Upload has already finished - completing future";
        m_Promise.set_value(std::move(m_Status));
    }
    else
    {
        DLOG(INFO) << "Received Finished from Server before Client has sent Done writing";
    }
    return true;
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateIdle(bool ok)
{
    LOG(FATAL) << "Your logic is bad - you should never have come here";
}

template<typename Request, typename Response>
bool ClientBidirectional<Request, Response>::StateInvalid(bool ok)
{
    LOG(FATAL) << "Your logic is bad - you should never have come here";
}
/*

    template<typename OnReturnFn>
    auto Enqueue(Request* request, Response* response, OnReturnFn on_return)
    {
        auto wrapped = this->Wrap(on_return);
        auto future = wrapped->Future();

        auto ctx = m_Queue.front();
        ctx->m_Request = request;
        ctx->m_Response = response;
        ctx->m_Callback = [ctx, wrapped]() mutable {
            (*wrapped)(*ctx->m_Request, *ctx->m_Response, ctx->m_Status);
        };

        ctx->m_Reader->StartCall();
        ctx->m_Reader->Finish(ctx->m_Response, &ctx->m_Status, ctx->Tag());

        return future.share();
    }

    template<typename OnReturnFn>
    auto Enqueue(Request&& request, OnReturnFn on_return)
    {
        auto req = std::make_shared<Request>(std::move(request));
        auto resp = std::make_shared<Response>();

        auto extended_on_return = [req, resp, on_return](Request& request, Response& response,
                                                         ::grpc::Status& status) mutable -> auto{
            return on_return(request, response, status);
        };

        return Enqueue(req.get(), resp.get(), extended_on_return);
    }
*/

} // namespace client
} // namespace nvrpc