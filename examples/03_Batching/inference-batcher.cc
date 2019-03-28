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
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

#include "nvrpc/context.h"
#include "nvrpc/executor.h"
#include "nvrpc/server.h"
#include "trtlab/core/thread_pool.h"

using nvrpc::Context;
using nvrpc::Executor;
using nvrpc::Server;
using trtlab::ThreadPool;

#include "moodycamel/blockingconcurrentqueue.h"

using moodycamel::BlockingConcurrentQueue;
using moodycamel::ConsumerToken;
using moodycamel::ProducerToken;

#include "echo.grpc.pb.h"
#include "echo.pb.h"

/**
 * @brief Batching Service for Unary Requests
 *
 * Exposes a Unary (send/recv) interface for a given RPC, but rather than
 * computing the RPC, the service simply batches the incoming requests and
 * forwards them via a gRPC stream to a service that implements the actual
 * compute portion of the RPC.
 *
 * The backend compute service is not a Unary service.  Rather it must
 * implemented the LifeCycleBatching service Context, i.e. BatchingContext.
 * The other application in this folder implements the backend service.
 *
 * Streams are used as a forwarding mechanism because of how they interact
 * with a load-balancer.  Unlike unary requests which get balanced on each
 * request, a stream only get balanced when it is opened.  All items of a stream
 * go to the same endpoint service.
 *
 * @tparam ServiceType
 * @tparam Request
 * @tparam Response
 */
template<class ServiceType, class Request, class Response>
struct BatchingService
{
    using Callback = std::function<void(bool)>;

    struct MessageType
    {
        Request* request;
        Response* response;
        Callback callback;
    };

    /**
     * @brief Forwards incoming Unary requests via a gRPC Stream to
     * a Batched Steaming Service that implements the actual RPC
     */
    class Client
    {
      public:
        using PrepareFunc =
            std::function<std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>>(
                ::grpc::ClientContext*, ::grpc::CompletionQueue*)>;

        Client(PrepareFunc prepare_func, std::shared_ptr<ThreadPool> thread_pool)
            : m_PrepareFunc(prepare_func), m_ThreadPool(thread_pool), m_CurrentCQ(0)
        {
            for(decltype(m_ThreadPool->Size()) i = 0; i < m_ThreadPool->Size(); i++)
            {
                LOG(INFO) << "Starting Client Progress Engine #" << i;
                m_CQs.emplace_back(new ::grpc::CompletionQueue);
                auto cq = m_CQs.back().get();
                m_ThreadPool->enqueue([this, cq] { ProgressEngine(*cq); });
            }
        }

        void WriteAndCloseStream(uint32_t messages_count, MessageType* messages)
        {
            auto cq = m_CQs[++m_CurrentCQ % m_CQs.size()].get();
            LOG(INFO) << "Client using CQ: " << (void*)cq;

            auto ctx = new Call;
            for(uint32_t i = 0; i < messages_count; i++)
            {
                ctx->Push(messages[i]);
            }

            ctx->m_Stream = m_PrepareFunc(&ctx->m_Context, cq);
            ctx->Start();
        }

      private:
        class Call
        {
          public:
            Call() : m_Started(false), m_NextState(&Call::StateInvalid) {}
            virtual ~Call() {}

            void Push(MessageType& message)
            {
                if(m_Started) LOG(FATAL) << "Stream started; No pushing allowed.";
                m_Requests.push(message.request);
                m_Responses.push(message.response);
                m_CallbackByResponse[message.response] = message.callback;
            }

            void Start()
            {
                LOG(INFO) << "Starting Batch Forwarding of Size " << m_Requests.size()
                          << " for Tag " << Tag();
                m_NextState = &Call::StateWriteDone;
                m_Stream->StartCall(Tag());
            }

          private:
            bool RunNextState(bool ok)
            {
                bool ret = (this->*m_NextState)(ok);
                if(!ret) DLOG(INFO) << "RunNextState returning false";
                return ret;
            }

            void* Tag() { return static_cast<void*>(this); }

            bool Fail()
            {
                LOG(FATAL) << "Fail";
                return false;
            }

            void WriteNext()
            {
                if(m_Requests.size())
                {
                    auto request = m_Requests.front();
                    m_Requests.pop();
                    DLOG(INFO) << "forwarding request";
                    m_NextState = &Call::StateWriteDone;
                    m_Stream->Write(*request, Tag());
                }
                else
                {
                    DLOG(INFO) << "closing client stream for writing";
                    m_NextState = &Call::StateCloseStreamDone;
                    m_Stream->WritesDone(Tag());
                }
            }

            void ReadNext()
            {
                if(m_Responses.size())
                {
                    DLOG(INFO) << "waiting on response";
                    auto response = m_Responses.front();
                    m_NextState = &Call::StateReadDone;
                    m_Stream->Read(response, Tag());
                }
                else
                {
                    DLOG(INFO) << "waiting on finished message from server";
                    m_NextState = &Call::StateFinishedDone;
                    m_Stream->Finish(&m_Status, Tag());
                }
            }

            bool StateWriteDone(bool ok)
            {
                if(!ok) return Fail();
                DLOG(INFO) << "request forwarded!";
                WriteNext();
                return true;
            }

            bool StateReadDone(bool ok)
            {
                if(!ok) return Fail();
                DLOG(INFO) << "response received";
                auto response = m_Responses.front();
                m_Responses.pop();
                auto search = m_CallbackByResponse.find(response);
                if(search == m_CallbackByResponse.end())
                    LOG(FATAL) << "Callback for response not found";
                ReadNext();
                // Execute callback which will complete the unary request for this stream item
                DLOG(INFO) << "triggering callback on held receive context";
                search->second(true);
                DLOG(INFO) << "callback completed";
                return true;
            }

            bool StateCloseStreamDone(bool ok)
            {
                if(!ok) return Fail();
                DLOG(INFO) << "closed client stream for writing";
                ReadNext();
                return true;
            }

            bool StateFinishedDone(bool ok)
            {
                if(m_Status.ok())
                    DLOG(INFO) << "ClientContext: " << Tag() << " finished with OK";
                else
                    DLOG(INFO) << "ClientContext: " << Tag() << " finished with CANCELLED";
                m_NextState = &Call::StateInvalid;
                LOG(INFO) << "Batch Forwarding Completed for Tag " << Tag();
                return false;
            }

            bool StateInvalid(bool ok) { LOG(FATAL) << "This should never be called"; }

          private:
            std::queue<Request*> m_Requests;
            std::queue<Response*> m_Responses;
            std::map<const Response*, Callback> m_CallbackByResponse;

            bool (Call::*m_NextState)(bool);

            ::grpc::Status m_Status;
            ::grpc::ClientContext m_Context;
            std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>> m_Stream;
            bool m_Started;

            friend class Client;
        };

        void ProgressEngine(::grpc::CompletionQueue& cq)
        {
            void* tag;
            bool ok = false;

            while(cq.Next(&tag, &ok))
            {
                CHECK(ok) << "not ok";
                Call* call = static_cast<Call*>(tag);
                if(!call->RunNextState(ok))
                {
                    DLOG(INFO) << "Deleting Stream: " << tag;
                    delete call;
                }
            }
        }

        int m_CurrentCQ;
        PrepareFunc m_PrepareFunc;
        std::shared_ptr<ThreadPool> m_ThreadPool;
        std::vector<std::unique_ptr<::grpc::CompletionQueue>> m_CQs;
    };

    class Resources : public ::trtlab::Resources
    {
      public:
        Resources(uint32_t max_batch_size, uint64_t timeout, std::shared_ptr<Client> client)
            : m_MaxBatchsize(max_batch_size), m_Timeout(timeout), m_Client(client)
        {
        }

        virtual void PreprocessRequest(Request* req) {}

        void Push(Request* req, Response* resp, Callback callback)
        {
            // thread_local ProducerToken token(m_MessageQueue);
            // m_MessageQueue.enqueue(token, MessageType(req, resp, callback));
            PreprocessRequest(req);
            m_MessageQueue.enqueue(MessageType{req, resp, callback});
        }

        void ProgressEngine()
        {
            constexpr uint64_t quanta = 100;
            const double timeout = static_cast<double>(m_Timeout - quanta) / 1000000.0;
            size_t total_count;
            size_t max_batch;

            thread_local ConsumerToken token(m_MessageQueue);
            for(;;)
            {
                MessageType messages[m_MaxBatchsize];
                max_batch = m_MaxBatchsize;
                total_count = 0;
                auto start = std::chrono::steady_clock::now();
                auto elapsed = [start]() -> double {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                        .count();
                };
                do
                {
                    auto count = m_MessageQueue.wait_dequeue_bulk_timed(
                        token, &messages[total_count], max_batch, quanta);
                    total_count += count;
                    max_batch -= count;
                } while(total_count && total_count < m_MaxBatchsize && elapsed() < timeout);
                if(total_count)
                {
                    m_Client->WriteAndCloseStream(total_count, messages);
                }
            }
        }

      private:
        size_t m_MaxBatchsize;
        uint64_t m_Timeout;
        std::shared_ptr<Client> m_Client;
        BlockingConcurrentQueue<MessageType> m_MessageQueue;
    };

    class ReceiveContext final : public ::nvrpc::Context<Request, Response, Resources>
    {
        void ExecuteRPC(Request& request, Response& response) final override
        {
            LOG(INFO) << "incoming unary request";
            this->GetResources()->Push(&request, &response, [this](bool ok) {
                if(ok)
                {
                    this->FinishResponse();
                }
                else
                {
                    LOG(INFO) << "shoot";
                    this->CancelResponse();
                }
            });
        }
    };
};

DEFINE_uint32(max_batch_size, 8, "Maximum batch size to collect and foward");
DEFINE_uint64(timeout_usecs, 2000, "Batching window timeout in microseconds");
DEFINE_uint32(max_batches_in_flight, 1, "Maximum number of forwarded batches");
DEFINE_uint32(receiving_threads, 1, "Number of Forwarding threads");
DEFINE_uint32(forwarding_threads, 1, "Number of Forwarding threads");
DEFINE_string(forwarding_target, "localhost:50051", "Batched Compute Service / Load-Balancer");

using InferenceBatchingService = BatchingService<simple::Inference, simple::Input, simple::Output>;

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("simpleBatchingService");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    auto forwarding_threads = std::make_shared<ThreadPool>(FLAGS_forwarding_threads);
    auto channel = grpc::CreateChannel(FLAGS_forwarding_target, grpc::InsecureChannelCredentials());
    auto stub = ::simple::Inference::NewStub(channel);
    auto forwarding_prepare_func = [&stub](::grpc::ClientContext * context,
                                           ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(stub->PrepareAsyncBatchedCompute(context, cq));
    };

    auto client = std::make_shared<InferenceBatchingService::Client>(forwarding_prepare_func,
                                                                     forwarding_threads);

    auto rpcResources = std::make_shared<InferenceBatchingService::Resources>(
        FLAGS_max_batch_size, FLAGS_timeout_usecs, client);

    Server server("0.0.0.0:50049");
    auto recvService = server.RegisterAsyncService<::simple::Inference>();
    auto rpcCompute = recvService->RegisterRPC<InferenceBatchingService::ReceiveContext>(
        &::simple::Inference::AsyncService::RequestCompute);

    uint64_t context_count = FLAGS_max_batch_size * FLAGS_max_batches_in_flight;
    uint64_t contexts_per_executor_thread = std::max(context_count / FLAGS_receiving_threads, 1UL);

    auto executor = server.RegisterExecutor(new Executor(FLAGS_receiving_threads));
    executor->RegisterContexts(rpcCompute, rpcResources, contexts_per_executor_thread);

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(1), [rpcResources] { rpcResources->ProgressEngine(); });

    return 0;
}