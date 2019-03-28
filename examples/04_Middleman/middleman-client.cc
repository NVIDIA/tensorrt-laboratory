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

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/malloc.h"

using trtlab::Allocator;
using trtlab::Malloc;

#include "nvrpc/context.h"
#include "nvrpc/executor.h"
#include "nvrpc/server.h"

using nvrpc::Context;
using nvrpc::Executor;
using nvrpc::Server;
using trtlab::ThreadPool;

#include "moodycamel/blockingconcurrentqueue.h"

using moodycamel::BlockingConcurrentQueue;
using moodycamel::ConsumerToken;
using moodycamel::ProducerToken;

// NVIDIA Inference Server Protos
#include "grpc_service.grpc.pb.h"
#include "grpc_service.pb.h"

namespace easter = ::nvidia::inferenceserver;
/*
using nvidia::inferenceserver::GRPCService;
using nvidia::inferenceserver::InferRequest;
using nvidia::inferenceserver::InferResponse;
*/

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
struct MiddlemanService
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
            std::function<std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>>(
                ::grpc::ClientContext*, const Request&, ::grpc::CompletionQueue*)>;

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
            DLOG(INFO) << "Client using CQ: " << (void*)cq;
            CHECK_EQ(1U, messages_count) << "forwarder; not batcher";

            auto ctx = new Call;
            for(uint32_t i = 0; i < messages_count; i++)
            {
                ctx->Push(messages[i]);
            }

            ctx->m_Reader = m_PrepareFunc(&ctx->m_Context, *ctx->m_Request, cq);
            ctx->m_Reader->StartCall();
            ctx->m_Reader->Finish(ctx->m_Response, &ctx->m_Status, ctx->Tag());
        }

      private:
        class Call
        {
          public:
            Call() : m_NextState(&Call::StateFinishedDone) {}
            virtual ~Call() {}

            void Push(MessageType& message)
            {
                m_Request = message.request;
                m_Response = message.response;
                m_Callback = message.callback;
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

            bool StateFinishedDone(bool ok)
            {
                if(m_Status.ok())
                    DLOG(INFO) << "ClientContext: " << Tag() << " finished with OK";
                else
                    DLOG(INFO) << "ClientContext: " << Tag() << " finished with CANCELLED";
                m_Callback(m_Status.ok());
                DLOG(INFO) << "Forwarding Completed for Tag " << Tag();
                return false;
            }

          private:
            Request* m_Request;
            Response* m_Response;
            Callback m_Callback;

            bool (Call::*m_NextState)(bool);

            ::grpc::Status m_Status;
            ::grpc::ClientContext m_Context;
            std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>> m_Reader;

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

  public:
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
                    CHECK_LE(count, max_batch);
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
            DLOG(INFO) << "incoming unary request";
            this->GetResources()->Push(&request, &response, [this](bool ok) {
                if(ok)
                    this->FinishResponse();
                else
                {
                    LOG(INFO) << "shoot";
                    this->CancelResponse();
                }
            });
        }
    };
};

DEFINE_uint32(max_batch_size, 1, "Maximum batch size to collect and foward");
DEFINE_uint64(timeout_usecs, 200, "Batching window timeout in microseconds");
DEFINE_uint32(max_batches_in_flight, 300, "Maximum number of forwarded batches");
DEFINE_uint32(receiving_threads, 2, "Number of Forwarding threads");
DEFINE_uint32(forwarding_threads, 2, "Number of Forwarding threads");
DEFINE_string(forwarding_target, "localhost:8001", "Batched Compute Service / Load-Balancer");

using InferMiddlemanService =
    MiddlemanService<easter::GRPCService, easter::InferRequest, easter::InferResponse>;
using StatusMiddlemanService =
    MiddlemanService<easter::GRPCService, easter::StatusRequest, easter::StatusResponse>;

class DemoMiddlemanService : public InferMiddlemanService
{
  public:
    class Resources : public InferMiddlemanService::Resources
    {
      public:
        using InferMiddlemanService::Resources::Resources;
        void PreprocessRequest(easter::InferRequest* req) override
        {
            static auto local_data = std::make_unique<Allocator<Malloc>>(10 * 1024 * 1024);
            DLOG(INFO) << "Boom - preprocess request here!";
            auto bytes = req->meta_data().batch_size() * req->meta_data().input(0).batch_byte_size();
            CHECK_EQ(0, req->raw_input_size());
            req->add_raw_input(local_data->Data(), bytes);
        }
    };
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("easterForwardingService");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    auto channel = grpc::CreateCustomChannel(FLAGS_forwarding_target,
                                             grpc::InsecureChannelCredentials(), ch_args);

    // GRPCService::Infer async forwarder
    auto forwarding_threads = std::make_shared<ThreadPool>(FLAGS_forwarding_threads);
    auto stub = ::easter::GRPCService::NewStub(channel);
    auto forwarding_prepare_func = [&stub](::grpc::ClientContext * context,
                                           const ::easter::InferRequest& request,
                                           ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(stub->PrepareAsyncInfer(context, request, cq));
    };
    auto client =
        std::make_shared<DemoMiddlemanService::Client>(forwarding_prepare_func, forwarding_threads);

    // GRPCService::Status async forwarder
    auto status_forwarding_threads = std::make_shared<ThreadPool>(1);
    auto status_stub = ::easter::GRPCService::NewStub(channel);
    auto status_forwarding_prepare_func = [&stub](::grpc::ClientContext * context,
                                                  const ::easter::StatusRequest& request,
                                                  ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(stub->PrepareAsyncStatus(context, request, cq));
    };
    auto status_client = std::make_shared<StatusMiddlemanService::Client>(
        status_forwarding_prepare_func, status_forwarding_threads);

    auto rpcResources = std::make_shared<DemoMiddlemanService::Resources>(
        FLAGS_max_batch_size, FLAGS_timeout_usecs, client);

    auto statusResources = std::make_shared<StatusMiddlemanService::Resources>(
        FLAGS_max_batch_size, FLAGS_timeout_usecs, status_client);

    Server server("0.0.0.0:50049");
    auto bytes = trtlab::StringToBytes("100MiB");
    server.Builder().SetMaxReceiveMessageSize(bytes);
    LOG(INFO) << "gRPC MaxReceiveMessageSize = " << trtlab::BytesToString(bytes);

    auto recvService = server.RegisterAsyncService<::easter::GRPCService>();
    auto rpcCompute = recvService->RegisterRPC<DemoMiddlemanService::ReceiveContext>(
        &::easter::GRPCService::AsyncService::RequestInfer);
    auto rpcStatus = recvService->RegisterRPC<StatusMiddlemanService::ReceiveContext>(
        &::easter::GRPCService::AsyncService::RequestStatus);

    uint64_t context_count = FLAGS_max_batch_size * FLAGS_max_batches_in_flight;
    uint64_t contexts_per_executor_thread = std::max(context_count / FLAGS_receiving_threads, 1UL);

    auto executor = server.RegisterExecutor(new Executor(FLAGS_receiving_threads));
    executor->RegisterContexts(rpcCompute, rpcResources, contexts_per_executor_thread);

    auto status_executor = server.RegisterExecutor(new Executor(1));
    status_executor->RegisterContexts(rpcStatus, statusResources, 1);

    auto executor_threads = std::make_shared<ThreadPool>(2);
    executor_threads->enqueue([rpcResources] { rpcResources->ProgressEngine(); });
    executor_threads->enqueue([statusResources] { statusResources->ProgressEngine(); });

    LOG(INFO) << "Running Server";
    server.Run(std::chrono::milliseconds(1), [] {});
}
