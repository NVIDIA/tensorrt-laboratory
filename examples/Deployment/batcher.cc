

template<typename Request, typename Response>
class BatchingService
{
  public:
    using PrepareFn =
        std::function<std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>>(
            ::grpc::ClientContext*, ::grpc::CompletionQueue*)>;
    using Callback = std::function<void(bool)>;

    struct MessageType
    {
        Request& request;
        Response& response;
        Callback callback;
    };

    class Resources
    {
      public:
        Resources(PrepareFn prepare_fn, std::shared_ptr<client::Executor> executor,
                  std::shared_ptr<::trtlab::ThreadPool> post_process uint32_t max_batch_size,
                  uint64_t timeout_in_us)
            : m_PrepareFn(prepare_fn), m_Executor(executor), m_WaitAndDone(post_process),
              m_MaxBatchSize(max_batch_size), m_Timeout(timeout_in_us)
        {
        }

        std::shared_ptr<ClientStreaming<Request, Response>>
            CreateClient(std::function<void(Response&&)> on_recv)
        {
            auto on_sent = [](Request&& request) {};

            return std::make_shared<client::ClientStreaming<Request, Response>>(
                m_PrepareFn, m_Executor, on_sent, on_recv);
        }

        void Enqueue(Request& req, Response& resp, Callback callback)
        {
            m_MessageQueue.enqueue(MessageType{req, resp, callback});
        }

      protected:
        void BatchingEngine()
        {
            constexpr uint64_t quanta = 100;
            const double timeout = static_cast<double>(m_Timeout - quanta) / 1000000.0;
            const size_t max_batch_size = m_MaxBatchSize;
            size_t total_count;
            size_t max_deque;
            std::shared_ptr<std::vector<MessageType>> messages;
            std::chrono::time_point<std::chrono::high_resolution_clock> start;
            thread_local ConsumerToken token(m_MessageQueue);

            // clang-format off
            auto elapsed_time =
                [](std::chrono::time_point<std::chrono::high_resolution_clock>& start) -> double {
                    return std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - start).count();
                };
            // clang-format on

            for(;;)
            {
                MessageType messages[m_MaxBatchsize];
                max_batch = m_MaxBatchsize;
                total_count = 0;

                // pull 1 element from the queue and start timer
                // if dequeue times outs, then restart the loop
                total_count = m_MessageQueue.wait_dequeue_bulk_timed(
                    token, &(*messages)[total_count], 1, quanta);
                max_deque = max_batch_size - total_count;

                if(count == 0)
                {
                    continue;
                }

                // Create a Corked Stream - Corked = buffered writes
                stream = CreateClient([messages](Response&& response) mutable {
                    CHECK(!messges.empty());
                    DLOG(INFO) << "Finishing Unary Response/Callback: " << message.size()
                               << " remain on queue";
                    auto m = messages.front();
                    m.response = std::move(response);
                    m.callback();
                    messages.erase(messages.begin());
                });
                stream->Corked(true);

                // Continue to collect inference requests until we reach a maximum batch size
                // or we hit the timeout.  We will eagerly forward our current batch items along
                // the stream so the preprocessor can get ahead start
                start = std::chrono::high_resolution_clock::now();
                while(total_count < max_batch_size && elapsed(start) <)
                {
                    total_count += total_count = m_MessageQueue.wait_dequeue_bulk_timed(
                        token, &(*messages)[total_count], max_deque, quanta);
                    max_deque = max_batch_size - total_count;

                    for(; isend < total_count; isend++)
                    {
                        auto& m = (*messages)[isend];
                        stream->Write(m.request));
                    }
                }

                // Batching complete
                if(total_count)
                {
                    messages->resize(total_count);
                    stream->Done();
                    m_WaitAndFinish.enqueue([stream]() mutable {
                        auto future = stream->Status();
                        future.wait();
                        streawm.reset();
                    });
                    messages.reset(new std::vector<MessageType>);
                    messages->resize(max_batch_size);
                }
            }
        }

      private:
        PrepareFn m_PrepareFn;
        std::shared_ptr<client::Executor> m_Executor;
        std::shared_ptr<::trtlab::ThreadPool> m_WaitAndDelete;
        size_t m_MaxBatchsize;
        uint64_t m_Timeout;
        BlockingConcurrentQueue<MessageType> m_MessageQueue;
    };

    class BatchingContext : public Context<Request, Response, Resources>
    {
        void ExecuteRPC(Request& request, Response& response) final override
        {
            LOG(INFO) << "incoming unary request";
            this->GetResources()->Enqueue(&request, &response, [this](bool ok) {
                if(ok)
                {
                    this->FinishResponse();
                }
                else
                {
                    LOG(ERROR) << "Upstream Error";
                    this->CancelResponse();
                }
            });
        }
    };
};
