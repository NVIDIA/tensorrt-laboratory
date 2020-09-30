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

#include "test_pingpong.h"

#include "nvrpc/server.h"

#include "test_build_client.h"
#include "test_build_server.h"

#include <nvrpc/fiber/executor.h>

#include <gtest/gtest.h>

#define PINGPONG_SEND_COUNT 10

using namespace nvrpc;
using namespace nvrpc::testing;

void PingPongUnaryContext::ExecuteRPC(Input& input, Output& output)
{
    auto headers    = ClientMetadata();
    auto model_name = headers.find("x-content-model");
    EXPECT_NE(model_name, headers.end());
    EXPECT_EQ(model_name->second, "flowers-152");
    output.set_batch_id(input.batch_id());
    FinishResponse();
}

void PingPongStreamingContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    EXPECT_EQ(++m_Counter, input.batch_id());

    EXPECT_NE(stream, nullptr);
    Output output;
    output.set_batch_id(input.batch_id());
    stream->WriteResponse(std::move(output));
}

void PingPongStreamingContext::StreamInitialized(std::shared_ptr<ServerStream> stream)
{
    m_Counter = 0;
}


/**
 * @brief Server->Client stream closes with OK before Client->Server stream
 *
 * In this test, the Server closes its server->client stream with a call to FinishStream.  This
 * essentially says, "me the server is happy with what it has gotten and this call was a success".
 *
 * The server will continue to process incoming requests from the client, but will not be able to
 * send back responses.
 *
 * In the EarlyCancel test, we call CancelStream, which will also immediately stop and drain the
 * processing of incoming requests.
 */
void PingPongStreamingEarlyFinishContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    //static size_t counter = 0;
    //m_Counter             = ++counter;
    m_Counter++;
    EXPECT_EQ(m_Counter, input.batch_id());

    if (stream && m_Counter > PINGPONG_SEND_COUNT / 2)
    {
        // We are closing the server->client portion of the stream early
        EXPECT_NE(stream, nullptr);
        stream->FinishStream();
    }
    if (!stream || !stream->IsConnected())
    {
        // Stream was closed
        EXPECT_GT(m_Counter, PINGPONG_SEND_COUNT / 2);
        return;
    }

    EXPECT_NE(stream, nullptr);
    Output output;
    output.set_batch_id(input.batch_id());
    stream->WriteResponse(std::move(output));
}

void PingPongStreamingEarlyFinishContext::StreamInitialized(std::shared_ptr<ServerStream> stream)
{
    m_Counter = 0;
}

void PingPongStreamingEarlyFinishContext::RequestsFinished(std::shared_ptr<ServerStream> stream)
{
    // The Server should still receive all incoming requests until the client sends WritesDone
    EXPECT_EQ(m_Counter, PINGPONG_SEND_COUNT);
}

/**
 * @brief Server->Client stream closes with CANCELLED before Client->Server stream
 *
 * In this test, the Server closes its server->client stream with a call to CancelStream.  This
 * essentially says, "me the server is unhappy with what it has gotten and its time to shut down"
 *
 * The server will stop processing incoming requests from the client and will not be able to
 * send back responses.
 */
void PingPongStreamingEarlyCancelContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    // static size_t counter = 0;
    // m_Counter             = ++counter;
    m_Counter++;
    EXPECT_EQ(m_Counter, input.batch_id());

    if (stream && m_Counter > PINGPONG_SEND_COUNT / 2)
    {
        // We are closing the server->client portion of the stream early
        EXPECT_NE(stream, nullptr);
        stream->CancelStream();
    }
    if (!stream || !stream->IsConnected())
    {
        // Stream was closed
        EXPECT_EQ(m_Counter, PINGPONG_SEND_COUNT / 2 + 1);
        return;
    }

    EXPECT_NE(stream, nullptr);
    Output output;
    output.set_batch_id(input.batch_id());
    stream->WriteResponse(std::move(output));
}

void PingPongStreamingEarlyCancelContext::StreamInitialized(std::shared_ptr<ServerStream> stream)
{
    m_Counter = 0;
}

void PingPongStreamingEarlyCancelContext::RequestsFinished(std::shared_ptr<ServerStream> stream)
{
    // The Server should still receive all incoming requests until the client sends WritesDone
    EXPECT_EQ(m_Counter, PINGPONG_SEND_COUNT / 2);
}

class PingPongTest : public ::testing::Test
{
    void SetUp() override {}

    void TearDown() override
    {
        if (m_Server)
        {
            m_Server->Shutdown();
            m_Server.reset();
        }
    }

protected:
    std::unique_ptr<Server> m_Server;
};

TEST_F(PingPongTest, UnaryTest)
{
    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingContext>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex  mutex;
    std::size_t count      = 0;
    std::size_t recv_count = 0;
    std::size_t send_count = PINGPONG_SEND_COUNT;

    auto client = BuildUnaryClient();

    std::vector<std::shared_future<void>> futures;

    for (int i = 1; i <= send_count; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        std::map<std::string, std::string> headers = {{"x-content-model", "flowers-152"}};
        futures.push_back(client->Enqueue(std::move(input),
                                          [&mutex, &count, &recv_count, i](Input& input, Output& output, ::grpc::Status& status) {
                                              EXPECT_EQ(output.batch_id(), i);
                                              EXPECT_TRUE(status.ok());
                                              std::lock_guard<std::mutex> lock(mutex);
                                              --count;
                                              ++recv_count;
                                          },
                                          headers));
    }

    for (auto& future : futures)
    {
        future.wait();
    }

    EXPECT_EQ(count, 0UL);
    EXPECT_EQ(send_count, recv_count);
    EXPECT_TRUE(m_Server->Running());

    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());
}

TEST_F(PingPongTest, FibersUnaryTest)
{
    // set up worker fiber pool
    trtlab::ThreadPool                    workers(1);
    bool                                  workers_running = true;
    std::mutex                            workers_mutex;
    boost::fibers::condition_variable_any workers_cv;
    for (int i = 0; i < workers.Size(); i++)
    {
        workers.enqueue([&workers_mutex, &workers_cv, &workers_running] {
            // start the fiber scheduler and put the main to deferred sleep
            LOG(INFO) << "fiber runner thread id: " << std::this_thread::get_id();
            boost::fibers::use_scheduling_algorithm<boost::fibers::algo::shared_work>();
            std::unique_lock<std::mutex> lock(workers_mutex);
            workers_cv.wait(lock, [&workers_running]() { return !workers_running; });
        });
    }

    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingContext, FiberExecutor>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex  mutex;
    std::size_t count      = 0;
    std::size_t recv_count = 0;
    std::size_t send_count = PINGPONG_SEND_COUNT;

    auto client = BuildUnaryClient();

    std::vector<std::shared_future<void>> futures;

    for (int i = 1; i <= send_count; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        std::map<std::string, std::string> headers = {{"x-content-model", "flowers-152"}};
        futures.push_back(client->Enqueue(std::move(input),
                                          [&mutex, &count, &recv_count, i](Input& input, Output& output, ::grpc::Status& status) {
                                              EXPECT_EQ(output.batch_id(), i);
                                              EXPECT_TRUE(status.ok());
                                              std::lock_guard<std::mutex> lock(mutex);
                                              --count;
                                              ++recv_count;
                                          },
                                          headers));
    }

    for (auto& future : futures)
    {
        future.wait();
    }

    EXPECT_EQ(count, 0UL);
    EXPECT_EQ(send_count, recv_count);
    EXPECT_TRUE(m_Server->Running());

    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());

    // shutdown worker fibers
    std::unique_lock<std::mutex> lock(workers_mutex);
    workers_running = false;
    lock.unlock();
    workers_cv.notify_all();
}

TEST_F(PingPongTest, StreamingTest)
{
    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingContext>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex  mutex;
    std::size_t count      = 0;
    std::size_t recv_count = 0;
    std::size_t send_count = PINGPONG_SEND_COUNT;

    auto on_recv = [&mutex, &count, &recv_count](Output&& response) {
        static size_t last = 0;
        EXPECT_EQ(++last, response.batch_id());
        std::lock_guard<std::mutex> lock(mutex);
        --count;
        ++recv_count;
    };

    auto stream = BuildStreamingClient([](Input&&) {}, on_recv);

    for (int i = 1; i <= send_count; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        EXPECT_TRUE(stream->Write(std::move(input)));
    }

    auto future = stream->Done();
    auto status = future.get();

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(count, 0UL);
    EXPECT_EQ(send_count, recv_count);
    EXPECT_TRUE(m_Server->Running());

    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());
}

TEST_F(PingPongTest, FibersStreamingTest)
{

    // set up worker fiber pool
    trtlab::ThreadPool                    workers(4);
    bool                                  workers_running = true;
    std::mutex                            workers_mutex;
    boost::fibers::condition_variable_any workers_cv;
    for (int i = 0; i < workers.Size(); i++)
    {
        workers.enqueue([&workers_mutex, &workers_cv, &workers_running] {
            // start the fiber scheduler and put the main to deferred sleep
            LOG(INFO) << "fiber runner thread id: " << std::this_thread::get_id();
            boost::fibers::use_scheduling_algorithm<boost::fibers::algo::shared_work>();
            std::unique_lock<std::mutex> lock(workers_mutex);
            workers_cv.wait(lock, [&workers_running]() { return !workers_running; });
        });
    }

    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingContext, FiberExecutor>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex  mutex;
    std::size_t count      = 0;
    std::size_t recv_count = 0;
    std::size_t send_count = PINGPONG_SEND_COUNT;

    auto on_recv = [&mutex, &count, &recv_count](Output&& response) {
        static size_t last = 0;
        EXPECT_EQ(++last, response.batch_id());
        std::lock_guard<std::mutex> lock(mutex);
        --count;
        ++recv_count;
    };

    auto stream = BuildStreamingClient([](Input&&) {}, on_recv);

    for (int i = 1; i <= send_count; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        EXPECT_TRUE(stream->Write(std::move(input)));
    }

    auto future = stream->Done();
    auto status = future.get();

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(count, 0UL);
    EXPECT_EQ(send_count, recv_count);
    EXPECT_TRUE(m_Server->Running());

    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());

    // shutdown worker fibers
    std::unique_lock<std::mutex> lock(workers_mutex);
    workers_running = false;
    lock.unlock();
    workers_cv.notify_all();
}

TEST_F(PingPongTest, ServerEarlyFinish)
{
    m_Server = BuildStreamingServer<PingPongStreamingEarlyFinishContext>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex  mutex;
    std::size_t count      = 0;
    std::size_t recv_count = 0;
    std::size_t send_count = PINGPONG_SEND_COUNT;

    auto on_recv = [&mutex, &count, &recv_count](Output&& response) {
        static size_t last = 0;
        EXPECT_EQ(++last, response.batch_id());
        std::lock_guard<std::mutex> lock(mutex);
        --count;
        ++recv_count;
    };

    auto stream = BuildStreamingClient([](Input&&) {}, on_recv);

    for (int i = 1; i <= send_count; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        EXPECT_TRUE(stream->Write(std::move(input)));
    }

    auto future = stream->Done();
    auto status = future.get();

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(send_count / 2, recv_count);
    EXPECT_TRUE(m_Server->Running());

    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());
}

TEST_F(PingPongTest, ServerEarlyCancel)
{
    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingEarlyCancelContext>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex  mutex;
    std::size_t count      = 0;
    std::size_t recv_count = 0;
    std::size_t send_count = PINGPONG_SEND_COUNT;

    auto on_recv = [&mutex, &count, &recv_count](Output&& response) {
        static size_t last = 0;
        EXPECT_EQ(++last, response.batch_id());
        std::lock_guard<std::mutex> lock(mutex);
        --count;
        ++recv_count;
    };

    auto stream = BuildStreamingClient([](Input&&) {}, on_recv);

    for (int i = 1; i <= send_count; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++count;
        }
        Input input;
        input.set_batch_id(i);
        EXPECT_TRUE(stream->Write(std::move(input)));
    }

    auto future = stream->Done();
    auto status = future.get();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(send_count / 2, recv_count);
    EXPECT_TRUE(m_Server->Running());

    // We need a sleep here - The Server's TryCancel() seems to
    // issue an OOB CANCELLED such that the Client receives the
    // status before the server actually flushes and shuts down.
    // This is expected behavior on gRPC cancelling.
    // The wait allows the server to complete it's testing
    // Since Client and Server are in the same process, we could
    // use a mutex+condition to synchronize this event. Anyone?
    std::this_thread::sleep_for(std::chrono::seconds(1));

    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());
}
