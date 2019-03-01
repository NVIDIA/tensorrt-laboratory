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

#include <gtest/gtest.h>

#define PINGPONG_SEND_COUNT 10

namespace nvrpc {
namespace testing {

void PingPongUnaryContext::ExecuteRPC(Input& input, Output& output)
{
    output.set_batch_id(input.batch_id());
    FinishResponse();
}

void PingPongStreamingContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    static size_t counter = 0;
    EXPECT_EQ(++counter, input.batch_id());

    EXPECT_NE(stream, nullptr);
    Output output;
    output.set_batch_id(input.batch_id());
    stream->WriteResponse(std::move(output));
}

void PingPongStreamingEarlyFinishContext::RequestReceived(Input&& input, std::shared_ptr<ServerStream> stream)
{
    static size_t counter = 0;
    m_Counter = ++counter;
    EXPECT_EQ(m_Counter, input.batch_id());

    if(stream && counter > PINGPONG_SEND_COUNT/2)
    {
        // We are closing the server->client portion of the stream early
        EXPECT_NE(stream, nullptr);
        stream->FinishStream();
    }
    if(!stream || !stream->IsConnected())
    {
        // Stream was closed
        EXPECT_GT(counter, PINGPONG_SEND_COUNT/2);
        return;
    }

    EXPECT_NE(stream, nullptr);
    Output output;
    output.set_batch_id(input.batch_id());
    stream->WriteResponse(std::move(output));
}

void PingPongStreamingEarlyFinishContext::RequestsFinished(std::shared_ptr<ServerStream> stream)
{
    // The Server should still receive all incoming requests until the client sends WritesDone
    EXPECT_EQ(m_Counter, PINGPONG_SEND_COUNT);
}

class PingPongTest : public ::testing::Test
{
    void SetUp() override
    {
    }

    void TearDown() override
    {
        if(m_Server)
        {
            m_Server->Shutdown();
            m_Server.reset();
        }
    }

  protected:
    std::unique_ptr<Server> m_Server;
};

TEST_F(PingPongTest, StreamingTest)
{
    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingContext>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex mutex;
    std::size_t count = 0;
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

    for(int i = 1; i <= send_count; i++)
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

TEST_F(PingPongTest, ServerEarlyFinish)
{
    m_Server = BuildServer<PingPongUnaryContext, PingPongStreamingEarlyFinishContext>();
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());

    std::mutex mutex;
    std::size_t count = 0;
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

    for(int i = 1; i <= send_count; i++)
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

} // namespace testing
} // namespace nvrpc