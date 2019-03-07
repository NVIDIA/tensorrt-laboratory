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
#include "test_resources.h"

namespace nvrpc {
namespace testing {

TestResources::TestResources(int numThreadsInPool) : m_ThreadPool(numThreadsInPool) {}

::trtlab::ThreadPool& TestResources::AcquireThreadPool() { return m_ThreadPool; }

void TestResources::StreamManagerInit()
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    m_Running = true;
    m_ThreadPool.enqueue([this]() mutable { StreamManagerWorker(); });
}

void TestResources::StreamManagerFini()
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    m_Running = false;
}

void TestResources::StreamManagerWorker()
{
    while(m_Running)
    {
        {
            std::lock_guard<std::mutex> lock(m_MessageMutex);
            for(auto& item : m_Streams)
            {
                LOG_FIRST_N(INFO, 10) << "Progress Engine";
                auto stream_id = item.first;
                auto& stream = item.second;

                for(size_t i = m_MessagesSent[stream_id] + 1; i <= m_MessagesRecv[stream_id]; i++)
                {
                    DLOG(INFO) << "Writing: " << i;
                    Output output;
                    output.set_batch_id(i);
                    stream->WriteResponse(std::move(output));
                }

                m_MessagesSent[stream_id] = m_MessagesRecv[stream_id];
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void TestResources::CloseStream(Stream stream)
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    auto stream_id = stream->StreamID();

    m_Streams.erase(stream_id);
    m_MessagesRecv.erase(stream_id);
    m_MessagesSent.erase(stream_id);

    DLOG(INFO) << "****** Client Closed ****** ";
    stream->FinishStream();
}

void TestResources::IncrementStreamCount(Stream stream)
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    auto stream_id = stream->StreamID();
    auto search = m_Streams.find(stream_id);
    if(search == m_Streams.end())
    {
        m_Streams[stream_id] = stream;
        m_MessagesRecv[stream_id] = 1;
    }
    else
    {
        m_MessagesRecv[stream_id]++;
    }
}

} // namespace testing
} // namespace nvrpc