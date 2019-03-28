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

#include "trtlab/core/resources.h"
#include "trtlab/core/thread_pool.h"

#include "nvrpc/life_cycle_streaming.h"

#include "testing.grpc.pb.h"
#include "testing.pb.h"

namespace nvrpc {
namespace testing {

struct TestResources : public ::trtlab::Resources
{
    TestResources(int numThreadsInPool = 3);

    using Stream = std::shared_ptr<LifeCycleStreaming<Input, Output>::ServerStream>;
    using StreamID = std::size_t;
    using Counter = std::size_t;

    ::trtlab::ThreadPool& AcquireThreadPool();

    void StreamManagerInit();
    void StreamManagerFini();
    void StreamManagerWorker();

    void IncrementStreamCount(Stream);
    void CloseStream(Stream);

  private:
    ::trtlab::ThreadPool m_ThreadPool;

    bool m_Running;
    std::mutex m_MessageMutex;
    std::map<StreamID, Stream> m_Streams;
    std::map<StreamID, Counter> m_MessagesRecv;
    std::map<StreamID, Counter> m_MessagesSent;

    std::mutex m_ShutdownMutex;
    bool m_ClientRunning;
    bool m_ServerRunning;
};

} // namespace testing
} // namespace nvrpc