/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#ifndef NVIS_EXECUTOR_H_
#define NVIS_EXECUTOR_H_
#pragma once

#include "YAIS/Interfaces.h"
#include "YAIS/Resources.h"

#include <thread>

#include <glog/logging.h>

using time_point = std::chrono::system_clock::time_point;

namespace yais
{

class Executor : public IExecutor
{
  public:
    Executor(int numThreads) : IExecutor(), m_ThreadCount(numThreads), m_TxCount(0) {}
    Executor() : Executor(1) {}
    ~Executor() override {}

    void Initialize(::grpc::ServerBuilder &builder) final override
    {
        for (int i = 0; i < m_ThreadCount; i++)
        {
            m_ServerCompletionQueues.emplace_back(builder.AddCompletionQueue());
        }
    }

    void RegisterContexts(IRPC *rpc, std::shared_ptr<Resources> resources, int numContextsPerThread) final override
    {
        auto base = dynamic_cast<IExecutor *>(this);
        CHECK_EQ(m_ThreadCount, m_ServerCompletionQueues.size()) << "Incorrect number of CQs";
        for (int i = 0; i < m_ThreadCount; i++)
        {
            auto cq = m_ServerCompletionQueues[i].get();
            for (int j = 0; j < numContextsPerThread; j++)
            {
                DLOG(INFO) << "Creating Context " << j << " on thread " << i;
                m_Contexts.emplace_back(
                    this->CreateContext(rpc, cq, resources));
            }
        }
    }

    void Run() final override
    {
        // Launch the threads polling on their CQs
        for (int i = 0; i < m_ThreadCount; i++)
        {
            m_Threads.emplace_back(&Executor::ProgressEngine, this, i);
        }
        // Queue the Execution Contexts in the recieve queue
        for (int i = 0; i < m_Contexts.size(); i++)
        {
            ResetContext(m_Contexts[i].get());
        }
    }

  protected:
    void RunTimeoutCycle();
    time_point GetTimeout();

  private:
    void ProgressEngine(int thread_id);

    size_t m_TxCount;
    int m_ThreadCount;
    std::vector<std::thread> m_Threads;
    std::vector<std::unique_ptr<IContext>> m_Contexts;
    std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>> m_ServerCompletionQueues;
    // std::vector<std::unique_ptr<PerThreadState>> m_ShutdownState;
};

} // end namespace yais

#endif // NVIS_EXECUTOR_H_
