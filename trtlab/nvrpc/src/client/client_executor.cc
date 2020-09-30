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
#include "nvrpc/client/executor.h"
#include "nvrpc/client/base_context.h"

#include <glog/logging.h>

using trtlab::ThreadPool;

namespace nvrpc {
namespace client {

Executor::Executor() : Executor(1) {}

Executor::Executor(int numThreads) : Executor(std::make_unique<ThreadPool>(numThreads)) {}

Executor::Executor(std::unique_ptr<ThreadPool> threadpool) : m_ThreadPool(std::move(threadpool)), m_Counter(0)
{
    // for(decltype(m_ThreadPool->Size()) i = 0; i < m_ThreadPool->Size(); i++)
    for(auto i = 0; i < m_ThreadPool->Size(); i++)
    {
        DVLOG(1) << "Starting Client Progress Engine #" << i;
        m_CQs.emplace_back(new ::grpc::CompletionQueue);
        auto cq = m_CQs.back().get();
        m_ThreadPool->enqueue([this, cq] { ProgressEngine(*cq); });
    }
}

Executor::~Executor() { ShutdownAndJoin(); }

void Executor::ShutdownAndJoin()
{
    for(auto& cq : m_CQs)
    {
        cq->Shutdown();
    }
    m_ThreadPool.reset();
}

void Executor::ProgressEngine(::grpc::CompletionQueue& cq)
{
    void* tag;
    bool ok = false;

    while(cq.Next(&tag, &ok))
    {
        // CHECK(ok);
        BaseContext* ctx = BaseContext::Detag(tag);
        DVLOG(3) << "executor issuing callback";
        auto should_delete = ctx->ExecutorShouldDeleteContext();
        if(!ctx->RunNextState(ok))
        {
            if(should_delete)
            {
                DVLOG(1) << "Deleting ClientContext: " << tag;
                delete ctx;
            }
        }
        DVLOG(3) << "executor callback complete";
    }
}

::grpc::CompletionQueue* Executor::GetNextCQ() const
{
    std::size_t idx = 0;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DCHECK_LT(m_Counter, m_ThreadPool->Size());
        if(++m_Counter == m_ThreadPool->Size()) { m_Counter = 0; }
        idx = m_Counter;
    }
    return m_CQs[idx].get();
}

} // namespace client
} // namespace nvrpc
