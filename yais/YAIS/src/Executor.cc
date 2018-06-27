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
#include "YAIS/Executor.h"

#include <glog/logging.h>

namespace yais
{

void Executor::ProgressEngine(int thread_id)
{
    bool ok;
    void *tag;
    auto deadline = GetTimeout();
    auto myCQ = m_ServerCompletionQueues[thread_id].get();
    using NextStatus = ::grpc::ServerCompletionQueue::NextStatus;

    while (true)
    {
        auto status = myCQ->AsyncNext(&tag, &ok, deadline);
        if (status == NextStatus::SHUTDOWN) return;
        else if (status == NextStatus::TIMEOUT)
        {
            RunTimeoutCycle();
            deadline = GetTimeout();
        }
        else if (status == NextStatus::GOT_EVENT)
        {
            auto ctx = static_cast<IContext *>(tag);
            if (!RunContext(ctx, ok)) 
            {
                ResetContext(ctx);
            }
        }
    }
}

void Executor::RunTimeoutCycle()
{
    static auto last = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();
    float elapsed = std::chrono::duration<float>(now - last).count();
    if (elapsed >= 5.0 && m_TxCount)
    {
        LOG(INFO) << m_TxCount << "transactions over " << elapsed << "seconds [" << (float)m_TxCount / elapsed << " txs/sec]";
        m_TxCount = 0;
        last = now;
    }
}

time_point Executor::GetTimeout()
{
    return std::chrono::system_clock::now() + std::chrono::seconds(1);
}

}
