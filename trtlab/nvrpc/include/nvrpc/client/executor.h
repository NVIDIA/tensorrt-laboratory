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
#include <memory>
#include <mutex>
#include <vector>

#include <grpc++/grpc++.h>

#include <trtlab/core/thread_pool.h>

namespace nvrpc {
namespace client {

class Executor : public std::enable_shared_from_this<Executor>
{
  public:
    Executor();
    Executor(int numThreads);
    Executor(std::unique_ptr<::trtlab::ThreadPool> threadpool);

    Executor(Executor&& other) noexcept = delete;
    Executor& operator=(Executor&& other) noexcept = delete;

    Executor(const Executor& other) = delete;
    Executor& operator=(const Executor& other) = delete;

    virtual ~Executor();

    void ShutdownAndJoin();
    ::grpc::CompletionQueue* GetNextCQ() const;

  private:
    void ProgressEngine(::grpc::CompletionQueue&);

    mutable std::size_t m_Counter;
    std::unique_ptr<::trtlab::ThreadPool> m_ThreadPool;
    std::vector<std::unique_ptr<::grpc::CompletionQueue>> m_CQs;
    mutable std::mutex m_Mutex;
};

} // namespace client
} // namespace nvrpc