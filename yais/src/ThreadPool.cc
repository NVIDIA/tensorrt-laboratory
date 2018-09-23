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
//
// Original Source: https://github.com/progschj/ThreadPool
//
// Original License:
//
// Copyright (c) 2012 Jakob Progsch, Václav Zeman
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//   distribution.
//
// Modifications: see header file
//
#include "YAIS/ThreadPool.h"
#include <glog/logging.h>

namespace yais
{

ThreadPool::ThreadPool(size_t nThreads) : ThreadPool(nThreads, Affinity::GetAffinity()) {}

ThreadPool::ThreadPool(size_t nThreads, const CpuSet &affinity_mask)
    : stop(false)
{
    for (size_t i = 0; i < nThreads; ++i)
    {
        InitThread(affinity_mask);
    }
}

ThreadPool::ThreadPool(const CpuSet &cpus) : stop(false)
{
    auto exclusive = cpus.GetAllocator();
    for (size_t i = 0; i < exclusive.size(); i++)
    {
        CpuSet affinity_mask;
        CHECK(exclusive.allocate(affinity_mask, 1)) << "Affinity Allocator failed on pass: " << i;
        InitThread(affinity_mask);
    }
}

void ThreadPool::InitThread(const CpuSet &affinity_mask)
{
    workers.emplace_back([this, affinity_mask]() {
        Affinity::SetAffinity(affinity_mask);
        DLOG(INFO) << "Initializing Thread " << std::this_thread::get_id() << " with CPU affinity "
                   << affinity_mask.GetCpuString();
        for (;;)
        {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                this->condition.wait(lock, [this]() {
                    return this->stop || !this->tasks.empty();
                });
                if (this->stop && this->tasks.empty())
                    return;
                task = move(this->tasks.front());
                this->tasks.pop();
            }

            task();
        }
    });
}

int ThreadPool::Size()
{
    return workers.size();
}

// the destructor joins all threads
ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();

    for (std::thread &worker : workers)
        worker.join();
}

} // end namespace yais
