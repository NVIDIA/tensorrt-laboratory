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
//
// Original Source: https://github.com/progschj/BaseThreadPool
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
// Modifications:
//   * Header-only file was split into .h/.cc files
//   * Added an extra safety check (lines 30-31) in the construction (.cc file).
//   * Added CPU affinity options to the constructor
//   * Added Size() method to get thread count
//   * Implemented transwarp::executor protocol
//
#pragma once

#include "trtlab/core/affinity.h"
#include "trtlab/core/utils.h"

#include <functional>
#include <future>
#include <queue>

#include <glog/logging.h>

namespace trtlab
{
    template <typename MutexType, typename ConditionType>
    class BaseThreadPool;

    using ThreadPool = BaseThreadPool<std::mutex, std::condition_variable>;

    /**
 * @brief Manages a Pool of Threads that consume a shared work Queue
 *
 * BaseThreadPool is the primary resoruce class for handling threads used throughout
 * the YAIS examples and tests.  The library is entirely a BYO-resources;
 * however, this implemenation is provided as a convenience class.  Many thanks
 * to the original authors for a beautifully designed class.
 */
    template <typename MutexType, typename ConditionType>
    class BaseThreadPool
    {
    public:
        /**
     * @brief Construct a new Thread Pool
     * @param nThreads Number of Worker Threads
     */
        BaseThreadPool(size_t nThreads);

        /**
     * @brief Construct a new Thread Pool with Shared CPU Affinity
     *
     * Creates a BaseThreadPool with nThreads where each thread sets its CPU affinity
     * to affinity_mask.
     *
     * @param nThreads
     * @param affinity_mask
     */
        BaseThreadPool(size_t nThreads, const cpu_set& affinity_mask);

        /**
     * @brief Construct a new Thread Pool with Exclusive CPU Affinity
     *
     * Creates a BaseThreadPool using a cpu_set such a for each CPU in the cpu_set a
     * thread is created and the affnity of that thread is dedicted to the
     * respective CPU from the cpu_set.
     *
     * @param cpu_set
     */
        BaseThreadPool(const cpu_set& cpu_set);
        virtual ~BaseThreadPool();

        DELETE_COPYABILITY(BaseThreadPool);
        DELETE_MOVEABILITY(BaseThreadPool);

        /**
     * @brief Enqueue Work to the BaseThreadPool by passing a Lambda Function
     *
     * Variadic template allows for an arbituary number of arguments to be passed
     * the captured lambda function.  Captures are still allowed and used
     * throughout the examples.
     *
     * The queue can grow larger than the number of threads.  A single worker
     * thread executues pulls a lambda function off the queue and executes it to
     * completion.  These are synchronous executions in an async messaging
     * library.  These synchronous pools can be swapped for truely async workers
     * using libevent or asio.  Happy to accept PRs to improve the async
     * abilities.
     *
     * @tparam F
     * @tparam Args
     * @param f
     * @param args
     * @return std::future<typename std::result_of<F(Args...)>::type>
     */
        template <class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

        void enqueue(std::function<void()> task);

        /**
     * @brief Number of Threads in the Pool
     */
        int Size();

#ifdef USE_TRANSWARP
        // transwarp interface: get_name, execute

        // The name of the executor
        std::string get_name() const final override
        {
            return "trtlab::BaseThreadPool";
        }

        // Only ever called on the thread of the caller to schedule()
        void execute(const std::function<void()>& functor, const std::shared_ptr<tw::node>& node) final override
        {
            {
                std::unique_lock<MutexType> lock(m_QueueMutex);
                tasks.push(functor);
            }
            m_Condition.notify_one();
        }
#endif

    private:
        void CreateThread(const cpu_set& affinity_mask);

        // need to keep track of threads so we can join them
        std::vector<std::thread> workers;
        // the task queue
        //std::queue<std::function<void()>> tasks;
        std::queue<std::packaged_task<void()>> tasks;

        // synchronization
        MutexType     m_QueueMutex;
        ConditionType m_Condition;
        bool          stop;
    };

    // add new work item to the pool
    template <typename MutexType, typename ConditionType>
    template <class F, class... Args>
    auto BaseThreadPool<MutexType, ConditionType>::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        std::packaged_task<return_type()> task(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task.get_future();
        {
            std::lock_guard<MutexType> lock(m_QueueMutex);

            // don't allow enqueueing after stopping the pool
            if (stop)
                throw std::runtime_error("enqueue on stopped BaseThreadPool");

            tasks.emplace(std::move(task));
        }
        m_Condition.notify_one();
        return res;
    }
/*
    template <typename MutexType, typename ConditionType>
    void BaseThreadPool<MutexType, ConditionType>::enqueue(std::function<void()> task)
    {
        {
            std::lock_guard<MutexType> lock(m_QueueMutex);

            // don't allow enqueueing after stopping the pool
            if (stop)
                throw std::runtime_error("enqueue on stopped BaseThreadPool");

            std::packaged_task
            tasks.push(task);
        }
        m_Condition.notify_one();
    }
*/
    template <typename MutexType, typename ConditionType>
    BaseThreadPool<MutexType, ConditionType>::BaseThreadPool(size_t nThreads)
    : BaseThreadPool(nThreads, affinity::this_thread::get_affinity())
    {
    }

    template <typename MutexType, typename ConditionType>
    BaseThreadPool<MutexType, ConditionType>::BaseThreadPool(size_t nThreads, const cpu_set& affinity_mask) : stop(false)
    {
        for (size_t i = 0; i < nThreads; ++i)
        {
            CreateThread(affinity_mask);
        }
    }

    template <typename MutexType, typename ConditionType>
    BaseThreadPool<MutexType, ConditionType>::BaseThreadPool(const cpu_set& cpus) : stop(false)
    {
        auto exclusive = cpus.get_allocator();
        for (size_t i = 0; i < exclusive.size(); i++)
        {
            cpu_set affinity_mask;
            CHECK(exclusive.allocate(affinity_mask, 1)) << "Affinity Allocator failed on pass: " << i;
            CreateThread(affinity_mask);
        }
    }

    template <typename MutexType, typename ConditionType>
    void BaseThreadPool<MutexType, ConditionType>::CreateThread(const cpu_set& affinity_mask)
    {
        workers.emplace_back([this, affinity_mask]() {
            affinity::this_thread::set_affinity(affinity_mask);
            for (;;)
            {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<MutexType> lock(this->m_QueueMutex);
                    this->m_Condition.wait(lock, [this]() { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }

    // the destructor joins all threads
    template <typename MutexType, typename ConditionType>
    BaseThreadPool<MutexType, ConditionType>::~BaseThreadPool()
    {
        {
            std::lock_guard<MutexType> lock(m_QueueMutex);
            stop = true;
        }
        m_Condition.notify_all();

        for (std::thread& worker : workers)
        {
            worker.join();
        }
    }

    template <typename MutexType, typename ConditionType>
    int BaseThreadPool<MutexType, ConditionType>::Size()
    {
        return workers.size();
    }

} // namespace trtlab
