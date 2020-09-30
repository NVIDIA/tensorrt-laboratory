/* Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <chrono>
#include <map>
#include <mutex>
#include <thread>

namespace trtlab
{
    class DeferredShortTaskPool
    {
    public:
        using clock_type = std::chrono::high_resolution_clock;
        using time_point = clock_type::time_point;

        DeferredShortTaskPool() : m_Thread(std::make_unique<std::thread>([this]() { ProgressThread(); })), m_Stop(false) {}

        virtual ~DeferredShortTaskPool()
        {
            shutdown();
            m_Thread->join();
        }

        DELETE_COPYABILITY(DeferredShortTaskPool);
        DELETE_MOVEABILITY(DeferredShortTaskPool);

        void enqueue_deferred(time_point deadline, std::function<void()> task)
        {
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                if(m_Stop) { throw std::runtime_error("shutting down task pool; new submission not allowed"); }
                m_Tasks[deadline] = std::move(task);
            }
            m_Condition.notify_one();
        }

        void shutdown()
        {
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Stop = true;
            }
            m_Condition.notify_all();
        }

    private:
        void ProgressThread()
        {
            for (;;)
            {
                std::function<void()>  task;
                clock_type::time_point deadline;
                {
                    std::unique_lock<std::mutex> lock(m_Mutex);
                    if (m_Tasks.empty())
                    {
                        m_Condition.wait(lock, [this]() { return m_Stop || !m_Tasks.empty(); });
                    }
                    else
                    {
                        auto keyval = m_Tasks.cbegin();
                        m_Condition.wait_until(lock, keyval->first, [this]() { return m_Stop; });
                    }

                    if (m_Stop && m_Tasks.empty()) { return; }
                    if (m_Tasks.empty()) { continue; }

                    auto it = m_Tasks.begin();
                    if(clock_type::now() < it->first) { continue; }
                    task    = std::move(it->second);
                    m_Tasks.erase(it->first);
                }
                //auto start = clock_type::now();
                task();
                //auto end = clock_type::now();
                //auto wall = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                //LOG_IF(WARNING, wall >= 5) << "task exceeded short duration limit (max 5us): " << wall << "us";
            }
        }

        std::unique_ptr<std::thread> m_Thread;
        bool                         m_Stop;

        std::map<time_point, std::function<void()>> m_Tasks;
        std::mutex                                  m_Mutex;
        std::condition_variable                     m_Condition;
    };

} // namespace trtlab
