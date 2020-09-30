#pragma once
#include <mutex>
#include <vector>
#include <boost/fiber/all.hpp>
#include <glog/logging.h>

namespace trtlab
{
    template <typename SchedulerType = boost::fibers::algo::shared_work, typename... Args>
    class FiberGroup
    {
    public:
        FiberGroup(std::size_t thread_count, Args&&... args) : m_thread_count(thread_count), m_running(true), m_thread_barrier(thread_count)
        {
            auto scheduler_init = std::bind(boost::fibers::use_scheduling_algorithm<SchedulerType>, std::forward<Args>(args)...);
            for (int i = 1; i < thread_count; i++)
            {
                m_threads.emplace_back([this, thread_count, scheduler_init] {
                    scheduler_init();
                    m_thread_barrier.wait();
                    std::unique_lock<std::mutex> lock(m_thread_mutex);
                    m_thread_cond.wait(lock, [this] { return !m_running; });
                });
            }
            boost::fibers::use_scheduling_algorithm<SchedulerType>(std::forward<Args>(args)...);
            m_thread_barrier.wait();
            // VLOG(1) << "thread and fiber scheduler initialized on " << thread_count << " threads";
        }

        ~FiberGroup()
        {
            {
                std::lock_guard<decltype(m_thread_mutex)> lock(m_thread_mutex);
                m_running = false;
            }
            m_thread_cond.notify_all();

            for (auto& t : m_threads)
            {
                t.join();
            }
        }

    private:
        bool                                  m_running;
        const std::size_t                     m_thread_count;
        boost::fibers::barrier                m_thread_barrier;
        std::vector<std::thread>              m_threads;
        std::mutex                            m_thread_mutex;
        boost::fibers::condition_variable_any m_thread_cond;
    };

} // namespace trtlab