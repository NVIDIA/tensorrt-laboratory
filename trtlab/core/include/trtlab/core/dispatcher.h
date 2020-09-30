

#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <queue>

#include <boost/container/static_vector.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>

#include <glog/logging.h>

#include "hybrid_condition.h"
#include "hybrid_mutex.h"
#include "thread_pool.h"
#include "task_pool.h"

#include "standard_threads.h"
#include "userspace_threads.h"

namespace trtlab
{
    template <typename Batcher>
    class Dispatcher;

    template <template <class, class> class Batcher, typename T>
    class Dispatcher<Batcher<T, standard_threads>> : private Batcher<T, standard_threads>
    {
        using batcher_type = Batcher<T, standard_threads>;
        using thread_type  = typename batcher_type::thread_type;
        using clock_type   = typename batcher_type::clock_type;

        using mutex_type   = std::mutex;
        using cv_type      = std::condition_variable;
        using thread_pool  = std::shared_ptr<ThreadPool>;
        using task_pool    = std::shared_ptr<DeferredShortTaskPool>;

        // extra performance can be achieved using hybrid mutex/condition variables
        // using thread_pool = BaseThreadPool<hybrid_mutex, hybrid_condition>;

    public:
        using future_type = typename batcher_type::future_type;
        using execute_fn  = std::function<void(const std::vector<typename batcher_type::batch_item>&, std::function<void()>)>;

        //template <typename... Args>
        Dispatcher(batcher_type&& batcher, std::chrono::nanoseconds batching_window, thread_pool workers, task_pool progress, execute_fn exec_fn)
        : batcher_type(std::move(batcher)),
          m_Workers(workers),
          m_Progress(progress),
          m_UserFn(exec_fn),
          m_ProgressTaskEnqueued(false),
          m_DispatchID(0),
          m_BatchingWindow(batching_window),
          m_Shutdown(false)
        {
        }

        virtual ~Dispatcher()
        {
            shutdown();
        }

        // can be moveable, but only if we inherit from std::enable_shared_from_this
        // and all thread/task offsets refer to this object via a shared_ptr and not this.
        Dispatcher(Dispatcher&&) = delete;
        Dispatcher& operator=(Dispatcher&&) = delete;

        // not copyable
        Dispatcher(const Dispatcher&) = delete;
        Dispatcher& operator=(const Dispatcher&) = delete;

        // enqueue will add the item to the current batch
        // the batcher controls all the in-process logic of batching
        // the dispatcher controls the out-of-process logic, like timeouts
        // access control is via the dispatcher
        future_type enqueue(T item)
        {
            std::lock_guard<mutex_type> lock(m_EnqueueMutex);
            if (m_Shutdown)
            {
                throw std::runtime_error("dispatcher shutting down; no new enqueues can be accepted");
            }

            // should we launch a deferred async task to ensure forward progress
            bool enqueue_progress_task = !m_ProgressTaskEnqueued && batcher_type::empty();

            // push current items, then query the batcher for a batch
            auto future = batcher_type::enqueue(std::move(item));
            if (auto batch = batcher_type::update())
            {
                QueueBatch(*batch);
                enqueue_progress_task = false;
            }

            if (enqueue_progress_task)
            {
                QueueProgressTask();
            }

            return future;
        }

        void shutdown()
        {
            std::condition_variable      cv;
            std::unique_lock<mutex_type> lock(m_EnqueueMutex);
            m_Shutdown = true;
            while (m_ProgressTaskEnqueued)
            {
                cv.wait_until(lock, clock_type::now() + m_BatchingWindow);
            }
        }

    private:
        // requires mutex
        void QueueBatch(typename batcher_type::Batch& batch)
        {
            DCHECK_GT(batch.items.size(), 0);
            auto work = [batch = std::move(batch), user_fn = m_UserFn]() {
                auto completer = [&batch]() mutable { batch.promise.set_value(); };
                user_fn(batch.items, completer);
            };
            m_Workers->enqueue(std::move(work));
            m_DispatchID++;
        }

        // requires mutex
        void QueueProgressTask()
        {
            DCHECK(!batcher_type::empty());
            auto deadline = batcher_type::start_time() + m_BatchingWindow;
            auto task     = [this, id = m_DispatchID]() { ProgressTask(id); };
            m_Progress->enqueue_deferred(deadline, std::move(task));
            m_ProgressTaskEnqueued = true;
        }

        void ProgressTask(std::size_t id)
        {
            std::lock_guard<mutex_type> lock(m_EnqueueMutex);
            m_ProgressTaskEnqueued = false;

            // if the ids match, then close and queue the current batch - it timed out!
            if (m_DispatchID == id)
            {
                DCHECK(!batcher_type::empty());
                if (auto batch = batcher_type::close_batch())
                {
                    QueueBatch(*batch);
                }
            }
            else
            {
                // if there is potential work, re-queue the progress task to ensure
                // that work will complete at some future time
                if (!batcher_type::empty())
                {
                    if (auto batch = batcher_type::update())
                    {
                        QueueBatch(*batch);
                    }
                    else
                    {
                        QueueProgressTask();
                    }
                }
            }
        }

        execute_fn                m_UserFn;
        thread_pool              m_Workers;
        task_pool                m_Progress;
        mutex_type               m_EnqueueMutex;
        bool                     m_ProgressTaskEnqueued;
        std::size_t              m_DispatchID;
        std::chrono::nanoseconds m_BatchingWindow;
        bool                     m_Shutdown;
    };

    // userspace_threads

    template <template <class, class> class Batcher, typename T>
    class Dispatcher<Batcher<T, userspace_threads>> : private Batcher<T, userspace_threads>
    {
        using batcher_type = Batcher<T, userspace_threads>;
        using thread_type  = typename batcher_type::thread_type;
        using clock_type   = typename batcher_type::clock_type;

        using mutex_type   = typename thread_type::mutex;
        using cv_type      = typename thread_type::cv;

    public:
        using batch_t     = std::vector<typename batcher_type::batch_item>;
        using future_type = typename batcher_type::future_type;

        //template <typename... Args>
        Dispatcher(batcher_type&& batcher, std::chrono::nanoseconds batching_window)
        : batcher_type(std::move(batcher)),
          m_ProgressTaskEnqueued(false),
          m_DispatchID(0),
          m_BatchingWindow(batching_window),
          m_Shutdown(false)
        {
        }

        virtual ~Dispatcher()
        {
            shutdown();
        }

        // can be moveable, but only if we inherit from std::enable_shared_from_this
        // and all thread/task offsets refer to this object via a shared_ptr and not this.
        Dispatcher(Dispatcher&&) = delete;
        Dispatcher& operator=(Dispatcher&&) = delete;

        // not copyable
        Dispatcher(const Dispatcher&) = delete;
        Dispatcher& operator=(const Dispatcher&) = delete;

        // enqueue will add the item to the current batch
        // the batcher controls all the in-process logic of batching
        // the dispatcher controls the out-of-process logic, like timeouts
        // access control is via the dispatcher
        future_type enqueue(T item)
        {
            std::lock_guard<mutex_type> lock(m_EnqueueMutex);
            if (m_Shutdown)
            {
                throw std::runtime_error("dispatcher shutting down; no new enqueues can be accepted");
            }

            // should we launch a deferred async task to ensure forward progress
            bool enqueue_progress_task = !m_ProgressTaskEnqueued && batcher_type::empty();

            // push current items, then query the batcher for a batch
            auto future = batcher_type::enqueue(std::move(item));
            if (auto batch = batcher_type::update())
            {
                DVLOG(2) << "got batch from batcher - queuing execution";
                QueueBatch(std::move(*batch));
                enqueue_progress_task = false;
                DVLOG(2) << "batch queued";
            }

            if (enqueue_progress_task)
            {
                DVLOG(2) << "queuing progress task to close window on timeout";
                QueueProgressTask();
            }

            return future;
        }

        void shutdown()
        {
            cv_type                      cv;
            std::unique_lock<mutex_type> lock(m_EnqueueMutex);
            m_Shutdown = true;
            while (m_ProgressTaskEnqueued)
            {
                cv.wait_until(lock, clock_type::now() + m_BatchingWindow);
            }
        }

    private:
        virtual void compute_batch_fn(const batch_t&, std::function<void()>) {}

        // requires mutex
        void QueueBatch(typename batcher_type::Batch&& batch)
        {
            DCHECK_GT(batch.items.size(), 0);

            DVLOG(3) << "QueueBatch";
            boost::fibers::fiber(boost::fibers::launch::dispatch, [this, batch = std::move(batch)] {
                auto completer = [&batch]() mutable { batch.promise.set_value(); };
                compute_batch_fn(batch.items, completer);
            }).detach();

            m_DispatchID++;
        }

        // requires mutex
        void QueueProgressTask()
        {
            DCHECK(!batcher_type::empty());
            auto deadline = batcher_type::start_time() + m_BatchingWindow;
            boost::fibers::fiber(boost::fibers::launch::dispatch, [this, id = m_DispatchID, deadline]() { 
                thread_type::sleep_until(deadline);
                ProgressTask(id); 
            }).detach();
            m_ProgressTaskEnqueued = true;
        }

        void ProgressTask(std::size_t id)
        {
            std::lock_guard<mutex_type> lock(m_EnqueueMutex);
            m_ProgressTaskEnqueued = false;

            // if the ids match, then close and queue the current batch - it timed out!
            if (m_DispatchID == id)
            {
                DCHECK(!batcher_type::empty());
                if (auto batch = batcher_type::close_batch())
                {
                    QueueBatch(std::move(*batch));
                }
            }
            else
            {
                // if there is potential work, re-queue the progress task to ensure
                // that work will complete at some future time
                if (!batcher_type::empty())
                {
                    if (auto batch = batcher_type::update())
                    {
                        QueueBatch(std::move(*batch));
                    }
                    else
                    {
                        QueueProgressTask();
                    }
                }
            }
        }

        mutex_type               m_EnqueueMutex;
        bool                     m_ProgressTaskEnqueued;
        std::size_t              m_DispatchID;
        std::chrono::nanoseconds m_BatchingWindow;
        bool                     m_Shutdown;
    };

} // namespace trtlab