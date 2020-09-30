#pragma once
#include <boost/fiber/all.hpp>

namespace trtlab
{
    struct userspace_threads
    {
        using mutex = boost::fibers::mutex;

        using cv = boost::fibers::condition_variable;

        template <typename T>
        using promise = boost::fibers::promise<T>;

        template <typename T>
        using future = boost::fibers::future<T>;

        template <typename T>
        using shared_future = boost::fibers::shared_future<T>;

        template <class R, class... Args>
        using packaged_task = boost::fibers::packaged_task<R(Args...)>;

        template <class Function, class... Args>
        static auto async(Function&& f, Args&&... args)
        {
            return boost::fibers::async(f, std::forward<Args>(args)...);
        }

        template <typename Rep, typename Period>
        static void sleep_for(std::chrono::duration<Rep, Period> const& timeout_duration)
        {
            boost::this_fiber::sleep_for(timeout_duration);
        }

        template <typename Clock, typename Duration>
        static void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time_point)
        {
            boost::this_fiber::sleep_until(sleep_time_point);
        }
    };
} // namespace trtlab