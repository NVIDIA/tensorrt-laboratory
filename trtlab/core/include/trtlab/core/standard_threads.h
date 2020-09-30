#pragma once
#include <future>

namespace trtlab
{
    struct standard_threads
    {

        using mutex = std::mutex;

        using cv = std::condition_variable;
        
        template <typename T>
        using promise = std::promise<T>;

        template <typename T>
        using future = std::future<T>;

        template <typename T>
        using shared_future = std::shared_future<T>;

        template <class Function, class... Args>
        static auto async(Function&& f, Args&&... args)
        {
            return std::async(f, std::forward<Args>(args)...);
        }

        template <typename Rep, typename Period>
        static void sleep_for(std::chrono::duration<Rep, Period> const& timeout_duration)
        {
            std::this_thread::sleep_for(timeout_duration);
        }

        template <typename Clock, typename Duration>
        static void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time_point)
        {
            std::this_thread::sleep_until(sleep_time_point);
        }
    };
}