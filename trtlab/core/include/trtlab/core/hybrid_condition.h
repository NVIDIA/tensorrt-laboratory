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

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>

#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include <x86intrin.h>

#include "trtlab/core/hybrid_mutex.h"

/**
 *
 */
class alignas(16) hybrid_condition final
{
    hybrid_condition(const hybrid_condition&) = delete;
    hybrid_condition& operator=(const hybrid_condition&) = delete;

  public:
    /**
     */
    constexpr hybrid_condition() noexcept : m_mutex(nullptr), m_sequence(0) {}

    /**
     */
    ~hybrid_condition() noexcept {}

    /** Wait for mutex to signal */
    void wait(std::unique_lock<hybrid_mutex>& lock) noexcept
    {
        (void)wait_for_impl(lock.mutex(), std::chrono::seconds(0), std::chrono::nanoseconds(0));
    }

    /**
     */
    template<typename TPredicate>
    void wait(std::unique_lock<hybrid_mutex>& lock, const TPredicate& pred)
    {
        while(!pred())
        {
            wait(lock);
        }
    }

    /**
     */
    template<typename TRep, typename TPeriod>
    std::cv_status wait_for(std::unique_lock<hybrid_mutex>& lock,
                            const std::chrono::duration<TRep, TPeriod>& rel_time)
    {
        auto rtime = rel_time;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(rtime);
        rtime -= std::chrono::duration_cast<std::chrono::duration<TRep, TPeriod>>(seconds);
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(rtime);
        return wait_for_impl(lock.mutex(), seconds, nanos);
    }

    /**
     */
    template<typename TRep, typename TPeriod, typename TPredicate>
    bool wait_for(std::unique_lock<hybrid_mutex>& lock,
                  const std::chrono::duration<TRep, TPeriod>& rel_time, TPredicate pred)
    {
        while(!pred())
        {
            if(wait_for(lock, rel_time) == std::cv_status::timeout)
            {
                return pred();
            }
        }

        return true;
    }

    /** Notify one waiting thread to wake */
    void notify_one() noexcept
    {
        // if no waiters, just return
        if(m_mutex == nullptr)
        {
            return;
        }

        // increment sequence for wakeup
        __atomic_fetch_add(&m_sequence, 1, __ATOMIC_ACQ_REL);

        // wake up one thread
        (void)sys_futex(&m_sequence, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    }

    /** Notify all waiting threads to wake */
    void notify_all() noexcept
    {
        hybrid_mutex* mutex = m_mutex;

        // if no waiters, just return
        if(mutex == nullptr)
        {
            return;
        }

        // increment sequence for wakeup
        __atomic_fetch_add(&m_sequence, 1, __ATOMIC_ACQ_REL);

        // wake one thread, requeue the rest, avoids thundering herd
        // wakes up one thread, requeues all remaining threads on mutex's queue
        (void)sys_futex(&m_sequence, FUTEX_CMP_REQUEUE_PRIVATE, 1,
                        reinterpret_cast<struct timespec*>(std::numeric_limits<int32_t>::max()),
                        &mutex->m_lock, m_sequence);
    }

  private:
    hybrid_mutex* m_mutex;
    int32_t m_sequence;

    /// wait for implementation
    std::cv_status wait_for_impl(hybrid_mutex* mutex, const std::chrono::seconds& seconds,
                                 const std::chrono::nanoseconds& nanoseconds) noexcept
    {
        // expected sequence number
        int sequence = m_sequence;

        if(m_mutex != mutex)
        {
            hybrid_mutex* expected = nullptr;

            // atomically set mutex ptr
            __atomic_compare_exchange_n(&m_mutex, &expected, mutex, false, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE);

            // make sure this condition variable is not
            // used by more than one mutex
            assert(m_mutex == mutex);
        }

        // unlock the calling mutex
        mutex->unlock();

        // setup timeout (outside of lock)
        struct timespec* timeoutptr = nullptr;
        struct timespec timeout;

        // if any timeout is set, setup time struct
        if(seconds.count() > 0 || nanoseconds.count() > 0)
        {
            timeout.tv_sec = seconds.count();
            timeout.tv_nsec = nanoseconds.count();
            timeoutptr = &timeout;
        }

        // put thread to sleep on the sequence address,
        // iff mSequence is still equal to sequence (value while lock was held)
        int ret = -1;
        std::cv_status status = std::cv_status::no_timeout;

        // if interrupted.. continue.. and try again..
        while((ret = sys_futex(&m_sequence, FUTEX_WAIT_PRIVATE, sequence, timeoutptr, nullptr,
                               0)) == -1 &&
              errno == EINTR)
        {
            continue;
        }

        // return false if we had timeout waiting to be notified
        if(ret == -1 && errno == ETIMEDOUT)
        {
            status = std::cv_status::timeout;
        }

        // awoke, we need to aquire then lock before we exit
        // slight bit of code duplication here with regard to hybrid_mutex::lock()
        while(__atomic_exchange_n(&mutex->m_lock.u, 0x101, __ATOMIC_ACQUIRE) & 0x1)
        {
            (void)sys_futex(&mutex->m_lock.u, FUTEX_WAIT_PRIVATE, 0x101, nullptr, nullptr, 0);
        }

        return status;
    }
};
