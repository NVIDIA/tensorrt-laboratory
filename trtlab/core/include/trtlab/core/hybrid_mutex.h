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
#include <cstdint>

#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include <x86intrin.h>

/**
 *
 */
inline int sys_futex(void* addr1, int op, int val1, const struct timespec* timeout, void* addr2,
                     int val3) noexcept
{
    return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}

/**
 *
 */
class alignas(16) hybrid_mutex final
{
    hybrid_mutex(const hybrid_mutex&) = delete;
    hybrid_mutex& operator=(const hybrid_mutex&) = delete;

    friend class hybrid_condition;

  public:
    /**
     */
    constexpr hybrid_mutex() noexcept : m_spins(1U), m_lock(0x0) {}

    /**
     */
    constexpr hybrid_mutex(uint32_t spins) noexcept : m_spins(spins), m_lock(0x0) {}

    /**
     */
    ~hybrid_mutex() noexcept { assert(m_lock.u == 0x0); }

    /**
     */
    void lock() noexcept
    {
        // try and spin first
        for(uint32_t i = 0U; i < m_spins; i++)
        {
            // do atomic exchange, returns previous value. if the previous
            // value was 0, i.e. unlocked, we acquired the lock.
            if(!__atomic_exchange_n(&m_lock.locked, 0x1, __ATOMIC_ACQUIRE))
            {
                return;
            }

            // be nice, tell the cpu we are spinning
            __pause();
        }

        // didn't get lock, we may need to sleep.
        // exchange with both the locked and contended bits set.
        // if the previous value is still locked, i.e. 0x1,
        // we need to go into the kernel and sleep
        while(__atomic_exchange_n(&m_lock.u, 0x101, __ATOMIC_ACQUIRE) & 0x1)
        {
            (void)sys_futex(&m_lock.u, FUTEX_WAIT_PRIVATE, 0x101, nullptr, nullptr, 0);
        }
    }

    /**
     */
    bool try_lock() noexcept
    {
        if(!__atomic_exchange_n(&m_lock.locked, 0x1, __ATOMIC_ACQUIRE))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
     */
    void unlock() noexcept
    {
        // locked and not contended
        // if we are locked without contention, i.e. only the locked flag is set,
        // attempt to atomically compare and swap to unlock
        if(m_lock.u == 0x1)
        {
            uint32_t locked = 0x1;

            if(__atomic_compare_exchange_n(&m_lock.u, &locked, 0x0, false, __ATOMIC_ACQ_REL,
                                           __ATOMIC_ACQUIRE))
            {
                return;
            }
        }

        // unlock, setting locked = 0, using release memory barrier
        __atomic_exchange_n(&m_lock.locked, 0x0, __ATOMIC_RELEASE);

        // spin, hoping someone takes the lock
        // if someone takes the lock under the spin
        // we can avoid going to the kernel
        // note: if m_spins * 2 overflows, there is no check here... though that many spins is
        // dumb...
        for(uint32_t i = 0U; i < m_spins * 2U; i++)
        {
            if(m_lock.locked)
            {
                return;
            }

            // be nice, tell the cpu we are spinning
            __pause();
        }

        // we need to wake someone up, go into the kernel
        // reset the contended flag. anyone actively waiting
        // on it, will set it again; otherwise, the kernel
        // will have no one to wake and the mutex will remain
        // in the unlocked, non-contended state (i.e. mLock.u == 0)
        m_lock.contended = 0x0;

        // tell the kernel to wake up 1 thread waiting on mLock address
        (void)sys_futex(&m_lock.u, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    }

  private:
    uint32_t m_spins; ///< number of spins before going to OS

    /*
     * lock data structure.
     *
     * The mutex in unlocked when the entire structure
     * has a value of 0. (i.e. mLock.u == 0)
     *
     * locked is the userland byte that is used for the
     * spinlock portion of the mutex.
     */
    union lock
    {
        constexpr lock(uint32_t _u) : u(_u) {}

        uint32_t u;
        struct
        {
            uint8_t locked;
            uint8_t contended;
            uint16_t pad;
        };
    } m_lock;
};
