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

#include <cstdint>
#include <cstring>
#include <mutex>

#include <gtest/gtest.h>

#define MUTEX_TIMEOUT_MS 1000

class TrackedTest : public ::testing::Test
{
  public:
    void EndTest();
    
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
};

struct log_tracker
{
    void on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept;

    void on_node_deallocation(void *ptr, std::size_t size, std::size_t alignment) noexcept;

    void on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;

    void on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;

    const char* name;
    static std::size_t node_total;
    static std::size_t node_count;
};

struct timeout_error : std::exception
{
    const char* what() const throw ()
    {
        return "timed out";
    }
};

class timeout_mutex : std::timed_mutex
{
  public:
    timeout_mutex() = default;
    timeout_mutex(const timed_mutex&) = delete;

    void lock();
    void unlock();
    bool try_lock();
};