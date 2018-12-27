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
#pragma once

#include <future>
#include <memory>

namespace yais {

template<typename CompleterFn>
struct AsyncCompute;

template<typename... Args>
struct AsyncCompute<void(Args...)>
{
    template<typename F>
    static auto Wrap(F&& f)
    {
        using ResultType = typename std::result_of<F(Args...)>::type;
        using UserFn = ResultType(Args...);
        return std::make_shared<AsyncCompute<UserFn>>(f);
    }
};

template<typename ResultType, typename... Args>
struct AsyncCompute<ResultType(Args...)>
{
    using CallingFn = std::function<ResultType(Args...)>;
    using WrappedFn = std::function<void(Args...)>;

    AsyncCompute(CallingFn calling_fn)
    {
        m_WrappedFn = [this, calling_fn](Args&&... args) {
            m_Promise.set_value(std::move(calling_fn(args...)));
        };
    }

    std::future<ResultType> Future()
    {
        return m_Promise.get_future();
    }

    void operator()(Args&&... args)
    {
        m_WrappedFn(args...);
    }

  private:
    WrappedFn m_WrappedFn;
    std::promise<ResultType> m_Promise;
};

} // namespace yais