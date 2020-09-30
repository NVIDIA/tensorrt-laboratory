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

#include <future>
#include <memory>

namespace trtlab
{
    namespace async
    {
        namespace detail
        {
            template <template <typename> class Promise, template <typename> class Future, typename CompleterFn>
            struct shared_packaged_task;

            template <template <typename> class Promise, template <typename> class Future, typename... Args>
            struct shared_packaged_task<Promise, Future, void(Args...)>
            {
                using CallingFn = std::function<void(Args...)>;
                using WrappedFn = std::function<void(Args...)>;

                shared_packaged_task(CallingFn calling_fn)
                {
                    m_WrappedFn = [this, calling_fn](Args&&... args) {
                        calling_fn(args...);
                        m_Promise.set_value();
                    };
                }

                Future<void> get_future()
                {
                    return m_Promise.get_future();
                }

                void operator()(Args&&... args)
                {
                    m_WrappedFn(args...);
                }

            private:
                WrappedFn     m_WrappedFn;
                Promise<void> m_Promise;
            };

            template <template <typename> class Promise, template <typename> class Future, typename ResultType, typename... Args>
            struct shared_packaged_task<Promise, Future, ResultType(Args...)>
            {
                using CallingFn = std::function<ResultType(Args...)>;
                using WrappedFn = std::function<void(Args...)>;

                shared_packaged_task(CallingFn calling_fn)
                {
                    m_WrappedFn = [this, calling_fn](Args&&... args) { m_Promise.set_value(std::move(calling_fn(args...))); };
                }

                std::future<ResultType> get_future()
                {
                    return m_Promise.get_future();
                }

                void operator()(Args&&... args)
                {
                    m_WrappedFn(args...);
                }

            private:
                WrappedFn                m_WrappedFn;
                std::promise<ResultType> m_Promise;
            };

        } // namespace detail

    } // namespace async

    template <typename CompleterFn>
    struct async_compute;

    template <typename... Args>
    struct async_compute<void(Args...)>
    {
        // create a shared object that holds both the promise and the user function
        // to call with some pre-defined arguments.
        // upon calling the () method on the created object, the value of the promise
        // is set by the return value of the wrapped  user function
        template <typename F>
        static auto wrap(F&& f)
        {
            using ResultType = typename std::result_of<F(Args...)>::type;
            using UserFn     = ResultType(Args...);
            //return std::make_shared<detail::async_compute_impl<UserFn>>(f);
            return std::make_shared<async::detail::shared_packaged_task<std::promise, std::future, UserFn>>(f);
        }
    };

} // namespace trtlab