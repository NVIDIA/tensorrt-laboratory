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
#include "tensorrt/playground/core/async_compute.h"
#include "gtest/gtest.h"

#include <glog/logging.h>

using namespace yais;

namespace {

class TestAsyncCompute : public ::testing::Test
{
};

TEST_F(TestAsyncCompute, EvenTest)
{
    auto compute =
        AsyncComputeWrapper<void(int)>::Wrap([](int i) -> bool { return (bool)((i % 2) == 0); });

    /*
    // fails to compile: the class was defined to accept a user function with only 1 int, not 2 
    auto compute2ints = AsyncComputeWrapper<void(int)>::Wrap([](int i, int j) -> bool {
         return (bool)((i % 2) == 0);
    });
    */

    auto future = compute->Future();
    (*compute)(42);
    // (*compute)(42, -2); // fails to compile, 2 ints instead of 1
    auto value = future.get();

    EXPECT_TRUE(value);
}

TEST_F(TestAsyncCompute, OddTest)
{
    auto compute =
        AsyncComputeWrapper<void(int)>::Wrap([](int i) -> bool { return (bool)((i % 2) == 0); });

    auto future = compute->Future();
    (*compute)(41);
    auto value = future.get();

    EXPECT_FALSE(value);
}

TEST_F(TestAsyncCompute, ReturnUniquePtr)
{
    auto compute = AsyncComputeWrapper<void(int)>::Wrap([](int i) -> std::unique_ptr<bool> {
        return std::make_unique<bool>((bool)((i % 2) == 0));
    });

    auto future = compute->Future();
    (*compute)(41);
    auto value = std::move(future.get());

    EXPECT_TRUE(value); // this the unique ptr
    EXPECT_FALSE(*value); // this the unique ptr
    EXPECT_ANY_THROW(future.get()); // this is the unique ptr - value moved out
}

TEST_F(TestAsyncCompute, ReturnVoid)
{
    auto compute = AsyncComputeWrapper<void(int)>::Wrap([](int i) {
        LOG(INFO) << "Inner";
    });

    auto future = compute->Future();
    (*compute)(41);
    future.wait();
}

/*
TEST_F(TestAsyncCompute, ReturnBoolInputs1xInt)
{
    struct ReturnBoolInputs1xInt : public AsyncComputeWrapper<bool(int)>
    {
        template<typename T, typename ...Args>
        auto Compute(T(Args...) UserFn) {
            std::vector<std::future<T>> futures;
            for(int i=0; i<10; i++) {
                auto compute = Wrap(UserFn);
                futures.push_back(std::move(compute->Future()));
                m_ThreadPool.enqueue([compute](int i) {
                    (*compute)(i);
                })
            }
            
        }
      protected:
        template<typename T, typename... Args>
        auto Enqueue(std::shared_ptr<AsyncComputeWrapper<T>> UserFn, Args&&... args)
        {
            auto compute = Wrap(UserFn);
            auto future = compute->Future();
            m_ThreadPool.enqueue([compute](Args&&... args) mutable {
                (*compute)(args...);
            });
        }

      private:
        ThreadPool m_ThreadPool;
    };

    ReturnBoolInputs1xInt compute()
}
*/

} // namespace