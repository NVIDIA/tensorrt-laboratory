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
#include "glog/logging.h"
#include "trtlab/core/thread_pool.h"
#include "gtest/gtest.h"

using namespace trtlab;

class TestThreadPool : public ::testing::Test
{
  protected:
    virtual void SetUp() { thread_pool = std::make_shared<ThreadPool>(3); }

    virtual void TearDown() {}

    std::shared_ptr<ThreadPool> thread_pool;
};

TEST_F(TestThreadPool, ReturnInt)
{
    auto should_be_1 = thread_pool->enqueue([] { return 1; });
    ASSERT_EQ(1, should_be_1.get());
}

TEST_F(TestThreadPool, ReturnChainedInt)
{
    auto should_be_1 =
        thread_pool->enqueue([this] { return thread_pool->enqueue([] { return 1; }); });
    ASSERT_EQ(1, should_be_1.get().get());
}

TEST_F(TestThreadPool, MakeUnique) { auto unqiue = std::make_unique<ThreadPool>(1); }

TEST_F(TestThreadPool, CaptureThis)
{
    class ObjectThatOwnsAThreadPool
    {
        class ValueObject;

      public:
        ObjectThatOwnsAThreadPool()
            : m_ThreadPool(std::move(std::make_unique<ThreadPool>(1))),
              m_Object(std::move(std::make_unique<ValueObject>()))
        {
        }

        DELETE_COPYABILITY(ObjectThatOwnsAThreadPool);
        DELETE_MOVEABILITY(ObjectThatOwnsAThreadPool);

        ~ObjectThatOwnsAThreadPool()
        {
            DVLOG(2) << "Destroying ObjectThatOwnsAThreadPool: " << this;
        }
        auto test()
        {
            DVLOG(2) << "[before queue] val = " << m_Object.get();
            return m_ThreadPool->enqueue([this]() {
                DVLOG(2) << "[before sleep] val = " << m_Object.get();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                DVLOG(2) << "[after  sleep] val = " << m_Object.get();
                return m_Object->value;
            });
        }

      private:
        class ValueObject
        {
          public:
            ValueObject() : value(42) {}
            DELETE_COPYABILITY(ValueObject);
            DELETE_MOVEABILITY(ValueObject);
            ~ValueObject() { DVLOG(2) << "Destroying ValueObject"; }
            int value;
        };
        std::unique_ptr<ValueObject> m_Object;
        std::unique_ptr<ThreadPool> m_ThreadPool;
    };

    auto obj = std::make_unique<ObjectThatOwnsAThreadPool>();
    auto future = obj->test();
    obj.reset();
    EXPECT_EQ(future.get(), 42);
}