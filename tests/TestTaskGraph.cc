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
#include "tensorrt/playground/core/thread_pool.h"
#include "gtest/gtest.h"

#include <list>

using namespace yais;

namespace
{

double adder(double x, int y) {
    return x + y;
}

class TestTaskGraph : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        executor = std::make_shared<ThreadPool>(2);
    }

    std::shared_ptr<ThreadPool> executor;
};

TEST_F(TestTaskGraph, TransWarpExample)
{
    auto task1 = tw::make_value_task("something", 13.3);
    auto task2 = tw::make_value_task("something else", 42);
    auto task3 = tw::make_task(tw::consume, "adder", adder, task1, task2);
    task3->schedule_all(&executor);  // Schedules all tasks for execution
    EXPECT_EQ(task3->get(), 55.3);

    task1->set_value(15.8);
    task2->set_value(43);
    task3->schedule_all(&executor);  // Re-schedules all tasks for execution
    EXPECT_EQ(task3->get(), 58.8);
}

TEST_F(TestTaskGraph, VectorOfTasks)
{
    std::vector<std::shared_ptr<tw::task<double>>> tasks;

    auto task1 = tw::make_value_task("something", 13.3);
    auto task2 = tw::make_value_task("something else", 42);

    auto adder1 = tw::make_task(tw::consume, "adder1", adder, task1, task2);
    auto adder2 = tw::make_task(tw::consume, "adder2", adder, task1, task2);
    tasks.push_back(adder1);
    tasks.push_back(adder2);
    // tasks.emplace_back(tw::make_task(tw::consume, "adder1", adder, task1, task2));
    // tasks.emplace_back(tw::make_task(tw::consume, "adder2", adder, task1, task2));

    auto wait_task = tw::make_task(tw::wait, []{ return 42; }, tasks);
    wait_task->schedule_all(&executor);  // Schedules all tasks for execution
    EXPECT_EQ(wait_task->get(), 42.0);
    EXPECT_EQ(adder1->get(), 55.3);
}

TEST_F(TestTaskGraph, VectorOfFutures)
{
    std::vector<std::future<void>> tasks;
    auto sleep = [] { std::this_thread::sleep_for(std::chrono::milliseconds(2)); };

    tasks.emplace_back(executor->enqueue(sleep));
    tasks.emplace_back(executor->enqueue(sleep));
    tasks.emplace_back(executor->enqueue([sleep]{ sleep(); }));

    for (auto & task : tasks)
    {
        task.wait();
    }
}

}