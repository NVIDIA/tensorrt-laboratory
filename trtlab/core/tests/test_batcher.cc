
#include <gtest/gtest.h>

#include <trtlab/core/standard_threads.h>
#include <trtlab/core/userspace_threads.h>

#include <trtlab/core/batcher.h>
#include <trtlab/core/dispatcher.h>
#include <trtlab/core/task_pool.h>

class TestBatcher : public ::testing::Test
{
};

using namespace trtlab;

TEST_F(TestBatcher, StandardBatcher)
{
    StandardBatcher<int, standard_threads> batcher(5);

    for (int i = 0; i < 9; i++)
    {
        auto f     = batcher.enqueue(i);
        auto batch = batcher.update();

        if (i == 4)
        {
            EXPECT_TRUE(batch);
            EXPECT_EQ(batch->items.size(), 5);

            EXPECT_EQ(f.wait_until(std::chrono::high_resolution_clock::now()), std::future_status::timeout);
            batch->promise.set_value();
            EXPECT_EQ(f.wait_until(std::chrono::high_resolution_clock::now()), std::future_status::ready);
        }
        else
        {
            EXPECT_FALSE(batch);
        }
    }

    auto batch = batcher.update();
    EXPECT_FALSE(batch);

    batch = batcher.close_batch();
    EXPECT_TRUE(batch);
    EXPECT_EQ(batch->items.size(), 4);
}

TEST_F(TestBatcher, FullBatcher)
{
    auto execute_on_batch = [](const std::vector<int>& batch, std::function<void()> release) {
        LOG(INFO) << "executing on " << batch.size() << " items";
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        LOG(INFO) << "freeing inputs ...";
        release();
    };

    StandardBatcher<int, standard_threads> batcher(5);
    auto                                   thread_pool = std::make_shared<ThreadPool>(1);
    auto                                   task_pool   = std::make_shared<DeferredShortTaskPool>();

    Dispatcher<decltype(batcher)> dispatcher(std::move(batcher), std::chrono::milliseconds(15), thread_pool, task_pool, execute_on_batch);

    std::queue<std::shared_future<void>> futures;

    for (int i = 0; i < 9; i++)
    {
        futures.push(dispatcher.enqueue(i));
    }

    // wait on first batch to complete
    for (int i = 0; i < 5; i++)
    {
        futures.front().wait();
        futures.pop();
    }

    EXPECT_EQ(futures.front().wait_until(std::chrono::high_resolution_clock::now()), std::future_status::timeout);
    //dispatcher.shutdown();
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    EXPECT_EQ(futures.front().wait_until(std::chrono::high_resolution_clock::now()), std::future_status::ready);
    futures.pop();

    LOG(INFO) << "waiting on futures";
    while (!futures.empty())
    {
        futures.front().wait();
        futures.pop();
    }
    LOG(INFO) << "futures complete";
}

TEST_F(TestBatcher, FullBatcherUserThreads)
{
    boost::fibers::use_scheduling_algorithm<boost::fibers::algo::shared_work>();

    auto execute_on_batch = [](const std::vector<int>& batch, std::function<void()> release) {
        LOG(INFO) << "exectue_fn: this_thread: " << std::this_thread::get_id() << "; this_fiber: " << boost::this_fiber::get_id();
        boost::this_fiber::sleep_for(std::chrono::milliseconds(2));
        LOG(INFO) << "exectue_fn - sleep complete: this_thread: " << std::this_thread::get_id() << "; this_fiber: " << boost::this_fiber::get_id();
        release();
    };

    StandardBatcher<int, userspace_threads> batcher(5);
    Dispatcher<decltype(batcher)> dispatcher(std::move(batcher), std::chrono::milliseconds(15), execute_on_batch);

    using dispatcher_type = Dispatcher<decltype(batcher)>;

    std::queue<typename dispatcher_type::future_type> futures;

    for (int i = 0; i < 9; i++)
    {
        LOG(INFO) << "enqueue: " << i;
        futures.push(dispatcher.enqueue(i));
    }

    // wait on first batch to complete
    for (int i = 0; i < 5; i++)
    {
        LOG(INFO) << "wait: " << i;
        futures.front().wait();
        futures.pop();
    }

    EXPECT_EQ(futures.front().wait_until(std::chrono::high_resolution_clock::now()), boost::fibers::future_status::timeout);
    boost::this_fiber::sleep_for(std::chrono::milliseconds(16));
    EXPECT_EQ(futures.front().wait_until(std::chrono::high_resolution_clock::now()), boost::fibers::future_status::ready);
    futures.pop();

    LOG(INFO) << "waiting on futures";
    while (!futures.empty())
    {
        futures.front().wait();
        futures.pop();
    }
    LOG(INFO) << "futures complete";
}

TEST_F(TestBatcher, ShortDeferredTaskPool)
{
    std::mutex              mu;
    std::condition_variable cv;
    std::size_t             count = 0;

    auto task = [&mu, &cv, &count](std::chrono::milliseconds ms) {
        DLOG(INFO) << "deferred for " << ms.count() << " ms";
        {
            std::lock_guard<std::mutex> lock(mu);
            count += ms.count();
        }
        cv.notify_one();
    };

    DeferredShortTaskPool pool;

    using namespace std::chrono_literals;
    using clock = std::chrono::high_resolution_clock;

    auto start = clock::now();

    pool.enqueue_deferred(clock::now() + 25ms, [task]() { task(25ms); });
    pool.enqueue_deferred(clock::now() + 5ms, [task]() { task(5ms); });
    pool.enqueue_deferred(clock::now() + 10ms, [task]() {
        task(10ms);
        std::this_thread::sleep_for(std::chrono::microseconds(5)); // should print a warning
    });

    std::unique_lock<std::mutex> lock(mu);
    ASSERT_EQ(count, 0);
    cv.wait(lock, [&count]() { return count != 0; });
    ASSERT_EQ(count, 5);
    cv.wait(lock, [&count]() { return count != 5; });
    ASSERT_EQ(count, 15);
    cv.wait(lock, [&count]() { return count != 15; });
    ASSERT_EQ(count, 40);

    auto elapsed = clock::now() - start;
    auto wall    = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    ASSERT_GE(wall, 25);
    ASSERT_LT(wall, 26);

    pool.shutdown();

    EXPECT_ANY_THROW(pool.enqueue_deferred(clock::now() + 3ms, [] {}));
}