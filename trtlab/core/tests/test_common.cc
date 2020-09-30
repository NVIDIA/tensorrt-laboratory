
#include "test_common.h"

#include <glog/logging.h>

std::size_t log_tracker::node_total = 0;
std::size_t log_tracker::node_count = 0;

void TrackedTest::SetUp()
{
    ASSERT_EQ(log_tracker::node_total, 0);
    ASSERT_EQ(log_tracker::node_count, 0);
}

void TrackedTest::TearDown()
{
    DVLOG(1) << "^^^ deallocation of stack messages from end of test to start of teardown ^^^";
    ASSERT_EQ(log_tracker::node_total, 0);
    ASSERT_EQ(log_tracker::node_count, 0);
}

void TrackedTest::EndTest()
{
    DVLOG(1) << "*-----------* end of test *-----------*";
}

void log_tracker::on_node_allocation(void* ptr, std::size_t size, std::size_t alignment) noexcept
{
    DVLOG(1) << name << ": node allocated: " << ptr << "; size: " << size << "; alignment: " << alignment;
    node_total += size;
    node_count++;
}

void log_tracker::on_node_deallocation(void *ptr, std::size_t size, std::size_t alignment) noexcept
{
    EXPECT_GT(node_total, 0);
    EXPECT_GT(node_count, 0);
    DVLOG(1) << name << ": node deallocated: " << ptr << "; " << size << "; " << alignment;
    node_count--;
    node_total -= size;
}

void log_tracker::on_array_allocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
{
    DVLOG(1) << name << ": array allocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
    node_count += count;
    node_total += count * size;
}

void log_tracker::on_array_deallocation(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept
{
    EXPECT_GT(node_total, 0);
    EXPECT_GT(node_count, 0);
    DVLOG(1) << name << ": array deallocated: " << ptr << " ( " << count << " * " << size << "; " << alignment << " )";
    node_count -= count;
    node_total -= count * size;
}

void timeout_mutex::lock()
{
    auto now = std::chrono::steady_clock::now();
    auto success = this->try_lock_until(now + std::chrono::milliseconds(MUTEX_TIMEOUT_MS));
    if(!success) throw timeout_error();
}

void timeout_mutex::unlock()
{
    std::timed_mutex::unlock();
}

bool timeout_mutex::try_lock()
{
    return std::timed_mutex::try_lock();
}

#include <trtlab/core/ranges.h>
using namespace trtlab;

class TestCore : public ::testing::Test {};

TEST_F(TestCore, FindRanges0)
{
    std::vector<int> a { 1 };
    std::vector<std::pair<int, int>> a_ranges { {1,1} };
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1");
}


TEST_F(TestCore, FindRanges1)
{
    std::vector<int> a { 1,2 };
    std::vector<std::pair<int, int>> a_ranges { {1,2} };
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-2");
}

TEST_F(TestCore, FindRanges2)
{
    std::vector<int> a { 1,2,3 };
    std::vector<std::pair<int, int>> a_ranges { {1,3} };
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-3");
}

TEST_F(TestCore, FindRanges3)
{
    std::vector<int> a { 1,3 };
    std::vector<std::pair<int, int>> a_ranges { {1,1}, {3,3} };
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1,3");
}

TEST_F(TestCore, FindRanges4)
{
    std::vector<int> a { 1,2,4,5,6,10 };
    std::vector<std::pair<int, int>> a_ranges { {1,2}, {4,6}, {10,10} };
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-2,4-6,10");
}

TEST_F(TestCore, FindRanges5)
{
    std::vector<int> a { 0,1,2,3,4,5,6 };
    std::vector<std::pair<int, int>> a_ranges { {0,6} };
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "0-6");
}