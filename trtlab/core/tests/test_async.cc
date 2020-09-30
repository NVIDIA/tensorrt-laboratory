#include <gtest/gtest.h>

#include <boost/fiber/all.hpp>

class TestAsync : public ::testing::Test
{
};

TEST_F(TestAsync, FibersHello)
{
    int i = 1;

    boost::fibers::fiber f([&i] { i=2; });
    f.join();

    ASSERT_EQ(i, 2);
}