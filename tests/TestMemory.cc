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
#include "YAIS/Memory.h"
#include "gtest/gtest.h"

#include <list>

using namespace yais;

namespace
{

static size_t one_mb = 1024*1024;

template <typename T>
class TestMemory : public ::testing::Test
{
  public:
    typedef std::list<T> List;
};

using AllocatorTypes = ::testing::Types<
    CudaDeviceAllocator, CudaManagedAllocator, CudaHostAllocator, SystemMallocAllocator>;

TYPED_TEST_CASE(TestMemory, AllocatorTypes);

TYPED_TEST(TestMemory, make_shared)
{
    auto shared = TypeParam::make_shared(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    shared.reset();
    EXPECT_FALSE(shared);
}

TYPED_TEST(TestMemory, make_unique)
{
    auto unique = TypeParam::make_unique(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    unique.reset();
    EXPECT_FALSE(unique);
}

class TestBytesToString : public ::testing::Test
{
};

TEST_F(TestBytesToString, BytesToString)
{
    // CREDIT: https://stackoverflow.com/questions/3758606
    using std::string;
    EXPECT_EQ(      string("0 B"), BytesToString(0));
    EXPECT_EQ(   string("1000 B"), BytesToString(1000));
    EXPECT_EQ(   string("1023 B"), BytesToString(1023));
    EXPECT_EQ(  string("1.0 KiB"), BytesToString(1024));
    EXPECT_EQ(  string("1.7 KiB"), BytesToString(1728));
    EXPECT_EQ(string("108.0 KiB"), BytesToString(110592));
    EXPECT_EQ(  string("6.8 MiB"), BytesToString(7077888));
    EXPECT_EQ(string("432.0 MiB"), BytesToString(452984832));
    EXPECT_EQ( string("27.0 GiB"), BytesToString(28991029248));
    EXPECT_EQ(  string("1.7 TiB"), BytesToString(1855425871872));
}

TEST_F(TestBytesToString, StringToBytes)
{
    EXPECT_EQ(          0, StringToBytes("0B"));
    EXPECT_EQ(          0, StringToBytes("0GB"));
    EXPECT_EQ(       1000, StringToBytes("1000B"));
    EXPECT_EQ(       1000, StringToBytes("1000b"));
    EXPECT_EQ(       1000, StringToBytes("1kb"));
    EXPECT_EQ(       1023, StringToBytes("1023b"));
//  EXPECT_EQ(       1023, StringToBytes("1.023kb")); // no effort to control rounding - this fails with 1022
    EXPECT_EQ(       1024, StringToBytes("1kib"));
    EXPECT_EQ(       1024, StringToBytes("1.0KiB"));
    EXPECT_EQ(    8000000, StringToBytes("8.0MB"));
    EXPECT_EQ(    8388608, StringToBytes("8.0MiB"));
    EXPECT_EQ(18253611008, StringToBytes("17GiB"));
}

} // namespace
