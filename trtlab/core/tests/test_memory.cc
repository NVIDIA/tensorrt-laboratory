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
#include "tensorrt/laboratory/core/memory/allocator.h"
#include "tensorrt/laboratory/core/memory/copy.h"
#include "tensorrt/laboratory/core/memory/malloc.h"
#include "tensorrt/laboratory/core/memory/system_v.h"
#include "tensorrt/laboratory/core/utils.h"

#include <list>

#include <gtest/gtest.h>

using namespace trtlab;
using namespace trtlab;

namespace {
static size_t one_mb = 1024 * 1024;

template<typename T>
class TestMemory : public ::testing::Test
{
};

using MemoryTypes = ::testing::Types<Malloc, SystemV>;

TYPED_TEST_CASE(TestMemory, MemoryTypes);

/*
TYPED_TEST(TestMemory, should_not_compile)
{
    TypeParam memory(one_mb);
}
*/

TYPED_TEST(TestMemory, make_shared)
{
    auto shared = std::make_shared<Allocator<TypeParam>>(one_mb);
    EXPECT_TRUE(shared->Data());
    EXPECT_EQ(one_mb, shared->Size());
    shared.reset();
    EXPECT_FALSE(shared);
}

TYPED_TEST(TestMemory, make_unique)
{
    auto unique = std::make_unique<Allocator<TypeParam>>(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    unique.reset();
    EXPECT_FALSE(unique);
}

TYPED_TEST(TestMemory, ctor)
{
    Allocator<TypeParam> memory(one_mb);
    EXPECT_TRUE(memory.Data());
    EXPECT_EQ(one_mb, memory.Size());
}

TYPED_TEST(TestMemory, move_ctor)
{
    Allocator<TypeParam> memory(one_mb);
    Allocator<TypeParam> host(std::move(memory));

    EXPECT_TRUE(host.Data());
    EXPECT_EQ(one_mb, host.Size());

    EXPECT_FALSE(memory.Data());
    EXPECT_EQ(0, memory.Size());
}

TYPED_TEST(TestMemory, move_to_shared_ptr)
{
    Allocator<TypeParam> memory(one_mb);
    auto ptr = std::make_shared<Allocator<TypeParam>>(std::move(memory));
    EXPECT_TRUE(ptr);
    EXPECT_TRUE(ptr->Data());
}

TYPED_TEST(TestMemory, alignment)
{
    EXPECT_EQ(TypeParam::DefaultAlignment(), 64);
    EXPECT_EQ(TypeParam::AllocationSizeWithAlignment(1), 64);
    EXPECT_EQ(TypeParam::AllocationSizeWithAlignment(64), 64);
    EXPECT_EQ(TypeParam::AllocationSizeWithAlignment(65), 128);

    /*
        EXPECT_EQ(TypeParam::AllocationSizeWithAlignment<float>(15), 64);
        EXPECT_EQ(TypeParam::AllocationSizeWithAlignment<float>(16), 64);
        EXPECT_EQ(TypeParam::AllocationSizeWithAlignment<float>(17), 128);

        EXPECT_EQ(TypeParam::AllocationSizeWithAlignment<double>(7), 64);
        EXPECT_EQ(TypeParam::AllocationSizeWithAlignment<double>(8), 64);
        EXPECT_EQ(TypeParam::AllocationSizeWithAlignment<double>(9), 128);
    */
}

class TestSystemVMemory : public ::testing::Test
{
};

TEST_F(TestSystemVMemory, same_process)
{
    Allocator<SystemV> master(one_mb);
    EXPECT_TRUE(master.ShmID());
    auto attached = SystemV::Attach(master.ShmID());
    EXPECT_EQ(master.ShmID(), attached->ShmID());
    EXPECT_EQ(master.Size(), attached->Size());
    // different virtual address pointing at the same memory
    EXPECT_NE(master.Data(), attached->Data());
    auto master_ptr = static_cast<long*>(master.Data());
    auto attach_ptr = static_cast<long*>(attached->Data());
    *master_ptr = 0xDEADBEEF;
    EXPECT_EQ(*master_ptr, *attach_ptr);
    EXPECT_EQ(*attach_ptr, 0xDEADBEEF);
}

TEST_F(TestSystemVMemory, smart_ptrs)
{
    auto master = std::make_unique<Allocator<SystemV>>(one_mb);
    EXPECT_TRUE(master->ShmID());
    auto attached = SystemV::Attach(master->ShmID());
    EXPECT_EQ(master->ShmID(), attached->ShmID());
    EXPECT_EQ(master->Size(), attached->Size());

    // different virtual address pointing at the same memory
    EXPECT_NE(master->Data(), attached->Data());

    // ensure both segments point to the same data
    auto master_ptr = static_cast<long*>(master->Data());
    auto attach_ptr = static_cast<long*>(attached->Data());
    *master_ptr = 0xDEADBEEF;
    EXPECT_EQ(*master_ptr, *attach_ptr);

    DLOG(INFO) << "releasing the attached segment";
    attached.reset();
    DLOG(INFO) << "released the attached segment";
}

TEST_F(TestSystemVMemory, TryAttachingToDeletedSegment)
{
    auto master = std::make_unique<Allocator<SystemV>>(one_mb);
    EXPECT_TRUE(master->ShmID());
    auto shm_id = master->ShmID();
    master.reset();
    DLOG(INFO) << "trying to attach to a deleted segment";
    EXPECT_DEATH(auto attached = SystemV::Attach(shm_id), "");
}

class TestCopy : public ::testing::Test
{
};

TEST_F(TestCopy, MallocToMalloc)
{
    char v0 = 111;
    char v1 = 222;

    Allocator<Malloc> m0(1024);
    auto m1 = std::make_unique<Allocator<Malloc>>(1024 * 1024);

    m0.Fill(v0);
    auto m0_array = m0.CastToArray<char>();
    EXPECT_EQ(m0_array[0], v0);
    EXPECT_EQ(m0_array[0], m0_array[1023]);

    m1->Fill(v1);
    auto m1_array = m1->CastToArray<char>();
    EXPECT_EQ(m1_array[0], v1);
    EXPECT_EQ(m1_array[0], m1_array[1024]);

    EXPECT_NE(m0_array[0], m1_array[0]);

    // Copy smaller into larger
    Copy(*m1, m0, 1024);

    EXPECT_EQ(m1_array[0], v0);
    EXPECT_EQ(m1_array[1024], v1);
}

class TestBytesToString : public ::testing::Test
{
};

TEST_F(TestBytesToString, BytesToString)
{
    // CREDIT: https://stackoverflow.com/questions/3758606
    using std::string;
    EXPECT_EQ(string("0 B"), BytesToString(0));
    EXPECT_EQ(string("1000 B"), BytesToString(1000));
    EXPECT_EQ(string("1023 B"), BytesToString(1023));
    EXPECT_EQ(string("1.0 KiB"), BytesToString(1024));
    EXPECT_EQ(string("1.7 KiB"), BytesToString(1728));
    EXPECT_EQ(string("108.0 KiB"), BytesToString(110592));
    EXPECT_EQ(string("6.8 MiB"), BytesToString(7077888));
    EXPECT_EQ(string("432.0 MiB"), BytesToString(452984832));
    EXPECT_EQ(string("27.0 GiB"), BytesToString(28991029248));
    EXPECT_EQ(string("1.7 TiB"), BytesToString(1855425871872));
}

TEST_F(TestBytesToString, StringToBytes)
{
    EXPECT_EQ(0, StringToBytes("0B"));
    EXPECT_EQ(0, StringToBytes("0GB"));
    EXPECT_EQ(1000, StringToBytes("1000B"));
    EXPECT_EQ(1000, StringToBytes("1000b"));
    EXPECT_EQ(1000, StringToBytes("1kb"));
    EXPECT_EQ(1023, StringToBytes("1023b"));
    //  EXPECT_EQ(       1023, StringToBytes("1.023kb")); // no effort to control rounding - this
    //  fails with 1022
    EXPECT_EQ(1024, StringToBytes("1kib"));
    EXPECT_EQ(1024, StringToBytes("1.0KiB"));
    EXPECT_EQ(8000000, StringToBytes("8.0MB"));
    EXPECT_EQ(8388608, StringToBytes("8.0MiB"));
    EXPECT_EQ(18253611008, StringToBytes("17GiB"));
    EXPECT_DEATH(StringToBytes("17G"), "");
    EXPECT_DEATH(StringToBytes("yais"), "");
}

} // namespace
