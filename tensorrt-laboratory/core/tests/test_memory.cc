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
static int64_t one_mb = 1024 * 1024;

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

    EXPECT_EQ(shared->DataType(), types::bytes);
    EXPECT_EQ(shared->DeviceInfo().device_type, kDLCPU);
    EXPECT_EQ(shared->DeviceInfo().device_id, 0);

    shared.reset();
    EXPECT_FALSE(shared);
}

TYPED_TEST(TestMemory, make_unique)
{
    auto unique = std::make_unique<Allocator<TypeParam>>(one_mb);
    EXPECT_TRUE(unique->Data());
    EXPECT_EQ(one_mb, unique->Size());
    EXPECT_EQ(unique->DataType(), types::bytes);
    unique.reset();
    EXPECT_FALSE(unique);
}

TYPED_TEST(TestMemory, ctor)
{
    Allocator<TypeParam> memory(one_mb);
    EXPECT_TRUE(memory.Data());
    EXPECT_EQ(memory.DataType(), types::bytes);
    EXPECT_EQ(one_mb, memory.Size());
}

TYPED_TEST(TestMemory, move_ctor)
{
    Allocator<TypeParam> memory(one_mb);
    Allocator<TypeParam> host(std::move(memory));

    EXPECT_TRUE(host.Data());
    EXPECT_EQ(one_mb, host.Size());
    EXPECT_EQ(host.Shape()[0], one_mb);

    EXPECT_FALSE(memory.Data());
    EXPECT_EQ(memory.Size(), 0);
    EXPECT_EQ(memory.Capacity(), 0);
    EXPECT_EQ(memory.Shape().size(), 0);
}

TYPED_TEST(TestMemory, move_ctor_with_reshape)
{
    Allocator<TypeParam> memory(one_mb);

    memory.Reshape({512, 512}, types::fp32);
    EXPECT_EQ(memory.Size(), one_mb);
    EXPECT_EQ(memory.Capacity(), one_mb);
    EXPECT_EQ(memory.Shape()[0], 512);
    EXPECT_EQ(memory.Shape()[1], 512);
    EXPECT_EQ(memory.DataType(), types::fp32);

    Allocator<TypeParam> host(std::move(memory));

    EXPECT_TRUE(host.Data());
    EXPECT_EQ(host.Size(), one_mb);
    EXPECT_EQ(host.Capacity(), one_mb);
    EXPECT_EQ(host.Shape()[0], 512);
    EXPECT_EQ(host.Shape()[1], 512);
    EXPECT_EQ(host.DataType(), types::fp32);

    EXPECT_FALSE(memory.Data());
    EXPECT_EQ(memory.Size(), 0);
    EXPECT_EQ(memory.Capacity(), 0);
    EXPECT_EQ(memory.DataType(), types::nil);
}

/*
TYPED_TEST(TestMemory, move_to_shared_ptr)
{
    Allocator<TypeParam> memory(one_mb);
    auto ptr = std::make_shared<Allocator<TypeParam>>(std::move(memory));
    EXPECT_TRUE(ptr);
    EXPECT_TRUE(ptr->Data());
}
*/

TYPED_TEST(TestMemory, smart_move)
{
    auto shared = std::make_shared<Allocator<TypeParam>>(one_mb);
    std::weak_ptr<TypeParam> weak = shared;

    {
        std::vector<std::shared_ptr<CoreMemory>> core_segs;
        std::vector<std::shared_ptr<HostMemory>> host_segs;

        core_segs.push_back(shared);
        host_segs.push_back(std::move(shared));

        EXPECT_FALSE(weak.expired());
    }

    EXPECT_TRUE(weak.expired());
}

TYPED_TEST(TestMemory, shape)
{
    Allocator<TypeParam> memory(one_mb);
    std::vector<int64_t> shape = {one_mb};

    EXPECT_EQ(memory.Shape(), shape);

    // Exact Size
    memory.Reshape({512, 512}, types::fp32);
    EXPECT_EQ(memory.Shape()[0], 512);
    EXPECT_EQ(memory.Shape()[1], 512);
    EXPECT_EQ(memory.DataType(), types::fp32);
    EXPECT_EQ(memory.Size(), memory.Capacity());

    // Exact Size
    memory.Reshape({1024, 512}, types::fp16);
    EXPECT_EQ(memory.Shape()[0], 1024);
    EXPECT_EQ(memory.Shape()[1], 512);
    EXPECT_EQ(memory.DataType(), types::fp16);

    // Exact Size
    memory.Reshape({1024, 1024}, types::int8);
    EXPECT_EQ(memory.Shape()[0], 1024);
    EXPECT_EQ(memory.Shape()[1], 1024);
    EXPECT_EQ(memory.DataType(), types::int8);

    // Reshape to smaller than allocated
    memory.Reshape({256, 128}, types::int8);
    EXPECT_EQ(memory.Shape()[0], 256);
    EXPECT_EQ(memory.Shape()[1], 128);
    EXPECT_LT(memory.Size(), memory.Capacity());

    // Reshape to larger than allocated
    try
    {
        memory.Reshape({512, 513}, types::fp32);
        FAIL() << "Expected std::length_error";
    }
    catch(std::length_error const& err)
    {
        EXPECT_EQ(err.what(), std::string("Reshape exceeds capacity"));
    }
    catch(...)
    {
        FAIL() << "Expected std::length_error";
    }
}

TYPED_TEST(TestMemory, alignment)
{
    EXPECT_EQ(TypeParam::DefaultAlignment(), 64);
    EXPECT_EQ(TypeParam::AlignedSize(1), 64);
    EXPECT_EQ(TypeParam::AlignedSize(64), 64);
    EXPECT_EQ(TypeParam::AlignedSize(65), 128);

    /*
        EXPECT_EQ(TypeParam::AlignedSize<float>(15), 64);
        EXPECT_EQ(TypeParam::AlignedSize<float>(16), 64);
        EXPECT_EQ(TypeParam::AlignedSize<float>(17), 128);

        EXPECT_EQ(TypeParam::AlignedSize<double>(7), 64);
        EXPECT_EQ(TypeParam::AlignedSize<double>(8), 64);
        EXPECT_EQ(TypeParam::AlignedSize<double>(9), 128);
    */
}

class TestSystemVMemory : public ::testing::Test
{
};

TEST_F(TestSystemVMemory, same_process)
{
    Allocator<SystemV> first(one_mb);
    ASSERT_GE(first.ShmID(), 0);
    ASSERT_EQ(first.Size(), one_mb);
    ASSERT_EQ(first.Capacity(), one_mb);

    // test move
    auto master = std::move(first);
    ASSERT_GE(master.ShmID(), 0);
    ASSERT_EQ(master.Size(), one_mb);
    ASSERT_EQ(master.Capacity(), one_mb);
    ASSERT_EQ(first.ShmID(), -1);

    {
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
        DLOG(INFO) << "finished with attached";
    }
    DLOG(INFO) << "finished with test";
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

/*
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
*/

class TestGeneric : public ::testing::Test
{
};

TEST_F(TestGeneric, AllocatedPolymorphism)
{
    auto malloc = std::make_shared<Allocator<Malloc>>(1024);
    auto sysv = std::make_shared<Allocator<SystemV>>(1024);

    std::vector<std::shared_ptr<CoreMemory>> memory;

    memory.push_back(std::move(malloc));
    memory.push_back(std::move(sysv));
}

/*
TEST_F(TestGeneric, HostDescriptorDLTensorLifecycle)
{
    void *ptr = (void*)0xDEADBEEF;
    mem_size_t size = 13370;
    DLTensor dltensor;
    std::shared_ptr<nextgen::SharedDescriptor<HostMemory>> shared_hdesc;

    {
        nextgen::HostDescriptor hdesc(ptr, size, [&ptr, &size]{
            ptr = nullptr;
            size = 0;
        });
        EXPECT_EQ(hdesc.Data(), (void*)0xDEADBEEF);
        EXPECT_EQ(hdesc.Size(), 13370);
        hdesc.Reshape({13370/2, 1}, types::fp16);

        // regular descriptors can not expose a dltensor
        // dltensor = (DLTensor)hdesc;

        shared_hdesc = std::make_shared<nextgen::SharedDescriptor<HostMemory>>(std::move(hdesc));
        dltensor = (DLTensor)(*shared_hdesc);

        EXPECT_EQ(shared_hdesc->Data(), dltensor.data);
        EXPECT_EQ(shared_hdesc->Capacity(), 13370);
        EXPECT_EQ(dltensor.ctx.device_type, kDLCPU);
        EXPECT_EQ(dltensor.dtype.code, kDLFloat);
        EXPECT_EQ(dltensor.dtype.bits, 16U);
        EXPECT_EQ(dltensor.dtype.lanes, 1U);
        EXPECT_EQ(dltensor.ndim, 2);
    }

    // dltensor is still valid as shared_hdesc is still valid
    EXPECT_EQ(ptr, (void*)0xDEADBEEF);
    EXPECT_EQ(size, 13370);
    EXPECT_EQ(dltensor.ndim, 2);
    EXPECT_EQ(dltensor.shape[0], 13370/2);
    EXPECT_EQ(dltensor.shape[1], 1);
/*
    {
        HostDescriptor hdesc(dltensor, [shared_hdesc]{});
        // shared_hdesc and hdesc do share the same memory;
        // however, they have difference DLPack descriptors that were equivalent
        // on instantiate, but can change with their respective objects
        // changes to hdesc are not seen by shared_hdesc
        EXPECT_EQ(hdesc.Data(), dltensor.data);
        EXPECT_EQ(hdesc.Capacity(), 13370);
        EXPECT_EQ(dltensor.ndim, 2);
        EXPECT_EQ(hdesc.Shape().size(), dltensor.ndim);
        hdesc.ReshapeToBytes();
        EXPECT_EQ(hdesc.Shape()[0], 13370);
        EXPECT_EQ(dltensor.shape[0], 13370/2);
    }
}
*/

TEST_F(TestGeneric, MallocAndHostDescriptors)
{
    // auto mdesc = nextgen::Malloc::Allocate(one_mb);

    // Unable to move from from Malloc -> HostMemory
    // Descriptor<HostMemory> hdesc(std::move(mdesc));

    // auto shared = std::make_shared<Descriptor<Malloc>>(std::move(mdesc));

    // auto shared_from = std::make_shared<Descriptor<Malloc>>(mdesc);
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
