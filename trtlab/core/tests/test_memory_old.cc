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
#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/bytes.h"
#include "trtlab/core/memory/bytes_allocator.h"
#include "trtlab/core/memory/tensor.h"
#include "trtlab/core/memory/copy.h"
#include "trtlab/core/memory/malloc.h"
#include "trtlab/core/memory/sysv_allocator.h"
#include "trtlab/core/memory/transactional_allocator.h"
#include "trtlab/core/utils.h"

#include <cstring>
#include <list>

#include <gtest/gtest.h>

#include <foonathan/memory/container.hpp> // vector, list, list_node_size
#include <foonathan/memory/memory_pool.hpp> // memory_pool
#include <foonathan/memory/smart_ptr.hpp> // allocate_unique
#include <foonathan/memory/static_allocator.hpp> // static_allocator_storage, static_block_allocator
#include <foonathan/memory/temporary_allocator.hpp> // temporary_allocator
#include <foonathan/memory/allocator_storage.hpp>
#include <foonathan/memory/tracking.hpp>

#include <foonathan/memory/namespace_alias.hpp>

using namespace trtlab;

namespace {

static mem_size_t one_kb = 1024;
static mem_size_t one_mb = one_kb * one_kb;
typedef std::vector<mem_size_t> shape_t;

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
    EXPECT_EQ(shared->Size(), one_mb);
    EXPECT_EQ(shared->Capacity(), one_mb);
    EXPECT_EQ(shared->Shape().size(), 1);
    EXPECT_EQ(shared->Shape()[0], one_mb);
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
    EXPECT_EQ(unique->Size(), one_mb);
    EXPECT_EQ(unique->Capacity(), one_mb);
    EXPECT_EQ(unique->Shape().size(), 1);
    EXPECT_EQ(unique->Shape()[0], one_mb);
    EXPECT_EQ(unique->DataType(), types::bytes);
    EXPECT_EQ(unique->DeviceInfo().device_type, kDLCPU);
    EXPECT_EQ(unique->DeviceInfo().device_id, 0);

    unique.reset();
    EXPECT_FALSE(unique);
}

TYPED_TEST(TestMemory, ctor)
{
    Allocator<TypeParam> memory(one_mb);

    EXPECT_TRUE(memory.Data());
    EXPECT_EQ(memory.Size(), one_mb);
    EXPECT_EQ(memory.Capacity(), one_mb);
    EXPECT_EQ(memory.Shape().size(), 1);
    EXPECT_EQ(memory.Shape()[0], one_mb);
    EXPECT_EQ(memory.DataType(), types::bytes);
    EXPECT_EQ(memory.DeviceInfo().device_type, kDLCPU);
    EXPECT_EQ(memory.DeviceInfo().device_id, 0);
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
    DLOG(INFO) << "reshaped";
    DLOG(INFO) << memory;
    DLOG(INFO) << "reshaped";
    EXPECT_EQ(memory.Size(), one_mb);
    EXPECT_EQ(memory.Capacity(), one_mb);
    EXPECT_EQ(memory.Shape()[0], 512);
    EXPECT_EQ(memory.Shape()[1], 512);
    EXPECT_EQ(memory.DataType(), types::fp32);

    Allocator<TypeParam> host(std::move(memory));
    DLOG(INFO) << "moved";

    // moved location
    EXPECT_TRUE(host.Data());
    EXPECT_EQ(host.Size(), one_mb);
    EXPECT_EQ(host.Capacity(), one_mb);
    EXPECT_EQ(host.Shape()[0], 512);
    EXPECT_EQ(host.Shape()[1], 512);
    EXPECT_EQ(host.DataType(), types::fp32);

    // original location
    EXPECT_FALSE(memory.Data());
    EXPECT_EQ(memory.Size(), 0);
    EXPECT_EQ(memory.Capacity(), 0);
    EXPECT_EQ(memory.DataType(), types::nil);
}

TYPED_TEST(TestMemory, move_to_shared_ptr)
{
    Allocator<TypeParam> memory(one_mb);
    auto ptr = std::make_shared<Allocator<TypeParam>>(std::move(memory));
    EXPECT_TRUE(ptr);
    EXPECT_TRUE(ptr->Data());
    EXPECT_FALSE(memory.Data());
}

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
    shape_t shape = {one_mb};

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
    EXPECT_EQ(memory.Capacity(), one_mb);
    EXPECT_EQ(memory.Size(), 256 * 128 * types::int8.bytes());
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

// This tests an object very similar to the pybind11
// DLPack Descriptor
template<typename MemoryType>
class DescFromSharedPointer : public Descriptor<MemoryType>
{
  public:
    DescFromSharedPointer(std::shared_ptr<MemoryType> shared)
        : Descriptor<MemoryType>(
              shared,
              std::string("SharedPointer<" + std::string(shared->TypeName()) + ">").c_str()),
          m_ManagedMemory(shared)
    {
    }

    ~DescFromSharedPointer() override {}

    std::function<void()> CaptureSharedObject()
    {
        return [mem = m_ManagedMemory] { DLOG(INFO) << *mem; };
    }

  private:
    std::shared_ptr<MemoryType> m_ManagedMemory;
};

TYPED_TEST(TestMemory, DescriptorFromSharedPointer)
{
    auto shared = std::make_shared<Allocator<TypeParam>>(one_mb);

    std::weak_ptr<TypeParam> weak = shared;
    ASSERT_EQ(weak.use_count(), 1);

    std::function<void()> captured_obj;

    {
        // the descriptor captures the shared ptr twice
        // once in the descriptors deleter and the other
        // as the m_ManagedMemory shared_ptr
        DescFromSharedPointer<TypeParam> desc(shared);
        ASSERT_EQ(weak.use_count(), 3);

        captured_obj = desc.CaptureSharedObject();
        ASSERT_EQ(weak.use_count(), 4);
    }

    // descriptor out of scope
    ASSERT_EQ(weak.use_count(), 2);

    shared.reset();
    ASSERT_EQ(weak.use_count(), 1);

    // calling does not release
    captured_obj();

    // release
    DLOG(INFO) << "ref count to zero";
    captured_obj = nullptr;

    ASSERT_TRUE(weak.expired());
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

TEST_F(TestCopy, MallocToMalloc)
{
    char v0 = 111;
    char v1 = 222;

    Allocator<Malloc> m0(one_kb);
    std::memset(m0.Data(), v0, one_kb);

    auto m0_array = static_cast<char*>(m0.Data());
    EXPECT_EQ(m0_array[0], v0);
    EXPECT_EQ(m0_array[0], m0_array[1023]);

    auto m1 = std::make_unique<Allocator<Malloc>>(one_mb);
    std::memset(m1->Data(), v1, one_mb);

    auto m1_array = static_cast<char*>(m1->Data());
    EXPECT_EQ(m1_array[0], v1);
    EXPECT_EQ(m1_array[0], m1_array[1024]);

    EXPECT_NE(m0_array[0], m1_array[0]);

    // Copy smaller into larger
    Copy(*m1, m0, 1024);

    EXPECT_EQ(m1_array[0], v0);
    EXPECT_EQ(m1_array[1024], v1);
}

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

template<typename T>
class TestProvider : public BytesProvider<T>
{
  public:
    TestProvider() : m_Memory(one_mb) {}

    Bytes<T> Allocate(size_t offset, size_t size)
    {
        CHECK(m_Memory[offset + size]);
        CHECK(m_Memory[offset]);
        return this->BytesFromThis(m_Memory[offset], size);
    }

  private:
    const void* BytesProviderData() const final override { return m_Memory.Data(); }
    mem_size_t BytesProviderSize() const final override { return m_Memory.Size(); }
    const DLContext& BytesProviderDeviceInfo() const final override
    {
        return m_Memory.DeviceInfo();
    }
    const T& BytesProviderMemory() const final override { return m_Memory; }

    Allocator<T> m_Memory;
};

#if ENABLED_BYTES_HANDLE
TEST_F(TestGeneric, BytesHandleLifecycle)
{
    auto provider = std::make_shared<TestProvider<Malloc>>();
    auto obj = provider->Allocate(one_kb, one_kb);
    auto d1 = obj.Handle();
    // auto p1 = detail::BytesHandleFactory::HostPinned((void*)0xFACEBEEF, 2048);
    // auto g1 = detail::BytesHandleFactory::Device((void*)0xFACEB000, 1024*1024);

    auto d2 = d1;
    ASSERT_EQ(d2.Data(), d1.Data());
    ASSERT_EQ(d2.Size(), d1.Size());

    auto d3 = std::move(d2);
    ASSERT_EQ(d3.Data(), d1.Data());
    ASSERT_EQ(d3.Size(), d1.Size());
    ASSERT_EQ(d2.Data(), nullptr);
    ASSERT_EQ(d2.Size(), 0);

    BytesHandle<Malloc> d4(std::move(d3));
    ASSERT_EQ(d4.Data(), d1.Data());
    ASSERT_EQ(d4.Size(), d1.Size());
    ASSERT_EQ(d3.Data(), nullptr);
    ASSERT_EQ(d3.Size(), 0);

    BytesHandle<Malloc> d5(d4);
    ASSERT_EQ(d4.Data(), d1.Data());
    ASSERT_EQ(d4.Size(), d1.Size());
    ASSERT_EQ(d5.Data(), d1.Data());
    ASSERT_EQ(d5.Size(), d1.Size());

    // Downcasting from Malloc to HostMemory is allowed
    auto h1 = d5.BaseHandle();

    // Upcasting from HostMemory to Malloc is forbidden
    // EXPECT_THROW(auto m1 = h1.Cast<Malloc>(), std::bad_cast);

    // compiler error; different types
    // h1 = d4;
    // h1 = std::move(d4);
    // BytesHandle<Malloc> copy_ctor(h1);
    // BytesHandle<Malloc> move_ctor(std::move(h1));
}
#endif

namespace middleman {
class base
{
};

template<typename T>
class derived_base : public base
{
};

template<typename T>
class derived : public derived_base<typename T::BaseType>
{
};
} // namespace middleman

TEST_F(TestGeneric, TemplatedMiddleman)
{
    auto t = middleman::derived<Malloc>();
    auto derived = std::make_shared<middleman::derived<Malloc>>(std::move(t));
    std::shared_ptr<middleman::derived_base<HostMemory>> host = derived;

    // derived<Malloc> : derived_base<HostName>

    // std::shared_ptr<middleman::derived<HostMemory>> host = derived;
}

TEST_F(TestGeneric, ProtectedInheritance)
{
    struct a
    {
    };
    struct b : protected a
    {
    };

    EXPECT_FALSE((std::is_convertible<b*, a*>::value));

    struct c : public a
    {
    };

    EXPECT_TRUE((std::is_convertible<c*, a*>::value));

    struct d : protected a
    {
        operator const a&() { return *this; }
    };

    d objd;
    a obja = (const a&)objd;

    EXPECT_FALSE((std::is_convertible<const d&, const a&>::value));
}

TEST_F(TestGeneric, BytesCapture)
{
    auto provider = std::make_shared<TestProvider<Malloc>>();
    CHECK(provider);
    std::weak_ptr<TestProvider<Malloc>> weak = provider;

    {
        auto bytes = std::move(provider->Allocate(0, one_mb));
        ASSERT_EQ(bytes.Size(), one_mb);

        provider.reset();
        ASSERT_FALSE(weak.expired());
    }
    ASSERT_TRUE(weak.expired());
}

TEST_F(TestGeneric, BytesCopyMoveAssignment)
{
    auto provider = std::make_shared<TestProvider<Malloc>>();
    CHECK(provider);
    std::weak_ptr<TestProvider<Malloc>> weak = provider;
    ASSERT_EQ(provider.use_count(), 1);
    const auto half_mb = one_mb / 2;

    {
        // move assignment
        auto mv_assign = std::move(provider->Allocate(0, one_mb));
        auto ptr = mv_assign.Data();
        ASSERT_EQ(weak.use_count(), 2);
        EXPECT_EQ(mv_assign.Size(), one_mb);
        EXPECT_EQ(mv_assign.NDims(), 1);
        EXPECT_EQ(mv_assign.Shape()[0], one_mb);
        EXPECT_EQ(mv_assign.Strides()[0], 1);

        Bytes<Malloc> mv_ctor(std::move(mv_assign));
        ASSERT_EQ(weak.use_count(), 2);
        ASSERT_EQ(mv_ctor.Data(), ptr);
        ASSERT_EQ(mv_assign.Data(), nullptr);

        // check to ensure we can get back to the Provider's Memory object
        ASSERT_EQ(mv_ctor.Memory().Data(), ptr);

        // release the shared_ptr to the provider; our Bytes object still has a ref
        provider.reset();
        ASSERT_EQ(weak.use_count(), 1);
        ASSERT_EQ(mv_ctor.Data(), ptr);

        auto derived = std::make_shared<Bytes<Malloc>>(std::move(mv_ctor));
        std::shared_ptr<BytesBase> base = derived;
        ASSERT_EQ(derived->Data(), ptr);
        ASSERT_EQ(base->Data(), ptr);

        auto take_away = std::move(*derived);
        EXPECT_EQ(take_away.Data(), ptr);
        EXPECT_EQ(derived->Data(), nullptr);
        EXPECT_EQ(base->Data(), nullptr);

        // should not compile
        // public inheritance; protected copy/move ctors/assignments
        // BytesBase base = std::move(*derived);

        // should not compile
        // copy ctor deleted
        // Bytes<Malloc> copy_ctor(*derived);
        ASSERT_FALSE(std::is_copy_constructible<Bytes<Malloc>>::value);
        ASSERT_TRUE(std::is_move_constructible<Bytes<Malloc>>::value);

        // should not compile
        // copy ctor/assignment deleted
        // auto copy = *derived
        ASSERT_FALSE(std::is_copy_assignable<Bytes<Malloc>>::value);
        ASSERT_TRUE(std::is_move_assignable<Bytes<Malloc>>::value);

        BytesBaseType<HostMemory> host = std::move(take_away);
        ASSERT_EQ(weak.use_count(), 1);
        EXPECT_EQ(host.Data(), ptr);
        EXPECT_EQ(take_away.Data(), nullptr);
        EXPECT_EQ(take_away.Size(), 0);
        EXPECT_EQ(take_away.NDims(), 0);
        EXPECT_EQ(take_away.Shape()[0], 0);
        EXPECT_EQ(take_away.Strides()[0], 0);

        host.Release();
        ASSERT_EQ(weak.use_count(), 0);
    }
    ASSERT_EQ(weak.use_count(), 0);
}


TEST_F(TestGeneric, MemoryProvider)
{
    auto sysv = Allocator<SystemV>(2*one_mb);
    auto mb = BytesAllocator<Malloc>::Allocate(one_mb);
    auto sb = BytesAllocator<SystemV>::Expose(std::move(sysv));

}



TEST_F(TestGeneric, FooNathanAlignOffset)
{
/*
    using memory::detail::align_offset;
    auto val = align_offset((void*)(1), 64);
    EXPECT_EQ(val, 63);

    val = align_offset((void*)128, 64);
    EXPECT_EQ(val, 0);

    // this line should not compile in debug mode - alignments must be powers of 2
    // auto compile = align_offset((void*)(1), 63);

    // ASSERT_EQ(memory::unordered_set_node_size<int>::value, 0);

    struct log_tracker
    {
        void on_node_allocation(void* ptr, std::size_t size, std::size_t) noexcept
        {
            LOG(INFO) << " node allocated: " << ptr << "; " << size;
        }

        void on_node_deallocation(void *ptr, std::size_t size, std::size_t) noexcept
        {
            LOG(INFO) << " node deallocated: " << ptr << "; " << size;
        }
    };

    //auto heap = memory::MallocAllocator();
    auto storage = memory::make_allocator_adapter(memory::MallocAllocator());
    auto tracked = memory::make_tracked_allocator(log_tracker{}, std::move(storage));

    void *ptr = tracked.allocate_node(4096u, 64u);
    tracked.deallocate_node(ptr, 4096u, 64u);

    auto unique = memory::allocate_unique<int>(tracked, 1024);
    LOG(INFO) << *unique;

    memory::static_allocator_storage<4096u> static_storage;
    using static_pool_t = memory::memory_pool<memory::node_pool, memory::static_block_allocator>;
    static_pool_t static_pool(64u, 4096u, static_storage);
    auto tracked_pool = memory::make_tracked_allocator(log_tracker{}, std::move(static_pool));
    auto v1 = memory::allocate_unique<int>(tracked_pool, 2048);
    auto v2 = memory::allocate_unique<int>(tracked_pool, 4096);
    v1.reset();
    v1 = memory::allocate_unique<int>(tracked_pool, 1024);
*/
}

static void* make_ptr(std::size_t addr)
{
    return reinterpret_cast<void*>(addr);
}

static std::size_t make_int(void* addr)
{
    return reinterpret_cast<std::size_t>(addr);
}

static std::size_t make_int(const char* addr)
{
    return std::size_t(addr);
}




TEST_F(TestGeneric, FooFixedSizedStackAllocator)
{
    /*
    memory::trtlab::transactional_detail::FixedSizeStackAllocator alloc(make_ptr(128), 1024);

    std::size_t remaining = 1024;
    auto a0 = alloc.Allocate(1, 1);
    EXPECT_EQ(alloc.Available(), remaining -= 1);
    auto a1 = alloc.Allocate(1, 1);
    EXPECT_EQ(alloc.Available(), remaining -= 1);
    auto a2 = alloc.Allocate(1, 8);
    EXPECT_EQ(alloc.Available(), remaining = 1024 - 8 - 1);
    auto out_of_range = alloc.Allocate(2048, 8);

    char p = 128;

    EXPECT_EQ(make_int(a0), 128);
    EXPECT_EQ(make_int(a1), 129);
    EXPECT_EQ(make_int(a2), 128 + 8);
    EXPECT_EQ(out_of_range, nullptr);

    auto moved_alloc = std::move(alloc);

    auto should_be_nullptr = alloc.Allocate(8, 8);
    EXPECT_EQ(should_be_nullptr, nullptr);
    EXPECT_EQ(alloc.Available(), 0u);

    EXPECT_EQ(moved_alloc.Available(), remaining);
    auto a3 = moved_alloc.Allocate(128, 64);
    EXPECT_EQ(moved_alloc.Available(), remaining = 1024 - 64 - 128);


    EXPECT_TRUE(moved_alloc.Contains(make_ptr(128)));
    EXPECT_TRUE(moved_alloc.Contains(make_ptr(1024+128-1)));
    EXPECT_FALSE(moved_alloc.Contains(make_ptr(1024+128)));
    EXPECT_FALSE(moved_alloc.Contains(make_ptr(127)));

    EXPECT_FALSE(alloc.Contains(make_ptr(128)));
    EXPECT_FALSE(alloc.Contains(make_ptr(1024+128-1)));
    EXPECT_FALSE(alloc.Contains(make_ptr(1024+128)));
    EXPECT_FALSE(alloc.Contains(make_ptr(127)));

    EXPECT_TRUE(alloc.Contains(nullptr));
    EXPECT_FALSE(moved_alloc.Contains(nullptr));
    */
}

TEST_F(TestGeneric, FooTransactionalStackAllocator)
{
    /*
    std::size_t remaining = 1024;
    memory::trtlab::transactional_detail::StackAllocator alloc(make_ptr(128), remaining);

    auto a0 = alloc.Allocate(1, 1);
    EXPECT_EQ(alloc.Available(), remaining -= 1);
    EXPECT_EQ(alloc.InUseCount(), 1);

    auto a1 = alloc.Allocate(1, 1);
    EXPECT_EQ(alloc.Available(), remaining -= 1);
    EXPECT_EQ(alloc.InUseCount(), 2);

    auto a2 = alloc.Allocate(1, 8);
    EXPECT_EQ(alloc.Available(), remaining = 1024 - 8 - 1);
    EXPECT_EQ(alloc.InUseCount(), 3);

    EXPECT_ANY_THROW(auto out_of_range = alloc.Allocate(2048, 8));
    EXPECT_EQ(alloc.InUseCount(), 3);

    EXPECT_FALSE(alloc.ShouldReleaseAfterDeallocate(a1));
    EXPECT_FALSE(alloc.ShouldReleaseAfterDeallocate(a2));
    EXPECT_TRUE(alloc.ShouldReleaseAfterDeallocate(a0));
    */
}

template<typename RawAllocator>
auto make_balloc(std::size_t block_size, std::size_t max_blocks, RawAllocator&& alloc)
{
    return memory::trtlab::GrowthCappedFixedSizeBlockAllocator<RawAllocator>(block_size, max_blocks, std::move(alloc));
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

        shared_hdesc =
std::make_shared<nextgen::SharedDescriptor<HostMemory>>(std::move(hdesc)); dltensor =
(DLTensor)(*shared_hdesc);

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
    // Edge cases inspired from example output: https://stackoverflow.com/questions/3758606
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
    //  EXPECT_EQ(       1023, StringToBytes("1.023kb")); // no effort to control rounding -
    //  this fails with 1022
    EXPECT_EQ(1024, StringToBytes("1kib"));
    EXPECT_EQ(1024, StringToBytes("1.0KiB"));
    EXPECT_EQ(8000000, StringToBytes("8.0MB"));
    EXPECT_EQ(8388608, StringToBytes("8.0MiB"));
    EXPECT_EQ(18253611008, StringToBytes("17GiB"));
    EXPECT_DEATH(StringToBytes("17G"), "");
    EXPECT_DEATH(StringToBytes("yais"), "");
}

} // namespace
