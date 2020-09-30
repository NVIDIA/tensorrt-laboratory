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
#include <chrono>
#include <thread>
#include <glog/logging.h>

#include "test_common.h"

#include "trtlab/core/memory/allocator.h"
#include "trtlab/core/memory/sysv_allocator.h"

#include <foonathan/memory/namespace_alias.hpp>

#include "test_transactional_allocator.h"

using namespace memory::literals;

class TestSysV : public TrackedTest
{
};

TEST_F(TestSysV, BlankTest) 
{
    ASSERT_TRUE(true);
    auto alloc = memory::trtlab::sysv_allocator();
    ASSERT_TRUE(true);
}

TEST_F(TestSysV, LifeCycle)
{
    auto raw = memory::trtlab::sysv_allocator();
    auto alloc = memory::make_allocator_adapter(std::move(raw));

    auto ptr = alloc.allocate_node(1024, 8);
    alloc.deallocate_node(ptr, 1024, 8);

    EndTest();
}

TEST_F(TestSysV, Attach)
{
    auto raw = memory::trtlab::sysv_allocator();
    auto alloc = memory::make_allocator_adapter(std::move(raw));

    auto ptr = alloc.allocate_node(1024, 8);
    ASSERT_NE(ptr, nullptr);

    auto info = alloc.get_allocator().sysv_info_for_pointer(ptr);
    ASSERT_NE(info.shm_id, -1);
    ASSERT_EQ(info.offset, 0);
    ASSERT_EQ(info.attachment_count, 1);

    auto attached_ptr = alloc.get_allocator().attach(info.shm_id);
    auto updated_info = alloc.get_allocator().sysv_info_for_pointer(attached_ptr);
    ASSERT_NE(attached_ptr, nullptr);
    ASSERT_NE(ptr, attached_ptr);
    ASSERT_EQ(updated_info.attachment_count, 2);

    alloc.deallocate_node(ptr, 1024, 8);

    auto info_post_dealloc = alloc.get_allocator().sysv_info_for_pointer(attached_ptr);
    ASSERT_EQ(info_post_dealloc.attachment_count, 1);

    alloc.deallocate_node(attached_ptr, 1024, 8);

    EndTest();
}

TEST_F(TestSysV, AttachShouldFailIfSegmentHasBeenReleased)
{
    auto raw = memory::trtlab::sysv_allocator();
    auto alloc = memory::make_allocator_adapter(std::move(raw));

    auto ptr = alloc.allocate_node(1024, 8);
    ASSERT_NE(ptr, nullptr);

    auto info = alloc.get_allocator().sysv_info_for_pointer(ptr);
    ASSERT_NE(info.shm_id, -1);
    ASSERT_EQ(info.offset, 0);
    ASSERT_EQ(info.attachment_count, 1);
    ASSERT_TRUE(info.is_attachable);

    alloc.get_allocator().release(info.shm_id);
    info = alloc.get_allocator().sysv_info_for_pointer(ptr);
    ASSERT_EQ(info.attachment_count, 1);
    ASSERT_FALSE(info.is_attachable);

    ASSERT_ANY_THROW(alloc.get_allocator().attach(info.shm_id));

    alloc.deallocate_node(ptr, 1024, 8);

    EndTest();
}


TEST_F(TestSysV, AsBaseForHighLevelAllocators)
{
    auto alloc = make_smart_transactional_allocator<memory::trtlab::sysv_allocator>(1_MiB, 2);

    auto ptr_0 = alloc.allocate_node(1024, 8);
    auto ptr_1 = alloc.allocate_node(2048, 8);
    auto ptr_2 = alloc.allocate_node(1_MiB, 8);

    auto info_0 = memory::trtlab::sysv_allocator::sysv_info_for_pointer(ptr_0);
    auto info_1 = memory::trtlab::sysv_allocator::sysv_info_for_pointer(ptr_1);
    auto info_2 = memory::trtlab::sysv_allocator::sysv_info_for_pointer(ptr_2);
    ASSERT_EQ(info_0.offset, 0);
    ASSERT_EQ(info_1.offset, 1024);
    ASSERT_EQ(info_0.shm_id, info_1.shm_id);
    ASSERT_NE(info_0.shm_id, info_2.shm_id);
    ASSERT_NE(info_1.shm_id, info_2.shm_id);

    alloc.deallocate_node(ptr_0, 1024, 8);
    alloc.deallocate_node(ptr_1, 2048, 8);
    alloc.deallocate_node(ptr_2, 1_MiB, 8);

    EndTest();
}