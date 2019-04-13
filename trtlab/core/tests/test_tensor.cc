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
#include "trtlab/core/memory/malloc.h"
#include "trtlab/core/memory/tensor.h"

#include <gtest/gtest.h>

using namespace trtlab;

static const uint64_t one_mb = 1024 * 1024;

class TestTensor : public ::testing::Test
{
};

TEST_F(TestTensor, StateFromBytes)
{
    auto bytes = BytesAllocator<Malloc>::Allocate(one_mb);
    auto tensor = TensorAllocator::FromBytes(std::move(bytes));

    ASSERT_NE(tensor.Data(), nullptr);
    ASSERT_EQ(tensor.NDims(), 1);
    ASSERT_EQ(tensor.Shape()[0], one_mb);
    ASSERT_EQ(tensor.Strides()[0], 1);
    ASSERT_EQ(tensor.NBytes(), one_mb);
    ASSERT_EQ(tensor.ItemSize(), 1);
    ASSERT_FALSE(tensor.IsShared());

    Tensor<HostMemory> moved(std::move(tensor));
    ASSERT_FALSE(moved.IsShared());
    ASSERT_THROW(tensor.IsShared(), std::runtime_error);

    {
        Tensor<HostMemory> copied(moved);
        ASSERT_TRUE(moved.IsShared());
        ASSERT_TRUE(copied.IsShared());
    }

    ASSERT_FALSE(moved.IsShared());

    {
        auto cp_assign = moved;
        ASSERT_TRUE(moved.IsShared());
        ASSERT_TRUE(cp_assign.IsShared());
    }
}

TEST_F(TestTensor, ReshapeView)
{
    auto bytes = BytesAllocator<Malloc>::Allocate(one_mb);
    auto tensor = TensorAllocator::FromBytes(std::move(bytes));

    ASSERT_NE(tensor.Data(), nullptr);
    ASSERT_EQ(tensor.NDims(), 1);
    ASSERT_EQ(tensor.Shape()[0], one_mb);
    ASSERT_EQ(tensor.Strides()[0], 1);
    ASSERT_EQ(tensor.NBytes(), one_mb);
    ASSERT_EQ(tensor.ItemSize(), 1);
    ASSERT_FALSE(tensor.IsShared());

    tensor.ReshapeView({512, 512}, types::fp32);
    ASSERT_EQ(tensor.NDims(), 2);
    ASSERT_EQ(tensor.Shape()[0], 512);
    ASSERT_EQ(tensor.Shape()[1], 512);
    ASSERT_EQ(tensor.Strides()[1], 1);
    ASSERT_EQ(tensor.Strides()[0], 512);
    ASSERT_EQ(tensor.ItemSize(), 4);
    ASSERT_EQ(tensor.NBytes(), one_mb);
    ASSERT_FALSE(tensor.IsShared());

    {
        Tensor<HostMemory> copied(tensor);
        ASSERT_TRUE(tensor.IsShared());
        ASSERT_TRUE(copied.IsShared());

        ASSERT_THROW(tensor.ReshapeView({512, 256}, types::fp32), std::runtime_error);
    }

    ASSERT_THROW(tensor.ReshapeView({512, 513}, types::fp32), std::runtime_error);

    tensor.ReshapeView({2, 128, 64}, types::int16);
    ASSERT_EQ(tensor.NDims(), 3);
    ASSERT_EQ(tensor.Shape()[0], 2);
    ASSERT_EQ(tensor.Shape()[1], 128);
    ASSERT_EQ(tensor.Shape()[2], 64);
    ASSERT_EQ(tensor.Strides()[2], 1);
    ASSERT_EQ(tensor.Strides()[1], 64);
    ASSERT_EQ(tensor.Strides()[0], 128*64);
    ASSERT_EQ(tensor.ItemSize(), 2);
    ASSERT_EQ(tensor.NBytes(), 32768);
}


TEST_F(TestTensor, Shapes1D)
{
    TensorShapeGeneric array(one_mb);
    ASSERT_EQ(array.NDims(), 1);
    ASSERT_EQ(array.Size(), one_mb);
    ASSERT_EQ(array.Shape()[0], one_mb);
    ASSERT_EQ(array.Strides()[0], 1);
    ASSERT_TRUE(array.IsCompact());
    ASSERT_FALSE(array.IsStrided());
    ASSERT_FALSE(array.IsBroadcasted());
}

TEST_F(TestTensor, ShapesNDGeneric)
{
    TensorShapeGeneric nd({8, 3, 224, 224});
    ASSERT_EQ(nd.NDims(), 4);
    ASSERT_EQ(nd.Size(), 1204224);
    ASSERT_EQ(nd.Strides()[3], 1);
    ASSERT_EQ(nd.Strides()[2], 224);
    ASSERT_EQ(nd.Strides()[1], 224 * 224);
    ASSERT_EQ(nd.Strides()[0], 224 * 224 * 3);
    ASSERT_TRUE(nd.IsCompact());
    ASSERT_FALSE(nd.IsStrided());
    ASSERT_FALSE(nd.IsBroadcasted());
}

TEST_F(TestTensor, ShapesNDGenericWithStrides)
{
    // exposed as NCHW
    // stored as NHWC
    TensorShapeGeneric nd({8, 3, 224, 224}, {224 * 224 * 3, 1, 224 * 3, 3});
    ASSERT_EQ(nd.NDims(), 4);
    ASSERT_EQ(nd.Size(), 1204224);
    ASSERT_TRUE(nd.IsCompact());
    ASSERT_FALSE(nd.IsStrided());
    ASSERT_FALSE(nd.IsBroadcasted());
}

TEST_F(TestTensor, ShapesEmpty)
{
    // exposed as NCHW
    // stored as NHWC
    ASSERT_ANY_THROW(TensorShapeGeneric nd({}));
}


