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
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "trtlab/core/types.h"

using namespace trtlab;
using namespace trtlab::types;

class TestTypes : public ::testing::Test
{
};

TEST_F(TestTypes, int8)
{
    EXPECT_EQ(types::int8.to_dlpack().code, (uint8_t)kDLInt);
    EXPECT_EQ(types::int8.to_dlpack().bits, (uint8_t)8U);
    EXPECT_EQ(types::int8.to_dlpack().lanes, (uint16_t)1U);
}

TEST_F(TestTypes, uint8)
{
    EXPECT_EQ(types::uint8.to_dlpack().code, (uint8_t)kDLUInt);
    EXPECT_EQ(types::uint8.to_dlpack().bits, (uint8_t)8U);
    EXPECT_EQ(types::uint8.to_dlpack().lanes, (uint16_t)1U);
}

TEST_F(TestTypes, fp32)
{
    EXPECT_EQ(types::fp32.to_dlpack().code, (uint8_t)kDLFloat);
    EXPECT_EQ(types::fp32.to_dlpack().bits, (uint8_t)32U);
    EXPECT_EQ(types::fp32.to_dlpack().lanes, (uint8_t)1U);
}

TEST_F(TestTypes, ctors_and_assignment)
{
    // copy assignment
    auto bytes = types::uint8;
    EXPECT_EQ(bytes.bytes(), 1);

    // copy ctor
    types::dtype bytes2(bytes);

    EXPECT_EQ(bytes, bytes2);

    // move ctor
    auto bytes3 = std::move(bytes2);
    EXPECT_EQ(bytes, bytes3);
    EXPECT_NE(bytes, bytes2);
    EXPECT_EQ(bytes2.to_dlpack().bits, 0);
    EXPECT_EQ(bytes2.bytes(), 0);

    // move assignment
    bytes2 = std::move(bytes3);
    EXPECT_EQ(bytes, bytes2);
    EXPECT_NE(bytes, bytes3);

}

TEST_F(TestTypes, Equivalence)
{
    auto bytes = types::uint8;
    EXPECT_EQ(bytes, types::uint8);
    EXPECT_NE(bytes, types::int8);
    EXPECT_NE(bytes, types::fp32);

    DLDataType dlpack = bytes.to_dlpack();
    EXPECT_EQ(dlpack.code, (uint8_t)kDLUInt);
    EXPECT_EQ(dlpack.bits, (uint8_t)8U);
    EXPECT_EQ(dlpack.lanes, (uint16_t)1U);

    types::dtype test(dlpack);
    EXPECT_EQ(test, types::uint8);
}

TEST_F(TestTypes, TypeVsObject)
{
    ASSERT_EQ(dtype::from<void>(), bytes);
}

TEST_F(TestTypes, ArbituaryDLDataTypes)
{
    DLDataType dt;

    dt.code = kDLFloat;
    dt.bits = 32;
    dt.lanes = 1;

    dtype float32(dt);
    ASSERT_TRUE(float32.is_compatible<float>());

    // todo: we could allow this to be compatible
    dt.lanes = 3;
    dtype float3(dt);
    ASSERT_FALSE(float3.is_compatible<float>());
    ASSERT_TRUE(float3.is_compatible<void>());

    dt.lanes = 1;
    dt.bits = 33;
    dtype float33(dt);
    ASSERT_EQ(float33.bytes(), 5); // round to nearest byte
    ASSERT_FALSE(float3.is_compatible<float>());
    ASSERT_TRUE(float3.is_compatible<void>());

    dt.bits = 32;
    dt.code = 7;
    ASSERT_THROW(dtype unknown(dt), std::runtime_error);
}

TEST_F(TestTypes, CheckAllForCompatibility)
{
    for(const auto& t : types::All)
    {
        int count = 0;
        count += (int)t.is_compatible<int8_t>();
        count += (int)t.is_compatible<int16_t>();
        count += (int)t.is_compatible<int32_t>();
        count += (int)t.is_compatible<int64_t>();
        count += (int)t.is_compatible<uint8_t>();
        count += (int)t.is_compatible<uint16_t>();
        count += (int)t.is_compatible<uint32_t>();
        count += (int)t.is_compatible<uint64_t>();
        count += (int)t.is_compatible<float>();
        count += (int)t.is_compatible<double>();

        if(t == types::fp16) { count++; }

        DVLOG(2) << t;
        ASSERT_EQ(count, 1);
    }
}