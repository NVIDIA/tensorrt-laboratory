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

    LOG(INFO) << bytes;
    types::dtype test(dlpack);
    LOG(INFO) << test;
    EXPECT_EQ(test, types::uint8);
}