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
#pragma once

#include <dlpack/dlpack.h>

#include <cstdint> // for int64_t etc
#include <functional> // for std::multiplies
#include <memory>
#include <numeric>
#include <vector>
#include <ostream>

namespace trtlab {
namespace types {

struct dtype
{
    dtype(const DLDataType&);
    dtype(uint8_t code, uint8_t bits, uint16_t lanes);

    dtype(dtype&&) noexcept;
    dtype& operator=(dtype&&) noexcept;

    dtype(const dtype&);
    dtype& operator=(const dtype&);

    virtual ~dtype() {}

    bool operator==(const dtype&) const;
    bool operator!=(const dtype& other) const { return !(*this == other); }

    int64_t bytes() const;
    const DLDataType& to_dlpack() const;

  private:
    dtype();
    DLDataType m_DLPackType;
    int64_t m_Bytes;

    friend std::ostream& operator<<(std::ostream& os, const dtype& dt);
};

static const auto nil = dtype(kDLInt, 0U, 0U);
static const auto bytes = dtype(kDLUInt, 8U, 1U);

static const auto int8 = dtype(kDLInt, 8U, 1U);
static const auto int16 = dtype(kDLInt, 16U, 1U);
static const auto int32 = dtype(kDLInt, 32U, 1U);
static const auto uint8 = dtype(kDLUInt, 8U, 1U);
static const auto uint16 = dtype(kDLUInt, 16U, 1U);
static const auto uint32 = dtype(kDLUInt, 32U, 1U);
static const auto fp16 = dtype(kDLFloat, 16U, 1U);
static const auto fp32 = dtype(kDLFloat, 32U, 1U);
static const auto fp64 = dtype(kDLFloat, 64U, 1U);

static const dtype All[] = {int8, int16, int32, uint8, uint16, uint32, fp16, fp32, fp64};

} // namespace types
} // namespace trtlab