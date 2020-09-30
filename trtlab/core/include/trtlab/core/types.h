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

#include <cstdint> // for int64_t etc
#include <functional> // for std::multiplies
#include <memory>
#include <numeric>
#include <vector>
#include <ostream>

namespace trtlab {
namespace types {

template<typename T>
struct ArrayType final
{
    using NativeType = T;
    static uint64_t ItemSize();
    static DLDataType DataTypeInfo();

  protected:
    ArrayType() = default;
};

template<typename T>
DLDataType ArrayType<T>::DataTypeInfo()
{
    DLDataType dtype;
    // clang-format off
    if(std::is_floating_point<T>::value) { dtype.code = kDLFloat; }
    else if(std::is_integral<T>::value)
    {
        dtype.code = kDLUInt;
        if(std::is_signed<unsigned int>::value) { dtype.code = kDLInt; }
    }
    else { throw std::runtime_error("Only integer or floating point types accepted"); }
    // clang-format on

    dtype.bits = 8 * ItemSize();
    dtype.lanes = 1;

    return dtype;
}

template<typename T>
uint64_t ArrayType<T>::ItemSize()
{
    return sizeof(T);
}

template<>
inline DLDataType ArrayType<void>::DataTypeInfo()
{
    return ArrayType<uint8_t>::DataTypeInfo();
}

template<>
inline uint64_t ArrayType<void>::ItemSize()
{
    return 1;
}

struct dtype final
{
    dtype(const DLDataType&);
    dtype(uint8_t code, uint8_t bits, uint16_t lanes);

    template<typename T>
    static dtype from() { return dtype(ArrayType<T>::DataTypeInfo()); }

    template<typename T>
    bool is_compatible() const;

    dtype(dtype&&) noexcept;
    dtype& operator=(dtype&&) noexcept;

    dtype(const dtype&);
    dtype& operator=(const dtype&);

    virtual ~dtype() {}

    bool operator==(const dtype&) const;
    bool operator!=(const dtype& other) const { return !(*this == other); }

    int64_t bytes() const;
    int64_t itemsize() const { return bytes(); };
    const DLDataType& to_dlpack() const;

    std::string Description() const;

  protected:
    uint32_t code() const { return m_DLPackType.code; }
    uint32_t bits() const { return m_DLPackType.bits; }
    uint32_t lanes() const { return m_DLPackType.lanes; }

  private:
    dtype();
    DLDataType m_DLPackType;
    int64_t m_Bytes;

    friend std::ostream& operator<<(std::ostream& os, const dtype& dt);
};

template<typename T>
bool dtype::is_compatible() const
{
    if(lanes() != 1) { return false; }
    if(bits() != sizeof(T) * 8) { return false; }
    if(std::is_integral<T>::value)
    {
        if(std::is_signed<T>::value) { if(code() != kDLInt) { return false; } }
        else{ if(code() != kDLUInt) { return false; }}
    }
    else if(std::is_floating_point<T>::value)
    {
        if(code() != kDLFloat) { return false; }
    }
    return true;
}

template<>
inline bool dtype::is_compatible<void>() const
{
    return true;
}


static const auto nil = dtype(kDLInt, 0U, 0U);
static const auto bytes = dtype(kDLUInt, 8U, 1U);

static const auto int8 = dtype(kDLInt, 8U, 1U);
static const auto int16 = dtype(kDLInt, 16U, 1U);
static const auto int32 = dtype(kDLInt, 32U, 1U);
static const auto int64 = dtype(kDLInt, 64U, 1U);
static const auto uint8 = dtype(kDLUInt, 8U, 1U);
static const auto uint16 = dtype(kDLUInt, 16U, 1U);
static const auto uint32 = dtype(kDLUInt, 32U, 1U);
static const auto uint64 = dtype(kDLUInt, 64U, 1U);
static const auto fp16 = dtype(kDLFloat, 16U, 1U);
static const auto fp32 = dtype(kDLFloat, 32U, 1U);
static const auto fp64 = dtype(kDLFloat, 64U, 1U);

static const dtype All[] = {int8, int16, int32, int64, uint8, uint16, uint32, uint64, fp16, fp32, fp64};

} // namespace types
} // namespace trtlab