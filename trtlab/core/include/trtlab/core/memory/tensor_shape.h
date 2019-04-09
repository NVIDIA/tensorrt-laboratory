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
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace trtlab {

struct ITensorShape
{
    using dims_t = int64_t;

    virtual uint32_t NDims() const = 0;
    virtual uint64_t Size() const = 0;
    virtual const dims_t* Shape() const = 0;
    virtual const dims_t* Strides() const = 0;
};

class TensorShapeGeneric : public ITensorShape
{
  public:
    using shape_t = std::vector<dims_t>;
    using strides_t = std::vector<dims_t>;

    TensorShapeGeneric(int64_t size) : m_Shape({size}), m_Strides({1}), m_Items(size) { CheckShape(); }
    TensorShapeGeneric(const shape_t& shape)
        : m_Shape(shape), m_Items(0) { Initialize(); }
    TensorShapeGeneric(const shape_t& shape, const strides_t& strides)
        : m_Shape(shape), m_Strides(strides), m_Items(0) { Initialize(); }

    uint32_t NDims() const final override { return m_Shape.size(); }
    const dims_t* Shape() const final override { return &m_Shape[0]; }
    const dims_t* Strides() const final override { return &m_Strides[0]; }
    uint64_t Size() const final override
    {
        return std::accumulate(m_Shape.cbegin(), m_Shape.cend(), uint64_t(1),
                               std::multiplies<dims_t>());
    }

    bool IsCompact() const { return (bool)(Size() == m_Items); }
    bool IsStrided() const { return (bool)(m_Items > Size()); }
    bool IsBroadcasted() const { return (bool)(m_Items < Size()); }

  protected:
    void Initialize();
    void CheckShape();
    void SetStridesCompactRowMajor();
    void ValidateDimensions();

  private:
    shape_t m_Shape;
    strides_t m_Strides;
    int64_t m_Items;
};

} // namespace trtlab