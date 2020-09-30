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
#include "trtlab/core/memory/tensor_shape.h"

#include <glog/logging.h>

namespace trtlab {

void TensorShapeGeneric::Initialize()
{
    CheckShape();
    if(m_Strides.size() == 0) { SetStridesCompactRowMajor(); }
    ValidateDimensions();
}

void TensorShapeGeneric::CheckShape()
{
    if(m_Shape.size() == 0)
    {
        throw std::runtime_error("Shape must have at least a dimension of 1");
    }
    for(const auto& d : m_Shape)
    {
        if(d <= 0)
        {
            throw std::runtime_error("Negative dimensions in Shape is not allowed");
        }
    }
}

void TensorShapeGeneric::ValidateDimensions()
{
    if(m_Shape.size() != m_Strides.size())
    {
        throw std::runtime_error("Shape and Strides must be of the same dimension");
    }

    if(Size())
    {
        for(int i = 0; i < m_Shape.size(); i++)
        {
            m_Items += (m_Shape[i] - 1) * m_Strides[i];
        }
        m_Items++;
    }
}

void TensorShapeGeneric::SetStridesCompactRowMajor()
{
    m_Strides.resize(m_Shape.size());

    int64_t offset = 1;
    for(int i = 1; i <= m_Shape.size(); i++)
    {
        m_Strides[m_Shape.size() - i] = offset;
        offset *= m_Shape[m_Shape.size() - i];
    }
}

} // namespace trtlab