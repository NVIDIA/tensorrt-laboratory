/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

namespace yais
{
template<class MemoryType>
BaseMemory<MemoryType>::BaseMemory(void* ptr, size_t size, bool allocated)
    : m_MemoryAddress(ptr), m_BytesAllocated(size), m_Allocated(allocated)
{
}

template<class MemoryType>
BaseMemory<MemoryType>::BaseMemory(BaseMemory&& other) noexcept
    : m_MemoryAddress{std::exchange(other.m_MemoryAddress, nullptr)},
      m_BytesAllocated{std::exchange(other.m_BytesAllocated, 0)}, 
      m_Allocated{std::exchange(other.m_Allocated, false)} 
{
}

template<class MemoryType>
void* BaseMemory<MemoryType>::Data() const
{
    return m_MemoryAddress;
}

template<class MemoryType>
size_t BaseMemory<MemoryType>::Size() const
{
    return m_BytesAllocated;
}

template<class MemoryType>
bool BaseMemory<MemoryType>::Allocated() const
{
    return m_Allocated;
}
/*
template<class MemoryType>
BaseMemory<MemoryType>::Memory(Memory&& other)
{
    m_MemoryAddress = other.m_MemoryAddress;
    m_BytesAllocated = other.m_BytesAllocated;
    other.m_BasePointer = nullptr;
    other.m_BytesAllocated = 0;
}

template<class MemoryType>
Memory& BaseMemory<MemoryType>::operator=(Memory&& other)
{
    m_MemoryAddress = other.m_MemoryAddress;
    m_BytesAllocated = other.m_BytesAllocated;
    other.m_BasePointer = nullptr;
    other.m_BytesAllocated = 0;
}
*/

} // namespace yais