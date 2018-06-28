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
#include "YAIS/MemoryStack.h"

#include <glog/logging.h>

namespace yais
{
/*
MemoryStack::MemoryStack(std::unique_ptr<IMemory> memory)
{
    auto alignment = memory->DefaultAlignment();
    MemoryStack(std::move(memory), alignment);
}

MemoryStack::MemoryStack(std::unique_ptr<IMemory> memory, size_t alignment)
    : m_Allocator{std::move(memory)}, m_Alignment(alignment), m_CurrentSize(0),
      m_CurrentPointer(m_Allocator->Data()) {}

void *
MemoryStack::Allocate(size_t size)
{
    CHECK_LE(m_CurrentSize + size, m_Allocator->Size())
        << "Allocation too large.  Memory Total: " << m_Allocator->Size() / (1024 * 1024) << "MB. "
        << "Used: " << m_CurrentSize / (1024 * 1024) << "MB. "
        << "Requested: " << size / (1024 * 1024) << "MB.";

    void *return_ptr = m_CurrentPointer;
    size_t remainder = size % m_Alignment;
    size = (remainder == 0) ? size : size + m_Alignment - remainder;
    m_CurrentPointer = (unsigned char *)m_CurrentPointer + size;
    m_CurrentSize += size;
    return return_ptr;
}

void
MemoryStack::Reset(bool writeZeros)
{
    m_CurrentPointer = m_Allocator->Data();
    m_CurrentSize = 0;
    if (writeZeros)
    {
        m_Allocator->WriteZeros();
    }
}

size_t
MemoryStack::Size()
{
    return m_Allocator->Size();
}

size_t
MemoryStack::Allocated()
{
    return m_CurrentSize;
}

size_t
MemoryStack::Available()
{
    return Size() - m_CurrentSize;
}
*/

// MemoryStackTracker

MemoryStackTracker::MemoryStackTracker(std::shared_ptr<IMemoryStack> stack)
    : m_Stack(stack) {}

void *
MemoryStackTracker::Allocate(size_t size)
{
    m_StackSize.push_back(size);
    m_StackPointers.push_back(m_Stack->Allocate(size));
    return m_StackPointers.back();
}

void *
MemoryStackTracker::GetPointer(uint32_t id)
{
    CHECK_LT(id, m_StackPointers.size()) << "Invalid Stack Pointer ID";
    return m_StackPointers[id];
}

size_t
MemoryStackTracker::GetSize(uint32_t id)
{
    CHECK_LT(id, m_StackSize.size()) << "Invalid Stack Pointer ID";
    return m_StackSize[id];
}

size_t
MemoryStackTracker::Count()
{
    return m_StackSize.size();
}

void **
MemoryStackTracker::GetPointers()
{
    return (void **)m_StackPointers.data();
}

} // namespace yais