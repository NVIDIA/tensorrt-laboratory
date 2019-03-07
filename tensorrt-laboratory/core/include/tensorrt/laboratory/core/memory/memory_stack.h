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
#include <memory>

#include <glog/logging.h>

#include "tensorrt/laboratory/core/memory/allocator.h"

namespace trtlab {

/**
 * @brief General MemoryStack
 *
 * Memory stack using a single memory allocation of MemoryType.  The stack pointer is advanced
 * by the Allocate function.  Each new stack pointer is aligned to either the DefaultAlignment
 * of the MemoryType or customized when the MemoryStack is instantiated.
 *
 * @tparam MemoryType
 */
template<class MemoryType>
class MemoryStack
{
  public:
    /**
     * @brief Construct a new MemoryStack object
     *
     * A stack using a single allocation of MemoryType.  The stack can only be advanced or
     * reset. Popping is not supported.
     *
     * @param size Size of the memory allocation
     * @param alignment Byte alignment for all pointer pushed on the stack
     */

    MemoryStack(std::unique_ptr<MemoryType> memory)
        : m_Memory(std::move(memory)), m_CurrentPointer(m_Memory->Data()), m_CurrentSize(0),
          m_Alignment(m_Memory->DefaultAlignment())
    {
        CHECK(m_Memory);
    }

    MemoryStack(size_t size) : MemoryStack(std::move(std::make_unique<Allocator<MemoryType>>(size)))
    {
    }

    virtual ~MemoryStack() {}

    using BaseType = typename MemoryType::BaseType;

    /**
     * @brief Advances the stack pointer
     *
     * Allocate advances the stack pointer by `size` plus some remainder to ensure subsequent
     * calles return aligned pointers
     *
     * @param size Number of bytes to reserve on the stack
     * @return void* Starting address of the stack reservation
     */
    void* Allocate(size_t size);

    /**
     * @brief Computes offset of a pointer to the base pointer of the stack
     *
     * Checks to ensure that the ptr passed is a member of the stack by ensuring that the offset
     * is never larger than the size of the stack.
     *
     * @param ptr
     * @return size_t
     */
    size_t Offset(void* ptr) const
    {
        char* base = static_cast<char*>(m_Memory->Data());
        size_t offset = static_cast<char*>(ptr) - base;
        CHECK_LT(offset, m_Memory->Size());
        return offset;
    }

    /**
     * @brief Reset the memory stack
     *
     * This operation resets the stack pointer to the base pointer of the memory allocation.
     */
    void Reset(bool writeZeros = false);

    /**
     * @brief Get Size of the Memory Stack
     */
    size_t Size() const { return m_Memory->Size(); }

    /**
     * @brief Get number of bytes currently allocated
     * @return size_t
     */
    size_t Allocated() const { return m_CurrentSize; }

    /**
     * @brief Get the number of free bytes that are not allocated
     * @return size_t
     */
    size_t Available() const { return Size() - Allocated(); }

    /**
     * @brief Byte alignment for stack allocations
     */
    size_t Alignment() const { return m_Alignment; }

    const MemoryType& Memory() const { return *m_Memory; }

  private:
    std::unique_ptr<MemoryType> m_Memory;
    void* m_CurrentPointer;
    size_t m_CurrentSize;
    size_t m_Alignment;
};

// Template Implementations

template<class MemoryType>
void* MemoryStack<MemoryType>::Allocate(size_t size)
{
    DLOG(INFO) << "Allocate pushes MemoryStack<" + m_Memory->Type() + "> Pointer by : " << size;

    CHECK_LE(m_CurrentSize + size, m_Memory->Size())
        << "Allocation too large.  Memory Total: " << m_Memory->Size() / (1024 * 1024) << "MB. "
        << "Used: " << m_CurrentSize / (1024 * 1024) << "MB. "
        << "Requested: " << size / (1024 * 1024) << "MB.";

    void* return_ptr = m_CurrentPointer;
    size_t remainder = size % m_Alignment;
    size = (remainder == 0) ? size : size + m_Alignment - remainder;
    m_CurrentPointer = static_cast<unsigned char*>(m_CurrentPointer) + size;
    m_CurrentSize += size;
    return return_ptr;
}

template<class MemoryType>
void MemoryStack<MemoryType>::Reset(bool writeZeros)
{
    m_CurrentPointer = m_Memory->Data();
    m_CurrentSize = 0;
    if(writeZeros)
    {
        m_Memory->Fill(0);
    }
}

} // namespace trtlab
