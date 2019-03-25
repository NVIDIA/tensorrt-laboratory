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

#include "tensorrt/laboratory/core/memory/allocator.h"
#include "tensorrt/laboratory/core/types.h"

namespace trtlab {

class CoreMemory
{
  protected:
    CoreMemory();
    CoreMemory(void*, mem_size_t);
    CoreMemory(void*, mem_size_t, const CoreMemory&);
    CoreMemory(const DLTensor&);
    CoreMemory(void*, std::vector<int64_t>, const types::dtype&);
    CoreMemory(void*, std::vector<int64_t>, std::vector<int64_t>, const types::dtype&);

    CoreMemory(CoreMemory&& other) noexcept;
    CoreMemory& operator=(CoreMemory&&) noexcept = delete;

    CoreMemory(const CoreMemory&) = delete;
    CoreMemory& operator=(const CoreMemory&) = delete;

  public:
    virtual ~CoreMemory();

    // pointer to the internal buffer
    inline void* Data() { return m_Handle.data; }
    inline const void* Data() const { return m_Handle.data; }

    // bytes used by the current shape
    mem_size_t Size() const { return m_Size; }

    // total bytes allocated
    mem_size_t Capacity() const { return m_Capacity; }

    types::dtype DataType() const;
    std::vector<int64_t> Shape() const;
    std::vector<int64_t> Strides() const;

    void Reshape(const std::vector<int64_t>& shape);
    void Reshape(const std::vector<int64_t>& shape, const types::dtype&);
    void ReshapeToBytes();

    void* operator[](size_t offset);
    const void* operator[](size_t offset) const;

    const DLContext& DeviceInfo() const { return m_Handle.ctx; }
    const DLTensor& TensorInfo() const { return m_Handle; }

    virtual bool IsHostMemory() const = 0;
    virtual bool IsPinnedMemory() const = 0;

    std::string Description() const;

  protected:
    // human readable typename
    virtual const char* TypeName() const = 0;

    void SetShape(const std::vector<int64_t>& shape, const types::dtype& dt, bool check_size);

  private:
    void SetDataAndSize(void*, mem_size_t);
    void SetHandle(const DLTensor&);
    bool ContiguousBytes() const;
    mem_size_t SizeOfDataType() const;
    static mem_size_t SizeFromShape(const std::vector<mem_size_t>& shape, mem_size_t sizeof_dtype);

    DLTensor m_Handle;
    mem_size_t m_Size;
    mem_size_t m_Capacity;
    mem_size_t m_Stride1;
    std::vector<mem_size_t> m_Shape;
    std::vector<mem_size_t> m_Strides;
    std::function<void()> m_Deleter;

    template<typename MemoryType>
    friend class Allocator;

    friend std::ostream& operator<<(std::ostream&, const CoreMemory&);
};

template<class MemoryType>
class BaseMemory : public CoreMemory
{
  public:
    using CoreMemory::CoreMemory;
    using BaseType = MemoryType;

    static size_t AlignedSize(size_t);
};

template<class MemoryType>
size_t BaseMemory<MemoryType>::AlignedSize(size_t size_in_bytes)
{
    size_t alignment = MemoryType::DefaultAlignment();
    size_t remainder = size_in_bytes % alignment;
    return (remainder == 0) ? size_in_bytes : size_in_bytes + alignment - remainder;
}

} // namespace trtlab
