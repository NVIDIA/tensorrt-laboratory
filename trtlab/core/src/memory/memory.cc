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
#include "trtlab/core/memory/memory.h"
#include "trtlab/core/utils.h"

#include <cstring>
#include <glog/logging.h>

namespace trtlab {

CoreMemory::CoreMemory() : m_Size(0), m_Capacity(0), m_Stride1(1)
{
    m_Handle.data = nullptr;
    m_Handle.ndim = 0;
    m_Handle.shape = &m_Size;
    m_Handle.strides = &m_Stride1;
    m_Handle.dtype.code = kDLUInt;
    m_Handle.dtype.bits = 8U;
    m_Handle.dtype.lanes = 1U;
    m_Handle.ctx.device_type = kDLCPU;
    m_Handle.ctx.device_id = 0;
    m_Handle.byte_offset = 0;
}

CoreMemory::CoreMemory(void* ptr, mem_size_t size) : CoreMemory() { SetDataAndSize(ptr, size); }

void CoreMemory::SetDataAndSize(void* ptr, mem_size_t size)
{
    m_Handle.data = ptr;
    m_Handle.ndim = 1U;
    m_Size = size;
    m_Capacity = size;
}

CoreMemory::CoreMemory(void* ptr, mem_size_t size, const CoreMemory& mem) : CoreMemory(ptr, size)
{
    if(!mem.ContiguousBytes())
    {
        throw std::runtime_error(
            "CoreMemory ctor required the reference to be a contiguous bytearray");
    }
    m_Handle.ctx = mem.m_Handle.ctx;
}

CoreMemory::CoreMemory(const DLTensor& handle) : CoreMemory() { SetHandle(handle); }

CoreMemory::CoreMemory(CoreMemory&& other) noexcept
    : m_Deleter(std::exchange(other.m_Deleter, nullptr))
{
    SetHandle(other.m_Handle);
    other.m_Handle = DLTensor();
    other.m_Size = 0;
    other.m_Capacity = 0;
}

CoreMemory::CoreMemory(void* ptr, std::vector<int64_t> shape, const types::dtype& dt)
{
    DCHECK(ptr);
    DCHECK(shape.size());

    m_Handle.data = ptr;
    m_Handle.ndim = shape.size();
    m_Handle.dtype = dt.to_dlpack();

    if(m_Handle.ndim == 1)
    {
        m_Size = m_Capacity = shape[0];
        m_Handle.shape = &m_Size;
    }
    else
    {
        m_Size = m_Capacity = SizeFromShape(shape, dt.bytes());
        m_Shape = shape;
        m_Handle.shape = &m_Shape[0];
    }
}

CoreMemory::~CoreMemory()
{
    if(m_Deleter)
    {
        DLOG(INFO) << "Deallocating ptr: " << m_Handle.data;
        m_Deleter();
    }
}

types::dtype CoreMemory::DataType() const { return types::dtype(m_Handle.dtype); }

std::vector<int64_t> CoreMemory::Shape() const
{
    if(m_Handle.ndim == 1)
    {
        std::vector<int64_t> shape = {m_Size};
        return shape;
    }
    return m_Shape;
}

std::vector<int64_t> CoreMemory::Strides() const
{
    if(m_Handle.ndim == 1)
    {
        std::vector<int64_t> strides = {m_Handle.strides[0]};
        return strides;
    }
    return m_Strides;
}

void CoreMemory::Reshape(const std::vector<int64_t>& shape) { Reshape(shape, DataType()); }

void CoreMemory::Reshape(const std::vector<int64_t>& shape, const types::dtype& dt)
{
    SetShape(shape, dt, true);
}

void CoreMemory::SetShape(const std::vector<int64_t>& shape, const types::dtype& dt,
                          bool check_size)
{
    DCHECK(shape.size());
    DCHECK(dt.bytes());

    auto size = SizeFromShape(shape, dt.bytes());
    if(check_size && size > m_Capacity)
    {
        throw std::length_error("Reshape exceeds capacity");
    }

    m_Handle.ndim = shape.size();
    m_Handle.dtype = dt.to_dlpack();
    m_Handle.shape = nullptr;
    m_Size = size;

    if(shape.size() == 1)
    {
        m_Handle.shape = &m_Size;
    }
    else
    {
        m_Shape = shape;
        m_Handle.shape = &m_Shape[0];
    }

    // set strides for fortran column major
    m_Strides.resize(m_Handle.ndim);
    m_Handle.strides = &m_Strides[0];
    int64_t offset = 1;
    for(int i=1; i<=m_Handle.ndim; i++)
    {
        m_Strides[m_Handle.ndim - i] = offset;
        offset *= shape[m_Handle.ndim - i];
    }
}

void CoreMemory::ReshapeToBytes() { Reshape({Capacity()}, types::bytes); }

void* CoreMemory::operator[](size_t offset)
{
    CHECK_LE(offset, Size());
    return static_cast<void*>(static_cast<char*>(Data()) + offset);
}

const void* CoreMemory::operator[](size_t offset) const
{
    CHECK_LE(offset, Size());
    return static_cast<const void*>(static_cast<const char*>(Data()) + offset);
}

void CoreMemory::SetHandle(const DLTensor& handle)
{
    m_Handle = handle;
    if(handle.ndim == 1)
    {
        m_Capacity = m_Size = handle.shape[0] * SizeOfDataType();
        m_Handle.shape = &m_Size;
    }
    else
    {
        m_Shape.resize(handle.ndim);
        m_Handle.shape = &m_Shape[0];
        std::memcpy(m_Handle.shape, handle.shape, handle.ndim * sizeof(int64_t));
        m_Size = SizeFromShape(m_Shape, SizeOfDataType());
        m_Capacity = m_Size;
    }
    if(handle.strides)
    {
        m_Strides.resize(handle.ndim);
        m_Handle.strides = &m_Strides[0];
        std::memcpy(m_Handle.strides, handle.strides, handle.ndim * sizeof(int64_t));
        // compute capacity from strides
        mem_size_t itemsize = DataType().bytes();
        mem_size_t offset_to_end = itemsize;
        const auto& shape = Shape();
        for(int i = 0; i < m_Handle.ndim; i++)
        {
            offset_to_end += m_Strides[i] * (shape[i] - 1) * itemsize;
        }
        m_Capacity = offset_to_end;
    }
}

mem_size_t CoreMemory::SizeOfDataType() const
{
    return ((mem_size_t)m_Handle.dtype.bits * (mem_size_t)m_Handle.dtype.lanes + 7) / 8;
}

bool CoreMemory::ContiguousBytes() const
{
    return ((m_Handle.ndim == 1) && (m_Size == m_Capacity) && (DataType() == types::bytes));
}

mem_size_t CoreMemory::SizeFromShape(const std::vector<mem_size_t>& shape, mem_size_t sizeof_dtype)
{
    mem_size_t size = std::accumulate(std::begin(shape), std::end(shape), mem_size_t(1),
                                      std::multiplies<mem_size_t>());
    size *= sizeof_dtype;
    return size;
}

std::string CoreMemory::Description() const
{
    std::ostringstream os;
    // clang-format off
    os << "[" << TypeName();
    if(m_Handle.ctx.device_type == kDLGPU) { os << " gpu:" << m_Handle.ctx.device_id; }
    os << " " << m_Handle.data << "; shape: (";
    for(int i=0; i< m_Handle.ndim; i++) { os << (i ? "," : "") << m_Handle.shape[i]; }
    os << "); strides: (";
    for(int i=0; i< m_Handle.ndim; i++) { os << (i ? "," : "") << m_Handle.strides[i]; }
    os << "); dtype: " << DataType() << "; size: " << BytesToString(Size());
    if(Size() != Capacity()) { os << "; capacity: " << BytesToString(Capacity()); }
    os << "]";
    // clang-format on
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const CoreMemory& core)
{
    // clang-format off
    os << core.Description();
    return os;
}

} // namespace trtlab