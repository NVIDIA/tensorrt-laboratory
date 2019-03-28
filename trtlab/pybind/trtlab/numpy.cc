
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
#include "numpy.h"

#include "trtlab/core/memory/host_memory.h"

namespace trtlab {

// Anonymous namespace for the DLPack Descriptor
namespace {

/*
template<typename MemoryType>
class NumPykDescriptor : public Descriptor<MemoryType>
{
  public:
    DLPackDescriptor(void* data, std::vector<int64_t> dims, const types::dtype& dt, std::function<void()> deleter)
        : Descriptor<MemoryType>(dltensor, deleter, "NumPy")
    {
    }

    ~DLPackDescriptor() override {}
};

*/

py::dtype NumPyDataType(const types::dtype& dt)
{
    if(dt == types::int8)
        return py::dtype::of<int8_t>();
    else if(dt == types::int16)
        return py::dtype::of<int16_t>();
    else if(dt == types::int32)
        return py::dtype::of<int32_t>();
    else if(dt == types::int64)
        return py::dtype::of<int64_t>();
    else if(dt == types::uint8)
        return py::dtype::of<uint8_t>();
    else if(dt == types::uint16)
        return py::dtype::of<uint16_t>();
    else if(dt == types::uint32)
        return py::dtype::of<uint32_t>();
    else if(dt == types::uint64)
        return py::dtype::of<uint64_t>();
    else if(dt == types::fp16)
        return py::dtype("float16");
    else if(dt == types::fp32)
        return py::dtype::of<float>();
    else if(dt == types::fp64)
        return py::dtype::of<double>();
    throw std::runtime_error("cannot convert to numpy dtype");
    return py::dtype();
}

types::dtype DatatTypeFromNumPy(py::dtype dt)
{

    for(auto& t : types::All)
    {
        if (dt.is(NumPyDataType(t)))
        {
            return t;
        }
    }
    throw std::runtime_error("Unsupported numpy.dtype");
}



} // namespace

namespace python {

py::array NumPy::Export(py::object obj)
{
    auto mem = obj.cast<std::shared_ptr<HostMemory>>();
    auto dltensor = mem->TensorInfo();
    auto np_dtype = NumPyDataType(mem->DataType());
    return py::array(np_dtype, mem->Shape(), mem->Data(), obj);
}

/*
std::shared_ptr<HostMemory> NumPy::Import(py::array array)
{
    auto handle = array.cast<py::handle>();
    handle.inc_ref();
    auto deleter = [handle] {
        DLOG(INFO) << "NumPy Wrapper Releasing Ownership of Arrary: " << handle.ptr();
        py::gil_scoped_acquire acquire;
        handle.dec_ref();
    };
}
*/

} // namespace python
} // namespace trtlab