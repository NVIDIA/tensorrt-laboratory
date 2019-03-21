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

#include "tensorrt/laboratory/core/memory/memory.h"

namespace trtlab {

template<typename MemoryType>
class Descriptor : public MemoryType
{
  public:
    virtual ~Descriptor() override;
    const char* TypeName() const final override;

    Descriptor(Descriptor<MemoryType>&&) noexcept;

  protected:
    Descriptor(void*, mem_size_t, std::function<void()>, const char*);
    Descriptor(void*, mem_size_t, const MemoryType&, std::function<void()>, const char*);
    Descriptor(const DLTensor&, std::function<void()> deleter, const char*);
    Descriptor(std::shared_ptr<MemoryType>, const char*);
    Descriptor(MemoryType&&, std::function<void()>, const char*);

    // Descriptor(Descriptor&&) noexcept;
    //Descriptor& operator=(Descriptor&&) noexcept = delete;

    // Descriptor(const Descriptor&) = delete;
    ////  Descriptor& operator=(const Descriptor&) = delete;

  private:
    void SetDescription(const std::string&);

    std::function<void()> m_Deleter;
    std::string m_Desc;
};

template<typename MemoryType>
using DescriptorHandle = std::unique_ptr<MemoryType>;


} // namespace trtlab

#include "tensorrt/laboratory/core/impl/memory/descriptor.h"