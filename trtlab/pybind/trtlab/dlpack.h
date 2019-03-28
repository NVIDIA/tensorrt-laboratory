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

#include <pybind11/pybind11.h>

#include "trtlab/core/memory/memory.h"
#include "trtlab/core/memory/descriptor.h"

namespace py = pybind11;

namespace trtlab {
namespace python {

class DLPack final
{
  public:
    static std::shared_ptr<CoreMemory> Import(py::capsule obj);
    static py::capsule Export(py::object);
    static py::capsule Export(std::shared_ptr<CoreMemory>);

  private:
    // An instance of DLPack::Manager is created on Export.  This object is
    // wrapped in a PyCapsule, though, the lifcyle of this object
    // is not necessarily own by the PyCapsule.
    //
    // The DLManagedTensor's deleter method is responsible for freeing
    // this object.
    //
    // The deleter is called by the PyCapsule if the PyCapsule is being
    // destroyed and no other entity has assumed ownership, i.e. the
    // name of the data object in the capsule is still "dltensor".
    //
    // Otherwise, the deleter will be called by the new owner of the
    // capsule data.
    class Manager
    {
      public:
        Manager(const DLTensor&, std::function<void()>);
        virtual ~Manager();

        // Creates a PyCasule from an instance of DLPack
        py::capsule Capsule();

      private:
        DLManagedTensor m_ManagedTensor;
        std::function<void()> m_Releaser;
    };
};

}
}