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
#include "metrics.h"

#include <glog/logging.h>
#include <ostream>

namespace trtlab {

void Metrics::Initialize(uint32_t port)
{
    auto singleton = GetSingleton();
    if(singleton->m_Exposer)
    {
        LOG(WARNING) << "Metrics already initialized.  This call is ignored";
        return;
    }
    std::ostringstream stream;
    stream << "0.0.0.0:" << port;
    singleton->m_Exposer = std::make_unique<Exposer>(stream.str());
    singleton->m_Exposer->RegisterCollectable(singleton->m_Registry);
}

auto Metrics::GetRegistry() -> Registry&
{
    auto singleton = Metrics::GetSingleton();
    return *(singleton->m_Registry);
}

Metrics* Metrics::GetSingleton()
{
    static Metrics singleton;
    return &singleton;
}

Metrics::Metrics() : m_Registry(std::make_shared<Registry>()) {}

Metrics::~Metrics() {}

} // namespace trtlab