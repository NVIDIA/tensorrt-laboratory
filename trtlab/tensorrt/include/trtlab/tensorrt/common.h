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

#include <functional>
#include <memory>

#include <NvInfer.h>

namespace trtlab
{
    namespace TensorRT
    {
        struct NvInferDeleter
        {
            NvInferDeleter()
            {
                NvInferDeleter([] {});
            }
            NvInferDeleter(std::function<void()> capture) : m_Capture{capture} {}

            template <typename T>
            void operator()(T* obj) const
            {
                if (obj)
                {
                    obj->destroy();
                    if (m_Capture)
                    {
                        m_Capture();
                    }
                }
            }

        private:
            std::function<void()> m_Capture;
        };

        template <typename T>
        std::shared_ptr<T> nv_shared(T* obj)
        {
            if (!obj)
            {
                throw std::runtime_error("Failed to create object");
            }
            return std::shared_ptr<T>(obj, NvInferDeleter());
        };

        template <typename T>
        std::shared_ptr<T> nv_shared(T* obj, std::function<void()> capture)
        {
            if (!obj)
            {
                throw std::runtime_error("Failed to create object");
            }
            return std::shared_ptr<T>(obj, NvInferDeleter(capture));
        };

        template <typename T>
        using unique_t = std::unique_ptr<T, NvInferDeleter>;

        template <typename T>
        std::unique_ptr<T, NvInferDeleter> nv_unique(T* obj)
        {
            if (!obj)
            {
                throw std::runtime_error("Failed to create object");
            }
            return std::unique_ptr<T, NvInferDeleter>(obj);
        }

    } // namespace TensorRT
} // namespace trtlab
