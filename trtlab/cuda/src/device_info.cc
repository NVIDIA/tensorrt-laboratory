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
#include "trtlab/cuda/device_info.h"

#include <algorithm>

#include <glog/logging.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

#define test_bit(_n, _p) (_n & (1UL << _p))

namespace
{
    struct nvmlState
    {
        nvmlState()
        {
            CHECK_EQ(nvmlInit(), NVML_SUCCESS) << "Failed to initialize NVML";
        }
        ~nvmlState()
        {
            CHECK_EQ(nvmlShutdown(), NVML_SUCCESS) << "Failed to Shutdown NVML";
        }
    };

    static auto nvmlInstatnce = std::make_unique<nvmlState>();

    nvmlDevice_t GetHandleById(unsigned int device_id)
    {
        nvmlDevice_t handle;
        CHECK_EQ(nvmlDeviceGetHandleByIndex(device_id, &handle), NVML_SUCCESS);
        return handle;
    }

} // namespace

namespace trtlab
{
    cpu_set DeviceInfo::Affinity(int device_id)
    {
        nvmlDevice_t  gpu      = GetHandleById(device_id);
        unsigned long cpu_mask = 0;
        cpu_set       cpus;

        CHECK_EQ(nvmlDeviceGetCpuAffinity(gpu, sizeof(cpu_mask), &cpu_mask), NVML_SUCCESS)
            << "Failed to retrieve CpusSet for GPU=" << device_id;

        for (unsigned int i = 0; i < 8 * sizeof(cpu_mask); i++)
        {
            if (test_bit(cpu_mask, i))
            {
                cpus.insert(affinity::system::cpu_from_logical_id(i));
            }
        }

        DLOG(INFO) << "CPU Affinity for GPU " << device_id << ": " << cpus;
        return std::move(cpus);
    }

    std::size_t DeviceInfo::Alignment()
    {
        struct cudaDeviceProp properties;
        CHECK_EQ(CUDA_SUCCESS, cudaGetDeviceProperties(&properties, 0));
        return properties.textureAlignment;
    }

    double DeviceInfo::PowerUsage(int device_id)
    {
        unsigned int power;
        CHECK_EQ(nvmlDeviceGetPowerUsage(GetHandleById(device_id), &power), NVML_SUCCESS);
        return static_cast<double>(power) * 0.001;
    }

    double DeviceInfo::PowerLimit(int device_id)
    {
        unsigned int limit;
        CHECK_EQ(nvmlDeviceGetPowerManagementLimit(GetHandleById(device_id), &limit), NVML_SUCCESS);
        return static_cast<double>(limit) * 0.001;
    }

    std::string DeviceInfo::UUID(int device_id)
    {
        char buffer[256];
        CHECK_EQ(nvmlDeviceGetUUID(GetHandleById(device_id), buffer, 256), NVML_SUCCESS);
        return buffer;
    }

    int DeviceInfo::Count()
    {
        int device_count;
        CHECK_EQ(cudaGetDeviceCount(&device_count), CUDA_SUCCESS);
        return device_count;
    }

    bool DeviceInfo::IsValidDeviceID(int device_id)
    {
        return (device_id > 0) && (device_id < DeviceInfo::Count());
    }

    nvmlMemory_t DeviceInfo::MemoryInfo(int device_id)
    {
        nvmlMemory_t info;
        CHECK_EQ(nvmlDeviceGetMemoryInfo(GetHandleById(device_id), &info), NVML_SUCCESS);
        return info;
    }

    namespace cuda
    {
        namespace nvml
        {
            std::size_t device_count()
            {
                return DeviceInfo::Count();
            }

            nvmlMemory_t memory_info(int device_id)
            {
                return DeviceInfo::MemoryInfo(device_id);
            }
        } // namespace nvml
    }     // namespace cuda

} // namespace trtlab
