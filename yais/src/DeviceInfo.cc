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
#include "YAIS/DeviceInfo.h"

#include <algorithm>
#include <glog/logging.h>
#include "nvml.h"

#define   test_bit(_n,_p)     ( _n & ( 1UL << _p))

namespace yais
{

struct nvmlState
{
    nvmlState() { CHECK_EQ(nvmlInit(), NVML_SUCCESS) << "Failed to initialize NVML"; }
    ~nvmlState() { CHECK_EQ(nvmlShutdown(), NVML_SUCCESS) << "Failed to Shutdown NVML"; }
};

static auto nvmlInstatnce = std::make_unique<nvmlState>();

std::string GetDeviceUUID(int device_id)
{
    char buffer[256];
    nvmlDevice_t gpu;

    CHECK_EQ(nvmlDeviceGetHandleByIndex(device_id, &gpu), NVML_SUCCESS);
    CHECK_EQ(nvmlDeviceGetUUID(gpu, buffer, 256), NVML_SUCCESS);
    return buffer;
}

double GetDevicePowerUsage(int device_id)
{
    nvmlDevice_t gpu;
    unsigned int power;
    CHECK_EQ(nvmlDeviceGetHandleByIndex(0, &gpu), NVML_SUCCESS);
    CHECK_EQ(nvmlDeviceGetPowerUsage(gpu, &power), NVML_SUCCESS);
    return static_cast<double>(power) * 0.001;
}

double GetDevicePowerLimit(int device_id)
{
    nvmlDevice_t gpu;
    unsigned int limit;
    CHECK_EQ(nvmlDeviceGetHandleByIndex(device_id, &gpu), NVML_SUCCESS);
    CHECK_EQ(nvmlDeviceGetPowerManagementLimit(gpu, &limit), NVML_SUCCESS);
    return static_cast<double>(limit) * 0.001;
}

CpuSet GetDeviceAffinity(int device_id)
{
    CpuSet cpus;
    nvmlDevice_t gpu;
    unsigned long cpu_mask = 0;

    CHECK_EQ(nvmlDeviceGetHandleByIndex(device_id, &gpu), NVML_SUCCESS)
        << "Failed to get Device for index=" << device_id;

    CHECK_EQ(nvmlDeviceGetCpuAffinity(gpu, sizeof(cpu_mask), &cpu_mask), NVML_SUCCESS)
        << "Failed to retrieve CpusSet for GPU=" << device_id;

    for (int i=0; i<8*sizeof(cpu_mask); i++)
    {
        if (test_bit(cpu_mask, i)) {
            cpus.insert(Affinity::GetCpuFromId(i));
            //cpus = cpus.Union(Affinity::GetCpuFromId(i));
        }
    }

    DLOG(INFO) << "CPU Affinity for GPU " << device_id << ": " << cpus.GetCpuString();
    return cpus;
}

} // namespace yais
