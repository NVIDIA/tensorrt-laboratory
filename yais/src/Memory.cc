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
#include "YAIS/Memory.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace yais
{



// HostMemory

size_t HostMemory::DefaultAlignment()
{
    return 64;
}

void HostMemory::WriteZeros()
{
    std::memset(Data(), 0, Size());
}

// DeviceMemory

size_t DeviceMemory::DefaultAlignment()
{
    return 256;
}

void DeviceMemory::WriteZeros()
{
    CHECK_EQ(cudaMemset(Data(), 0, Size()), CUDA_SUCCESS) << "WriteZeros failed on Device Allocation";
}

// CudaManagedMemory

CudaManagedMemory::CudaManagedMemory(size_t size) : DeviceMemory(size)
{
    CHECK_EQ(cudaMallocManaged((void **)&m_BasePointer, size, cudaMemAttachGlobal), CUDA_SUCCESS) << "cudaMallocManaged " << size << " bytes failed";
    DLOG(INFO) << "Allocated Cuda Managed Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

CudaManagedMemory::~CudaManagedMemory()
{
    CHECK_EQ(cudaFree(m_BasePointer), CUDA_SUCCESS) << "cudaFree(" << m_BasePointer << ") failed";
    DLOG(INFO) << "Deleted Cuda Manged Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

// CudaDeviceMemory

CudaDeviceMemory::CudaDeviceMemory(size_t size) : DeviceMemory(size)
{
    CHECK_EQ(cudaMalloc((void **)&m_BasePointer, size), CUDA_SUCCESS) << "cudaMalloc " << size << " bytes failed";
    DLOG(INFO) << "Allocated Cuda Device Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

CudaDeviceMemory::~CudaDeviceMemory()
{
    CHECK_EQ(cudaFree(m_BasePointer), CUDA_SUCCESS) << "cudaFree(" << m_BasePointer << ") failed";
    DLOG(INFO) << "Deleted Cuda Device Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

// CudaHostMemory

CudaHostMemory::CudaHostMemory(size_t size) : HostMemory(size)
{
    CHECK_EQ(cudaMallocHost((void **)&m_BasePointer, size), CUDA_SUCCESS) << "cudaMalloc " << size << " bytes failed";
    DLOG(INFO) << "Allocated Cuda Host Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

CudaHostMemory::~CudaHostMemory()
{
    CHECK_EQ(cudaFreeHost(m_BasePointer), CUDA_SUCCESS) << "cudaFree(" << m_BasePointer << ") failed";
    DLOG(INFO) << "Deleted Cuda Host Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

// SystemMallocMemory

SystemMallocMemory::SystemMallocMemory(size_t size) : HostMemory(size)
{
    m_BasePointer = malloc(size);
    CHECK(m_BasePointer) << "malloc(" << size << ") failed";
    DLOG(INFO) << "malloc(" << m_BytesAllocated << ") returned " << m_BasePointer;
}

SystemMallocMemory::~SystemMallocMemory()
{
    free(m_BasePointer);
    DLOG(INFO) << "Deleted Malloc'ed Memory (" << m_BasePointer << ", " << m_BytesAllocated << ")";
}

/**
 * @brief Converts bytes into a more friend human readable format
 * 
 * @param bytes 
 * @return std::string 
 */
std::string BytesToString(size_t bytes)
{
    // Credits: https://stackoverflow.com/questions/3758606
    char buffer[50];
    int unit = 1024;
    const char prefixes[] = "KMGTPE";
    if (bytes < unit)
    {
        sprintf(buffer, "%ld B", bytes);
        return std::string(buffer);
    }
    int exp = (int) (log(bytes) / log(unit));
    sprintf(buffer, "%.1f %ciB", bytes / pow(unit, exp), prefixes[exp-1]);
    return std::string(buffer);
}

std::uint64_t StringToBytes(const std::string str)
{
    // https://regex101.com/r/UVm5wT/1
    std::smatch m;
    std::regex r("(\\d+[.\\d+]*)([KMGTkmgt]*)([i]*)[bB]");
    std::map<char, int> prefix = {
        {'k', 1}, {'m', 2}, {'g', 3}, {'t', 4},
        {'K', 1}, {'M', 2}, {'G', 3}, {'T', 4},
    };

    if (!std::regex_search(str, m, r))
	LOG(FATAL) << "Unable to convert \"" << str << "\" to bytes. "
                   << "Expected format: 10b, 1024B, 1KiB, 10MB, 2.4gb, etc.";

    std::uint64_t base = m[3] == "" ? 1000 : 1024;
    auto exponent = prefix[m[2].str()[0]];
    auto scalar = std::stod(m[1]);
    return (std::uint64_t)(scalar * pow(base, exponent));
}

} // namespace yais
