#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <trtlab/core/standard_threads.h>
#include <trtlab/core/userspace_threads.h>

#include <trtlab/cuda/common.h>

namespace trtlab
{
    template <typename ThreadType>
    struct cuda_sync;

    template <>
    struct cuda_sync<userspace_threads>
    {
        static void event_sync(cudaEvent_t event)
        {
            cudaError_t rc = cudaEventQuery(event);
            while (rc != cudaSuccess)
            {
                if (rc != cudaErrorNotReady)
                {
                    LOG(ERROR) << cudaGetErrorName(rc);
                    throw std::runtime_error("cuda event query failed");
                }
                boost::this_fiber::yield();
                rc = cudaEventQuery(event);
            }
        }

        static void stream_sync(cudaStream_t stream)
        {
            cudaError_t rc = cudaStreamQuery(stream);
            while (rc != cudaSuccess)
            {
                if (rc != cudaErrorNotReady)
                {
                    LOG(ERROR) << cudaGetErrorName(rc);
                    throw std::runtime_error("cuda stream query failed");
                }
                boost::this_fiber::yield();
                rc = cudaStreamQuery(stream);
            }
        }
    };

    template<>
    struct cuda_sync<standard_threads>
    {
        static void event_sync(cudaEvent_t event)
        {
            CHECK_CUDA(cudaEventSynchronize(event));
        }

        static void stream_sync(cudaStream_t stream)
        {
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
    };

} // namespace trtlab