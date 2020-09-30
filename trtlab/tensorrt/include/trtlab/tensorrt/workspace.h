#pragma once

#include <vector>

#include <cuda_runtime.h>

#include <trtlab/core/utils.h>

#include <trtlab/memory/descriptor.h>

#include <trtlab/tensorrt/model.h>
#include <trtlab/tensorrt/execution_context.h>

namespace trtlab
{
    namespace TensorRT
    {
        class WorkspaceBase
        {
            WorkspaceBase();
            virtual ~WorkspaceBase();

            cudaStream_t stream() { return m_Stream; }

        private:
            cudaStream_t m_Stream;
        };

        class StaticSingleModelGraphWorkspace
        {
        public:
            using descriptor_t = trtlab::memory::descriptor;

            StaticSingleModelGraphWorkspace(std::shared_ptr<Model>);
            virtual ~StaticSingleModelGraphWorkspace();

            DELETE_COPYABILITY(StaticSingleModelGraphWorkspace);
            DELETE_MOVEABILITY(StaticSingleModelGraphWorkspace);

            void enqueue();

            descriptor_t& binding(std::uint32_t binding_id);

            cudaStream_t stream()
            {
                return m_Stream;
            }

            std::size_t batch_size();

            std::string name() const
            {
                return m_Name;
            }

        protected:
            ExecutionContext& exec_ctx() { return m_Context; }

        private:
            ExecutionContext          m_Context;
            std::vector<descriptor_t> m_Bindings;
            std::vector<void*>        m_BindingPointers;
            descriptor_t              m_DeviceMemory;
            cudaStream_t              m_Stream;
            cudaGraph_t               m_Graph;
            cudaGraphExec_t           m_GraphExecutor;
            std::string               m_Name;
        };

        class BenchmarkWorkspace : public StaticSingleModelGraphWorkspace
        {
        public:
            BenchmarkWorkspace(std::shared_ptr<Model>);
            ~BenchmarkWorkspace() override = default;

            descriptor_t& host_binding(std::uint32_t binding_id);

            void async_h2d();
            void async_d2h();

        private:
            std::vector<descriptor_t> m_HostBindings;

        };

        class TimedBenchmarkWorkspace : private BenchmarkWorkspace
        {
        public:
            TimedBenchmarkWorkspace(std::shared_ptr<Model>);
            ~TimedBenchmarkWorkspace() override = default;

            void enqueue_pipeline();

            float get_compute_time_ms();
            float get_h2d_time_ms();
            float get_d2h_time_ms();

            using BenchmarkWorkspace::binding;
            using BenchmarkWorkspace::stream;

        private:
            cudaEvent_t m_Start;
            cudaEvent_t m_CompleteAsyncH2D;
            cudaEvent_t m_CompleteCompute;
            cudaEvent_t m_CompleteAsyncD2H;
        };

    } // namespace TensorRT
} // namespace trtlab
