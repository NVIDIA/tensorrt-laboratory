#pragma once

#include <trtlab/memory/descriptor.h>
#include <trtlab/core/utils.h>

#include "NvInfer.h"
#include "trtlab/tensorrt/model.h"
#include "trtlab/tensorrt/common.h"

namespace trtlab
{
    namespace TensorRT
    {
        class ExecutionContext
        {
        public:
            using model_t   = std::shared_ptr<Model>;
            using context_t = nvinfer1::IExecutionContext;

            ExecutionContext(model_t);
            virtual ~ExecutionContext();

            ExecutionContext(ExecutionContext&&) noexcept = default;
            ExecutionContext& operator=(ExecutionContext&&) noexcept = default;

            DELETE_COPYABILITY(ExecutionContext);

            context_t& context()
            {
                return *m_Context;
            }

            const Model& model() const
            {
                return *m_Model;
            }

            const nvinfer1::ICudaEngine& engine() const
            {
                return m_Context->getEngine();
            }

            std::string binding_info(std::uint32_t binding_id);
            std::string profile_info(std::uint32_t profile_id);

            std::size_t binding_size_in_bytes(std::uint32_t binding_id);

        private:
            unique_t<context_t> m_Context;
            model_t             m_Model;
        };

    } // namespace TensorRT
} // namespace trtlab