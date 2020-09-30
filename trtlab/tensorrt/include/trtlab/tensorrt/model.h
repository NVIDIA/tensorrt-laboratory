#pragma once
#include <memory>
#include <vector>

#include "NvInfer.h"

#include "trtlab/tensorrt/allocator.h"

namespace trtlab
{
    namespace TensorRT
    {
        struct BaseModel
        {
        };

        class Model : public BaseModel
        {
        public:
            using engine_t  = std::shared_ptr<nvinfer1::ICudaEngine>;
            using weights_t = std::vector<typename NvAllocator::Pointer>;

            Model(engine_t, const weights_t&);
            virtual ~Model();

            nvinfer1::ICudaEngine& engine()
            {
                return *m_Engine;
            }

            std::string profiles_info() const;
            std::string profile_info(std::uint32_t profile_id, nvinfer1::OptProfileSelector) const;

            std::string bindings_info() const;
            std::string binding_info(std::uint32_t) const;
            static std::string dims_info(const nvinfer1::Dims&);

            std::size_t binding_element_count(std::uint32_t binding_id) const;
            std::size_t binding_size_in_bytes(std::uint32_t binding_id) const;

        protected:


        private:
            engine_t  m_Engine;
            weights_t m_Weights;
        };

    } // namespace TensorRT
} // namespace trtlab
