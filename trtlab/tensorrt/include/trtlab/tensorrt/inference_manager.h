#pragma once
#include <memory>

#include <trtlab/tensorrt/model.h>

namespace trtlab
{
    namespace TensorRT
    {
        class InferenceManager : public std::enable_shared_from_this<InferenceManager>
        {
            struct key {};

        public:
            static std::shared_ptr<InferenceManager> Create();

            InferenceManager(key);
            virtual ~InferenceManager();

            void RegisterModel(std::shared_ptr<Model>);
        };

    } // namespace TensorRT
} // namespace trtlab