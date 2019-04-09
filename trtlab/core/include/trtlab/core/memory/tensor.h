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
#include <type_traits>
#include <utility>

#include "trtlab/core/memory/bytes.h"
#include "trtlab/core/memory/tensor_shape.h"
#include "trtlab/core/types.h"
#include "trtlab/core/utils.h"

namespace trtlab {

struct ITensor : public ITensorShape
{
    virtual void* Data() = 0;
    virtual const void* Data() const = 0;
    virtual uint64_t NBytes() const = 0;
    virtual uint64_t ItemSize() const = 0;
    virtual const types::dtype& DataType() const = 0;
};

class ITensorProvider : public ITensor
{
    virtual void SetShape(std::unique_ptr<ITensorShape>, const types::dtype&) = 0;

    template<typename MemoryType>
    friend class Tensor;
};

template<typename MemoryType>
class Tensor;

template<typename MemoryType>
class TensorProvider : public IBytesProvider,
                       public ITensorProvider,
                       public std::enable_shared_from_this<TensorProvider<MemoryType>>
{
  protected:
    Tensor<MemoryType> TensorFromThis();
};

template<typename MemoryType>
class Tensor final : public ITensor
{
  public:
    Tensor(Tensor&& other) noexcept : m_Provider(std::exchange(other.m_Provider, nullptr)) {}
    Tensor(const Tensor& other) : m_Provider(other.m_Provider) {}

    Tensor& operator=(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;

    virtual ~Tensor() {}

    void* Data() final override { return TensorInfo().Data(); }
    const void* Data() const final override { return TensorInfo().Data(); }
    uint32_t NDims() const final override { return TensorInfo().NDims(); }
    const dims_t* Shape() const final override { return TensorInfo().Shape(); }
    const dims_t* Strides() const final override { return TensorInfo().Strides(); }
    uint64_t Size() const final override { return TensorInfo().Size(); }
    uint64_t NBytes() const final override { return TensorInfo().NBytes(); }
    uint64_t ItemSize() const final override { return TensorInfo().ItemSize(); }
    const types::dtype& DataType() const final override { return TensorInfo().DataType(); }

    bool IsShared() const
    {
        TensorInfo();
        return (bool)(m_Provider.use_count() > 1);
    }

    // Modify the TensorState
    void ReshapeView(const std::vector<int64_t>&, const types::dtype&);
    // void ReshapeView(const Shape&, const Strides&, const dtype&);

  protected:
    Tensor(std::shared_ptr<ITensorProvider> provider) : m_Provider(provider) {}
    ITensor& TensorInfo() const; // throws exception if m_Provider is nullptr
    /*
        template<typename T>
        class Accessor
        {
          public:
            using container_type = Accessor<T>;
            using shape_type = std::vector<int64_t>;

            T* begin()

            T* Data() final override { return TensorInfo().Data(); }
            const T* Data() const final override { return TensorInfo().Data(); }
            uint32_t NDims() const final override { return TensorInfo().NDims(); }
            const shape_type& Shape() const final override { return TensorInfo().Shape(); }
            const shape_type& Strides() const final override { return TensorInfo().Strides(); }
            uint64_t Size() const final override { return TensorInfo().Size(); }
            uint64_t NBytes() const final override { return TensorInfo().NBytes(); }
            uint64_t ItemSize() const final override { return TensorInfo().ItemSize(); }
            const types::dtype& DataType() const final override { return TensorInfo().DataType(); }
        };
    */
  private:
    std::shared_ptr<ITensorProvider> m_Provider;

    friend class TensorProvider<MemoryType>;
};

template<typename MemoryType>
ITensor& Tensor<MemoryType>::TensorInfo() const
{
    if(m_Provider)
    {
        return *m_Provider;
    }
    throw std::runtime_error("No internal TensorState");
}

template<typename MemoryType>
void Tensor<MemoryType>::ReshapeView(const std::vector<int64_t>& s, const types::dtype& dt)
{
    if(IsShared())
    {
        throw std::runtime_error("Cannot reshape shared tensor; ownership must be unique");
    }
    auto shape = std::make_unique<TensorShapeGeneric>(s);
    m_Provider->SetShape(std::move(shape), dt);
}

class TensorAllocator
{
    template<typename BaseType>
    class TensorState;

  public:
    // using Shape = std::vector<dims_t>;
    // using Strides = std::vector<dims_t>;

    // static Tensor<MemoryType> Allocate(const Shape& shape, const dtype& dtype);
    // static Tensor<MemoryType> Allocate(const Shape& shape, const Strides& strides,
    //                                    const dtype& dtype);

    // static Tensor<MemoryType> Convert(Bytes<MemoryType>&&, const dtype& dtype);
    // static Tensor<MemoryType> Convert(Bytes<MemoryType>&&, const Shape& shape, const dtype&
    // dtype)); static Tensor<MemoryType> Convert(Bytes<MemoryType>&&, const Shape& shape, const
    // Strides& strides, const dtype& dtype));

    template<typename MemoryType>
    static Tensor<typename MemoryType::BaseType> FromBytes(Bytes<MemoryType>&& bytes)
    {
        using TS = TensorState<typename MemoryType::BaseType>;
        auto state = std::make_shared<TS>(std::move(bytes), typename TS::ctor_guard());
        return state->GetTensor();
    }

  private:
    template<typename BaseType>
    class TensorState final : public TensorProvider<BaseType>
    {
        struct ctor_guard
        {
        };

      public:
        using dims_t = typename ITensor::dims_t;

        TensorState(BytesBaseType<BaseType>&& bytes, ctor_guard)
            : m_Bytes(std::move(bytes)), m_Shape(&m_Bytes), m_dtype(types::bytes)
        {
        }

        Tensor<BaseType> GetTensor() { return this->TensorFromThis(); }

        // ITensorShape Interface
        uint32_t NDims() const final override { return m_Shape->NDims(); }
        const dims_t* Shape() const final override { return m_Shape->Shape(); }
        const dims_t* Strides() const final override { return m_Shape->Strides(); }
        uint64_t Size() const final override { return m_Shape->Size(); }

        // ITensor Interface
        void* Data() final override { return m_Bytes.Data(); }
        const void* Data() const final override { return m_Bytes.Data(); }
        uint64_t NBytes() const final override { return m_Shape->Size() * m_dtype.itemsize(); }
        uint64_t ItemSize() const final override { return m_dtype.itemsize(); }
        const types::dtype& DataType() const final override { return m_dtype; }

        // IBytesProvider Interface
        const void* BytesProviderData() const final override { return m_Bytes.Data(); }
        mem_size_t BytesProviderSize() const final override { return m_Bytes.Size(); }
        const DLContext& BytesProviderDeviceInfo() const final override
        {
            return m_Bytes.DeviceInfo();
        }

      protected:
        void SetShape(std::unique_ptr<ITensorShape> shape, const types::dtype& dt) final override
        {
            auto required = shape->Size() * dt.itemsize();
            if(required > m_Bytes.Size())
            {
                throw std::runtime_error(
                    "Requested shape/dtype requires " + BytesToString(required) +
                    " which is greater than the capactiy of " + BytesToString(m_Bytes.Size()));
            }
            m_ShapeHolder = std::move(shape);
            m_Shape = m_ShapeHolder.get();
            m_dtype = dt;
        }

      private:
        BytesBaseType<BaseType> m_Bytes;
        std::unique_ptr<ITensorShape> m_ShapeHolder;
        ITensorShape* m_Shape;
        types::dtype m_dtype;

        friend class TensorProvider<BaseType>;
        friend class TensorAllocator;
    };
};

template<typename T>
Tensor<T> TensorProvider<T>::TensorFromThis()
{
    return Tensor<T>(std::enable_shared_from_this<TensorProvider<T>>::shared_from_this());
}

/*
template <typename MemoryType, typename T>
class TensorAdaptor;

template <typename MemoryType, typename T>
struct xcontainer_inner_types<TensorAdaptor<T>>
{
    using container_type = typename MemoryType::vector_type;
}
*/

} // namespace trtlab