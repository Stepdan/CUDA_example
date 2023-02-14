#pragma once

#include <tracv/mesh/points.hpp>

#include <tracv/utils/cuda/memory.hpp>
#include <tracv/utils/cuda/convolution.hpp>

#include <tracv/utils/throw_with_trace.hpp>

#include <memory>
#include <type_traits>

namespace tracv { namespace utils {

enum class ConvType
{
    Undefined,
    ScharrHoriz,
    ScharrVert,
};

template <typename T>
class ConvolutionKernel
{
public:
    using DataType = T;
    static_assert(std::is_floating_point<DataType>::value || std::is_integral<DataType>::value);

public:
    explicit ConvolutionKernel(size_t kernel_size, std::vector<DataType>&& kernel_data)
        : size_(kernel_size)
    {
        if (kernel_size <= 0 || kernel_size % 2 == 0 ||
            kernel_data.size() != kernel_size * kernel_size)
        {
            throw_with_trace(std::runtime_error("Kernel size is not similar with data size"));
        }

        data_ = std::make_unique<tracv::utils::CudaMemoryWrapper<DataType>>(
            kernel_data.data(), kernel_size * kernel_size);

        if (!data_)
            throw_with_trace(std::runtime_error("Invalid kernel data"));
    }

    size_t kernel_size() const noexcept { return size_; }

    size_t size() { return data_->size(); }

    DataType* ptr() { return data_->ptr(); }

private:
    std::unique_ptr<tracv::utils::CudaMemoryWrapper<DataType>> data_;
    size_t size_;
};

template <typename DataType>
ConvolutionKernel<DataType> create_conv_kernel(ConvType type)
{
    switch (type)
    {
        /* clang-format off */
        case ConvType::ScharrHoriz:
            return ConvolutionKernel<DataType>(3, {
                3, 0, -3,
                10, 0, -10,
                3, 0, -3
            });
        case ConvType::ScharrVert:
            return ConvolutionKernel<DataType>(3, {
                3, 10, 3,
                0, 0, 0,
                -3, -10, -3
            });
            /* clang-format on */
        default:
            break;
    }

    throw_with_trace(std::runtime_error("Invalid kernel type"));
}

}}  // namespace tracv::utils