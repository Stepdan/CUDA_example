#include <tracv/utils/cuda/scharr.hpp>

#include <tracv/utils/cuda/convolution.hpp>
#include <tracv/utils/cuda/algorithms/convolution.h>

#include <log/log.h>

namespace tracv { namespace utils {

bool sharr(const void* plane_data, int width, int height, float* output_dx, float* output_dy)
{
    auto kernel_scharr_horiz = create_conv_kernel<float>(ConvType::ScharrHoriz);
    const auto status_dx =
        cuda_convolution(static_cast<int>(kernel_scharr_horiz.kernel_size()),
                         kernel_scharr_horiz.ptr(), plane_data, width, height, output_dx);

    if (!status_dx)
    {
        LOG(L_ERROR, "CUDA Scharr convolution dx error!");
        return false;
    }

    auto kernel_scharr_vert = create_conv_kernel<float>(ConvType::ScharrVert);
    const auto status_dy =
        cuda_convolution(static_cast<int>(kernel_scharr_vert.kernel_size()),
                         kernel_scharr_vert.ptr(), plane_data, width, height, output_dy);

    if (!status_dy)
    {
        LOG(L_ERROR, "CUDA Scharr convolution dy error!");
        return false;
    }

    return true;
}

}}  // namespace tracv::utils