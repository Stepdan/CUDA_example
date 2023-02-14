#include <tracv/utils/cuda/algorithms/gradient_utils.h>

#include <tracv/utils/cuda/device.hpp>

#include <cuda_runtime.h>
#include <npp.h>

#include "cudaDmy.cuh"
#include "utils.cuh"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#include <limits>

constexpr unsigned int kBlockSize = 16;

namespace {

__global__ void cuda_gradient_directions_and_magnitudes_calculation_kernel(
    float* dx_to_directions_data, float* dy_to_magnitudes_data, int width, int height,
    float* min_mag, float* max_mag)
{
    const int xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int yi = blockIdx.y * blockDim.y + threadIdx.y;
    if (!(xi < width && yi < height))
        return;

    const int index = width * yi + xi;
    const float dx = dx_to_directions_data[index];
    const float dy = dy_to_magnitudes_data[index];
    const float abs_dx = fabs(dx);
    const float abs_dy = fabs(dy);

    dx_to_directions_data[index] = atan2(dy, dx);
    dy_to_magnitudes_data[index] = abs_dx + abs_dy;
    tracv::utils::cuda_atomic_min(min_mag, dy_to_magnitudes_data[index]);
    tracv::utils::cuda_atomic_max(max_mag, dy_to_magnitudes_data[index]);
}

}  // namespace

namespace tracv { namespace utils {

bool cuda_gradient_directions_and_magnitudes_calculation(float* dx_to_directions_plane_data,
                                                         float* dy_to_magnitudes_plane_data,
                                                         int width, int height)
{
    cudaStream_t working_stream;
    CUDA_CHECK(cudaStreamCreate(&working_stream), false);

    auto div_up = [](size_t x, size_t y) -> unsigned int { return (x + y - 1) / y; };
    dim3 grid_extent{div_up(width, kBlockSize), div_up(height, kBlockSize), 1};
    dim3 block_extent{kBlockSize, kBlockSize, 1};

    CudaManagedValue<float> min_mag(std::numeric_limits<float>::max());
    CudaManagedValue<float> max_mag(-min_mag.value());

    cuda_gradient_directions_and_magnitudes_calculation_kernel<<<grid_extent, block_extent, 0,
                                                                 working_stream>>>(
        dx_to_directions_plane_data, dy_to_magnitudes_plane_data, width, height, min_mag.ptr(),
        max_mag.ptr());
    CUDA_CHECK(cudaPeekAtLastError(), false);

    CUDA_CHECK(cudaStreamSynchronize(working_stream), false);
    CUDA_CHECK(cudaStreamDestroy(working_stream), false);

    const auto status =
        nppsNormalize_32f(dy_to_magnitudes_plane_data, dy_to_magnitudes_plane_data, width * height,
                          min_mag.value(), max_mag.value() - min_mag.value());

    if (status != NPP_SUCCESS)
    {
        printf("npp Normalize has error: %d\n", status);
        return false;
    }

    return true;
}

}}  // namespace tracv::utils