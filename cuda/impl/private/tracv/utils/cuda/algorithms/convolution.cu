#include <tracv/utils/cuda/algorithms/convolution.h>

#include <tracv/utils/cuda/device.hpp>

#include <cuda_runtime.h>

#include "cudaDmy.cuh"

#include <cstdio>
#include <memory>
#include <vector>

constexpr unsigned int kBlockSize = 16;

namespace {

texture<float, cudaTextureType2D, cudaReadModeElementType> frame_texture;

__global__ void cuda_convolution_kernel(int width, int height, int kernel_size,
                                        int half_kernel_size, float* kernel_data, float* output)
{
    const int xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int yi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= width || yi >= height)
        return;

    float result = 0.0f;
    const int x0_orig = xi - half_kernel_size;
    const int y0_orig = yi - half_kernel_size;
    const int x0 = (x0_orig >= 0) * x0_orig;
    const int y0 = (y0_orig >= 0) * y0_orig;

    const int xi_with_half = xi + half_kernel_size;
    const int yi_with_half = yi + half_kernel_size;

    const bool is_x_lesser_than_max = xi_with_half < width;
    const bool is_y_lesser_than_max = yi_with_half < height;

    const int x1 = !is_x_lesser_than_max * width + is_x_lesser_than_max * (xi_with_half + 1);
    const int y1 = !is_y_lesser_than_max * height + is_y_lesser_than_max * (yi_with_half + 1);

    for (int x = x0, i = x - x0_orig; x < x1; ++x, ++i)
        for (int y = y0, j = y - y0_orig; y < y1; ++y, ++j)
        {
            result += kernel_data[j * kernel_size + i] * tex2D(frame_texture, x + 0.5f, y + 0.5f);
        }

    output[width * yi + xi] = result;
}

}  // namespace

namespace tracv { namespace utils {

bool cuda_convolution(int kernel_size, float* kernel_data, const void* plane_data, int width,
                      int height, float* output)
{
    cudaStream_t working_stream;
    CUDA_CHECK(cudaStreamCreate(&working_stream), {});

    frame_texture.addressMode[0] = cudaAddressModeClamp;
    frame_texture.addressMode[1] = cudaAddressModeClamp;
    frame_texture.filterMode = cudaFilterModePoint;
    frame_texture.normalized = false;

    size_t bytes_width = width * sizeof(float);

    static const auto deleter = +[](float* pitched) { CUDA_CHECK(cudaFree(pitched)); };

    size_t pitch = 0;
    std::unique_ptr<float, decltype(deleter)> pitched{
        [&]() -> float* {
            float* pitched = nullptr;
            CUDA_CHECK(cudaMallocPitch(&pitched, &pitch, bytes_width, height), nullptr);
            return pitched;
        }(),
        deleter};

    if (!pitched)
    {
        printf("!pitched\n");
        return false;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(pitched.get(), pitch, plane_data, bytes_width, bytes_width, height,
                                 cudaMemcpyDeviceToDevice, working_stream),
               false);

    size_t offset = 0;
    CUDA_CHECK(cudaBindTexture2D(&offset, &frame_texture, pitched.get(), &frame_texture.channelDesc,
                                 width, height, pitch),
               false);

    if (offset != 0)
    {
        printf("Texture offset is %zu but 0 expected\n", offset);
        return false;
    }

    auto div_up = [](size_t x, size_t y) -> unsigned int { return (x + y - 1) / y; };
    dim3 grid_extent{div_up(width, kBlockSize), div_up(height, kBlockSize), 1};
    dim3 block_extent{kBlockSize, kBlockSize, 1};

    cuda_convolution_kernel<<<grid_extent, block_extent, 0, working_stream>>>(
        width, height, kernel_size, kernel_size / 2, kernel_data, output);

    CUDA_CHECK(cudaStreamSynchronize(working_stream), false);
    CUDA_CHECK(cudaStreamDestroy(working_stream), false);

    return true;
}

}}  // namespace tracv::utils