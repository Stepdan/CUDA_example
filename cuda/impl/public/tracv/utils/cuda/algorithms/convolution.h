#include <tracv/utils/cuda/memory.hpp>

#include <tracv/mesh/points.hpp>

namespace tracv { namespace utils {

bool cuda_convolution(int kernel_size, float* kernel_data, const void* plane_data, int width,
                      int height, float* output);

}}  // namespace tracv::utils
