#include <tracv/utils/cuda/memory.hpp>

namespace tracv { namespace utils {

bool sharr(const void* plane_data, int width, int height, float* output_dx, float* output_dy);

}}  // namespace tracv::utils