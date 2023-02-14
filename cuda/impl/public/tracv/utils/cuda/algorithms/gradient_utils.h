#include <tracv/utils/cuda/memory.hpp>

namespace tracv { namespace utils {

/*
    Computes gradient's directions and magnitudes images based on DX and DY images.
    Rewrites DX/DY memory (for speed and memory-saving purposes):
    DX -> directions
    DY -> magnitudes
*/
bool cuda_gradient_directions_and_magnitudes_calculation(float* dx_to_directions_data,
                                                         float* dy_to_magnitudes_data, int width,
                                                         int height);

}}  // namespace tracv::utils