#pragma once

#include <cuda_runtime.h>

namespace tracv { namespace utils {

__forceinline__ __device__ float cuda_atomic_min(float* addr, float val)
{
    static_assert(sizeof(int) == sizeof(float));

    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__forceinline__ __device__ float cuda_atomic_max(float* addr, float val)
{
    static_assert(sizeof(int) == sizeof(float));

    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

}}  // namespace tracv::utils