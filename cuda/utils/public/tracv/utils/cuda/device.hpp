#pragma once

/* 
 * THIS HEADER MAY BE USED IN *.cu SOURCES
 * WHICH IS COMPILED BY nvcc NOT SUPPORTED C++17
 * SO DO NOT USE C++17 FEATURES IN THIS HEADER!
 */

#define CUDA_CHECK(X, ...)                                                                         \
    if (!tracv::utils::cuda_check(X, __FILE__, __LINE__))                                          \
        return __VA_ARGS__;

namespace tracv { namespace utils {
bool gpu_device_available();
bool cuda_check(int err, const char* file = "", int line = -1);
void cuda_reset_error();

void log_device_memory();
void log_device_memory_info();
}}  // namespace tracv::utils