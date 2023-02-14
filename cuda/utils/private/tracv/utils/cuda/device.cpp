#include <tracv/utils/cuda/device.hpp>

#include <log/log.h>

#include <easy/profiler.h>
#include <easy/arbitrary_value.h>

#include <cuda_runtime.h>

namespace tracv::utils {

namespace {
constexpr float kMB = 1.f / (1024 * 1024);
}

bool gpu_device_available()
{
    static bool result = [] {
        int count{};
        cudaError_t error = cudaGetDeviceCount(&count);

        return error == cudaSuccess && count > 0;
    }();
    return result;
}

bool cuda_check(int code, const char* file, int line)
{
    cudaError_t err = static_cast<cudaError_t>(code);
    if (err == cudaSuccess)
        return true;

    LOG(L_CRITICAL, "CUDA error {}; msg {}; {}:{}", err, cudaGetErrorName(err), file, line);
    cuda_reset_error();

    return false;
}

void cuda_reset_error() { cudaGetLastError(); /* reset state */ }

void log_device_memory_impl(LOG_LEVEL level)
{
    if (tracv::log::Logger::Instance().GetLogLevel() > level && !profiler::isEnabled())
        return;

    if (!gpu_device_available())
        return;

    EASY_FUNCTION(profiler::colors::Green500);

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    size_t free;
    size_t total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));

    free *= kMB;
    total *= kMB;

    EASY_VALUE("free", free);
    EASY_VALUE("total", total);

    LOG(level, "GPU {} memory: free={}, total={}", id, free, total);
}

void log_device_memory() { log_device_memory_impl(L_TRACE); }

void log_device_memory_info() { log_device_memory_impl(L_INFO); }

}  // namespace tracv::utils