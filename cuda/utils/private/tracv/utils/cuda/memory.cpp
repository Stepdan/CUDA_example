#include <tracv/utils/cuda/memory.hpp>

#include <tracv/utils/cuda/device.hpp>

#include <tracv/utils/scope_guard.hpp>

#include <algorithm>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace tracv::utils {

CudaHostMemory::CudaHostMemory() = default;

CudaHostMemory::CudaHostMemory(CudaHostMemory&& other)
{
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
}
void CudaHostMemory::operator=(CudaHostMemory&& other)
{
    CudaHostMemory tmp(std::move(other));
    std::swap(ptr_, tmp.ptr_);
    std::swap(size_, tmp.size_);
}

CudaHostMemory::~CudaHostMemory()
{
    if (ptr_)
    {
        cudaFreeHost(ptr_);
    }
}

bool CudaHostMemory::reserve(size_t size)
{
    bool res = true;
    if (size_ < size)
    {
        if (ptr_)
        {
            cudaFreeHost(ptr_);
        }
        cudaError_t error = cudaMallocHost(&ptr_, size);
        if (error != cudaSuccess)
        {
            ptr_ = nullptr;
            size_ = 0;
            res = false;
        }
        else
        {
            size_ = size;
        }
    }
    return res;
}

uint8_t* CudaHostMemory::ptr() { return static_cast<uint8_t*>(ptr_); }
size_t CudaHostMemory::size() { return size_; }

CudaMemory::CudaMemory(const ConstructorGuard&, size_t bytes)
{
    utils::ScopeGuard sg(tracv::utils::log_device_memory, tracv::utils::log_device_memory);
    size_ = bytes;
    if (!cuda_check(cudaMalloc(&ptr_, size_)))
    {
        throw std::runtime_error("CudaMemory::CudaMemory(): cudaMalloc failed");
    }
}

CudaMemory::CudaMemory(const ConstructorGuard& cg, void* src_data, size_t bytes)
    : CudaMemory(cg, bytes)
{
    if (!cuda_check(cudaMemcpy(ptr_, src_data, size_, cudaMemcpyHostToDevice)))
    {
        throw std::runtime_error("CudaMemory::CudaMemory(): cudaMemcpy failed");
    }
}

CudaMemory::CudaMemory(CudaMemory&& other)
{
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
}
CudaMemory& CudaMemory::operator=(CudaMemory&& other)
{
    if (this == &other)
    {
        return *this;
    }
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
    return *this;
}

CudaMemory::~CudaMemory() { cudaFree(ptr_); }

void* CudaMemory::ptr() { return ptr_; }
size_t CudaMemory::size() { return size_; }

bool CudaMemory::download(size_t size, CudaHostMemory* dst)
{
    if (size > size_)
        throw std::range_error("too many bytes acquired");

    if (!dst->reserve(size))
        return false;

    cudaMemcpy(dst->ptr(), ptr_, size, cudaMemcpyDeviceToHost);
    return true;
}

}  // namespace tracv::utils
