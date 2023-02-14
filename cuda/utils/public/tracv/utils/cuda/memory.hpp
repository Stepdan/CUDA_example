#pragma once

/* 
 * THIS HEADER MAY BE USED IN *.cu SOURCES
 * WHICH IS COMPILED BY nvcc NOT SUPPORTED C++17
 * SO DO NOT USE C++17 FEATURES IN THIS HEADER!
 */

#include "device.hpp"

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace tracv { namespace utils {

class CudaHostMemory
{
public:
    CudaHostMemory();

    CudaHostMemory(const CudaHostMemory& other) = delete;
    void operator=(const CudaHostMemory& other) = delete;

    CudaHostMemory(CudaHostMemory&& other);
    void operator=(CudaHostMemory&& other);

    ~CudaHostMemory();

    bool reserve(size_t size);
    uint8_t* ptr();
    size_t size();

private:
    void* ptr_{};
    size_t size_{};
};

struct CudaMemory
{
private:
    struct ConstructorGuard
    {
    };

public:
    CudaMemory(const ConstructorGuard&, size_t bytes);
    CudaMemory(const ConstructorGuard&, void* src_data, size_t bytes);

    CudaMemory(const CudaMemory& other) = delete;
    void operator=(const CudaMemory& other) = delete;

    CudaMemory(CudaMemory&& other);
    CudaMemory& operator=(CudaMemory&& other);

    ~CudaMemory();

    void* ptr();
    size_t size();

    template <typename... Args>
    static auto create(Args&&... args)
    {
        return std::make_shared<CudaMemory>(ConstructorGuard{}, std::forward<Args>(args)...);
    }

    bool download(size_t size, CudaHostMemory* dst);

private:
    void* ptr_{};
    size_t size_{};
};

template <typename T>
struct CudaMemoryWrapper
{
    using DataType = T;
    explicit CudaMemoryWrapper(void* data, size_t count)
        : CudaMemoryWrapper(CudaMemory::create(data, count * sizeof(DataType)))
    {
    }
    explicit CudaMemoryWrapper(std::shared_ptr<CudaMemory> memory) : memory_(memory) {}
    DataType* ptr() { return static_cast<DataType*>(memory_->ptr()); }
    size_t size() { return memory_->size() / sizeof(DataType); }

private:
    std::shared_ptr<CudaMemory> memory_;
};

template <typename T>
struct CudaManagedValue
{
public:
    using DataType = T;
    static_assert(std::is_floating_point<DataType>::value || std::is_integral<DataType>::value);

public:
    explicit CudaManagedValue(DataType value)
    {
        if (!cuda_check(cudaMallocManaged(&data, sizeof(DataType))))
            throw std::bad_alloc();
        data[0] = value;
    }
    ~CudaManagedValue() { cudaFree(data); }

    DataType value() const { return data[0]; }
    DataType* ptr() { return data; }

    CudaManagedValue(const CudaManagedValue& other) = delete;
    void operator=(const CudaManagedValue& other) = delete;

private:
    DataType* data;
};

}}  // namespace tracv::utils
