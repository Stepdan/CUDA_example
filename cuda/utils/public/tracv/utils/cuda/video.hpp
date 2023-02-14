#pragma once

#include <tracv/video/frame.hpp>
#include <tracv/video/utils.hpp>
#include <tracv/utils/cuda/memory.hpp>

#include <cassert>
#include <memory>

namespace tracv::utils {

class SinglePlaneAdapter : public video::FrameMetaStorage<video::RefCountedObject<video::Frame>>
{
private:
    SinglePlaneAdapter(std::shared_ptr<CudaMemory> memory, video::Frame::Format format,
                       uint32_t width, uint32_t height, uint32_t bpp, int64_t timestamp);

public:
    Format format() override;

    uint32_t width() override;
    uint32_t height() override;

    bool map(PlaneId id, MapFlagsBitField flags, Plane* plane) override;

    void unmap(Plane* plane) override;

    template <typename... Args>
    static auto create(Args&&... args)
    {
        return video::RefPtr<video::Frame>(new SinglePlaneAdapter(std::forward<Args>(args)...));
    }

    static video::RefPtr<video::Frame> create(video::Frame* frame);

private:
    std::shared_ptr<CudaMemory> memory_;
    Format format_;
    PlaneId plane_id_;
    uint32_t width_;
    uint32_t height_;
    uint32_t bpp_;
};

}  // namespace tracv::utils
