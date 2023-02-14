#include <tracv/utils/cuda/video.hpp>

#include <tracv/utils/cuda/device.hpp>
#include <tracv/utils/throw_with_trace.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <log/log.h>

namespace tracv::utils {

namespace {
video::Frame::PlaneId extract_plane_id(video::Frame::Format format)
{
    const int format_value = static_cast<int>(format);
    if (format_value < 0 || format_value >= video::Frame::kFormatEnd)
        throw_with_trace(std::runtime_error("format is invalid"));

    const video::Frame::PlaneId* planes = video::Frame::list_planes(format);
    if (planes[0] == video::Frame::kPlaneInvalid)
        throw_with_trace(std::runtime_error("format doesn't contain any planes"));
    if (planes[1] != video::Frame::kPlaneInvalid)
        throw_with_trace(std::runtime_error("format contains multiple planes"));

    return planes[0];
}

}  // namespace

SinglePlaneAdapter::SinglePlaneAdapter(std::shared_ptr<CudaMemory> memory,
                                       video::Frame::Format format, uint32_t width, uint32_t height,
                                       uint32_t bpp, int64_t timestamp)
    : memory_(memory)
    , format_(format)
    , plane_id_(extract_plane_id(format_))
    , width_(width)
    , height_(height)
    , bpp_(bpp)
{
    video::set_frame_meta_value<video::meta::Timestamp>(this, timestamp);
}

video::RefPtr<video::Frame> SinglePlaneAdapter::create(video::Frame* frame)
{
    const auto plane_id = extract_plane_id(frame->format());
    video::Frame::Plane plane{};
    if (!frame->map(plane_id, video::Frame::MapFlags::kMapRead, &plane))
        throw std::runtime_error("Can't map frame for SinglePlaneAdapter with PlaneId " +
                                 std::to_string(plane_id));

    if (plane.flags & video::Frame::MapFlags::kMapCuda)
        throw std::runtime_error("Can't create SinglePlaneAdapter with raw CUDA frame");

    try
    {
        return SinglePlaneAdapter::create(
            CudaMemory::create(plane.data, plane.width * plane.height * plane.bpp / CHAR_BIT),
            frame->format(), plane.width, plane.height, plane.bpp,
            video::get_frame_meta_value<video::meta::Timestamp>(frame));
    }
    catch (const std::exception& e)
    {
        LOG(L_WARN, "Exception caught; {}", e.what());
    }
    return video::RefPtr<video::Frame>(nullptr);
}

video::Frame::Format SinglePlaneAdapter::format() { return format_; }

uint32_t SinglePlaneAdapter::width() { return width_; }

uint32_t SinglePlaneAdapter::height() { return height_; }

bool SinglePlaneAdapter::map(PlaneId id, MapFlagsBitField flags, Plane* plane)
{
    if (id != plane_id_)
        return false;

    uint32_t stride = width_ * bpp_ / CHAR_BIT;
    size_t size = stride * height_;
    if (flags & kMapCuda)
    {
        plane->data = memory_->ptr();
    }
    else
    {
        plane->data = malloc(size);
        if (!plane->data)
        {
            LOG(L_ERROR, "{}: malloc failed allocating {}", __func__, size);
            return false;
        }
        cudaError_t err = cudaMemcpy(plane->data, memory_->ptr(), size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            LOG(L_ERROR, "{}: cudaMemcpy failed with error: {}", __func__, err);
            return false;
        }
    }
    plane->width = width_;
    plane->height = height_;
    plane->bpp = bpp_;
    plane->stride = stride;
    plane->id = id;
    plane->flags = flags;
    return true;
}

void SinglePlaneAdapter::unmap(Plane* plane)
{
    if (!(plane->flags & kMapCuda))
    {
        free(plane->data);
        plane->data = nullptr;
    }
}

}  // namespace tracv::utils