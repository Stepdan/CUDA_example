#include <tracv/utils/cuda/image_statistics.hpp>
#include <tracv/utils/cuda/memory.hpp>
#include <tracv/utils/cuda/device.hpp>
#include <tracv/utils/cuda/scharr.hpp>
#include <tracv/utils/cuda/algorithms/gradient_utils.h>
#include <tracv/utils/cuda/algorithms/interpolation.h>

#include <tracv/utils/video.hpp>
#include <tracv/utils/scope_guard.hpp>
#include <tracv/utils/cuda/video.hpp>

#include <log/log.h>

#include <opencv2/opencv.hpp>

#include <easy/profiler.h>

namespace tracv::utils {

std::shared_ptr<ImageStatistics> ImageStatistics::create(const cv::Mat& image)
{
    try
    {
        return std::shared_ptr<ImageStatistics>(new ImageStatistics(image));
    }
    catch (const std::exception& e)
    {
        LOG(L_ERROR, "ImageStatistics::create error: {}", e.what());
        return nullptr;
    }
    catch (...)
    {
        LOG(L_ERROR, "ImageStatistics::create: unknown error");
        return nullptr;
    }
}

ImageStatistics::ImageStatistics(const cv::Mat& image) { calculate_data(image); }

bool ImageStatistics::out_of_image(float val) const noexcept
{
    return val == std::numeric_limits<float>::max();
}

std::vector<float> ImageStatistics::get_gradient_directions(const tracv::scene::Points2f& coords)
{
    video::Frame::Plane plane{};
    video::Frame::MapFlagsBitField flags{video::Frame::MapFlags::kMapRead |
                                         video::Frame::MapFlags::kMapCuda};
    if (!gradient_directions_img_ ||
        !gradient_directions_img_->map(video::Frame::PlaneId::kPlaneY, flags, &plane))
    {
        LOG_ERROR("getGradientDirections: Can't map needed plane");
        return {};
    };

    return cuda_bilinear_interpolation_kernel(plane.data, plane.width, plane.height, coords);
}

std::vector<float> ImageStatistics::get_gradient_magnitudes(const tracv::scene::Points2f& coords)
{
    video::Frame::Plane plane{};
    video::Frame::MapFlagsBitField flags{video::Frame::MapFlags::kMapRead |
                                         video::Frame::MapFlags::kMapCuda};
    if (!gradient_magnitudes_img_ ||
        !gradient_magnitudes_img_->map(video::Frame::PlaneId::kPlaneY, flags, &plane))
    {
        LOG_ERROR("getGradientMagnitudes: Can't map needed plane");
        return {};
    };

    return cuda_bilinear_interpolation_kernel(plane.data, plane.width, plane.height, coords);
}

void ImageStatistics::calculate_data(const cv::Mat& image)
{
    EASY_FUNCTION(profiler::colors::Red500);

    EASY_BLOCK("convertTo", profiler::colors::Cyan500);
    cv::Mat float_image;
    image.convertTo(float_image, cv::DataType<float>::type);
    EASY_END_BLOCK;

    EASY_BLOCK("create adapter", profiler::colors::Cyan500);
    auto frame = utils::F32MatFrameAdapter::create(float_image);
    if (!utils::gpu_device_available())
    {
        // todo - replace CPU imageStatistics initialization here after refactoring
        LOG_ERROR("GPU device unavailable");
        return;
    }
    EASY_END_BLOCK;

    EASY_BLOCK("create plane adapter", profiler::colors::Indigo500);
    frame = utils::SinglePlaneAdapter::create(frame.get());
    video::Frame::Plane orig_plane{};
    video::Frame::MapFlagsBitField flags{video::Frame::MapFlags::kMapRead |
                                         video::Frame::MapFlags::kMapCuda};
    if (!frame || !frame->map(video::Frame::PlaneId::kPlaneY, flags, &orig_plane))
    {
        LOG_ERROR("Can't map needed plane");
    };
    EASY_END_BLOCK;

    const auto size = orig_plane.width * orig_plane.height * sizeof(float);

    auto dx_to_directions_memory = utils::CudaMemory::create(size);
    auto dy_to_magnitudes_memory = utils::CudaMemory::create(size);
    tracv::utils::CudaMemoryWrapper<float> dx_to_directions_wrapper(dx_to_directions_memory);
    tracv::utils::CudaMemoryWrapper<float> dy_to_magnitudes_wrapper(dy_to_magnitudes_memory);

    EASY_BLOCK("scharr", profiler::colors::Orange500);
    const auto derivative_result =
        sharr(orig_plane.data, orig_plane.width, orig_plane.height, dx_to_directions_wrapper.ptr(),
              dy_to_magnitudes_wrapper.ptr());
    EASY_END_BLOCK;
    if (!derivative_result)
    {
        LOG_ERROR("Scharr error");
        return;
    }

    EASY_BLOCK("grad_utils", profiler::colors::Blue500);
    const auto grad_utils_result = cuda_gradient_directions_and_magnitudes_calculation(
        dx_to_directions_wrapper.ptr(), dy_to_magnitudes_wrapper.ptr(), orig_plane.width,
        orig_plane.height);
    EASY_END_BLOCK;
    if (!grad_utils_result)
    {
        LOG_ERROR("grad_utils error");
        return;
    }

    EASY_BLOCK("create frames", profiler::colors::Green500);
    gradient_directions_img_ = utils::SinglePlaneAdapter::create(
        std::move(dx_to_directions_memory), video::Frame::Format::kFormatGray, orig_plane.width,
        orig_plane.height, sizeof(float) * CHAR_BIT, 0);

    gradient_magnitudes_img_ = utils::SinglePlaneAdapter::create(
        std::move(dy_to_magnitudes_memory), video::Frame::Format::kFormatGray, orig_plane.width,
        orig_plane.height, sizeof(float) * CHAR_BIT, 0);
    EASY_END_BLOCK;

    // Example of image saving.

    // Example of image saving.
    //FrameF32MatAccessor dx_mat(frame.get());
    //cv::imwrite("/root/cv_platform/source/build/bin/temp/my_dx.png", dx_mat.mat());
}

}  // namespace tracv::utils