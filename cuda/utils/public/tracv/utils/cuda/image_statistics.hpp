#pragma once

#include <tracv/video/frame.hpp>
#include <tracv/video/utils.hpp>

#include <tracv/mesh/points.hpp>

#include <opencv2/opencv.hpp>

#include <memory>
#include <vector>

namespace tracv::utils {

class ImageStatistics
{
public:
    static std::shared_ptr<ImageStatistics> create(const cv::Mat& image);

    std::vector<float> get_gradient_directions(const tracv::scene::Points2f& coords);
    std::vector<float> get_gradient_magnitudes(const tracv::scene::Points2f& coords);

    bool out_of_image(float val) const noexcept;

private:
    ImageStatistics(const cv::Mat& image);

    void calculate_data(const cv::Mat& image);

private:
    tracv::video::RefPtr<tracv::video::Frame> gradient_directions_img_;
    tracv::video::RefPtr<tracv::video::Frame> gradient_magnitudes_img_;
};

}  // namespace tracv::utils