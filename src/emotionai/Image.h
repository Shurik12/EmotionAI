#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <emotiefflib/facial_analysis.h>

class Image
{
public:
	// Constructor from filename
	Image(const std::string &filename);

	// Constructor from OpenCV Mat
	Image(const cv::Mat &image, const std::string &format = ".jpg");

	// Get the image buffer as string
	std::string get_buffer() const;

	// Get MIME bundle representation for display
	nlohmann::json mime_bundle_repr() const;

	// Get OpenCV Mat from buffer
	cv::Mat to_cv_mat() const;

	std::vector<cv::Mat> recognizeFaces(const cv::Mat &frame, int downscaleWidth = 500);
	std::pair<cv::Mat, nlohmann::json> process_image(const cv::Mat &image, EmotiEffLib::EmotiEffLibRecognizer *fer);

private:
	std::stringstream m_buffer;
};

// Utility functions
cv::Mat resizeWithAspectRatio(const cv::Mat &img, int target_width = 0, int target_height = 0);
cv::Mat downscaleImageToWidth(const cv::Mat &inputImage, int targetWidth);