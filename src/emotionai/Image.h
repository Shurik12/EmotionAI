#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <emotiefflib/facial_analysis.h>

class Image
{
public:
	// Constructor from OpenCV Mat
	Image(const cv::Mat &image, const std::string &format = ".jpg");

	std::vector<cv::Mat> recognizeFaces(const cv::Mat &frame, int downscaleWidth = 500);
	std::pair<cv::Mat, nlohmann::json> process_image(const cv::Mat &image, EmotiEffLib::EmotiEffLibRecognizer *fer);

private:
	std::stringstream m_buffer;
};

// Utility functions
cv::Mat downscaleImageToWidth(const cv::Mat &inputImage, int targetWidth);