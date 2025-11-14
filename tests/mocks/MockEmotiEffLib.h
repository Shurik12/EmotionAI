#pragma once

#include <gmock/gmock.h>
#include <emotiefflib/facial_analysis.h>

class MockEmotiEffLibRecognizer : public EmotiEffLib::EmotiEffLibRecognizer
{
public:
	MOCK_METHOD(EmotiEffLib::EmotiEffLibRes, predictEmotions, (const cv::Mat &, bool), (override));
	MOCK_METHOD(std::vector<EmotiEffLib::EmotiEffLibRes>, predictEmotionsBatch, (const std::vector<cv::Mat> &, bool), (override));

	// Static method mock - we'll handle this differently
	static std::unique_ptr<MockEmotiEffLibRecognizer> createMock()
	{
		return std::make_unique<MockEmotiEffLibRecognizer>();
	}
};

// Mock factory function
namespace EmotiEffLib
{
	std::unique_ptr<EmotiEffLibRecognizer> EmotiEffLibRecognizer::createInstance(const std::string &backend, const std::string &modelPath)
	{
		// Return a mock instance for testing
		return std::make_unique<MockEmotiEffLibRecognizer>();
	}
}