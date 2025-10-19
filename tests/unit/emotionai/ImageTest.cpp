#include <gtest/gtest.h>
#include <emotionai/Image.h>
#include "../../utils/TestUtils.h"

class ImageTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("image_test");
		testImagePath = testDir / "test_image.jpg";
		TestUtils::createTestImage(testImagePath);
		TestUtils::setupTestLogging();
	}

	void TearDown() override
	{
		TestUtils::cleanupTestDir(testDir);
	}

	std::filesystem::path testDir;
	std::filesystem::path testImagePath;
};

TEST_F(ImageTest, ConstructorFromFilename)
{
	EXPECT_NO_THROW({
		EmotionAI::Image image(testImagePath.string());
		EXPECT_FALSE(image.get_buffer().empty());
	});
}

TEST_F(ImageTest, ConstructorFromCVMat)
{
	cv::Mat testMat(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));

	EXPECT_NO_THROW({
		EmotionAI::Image image(testMat, ".jpg");
		EXPECT_FALSE(image.get_buffer().empty());
	});
}

TEST_F(ImageTest, ToCVMatConversion)
{
	EmotionAI::Image image(testImagePath.string());
	cv::Mat cvImage = image.to_cv_mat();

	EXPECT_FALSE(cvImage.empty());
	EXPECT_EQ(cvImage.cols, 100);
	EXPECT_EQ(cvImage.rows, 100);
}

TEST_F(ImageTest, MimeBundleRepr)
{
	EmotionAI::Image image(testImagePath.string());
	auto bundle = image.mime_bundle_repr();

	EXPECT_TRUE(bundle.contains("image/jpeg"));
	EXPECT_FALSE(bundle["image/jpeg"].get<std::string>().empty());
}

TEST_F(ImageTest, ImageResizeUtilities)
{
	cv::Mat testImage(200, 300, CV_8UC3, cv::Scalar(255, 255, 255));

	// Test resize with aspect ratio
	cv::Mat resized = EmotionAI::resizeWithAspectRatio(testImage, 150, 0);
	EXPECT_EQ(resized.cols, 150);
	EXPECT_EQ(resized.rows, 100); // 150 / (300/200) = 100

	// Test downscale
	cv::Mat downscaled = EmotionAI::downscaleImageToWidth(testImage, 150);
	EXPECT_EQ(downscaled.cols, 150);
	EXPECT_EQ(downscaled.rows, 100);
}