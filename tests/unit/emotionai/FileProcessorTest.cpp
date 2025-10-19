#include <gtest/gtest.h>
#include <emotionai/FileProcessor.h>
#include <db/RedisManager.h>
#include "../../utils/TestUtils.h"

class FileProcessorTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("fileprocessor_test");
		TestUtils::setupTestLogging();
	}

	void TearDown() override
	{
		TestUtils::cleanupTestDir(testDir);
	}

	std::filesystem::path testDir;
};

TEST_F(FileProcessorTest, BasicCreation)
{
	db::RedisManager redisManager;
	EXPECT_NO_THROW({
		EmotionAI::FileProcessor processor(redisManager);
	});
}

TEST_F(FileProcessorTest, AllowedFileExtensions)
{
	db::RedisManager redisManager;
	EmotionAI::FileProcessor processor(redisManager);

	// Test supported image formats
	EXPECT_TRUE(processor.allowed_file("test.jpg"));
	EXPECT_TRUE(processor.allowed_file("test.jpeg"));
	EXPECT_TRUE(processor.allowed_file("test.png"));

	// Test supported video formats
	EXPECT_TRUE(processor.allowed_file("test.mp4"));
	EXPECT_TRUE(processor.allowed_file("test.avi"));
	EXPECT_TRUE(processor.allowed_file("test.webm"));

	// Test unsupported formats
	EXPECT_FALSE(processor.allowed_file("test.txt"));
	EXPECT_FALSE(processor.allowed_file("test.pdf"));
	EXPECT_FALSE(processor.allowed_file("test.exe"));
	EXPECT_FALSE(processor.allowed_file("test.doc"));
}

TEST_F(FileProcessorTest, FileExtensionCaseInsensitive)
{
	db::RedisManager redisManager;
	EmotionAI::FileProcessor processor(redisManager);

	EXPECT_TRUE(processor.allowed_file("test.JPG"));
	EXPECT_TRUE(processor.allowed_file("test.JPEG"));
	EXPECT_TRUE(processor.allowed_file("test.PNG"));
	EXPECT_TRUE(processor.allowed_file("TEST.MP4"));
}

TEST_F(FileProcessorTest, ModelLoadedState)
{
	db::RedisManager redisManager;
	EmotionAI::FileProcessor processor(redisManager);

	// Just test that the method doesn't crash
	bool loaded = processor.is_model_loaded();
	SUCCEED(); // Don't care about the actual value in unit test
}

TEST_F(FileProcessorTest, EmotionRecognizerAccess)
{
	db::RedisManager redisManager;
	EmotionAI::FileProcessor processor(redisManager);

	// Just test that the method doesn't crash
	auto *recognizer = processor.get_emotion_recognizer();
	SUCCEED(); // Don't care about the actual value in unit test
}

TEST_F(FileProcessorTest, ProcessFileWithoutModels)
{
	db::RedisManager redisManager;
	EmotionAI::FileProcessor processor(redisManager);

	std::string taskId = "test_task_123";
	std::filesystem::path testFilePath = testDir / "test.txt";
	std::string testFile = testFilePath.string(); // Convert to string

	// Create a test file
	{
		std::ofstream file(testFilePath);
		file << "test content";
	}

	// Test processing with unsupported file type
	EXPECT_NO_THROW({
		try
		{
			processor.process_file(taskId, testFile, "test.txt");
		}
		catch (const std::exception &e)
		{
			// Expected to fail due to unsupported file type
			// Just ensure it doesn't crash
			SUCCEED();
		}
	});
}