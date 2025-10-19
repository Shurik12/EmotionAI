#pragma once

#include <gmock/gmock.h>
#include <emotionai/FileProcessor.h>

class MockFileProcessor : public EmotionAI::FileProcessor
{
public:
	MockFileProcessor(db::RedisManager &redis_manager)
		: EmotionAI::FileProcessor(redis_manager) {}

	MOCK_METHOD(bool, allowed_file, (const std::string &), (override));
	MOCK_METHOD(void, process_file, (const std::string &, const std::string &, const std::string &), (override));
	MOCK_METHOD(bool, is_model_loaded, (), (const, override));
};