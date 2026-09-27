#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <logging/Logger.h>

int main(int argc, char **argv)
{
	// Initialize Google Test
	spdlog::set_level(spdlog::level::err);
	::testing::InitGoogleTest(&argc, argv);

	// The LOG_* macros dereference a shared_ptr that is only created by
	// Logger::initialize(). Without this, any unit test that constructs a
	// class which logs (e.g. BurnoutAnalyzer) segfaults in its constructor.
	try
	{
		Logger::instance().initialize("/tmp/emotionai_tests_logs",
									  "emotionai_tests",
									  spdlog::level::err);
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
	}

	// Run tests
	return RUN_ALL_TESTS();
}