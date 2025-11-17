#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

int main(int argc, char **argv)
{
	// Initialize Google Test
	spdlog::set_level(spdlog::level::err);
	::testing::InitGoogleTest(&argc, argv);

	// Run tests
	return RUN_ALL_TESTS();
}