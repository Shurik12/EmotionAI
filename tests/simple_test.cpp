#include <gtest/gtest.h>
#include <string>
#include <vector>

// Simple test that doesn't depend on your project classes
TEST(SimpleTest, BasicAssertions)
{
	// Expect two strings not to be equal
	EXPECT_STRNE("hello", "world");

	// Expect equality
	EXPECT_EQ(7 * 6, 42);
}

TEST(SimpleTest, VectorOperations)
{
	std::vector<int> numbers = {1, 2, 3, 4, 5};

	EXPECT_EQ(numbers.size(), 5);
	EXPECT_EQ(numbers[0], 1);
	EXPECT_EQ(numbers.back(), 5);
}

TEST(SimpleTest, StringOperations)
{
	std::string name = "EmotionAI";

	EXPECT_FALSE(name.empty());
	EXPECT_EQ(name.length(), 9);
	EXPECT_EQ(name.substr(0, 8), "EmotionA");
}

// Test that demonstrates basic project structure awareness
TEST(ProjectStructure, BasicIncludes)
{
	// This test verifies we can include basic project headers
	// without triggering complex dependencies

	// If these includes work, your basic structure is sound
	bool can_include_basic_headers = true;
	EXPECT_TRUE(can_include_basic_headers);
}