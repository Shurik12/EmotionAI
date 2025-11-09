#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

class Logger
{
public:
	static Logger &instance();

	void initialize(const std::string &log_dir,
					const std::string &app_name = "EmotionAI",
					spdlog::level::level_enum level = spdlog::level::info);

	std::shared_ptr<spdlog::logger> getLogger() { return logger_; }

	// Direct logging methods
	template <typename... Args>
	void trace(spdlog::format_string_t<Args...> fmt, Args &&...args)
	{
		logger_->trace(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	void debug(spdlog::format_string_t<Args...> fmt, Args &&...args)
	{
		logger_->debug(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	void info(spdlog::format_string_t<Args...> fmt, Args &&...args)
	{
		logger_->info(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	void warn(spdlog::format_string_t<Args...> fmt, Args &&...args)
	{
		logger_->warn(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	void error(spdlog::format_string_t<Args...> fmt, Args &&...args)
	{
		logger_->error(fmt, std::forward<Args>(args)...);
	}

	template <typename... Args>
	void critical(spdlog::format_string_t<Args...> fmt, Args &&...args)
	{
		logger_->critical(fmt, std::forward<Args>(args)...);
	}

private:
	Logger() = default;
	~Logger() = default;

	std::shared_ptr<spdlog::logger> logger_;
};

// Convenience logging macros
#define LOG_TRACE(...) Logger::instance().trace(__VA_ARGS__)
#define LOG_DEBUG(...) Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...) Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...) Logger::instance().warn(__VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)
#define LOG_CRITICAL(...) Logger::instance().critical(__VA_ARGS__)

// Conditional logging macros
#define LOG_IF(condition, ...)                    \
	do                                            \
	{                                             \
		if (condition)                            \
		{                                         \
			Logger::instance().info(__VA_ARGS__); \
		}                                         \
	} while (0)

#define LOG_ERROR_IF(condition, ...)               \
	do                                             \
	{                                              \
		if (condition)                             \
		{                                          \
			Logger::instance().error(__VA_ARGS__); \
		}                                          \
	} while (0)

// Scope-based logging
#define LOG_SCOPE(name) LogScope logScope##__LINE__(name)
#define LOG_FUNCTION_SCOPE() LOG_SCOPE(__FUNCTION__)

// Helper class for scope-based logging
class LogScope
{
public:
	explicit LogScope(const std::string &scope_name)
		: scope_name_(scope_name)
	{
		Logger::instance().debug("ENTER: {}", scope_name_);
	}

	~LogScope()
	{
		Logger::instance().debug("EXIT: {}", scope_name_);
	}

private:
	std::string scope_name_;
};