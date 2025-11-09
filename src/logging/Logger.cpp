#include <filesystem>

#include "Logger.h"

namespace fs = std::filesystem;

Logger &Logger::instance()
{
	static Logger instance;
	return instance;
}

void Logger::initialize(const std::string &log_dir,
						const std::string &app_name,
						spdlog::level::level_enum level)
{
	try
	{
		// Create log directory if it doesn't exist
		fs::create_directories(log_dir);

		// Create rotating file sink (20MB max, 5 rotated files)
		auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
			log_dir + "/" + app_name + ".log", 1024 * 1024 * 20, 5);

		// Create console sink with colors
		auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

		// Create logger with both sinks
		logger_ = std::make_shared<spdlog::logger>(app_name,
												   spdlog::sinks_init_list{console_sink, file_sink});

		// Different patterns for console (with colors) and file (without colors)
		console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
		file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");

		// Set level
		logger_->set_level(level);

		// Flush on every log call for development
		logger_->flush_on(level);

		logger_->info("Logger initialized successfully");
	}
	catch (const spdlog::spdlog_ex &ex)
	{
		std::cerr << "Log initialization failed: " << ex.what() << std::endl;
		throw;
	}
}