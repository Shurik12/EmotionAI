#include <csignal>
#include <iostream>
#include <memory>

#include <config/Config.h>
#include <logging/Logger.h>
#include <server/MultiplexingServer.h>

std::unique_ptr<MultiplexingServer> web_server;

void signal_handler(int signal)
{
	Logger::instance().info("Received signal {}, shutting down gracefully...", signal);
	if (web_server)
	{
		web_server->stop();
	}
}

int main()
{
	try
	{
		// Set up signal handlers
		std::signal(SIGINT, signal_handler);
		std::signal(SIGTERM, signal_handler);

		std::cout << "Starting EmotionAI Multiplexing Server..." << std::endl;

		// Initialize configuration first
		auto &config = Common::Config::instance();
		if (!config.loadFromFile("config.yaml"))
		{
			std::cerr << "Warning: Failed to load config, using defaults" << std::endl;
		}

		// Initialize logger
		Logger::instance().initialize(config.logPath(), "EmotionAI-Server");
		Logger::instance().info("Logger initialized successfully");

		// Create and start server
		web_server = std::make_unique<MultiplexingServer>();
		web_server->initialize();
		web_server->start();

		Logger::instance().info("Server stopped gracefully");
		return 0;
	}
	catch (const std::exception &e)
	{
		Logger::instance().critical("Fatal error: {}", e.what());
		return 1;
	}
	catch (...)
	{
		Logger::instance().critical("Unknown fatal error occurred");
		return 1;
	}
}