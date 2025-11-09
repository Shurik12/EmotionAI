// main.cpp (updated)
#include <csignal>
#include <iostream>
#include <memory>

#include <config/Config.h>
#include <logging/Logger.h>
#include <server/ServerFactory.h>
#include <server/IServer.h>

std::unique_ptr<IServer> web_server;

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

		std::cout << "Starting EmotionAI Server..." << std::endl;

		// Initialize configuration first
		auto &config = Common::Config::instance();
		if (!config.loadFromFile("config.yaml"))
		{
			std::cerr << "Warning: Failed to load config, using defaults" << std::endl;
		}

		// Initialize logger
		Logger::instance().initialize(config.logPath(), "EmotionAI-Server");
		Logger::instance().info("Logger initialized successfully");

		// Create server based on configuration
		std::string server_type = config.server().type;
		Logger::instance().info("Creating server type: {}", server_type);

		web_server = ServerFactory::createServer(server_type);

		// Initialize and start server
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