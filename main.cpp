#include <csignal>
#include <iostream>
#include <memory>

#include <config/Config.h>
#include <logging/Logger.h>
#include <server/Server.h>

std::unique_ptr<Server> web_server;

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
		auto &config = Config::instance();
		if (!config.loadFromFile("config.yaml"))
		{
			std::cerr << "Warning: Failed to load config, using defaults" << std::endl;
		}

		// Setup application environment (create directories)
		if (!config.setupApplicationEnvironment())
		{
			std::cerr << "Failed to setup application environment" << std::endl;
			return 1;
		}

		// Validate configuration
		if (!config.validate())
		{
			std::cerr << "Configuration validation failed" << std::endl;
			return 1;
		}

		// Initialize logger using configuration
		Logger::instance().initialize(
			config.paths().logs,
			"EmotionAI-Server",
			config.getSpdLogLevel()
		);

		// Log startup information
		Logger::instance().info("=== EmotionAI Server Starting ===");
		Logger::instance().info("Host: {}", config.server().host);
		Logger::instance().info("Port: {}", config.server().port);
		Logger::instance().info("Log level: {}", config.logging().level);
		Logger::instance().info("Upload path: {}", config.paths().uploads);
		Logger::instance().info("Results path: {}", config.paths().results);

		// Create server (now only MultiplexingServer)
		Logger::instance().info("Creating server (MultiplexingServer)");
		web_server = std::make_unique<Server>();

		if (!web_server)
		{
			Logger::instance().error("Failed to create server");
			return 1;
		}

		// Initialize server
		Logger::instance().info("Initializing server...");
		web_server->initialize();

		// Start server
		Logger::instance().info("Starting server...");
		web_server->start();

		Logger::instance().info("Server stopped gracefully");
		return 0;
	}
	catch (const std::exception &e)
	{
		// If logger might not be initialized, also write to stderr
		try {
			Logger::instance().critical("Fatal error: {}", e.what());
		} catch (...) {
			std::cerr << "Fatal error: " << e.what() << std::endl;
		}
		return 1;
	}
	catch (...)
	{
		try {
			Logger::instance().critical("Unknown fatal error occurred");
		} catch (...) {
			std::cerr << "Unknown fatal error occurred" << std::endl;
		}
		return 1;
	}
}