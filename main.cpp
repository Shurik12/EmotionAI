#include <iostream>
#include <memory>
#include <csignal>
#include <server/WebServer.h>
#include <common/Config.h>

std::unique_ptr<WebServer> web_server;

// Signal handler for graceful shutdown
void signal_handler(int signal)
{
    std::cout << "Received signal " << signal << ", shutting down gracefully..." << std::endl;
    if (web_server)
    {
        web_server->stop();
    }
}

int main()
{
    try
    {
        // Set up signal handlers for graceful shutdown
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::cout << "Starting EmotionAI Web Server..." << std::endl;

        // Create and start the web server
        web_server = std::make_unique<WebServer>();
        web_server->initialize();
        web_server->start();

        std::cout << "Server stopped." << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}