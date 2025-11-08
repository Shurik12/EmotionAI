#include <iostream>
#include <memory>
#include <csignal>
#include <server/MultiplexingServer.h>
#include <common/Config.h>

std::unique_ptr<MultiplexingServer> web_server;

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

        std::cout << "Starting EmotionAI Multiplexing Server..." << std::endl;

        // Create and start the multiplexing server
        web_server = std::make_unique<MultiplexingServer>();
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