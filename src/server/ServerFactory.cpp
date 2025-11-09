#include <server/ServerFactory.h>
#include <server/WebServer.h>
#include <server/MultiplexingServer.h>
#include <logging/Logger.h>

ServerType ServerFactory::serverTypeFromString(const std::string &type)
{
	if (type == "blocking" || type == "BLOCKING")
	{
		return ServerType::BLOCKING;
	}
	else if (type == "non-blocking" || type == "NON_BLOCKING" || type == "non_blocking")
	{
		return ServerType::NON_BLOCKING;
	}
	else
	{
		LOG_WARN("Unknown server type: {}, defaulting to NON_BLOCKING", type);
		return ServerType::NON_BLOCKING;
	}
}

std::string ServerFactory::serverTypeToString(ServerType type)
{
	switch (type)
	{
	case ServerType::BLOCKING:
		return "blocking";
	case ServerType::NON_BLOCKING:
		return "non-blocking";
	default:
		return "unknown";
	}
}

std::unique_ptr<IServer> ServerFactory::createServer(ServerType type)
{
	switch (type)
	{
	case ServerType::BLOCKING:
		LOG_INFO("Creating blocking server (WebServer)");
		return std::make_unique<WebServer>();

	case ServerType::NON_BLOCKING:
		LOG_INFO("Creating non-blocking server (MultiplexingServer)");
		return std::make_unique<MultiplexingServer>();

	default:
		LOG_WARN("Unknown server type, defaulting to non-blocking");
		return std::make_unique<MultiplexingServer>();
	}
}

std::unique_ptr<IServer> ServerFactory::createServer(const std::string &type)
{
	return createServer(serverTypeFromString(type));
}