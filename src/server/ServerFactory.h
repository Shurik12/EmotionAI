#pragma once

#include <memory>
#include <string>
#include <server/IServer.h>

enum class ServerType
{
	BLOCKING,	 // WebServer (httplib-based)
	NON_BLOCKING // MultiplexingServer (epoll-based)
};

class ServerFactory
{
public:
	static ServerType serverTypeFromString(const std::string &type);
	static std::string serverTypeToString(ServerType type);
	static std::unique_ptr<IServer> createServer(ServerType type);
	static std::unique_ptr<IServer> createServer(const std::string &type);
};