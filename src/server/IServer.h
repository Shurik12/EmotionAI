#pragma once

#include <memory>
#include <string>

class IServer
{
public:
	virtual ~IServer() = default;

	virtual void initialize() = 0;
	virtual void start() = 0;
	virtual void stop() noexcept = 0;

	// Non-copyable, non-movable
	IServer(const IServer &) = delete;
	IServer &operator=(const IServer &) = delete;
	IServer(IServer &&) = delete;
	IServer &operator=(IServer &&) = delete;

protected:
	IServer() = default;
};