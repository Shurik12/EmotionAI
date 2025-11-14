#pragma once

#include <string>
#include <map>
#include <memory>
#include <optional>

#include <nlohmann/json.hpp>

class HttpClient
{
public:
	struct Response
	{
		int status_code;
		std::string body;
		std::map<std::string, std::string> headers;

		Response() : status_code(0) {}
		Response(int code, const std::string &b) : status_code(code), body(b) {}

		// Helper to parse JSON response
		nlohmann::json json() const
		{
			return nlohmann::json::parse(body);
		}

		// Check if request was successful
		bool success() const
		{
			return status_code >= 200 && status_code < 300;
		}
	};

	HttpClient(const std::string &host, int port, int timeout_seconds = 10);
	~HttpClient();

	// Basic HTTP methods
	Response sendRequest(const std::string &path,
						 const std::string &method,
						 const std::string &body = "",
						 const std::string &content_type = "application/json");

	Response get(const std::string &path);
	Response post(const std::string &path, const std::string &body, const std::string &content_type = "application/json");
	Response post(const std::string &path, const nlohmann::json &json_body);
	Response put(const std::string &path, const std::string &body, const std::string &content_type = "application/json");
	Response del(const std::string &path);

	// File upload methods
	Response uploadFile(const std::string &path,
						const std::string &file_content,
						const std::string &filename,
						const std::string &field_name = "file");

	Response uploadFileFromDisk(const std::string &path,
								const std::string &filepath,
								const std::string &field_name = "file");

	// Multipart form data with multiple fields
	Response uploadMultipart(const std::string &path,
							 const std::map<std::string, std::string> &fields,
							 const std::map<std::string, std::pair<std::string, std::string>> &files = {});

	// Set headers
	void setHeader(const std::string &key, const std::string &value);
	void setHeaders(const std::map<std::string, std::string> &headers);
	void clearHeaders();

	// Authentication
	void setBasicAuth(const std::string &username, const std::string &password);
	void setBearerToken(const std::string &token);

	// Connection settings
	void setTimeout(int seconds);
	void setFollowRedirects(bool follow);

	// Utility methods
	static std::string urlEncode(const std::string &value);
	static std::string generateBoundary();
	static std::map<std::string, std::string> parseQueryString(const std::string &query);

private:
	class Impl;
	std::unique_ptr<Impl> pimpl_;

	std::string host_;
	int port_;
	int timeout_seconds_;
	std::map<std::string, std::string> default_headers_;

	std::string buildMultipartBody(const std::map<std::string, std::string> &fields,
								   const std::map<std::string, std::pair<std::string, std::string>> &files,
								   const std::string &boundary) const;
};