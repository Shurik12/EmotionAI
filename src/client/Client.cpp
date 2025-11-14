#include "Client.h"
#include <common/httplib.h>
#include <sstream>
#include <iomanip>
#include <random>
#include <filesystem>
#include <fstream>

class HttpClient::Impl
{
public:
	Impl(const std::string &host, int port, int timeout_seconds)
		: client_(host, port)
	{
		client_.set_connection_timeout(timeout_seconds);
		client_.set_read_timeout(timeout_seconds);
		client_.set_write_timeout(timeout_seconds);
		client_.set_follow_location(true);
	}

	httplib::Client client_;
};

HttpClient::HttpClient(const std::string &host, int port, int timeout_seconds)
	: host_(host), port_(port), timeout_seconds_(timeout_seconds),
	  pimpl_(std::make_unique<Impl>(host, port, timeout_seconds))
{
	// Set default headers
	default_headers_["User-Agent"] = "EmotionAI-HTTP-Client/1.0";
	default_headers_["Accept"] = "*/*";
	default_headers_["Connection"] = "close";
}

HttpClient::~HttpClient() = default;

HttpClient::Response HttpClient::sendRequest(const std::string &path,
											 const std::string &method,
											 const std::string &body,
											 const std::string &content_type)
{
	// Prepare headers
	httplib::Headers headers;
	for (const auto &[key, value] : default_headers_)
	{
		headers.emplace(key, value);
	}

	if (!body.empty() && !content_type.empty())
	{
		headers.emplace("Content-Type", content_type);
	}

	httplib::Result response;

	if (method == "GET")
	{
		response = pimpl_->client_.Get(path.c_str(), headers);
	}
	else if (method == "POST")
	{
		response = pimpl_->client_.Post(path.c_str(), headers, body, content_type.c_str());
	}
	else if (method == "PUT")
	{
		response = pimpl_->client_.Put(path.c_str(), headers, body, content_type.c_str());
	}
	else if (method == "DELETE")
	{
		response = pimpl_->client_.Delete(path.c_str(), headers);
	}
	else if (method == "OPTIONS")
	{
		response = pimpl_->client_.Options(path.c_str(), headers);
	}
	else
	{
		return Response(0, "Unsupported HTTP method: " + method);
	}

	if (!response)
	{
		return Response(0, "No response from server or connection failed");
	}

	Response result;
	result.status_code = response->status;
	result.body = response->body;

	for (const auto &[key, value] : response->headers)
	{
		result.headers[key] = value;
	}

	return result;
}

HttpClient::Response HttpClient::get(const std::string &path)
{
	return sendRequest(path, "GET");
}

HttpClient::Response HttpClient::post(const std::string &path, const std::string &body, const std::string &content_type)
{
	return sendRequest(path, "POST", body, content_type);
}

HttpClient::Response HttpClient::post(const std::string &path, const nlohmann::json &json_body)
{
	return post(path, json_body.dump(), "application/json");
}

HttpClient::Response HttpClient::put(const std::string &path, const std::string &body, const std::string &content_type)
{
	return sendRequest(path, "PUT", body, content_type);
}

HttpClient::Response HttpClient::del(const std::string &path)
{
	return sendRequest(path, "DELETE");
}

HttpClient::Response HttpClient::uploadFile(const std::string &path,
											const std::string &file_content,
											const std::string &filename,
											const std::string &field_name)
{
	// For this httplib version, use UploadFormDataItems
	httplib::UploadFormDataItems items = {
		{field_name, file_content, filename, "application/octet-stream"}};

	// Prepare headers
	httplib::Headers headers;
	for (const auto &[key, value] : default_headers_)
	{
		headers.emplace(key, value);
	}

	auto response = pimpl_->client_.Post(path.c_str(), headers, items);

	if (!response)
	{
		return Response(0, "No response from server or connection failed");
	}

	Response result;
	result.status_code = response->status;
	result.body = response->body;

	for (const auto &[key, value] : response->headers)
	{
		result.headers[key] = value;
	}

	return result;
}

HttpClient::Response HttpClient::uploadFileFromDisk(const std::string &path,
													const std::string &filepath,
													const std::string &field_name)
{
	std::ifstream file(filepath, std::ios::binary);
	if (!file)
	{
		return Response(0, "Cannot open file: " + filepath);
	}

	std::vector<char> file_data((std::istreambuf_iterator<char>(file)),
								std::istreambuf_iterator<char>());

	std::string filename = std::filesystem::path(filepath).filename().string();
	std::string file_content(file_data.begin(), file_data.end());

	return uploadFile(path, file_content, filename, field_name);
}

HttpClient::Response HttpClient::uploadMultipart(const std::string &path,
												 const std::map<std::string, std::string> &fields,
												 const std::map<std::string, std::pair<std::string, std::string>> &files)
{
	httplib::UploadFormDataItems items;

	// Add form fields
	for (const auto &[name, value] : fields)
	{
		items.push_back({name, value, "", "text/plain"});
	}

	// Add files
	for (const auto &[field_name, file_info] : files)
	{
		const auto &[filename, content] = file_info;
		items.push_back({field_name, content, filename, "application/octet-stream"});
	}

	// Prepare headers
	httplib::Headers headers;
	for (const auto &[key, value] : default_headers_)
	{
		headers.emplace(key, value);
	}

	auto response = pimpl_->client_.Post(path.c_str(), headers, items);

	if (!response)
	{
		return Response(0, "No response from server or connection failed");
	}

	Response result;
	result.status_code = response->status;
	result.body = response->body;

	for (const auto &[key, value] : response->headers)
	{
		result.headers[key] = value;
	}

	return result;
}

void HttpClient::setHeader(const std::string &key, const std::string &value)
{
	default_headers_[key] = value;
}

void HttpClient::setHeaders(const std::map<std::string, std::string> &headers)
{
	for (const auto &[key, value] : headers)
	{
		default_headers_[key] = value;
	}
}

void HttpClient::clearHeaders()
{
	default_headers_.clear();
	// Restore minimal defaults
	default_headers_["User-Agent"] = "EmotionAI-HTTP-Client/1.0";
	default_headers_["Accept"] = "*/*";
}

void HttpClient::setBasicAuth(const std::string &username, const std::string &password)
{
	pimpl_->client_.set_basic_auth(username.c_str(), password.c_str());
}

void HttpClient::setBearerToken(const std::string &token)
{
	default_headers_["Authorization"] = "Bearer " + token;
}

void HttpClient::setTimeout(int seconds)
{
	timeout_seconds_ = seconds;
	pimpl_->client_.set_connection_timeout(seconds);
	pimpl_->client_.set_read_timeout(seconds);
	pimpl_->client_.set_write_timeout(seconds);
}

void HttpClient::setFollowRedirects(bool follow)
{
	pimpl_->client_.set_follow_location(follow);
}

std::string HttpClient::urlEncode(const std::string &value)
{
	std::ostringstream escaped;
	escaped.fill('0');
	escaped << std::hex;

	for (auto c : value)
	{
		if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
		{
			escaped << c;
		}
		else
		{
			escaped << '%' << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c));
		}
	}

	return escaped.str();
}

std::string HttpClient::generateBoundary()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 15);

	std::stringstream ss;
	ss << "----WebKitFormBoundary";
	for (int i = 0; i < 16; ++i)
	{
		ss << std::hex << dis(gen);
	}
	return ss.str();
}

std::map<std::string, std::string> HttpClient::parseQueryString(const std::string &query)
{
	std::map<std::string, std::string> result;
	std::istringstream iss(query);
	std::string pair;

	while (std::getline(iss, pair, '&'))
	{
		size_t pos = pair.find('=');
		if (pos != std::string::npos)
		{
			std::string key = pair.substr(0, pos);
			std::string value = pair.substr(pos + 1);
			result[urlEncode(key)] = urlEncode(value);
		}
	}

	return result;
}