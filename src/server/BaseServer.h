#pragma once

#include <map>
#include <set>
#include <string>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>

#include <nlohmann/json.hpp>
#include <server/IServer.h>
#include <db/DragonflyManager.h>
#include <emotionai/FileProcessor.h>
#include <server/ThreadPool.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

class BaseServer : public IServer
{
public:
    virtual ~BaseServer() override = default;

protected:
    explicit BaseServer();
    
    // Common initialization methods
    void loadConfiguration();
    void ensureDirectoriesExist();
    void initializeComponents();
    
    // Common route handler implementations - now returns task_id
    std::string handleUploadCommon(const std::string& file_content, const std::string& filename, bool realtime = false);
    std::string handleSubmitApplicationCommon(const std::string& body);
    void validateJsonDocument(const nlohmann::json &json);
    
    // Common file serving
    std::string getMimeType(const std::string &filename) const;
    bool isApiEndpoint(const std::string &path) const;
    
    // Multipart form data parsing
    std::map<std::string, std::string> parseMultipartFormData(const std::string &body, 
                                                             const std::string &boundary);
    std::string extractBoundary(const std::string &content_type);
    
    // Common components
    std::shared_ptr<DragonflyManager> dragonfly_manager_;
    std::unique_ptr<FileProcessor> file_processor_;
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // Common paths
    fs::path static_files_root_;
    fs::path upload_folder_;
    fs::path results_folder_;
    fs::path log_folder_;
};