#include "LocalFileStorage.h"
#include <fstream>
#include <sstream>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <iomanip>
#include <logging/Logger.h>

LocalFileStorage::LocalFileStorage(const std::string &base_path)
    : base_path_(base_path.empty() ? fs::current_path() : fs::path(base_path))
{
    // Ensure base directory exists
    createDirectory("");
    LOG_INFO("LocalFileStorage initialized with base path: {}", base_path_.string());
}

fs::path LocalFileStorage::resolvePath(const std::string &file_path)
{
    if (file_path.empty() || file_path == "/")
    {
        return base_path_;
    }

    // Prevent directory traversal attacks
    fs::path resolved_path = base_path_ / file_path;
    fs::path canonical_path = fs::weakly_canonical(resolved_path);

    // Ensure the resolved path is within base path
    if (canonical_path.string().find(base_path_.string()) != 0)
    {
        throw std::runtime_error("Path traversal attempt detected: " + file_path);
    }

    return canonical_path;
}

bool LocalFileStorage::saveFile(const std::string &file_content, const std::string &file_path)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    try
    {
        fs::path full_path = resolvePath(file_path);

        // Create parent directories if they don't exist
        fs::create_directories(full_path.parent_path());

        std::ofstream file(full_path, std::ios::binary);
        if (!file.is_open())
        {
            LOG_ERROR("Failed to open file for writing: {}", full_path.string());
            return false;
        }

        file.write(file_content.data(), file_content.size());
        file.close();

        LOG_DEBUG("File saved successfully: {} ({} bytes)", full_path.string(), file_content.size());
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error saving file {}: {}", file_path, e.what());
        return false;
    }
}

bool LocalFileStorage::saveFile(const std::vector<uint8_t> &file_content, const std::string &file_path)
{
    std::string content_str(file_content.begin(), file_content.end());
    return saveFile(content_str, file_path);
}

std::string LocalFileStorage::readFile(const std::string &file_path)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    try
    {
        fs::path full_path = resolvePath(file_path);

        if (!fs::exists(full_path) || !fs::is_regular_file(full_path))
        {
            LOG_ERROR("File not found: {}", full_path.string());
            return "";
        }

        std::ifstream file(full_path, std::ios::binary);
        if (!file.is_open())
        {
            LOG_ERROR("Failed to open file for reading: {}", full_path.string());
            return "";
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        LOG_DEBUG("File read successfully: {} ({} bytes)", full_path.string(), buffer.str().size());
        return buffer.str();
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error reading file {}: {}", file_path, e.what());
        return "";
    }
}

std::vector<uint8_t> LocalFileStorage::readFileBinary(const std::string &file_path)
{
    std::string content = readFile(file_path);
    return std::vector<uint8_t>(content.begin(), content.end());
}

bool LocalFileStorage::deleteFile(const std::string &file_path)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    try
    {
        fs::path full_path = resolvePath(file_path);

        if (!fs::exists(full_path))
        {
            LOG_WARN("File not found for deletion: {}", full_path.string());
            return false;
        }

        if (!fs::is_regular_file(full_path))
        {
            LOG_ERROR("Path is not a regular file: {}", full_path.string());
            return false;
        }

        bool success = fs::remove(full_path);
        if (success)
        {
            LOG_DEBUG("File deleted successfully: {}", full_path.string());
        }
        else
        {
            LOG_ERROR("Failed to delete file: {}", full_path.string());
        }

        return success;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error deleting file {}: {}", file_path, e.what());
        return false;
    }
}

bool LocalFileStorage::fileExists(const std::string &file_path)
{
    try
    {
        fs::path full_path = resolvePath(file_path);
        return fs::exists(full_path) && fs::is_regular_file(full_path);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error checking file existence {}: {}", file_path, e.what());
        return false;
    }
}

bool LocalFileStorage::createDirectory(const std::string &path)
{
    try
    {
        fs::path full_path = resolvePath(path);
        return fs::create_directories(full_path);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error creating directory {}: {}", path, e.what());
        return false;
    }
}

bool LocalFileStorage::deleteDirectory(const std::string &path)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    try
    {
        fs::path full_path = resolvePath(path);

        if (!fs::exists(full_path))
        {
            return true; // Directory doesn't exist, consider it deleted
        }

        if (!fs::is_directory(full_path))
        {
            LOG_ERROR("Path is not a directory: {}", full_path.string());
            return false;
        }

        // Remove directory and all contents
        std::uintmax_t removed_count = fs::remove_all(full_path);
        LOG_DEBUG("Directory deleted: {} ({} items removed)", full_path.string(), removed_count);
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error deleting directory {}: {}", path, e.what());
        return false;
    }
}

std::vector<std::string> LocalFileStorage::listFiles(const std::string &path)
{
    std::vector<std::string> files;

    try
    {
        fs::path full_path = resolvePath(path);

        if (!fs::exists(full_path) || !fs::is_directory(full_path))
        {
            LOG_ERROR("Directory not found: {}", full_path.string());
            return files;
        }

        for (const auto &entry : fs::directory_iterator(full_path))
        {
            if (entry.is_regular_file())
            {
                files.push_back(entry.path().filename().string());
            }
        }

        LOG_DEBUG("Listed {} files from directory: {}", files.size(), full_path.string());
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error listing files in {}: {}", path, e.what());
    }

    return files;
}

size_t LocalFileStorage::getFileSize(const std::string &file_path)
{
    try
    {
        fs::path full_path = resolvePath(file_path);

        if (!fs::exists(full_path) || !fs::is_regular_file(full_path))
        {
            return 0;
        }

        return fs::file_size(full_path);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error getting file size for {}: {}", file_path, e.what());
        return 0;
    }
}

std::string LocalFileStorage::calculateFileHash(const std::string &file_path)
{
    try
    {
        std::ifstream file(fs::path(base_path_) / file_path, std::ios::binary);
        if (!file)
        {
            LOG_ERROR("Failed to open file for hashing: {}", file_path);
            return "";
        }

        EVP_MD_CTX *context = EVP_MD_CTX_new();
        if (!context)
        {
            LOG_ERROR("Failed to create EVP context");
            return "";
        }

        if (EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1)
        {
            LOG_ERROR("Failed to initialize digest");
            EVP_MD_CTX_free(context);
            return "";
        }

        char buffer[4096];
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0)
        {
            if (EVP_DigestUpdate(context, buffer, file.gcount()) != 1)
            {
                LOG_ERROR("Failed to update digest");
                EVP_MD_CTX_free(context);
                return "";
            }
        }

        unsigned char hash[SHA256_DIGEST_LENGTH];
        unsigned int length = 0;
        if (EVP_DigestFinal_ex(context, hash, &length) != 1)
        {
            LOG_ERROR("Failed to finalize digest");
            EVP_MD_CTX_free(context);
            return "";
        }

        EVP_MD_CTX_free(context);

        // Convert to hex string
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        {
            ss << std::setw(2) << static_cast<unsigned>(hash[i]);
        }

        return ss.str();
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error calculating file hash for {}: {}", file_path, e.what());
        return "";
    }
}

std::string LocalFileStorage::getFileHash(const std::string &file_path)
{
    return calculateFileHash(file_path);
}

std::string LocalFileStorage::getFileUrl(const std::string &file_path)
{
    // For local storage, return a relative URL that can be served by the web server
    return "/api/storage/" + file_path;
}

nlohmann::json LocalFileStorage::getStorageInfo()
{
    try
    {
        nlohmann::json info;
        info["type"] = "local";
        info["base_path"] = base_path_.string();

        // Calculate storage usage
        size_t total_size = 0;
        size_t file_count = 0;

        for (const auto &entry : fs::recursive_directory_iterator(base_path_))
        {
            if (entry.is_regular_file())
            {
                total_size += entry.file_size();
                file_count++;
            }
        }

        info["total_files"] = file_count;
        info["total_size_bytes"] = total_size;
        info["total_size_mb"] = total_size / (1024.0 * 1024.0);

        return info;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error getting storage info: {}", e.what());
        return {{"error", e.what()}};
    }
}

bool LocalFileStorage::copyFile(const std::string &source_path, const std::string &dest_path)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    try
    {
        fs::path source_full = resolvePath(source_path);
        fs::path dest_full = resolvePath(dest_path);

        if (!fs::exists(source_full) || !fs::is_regular_file(source_full))
        {
            LOG_ERROR("Source file not found: {}", source_full.string());
            return false;
        }

        // Create destination directory if it doesn't exist
        fs::create_directories(dest_full.parent_path());

        fs::copy_file(source_full, dest_full, fs::copy_options::overwrite_existing);
        LOG_DEBUG("File copied from {} to {}", source_full.string(), dest_full.string());
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error copying file from {} to {}: {}", source_path, dest_path, e.what());
        return false;
    }
}

bool LocalFileStorage::moveFile(const std::string &source_path, const std::string &dest_path)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    try
    {
        fs::path source_full = resolvePath(source_path);
        fs::path dest_full = resolvePath(dest_path);

        if (!fs::exists(source_full) || !fs::is_regular_file(source_full))
        {
            LOG_ERROR("Source file not found: {}", source_full.string());
            return false;
        }

        // Create destination directory if it doesn't exist
        fs::create_directories(dest_full.parent_path());

        fs::rename(source_full, dest_full);
        LOG_DEBUG("File moved from {} to {}", source_full.string(), dest_full.string());
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error moving file from {} to {}: {}", source_path, dest_path, e.what());
        return false;
    }
}