#pragma once

#include "FileStorage.h"
#include <miniocpp/client.h>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>

class S3FileStorage : public FileStorage
{
public:
    struct Config
    {
        std::string endpoint;
        std::string access_key;
        std::string secret_key;
        std::string bucket;
        bool use_ssl = true;
        std::string region = "us-east-1";
        std::chrono::seconds timeout = std::chrono::seconds(30);
        bool create_bucket_if_not_exists = true;
    };

    S3FileStorage(const Config &config);
    S3FileStorage(const std::string &endpoint,
                  const std::string &access_key,
                  const std::string &secret_key,
                  const std::string &bucket,
                  bool use_ssl = true,
                  const std::string &region = "us-east-1");

    ~S3FileStorage() override;

    // Prevent copying
    S3FileStorage(const S3FileStorage &) = delete;
    S3FileStorage &operator=(const S3FileStorage &) = delete;

    // Allow moving
    S3FileStorage(S3FileStorage &&) noexcept = default;
    S3FileStorage &operator=(S3FileStorage &&) noexcept = default;

    // Connection management
    bool isConnected() const { return state_->connected.load(); }
    bool reconnect();

    // FileStorage interface implementation
    bool saveFile(const std::string &file_content, const std::string &file_path) override;
    bool saveFile(const std::vector<uint8_t> &file_content, const std::string &file_path) override;
    std::string readFile(const std::string &file_path) override;
    std::vector<uint8_t> readFileBinary(const std::string &file_path) override;
    bool deleteFile(const std::string &file_path) override;
    bool fileExists(const std::string &file_path) override;

    // Directory operations
    bool createDirectory(const std::string &path) override;
    bool deleteDirectory(const std::string &path) override;
    std::vector<std::string> listFiles(const std::string &path) override;

    // File info
    size_t getFileSize(const std::string &file_path) override;
    std::string getFileHash(const std::string &file_path) override;

    // URL generation for web access
    std::string getFileUrl(const std::string &file_path) override;
    std::string generatePresignedUrl(const std::string &file_path,
                                     std::chrono::seconds expiration = std::chrono::hours(24));

    // Storage info
    std::string getStorageType() const override { return "s3"; }
    nlohmann::json getStorageInfo() override;

    // Copy/move operations between storage systems
    bool copyFile(const std::string &source_path, const std::string &dest_path) override;
    bool moveFile(const std::string &source_path, const std::string &dest_path) override;

    // Enhanced operations
    bool copyFileBetweenBuckets(const std::string &source_path,
                                const std::string &dest_bucket,
                                const std::string &dest_path);

    bool uploadFromStream(std::istream &stream,
                          const std::string &file_path,
                          const std::string &content_type = "");

    bool downloadToStream(const std::string &file_path,
                          std::ostream &stream);

private:
    struct InternalState
    {
        std::shared_ptr<minio::creds::StaticProvider> credential_provider;
        std::unique_ptr<minio::s3::Client> client;
        std::string bucket_name;
        std::string region;
        std::string endpoint;
        bool use_ssl;
        std::atomic<bool> connected{false};
    };

    void initializeClient();
    bool ensureBucketExists();
    std::string ensureTrailingSlash(const std::string &path) const;

    // Thread-safe client access with error handling
    template <typename Func>
    auto withClient(Func func) -> decltype(func(std::declval<minio::s3::Client &>()));

    template <typename Func>
    auto withClient(Func func) const -> decltype(func(std::declval<const minio::s3::Client &>()));

    // Direct client access for initialization (no connection check)
    minio::s3::Client &getClientDirect();

    std::unique_ptr<InternalState> state_;
    mutable std::mutex mutex_;
};