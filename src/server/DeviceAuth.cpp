#include "DeviceAuth.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <openssl/evp.h>

#include <config/Config.h>
#include <db/DragonflyManager.h>
#include <logging/Logger.h>
#include <nlohmann/json.hpp>

namespace
{
    // Tokens are stored under a separate namespace so they can never collide
    // with task keys.
    constexpr const char *kTokenPrefix = "device:token:";
}

DeviceAuth::DeviceAuth(DragonflyManager *dragonfly)
    : dragonfly_(dragonfly)
{
}

std::string DeviceAuth::hashToken(const std::string &token)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx)
        return "";

    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, token.data(), token.size());
    EVP_DigestFinal_ex(ctx, digest, &length);
    EVP_MD_CTX_free(ctx);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < length; ++i)
        out << std::setw(2) << static_cast<int>(digest[i]);
    return out.str();
}

bool DeviceAuth::authenticate(const std::string &token, std::string &user_id) const
{
    user_id.clear();
    if (!dragonfly_ || token.empty())
        return false;

    const auto raw = dragonfly_->get_raw(std::string(kTokenPrefix) + hashToken(token));
    if (!raw)
        return false;

    try
    {
        const auto data = nlohmann::json::parse(*raw);
        user_id = data.value("user_id", "");
    }
    catch (const std::exception &e)
    {
        LOG_WARN("Malformed device token record: {}", e.what());
        return false;
    }

    return !user_id.empty();
}

std::string DeviceAuth::issue(std::string &user_id) const
{
    user_id.clear();
    if (!dragonfly_)
        return "";

    // Two random UUIDs, dashes stripped: ~244 bits of entropy, URL/header safe.
    std::string token = DragonflyManager::generate_uuid() + DragonflyManager::generate_uuid();
    token.erase(std::remove(token.begin(), token.end(), '-'), token.end());
    if (token.size() < 32)
        return "";

    user_id = DragonflyManager::generate_uuid();

    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const nlohmann::json record = {{"user_id", user_id}, {"created_ms", now_ms}};

    const int ttl = Config::instance().security().device_token_ttl_seconds;
    if (!dragonfly_->set_with_ttl(std::string(kTokenPrefix) + hashToken(token),
                                  record.dump(), ttl))
    {
        LOG_ERROR("Failed to persist a device token");
        user_id.clear();
        return "";
    }

    return token;
}

std::string DeviceAuth::revoke(const std::string &token) const
{
    if (!dragonfly_ || token.empty())
        return "";

    const std::string key = std::string(kTokenPrefix) + hashToken(token);

    std::string user_id;
    if (const auto raw = dragonfly_->get_raw(key))
    {
        try
        {
            user_id = nlohmann::json::parse(*raw).value("user_id", "");
        }
        catch (const std::exception &)
        {
            // Corrupt record: still delete the key.
        }
    }

    dragonfly_->del_raw(key);
    return user_id;
}
