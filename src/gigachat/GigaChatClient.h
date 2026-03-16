#pragma once

#include <string>
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>

namespace emotionai {
namespace gigachat {

struct EmotionData {
    float anger;
    float disgust;
    float fear;
    float happiness;
    float neutral;
    float sadness;
    float surprise;
    
    nlohmann::json toJson() const {
        return {
            {"anger", anger},
            {"disgust", disgust},
            {"fear", fear},
            {"happiness", happiness},
            {"neutral", neutral},
            {"sadness", sadness},
            {"surprise", surprise}
        };
    }
};

struct GigaChatConfig {
    bool enabled;
    std::string authKey;
    std::string model;
    std::string apiUrl;
    std::string authUrl;
    bool verifySsl;
};

class GigaChatClient {
public:
    explicit GigaChatClient(const GigaChatConfig& config);
    ~GigaChatClient();
    
    bool isEnabled() const { return m_config.enabled; }
    
    // Simple sync method that returns JSON string
    std::string analyzeEmotions(const EmotionData& emotions, const std::string& sessionId);
    
private:
    GigaChatConfig m_config;
    std::string m_accessToken;
    long m_expiresAt;
    
    bool ensureValidToken();
    bool fetchNewToken();
    std::string buildPrompt(const EmotionData& emotions) const;
};

} // namespace gigachat
} // namespace emotionai