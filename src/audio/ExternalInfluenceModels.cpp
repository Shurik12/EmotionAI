#include "ExternalInfluenceModels.h"

namespace audio {

std::string statusToString(ExternalInfluenceStatus status) {
    switch (status) {
        case ExternalInfluenceStatus::INSUFFICIENT_DATA:
            return "INSUFFICIENT_DATA";
        case ExternalInfluenceStatus::LOW:
            return "LOW";
        case ExternalInfluenceStatus::ELEVATED_TENSION:
            return "ELEVATED_TENSION";
        case ExternalInfluenceStatus::POSSIBLE_EXTERNAL_PRESSURE:
            return "POSSIBLE_EXTERNAL_PRESSURE";
        case ExternalInfluenceStatus::PROBABLE_EXTERNAL_INFLUENCE:
            return "PROBABLE_EXTERNAL_INFLUENCE";
        case ExternalInfluenceStatus::HIGH_EXTERNAL_INFLUENCE_RISK:
            return "HIGH_EXTERNAL_INFLUENCE_RISK";
        default:
            return "INSUFFICIENT_DATA";
    }
}

std::string managerActionForStatus(ExternalInfluenceStatus status) {
    switch (status) {
        case ExternalInfluenceStatus::INSUFFICIENT_DATA:
        case ExternalInfluenceStatus::LOW:
            return "STANDARD_PROCESS";
        case ExternalInfluenceStatus::ELEVATED_TENSION:
            return "CONTINUE_NO_BLOCK";
        case ExternalInfluenceStatus::POSSIBLE_EXTERNAL_PRESSURE:
            return "ASK_CONTROL_QUESTIONS";
        case ExternalInfluenceStatus::PROBABLE_EXTERNAL_INFLUENCE:
            return "ENHANCED_ANTIFRAUD_CHECK";
        case ExternalInfluenceStatus::HIGH_EXTERNAL_INFLUENCE_RISK:
            return "PRIORITY_ANTIFRAUD_REVIEW";
        default:
            return "STANDARD_PROCESS";
    }
}

nlohmann::json ExternalInfluenceResult::toJson() const {
    nlohmann::json j;

    j["status"] = statusToString(status);
    j["score"] = score;
    j["confidence"] = confidence;
    j["persistentPattern"] = persistentPattern;
    j["contextConfirmed"] = contextConfirmed;
    j["topFactors"] = topFactors;
    j["managerAction"] = managerAction;

    return j;
}

nlohmann::json ExternalInfluenceResult::diagnosticsToJson() const {
    nlohmann::json j;

    nlohmann::json comps = nlohmann::json::object();
    for (const auto& [key, value] : components) {
        comps[key] = value;
    }
    j["components"] = comps;
    j["fragmentCount"] = fragmentCount;
    j["totalFragmentCount"] = totalFragmentCount;
    j["fragmentScores"] = fragmentScores;
    j["rollingScores"] = rollingScores;
    j["bestWindowIndex"] = bestWindowIndex;
    j["windowSize"] = windowSize;
    j["contextFlags"] = contextFlags;
    j["strongContextCount"] = strongContextCount;
    j["moderateContextCount"] = moderateContextCount;
    j["audioQuality"] = audioQuality;
    j["meanModelProbability"] = meanModelProbability;
    j["speechCoverage"] = speechCoverage;
    j["baselineReliability"] = baselineReliability;
    j["insufficientReason"] = insufficientReason;

    return j;
}

} // namespace audio
