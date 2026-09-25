// tools/burnout_harness.cpp
//
// Offline measurement harness for the audio burnout pipeline.
//
// It reuses the production code paths (Audio::process_audio,
// Audio::extract_acoustic_features, Audio::add_burnout_analysis) but writes a
// per-file machine-readable record and an aggregate report instead of serving
// an HTTP response. This lets us measure the SAME scoring logic before/after a
// change on a fixed set of recordings without restarting the server.
//
// Usage:
//   burnout_harness --model models/audio_model.pt --out /tmp/opencode/run \
//                   [--baseline baseline.json] \
//                   name1=dir1 [name2=dir2 ...]
//
// For every file the harness runs inference + feature extraction ONCE and then
// scores it with each baseline variant (always "default", plus "custom" when
// --baseline is given), so baseline experiments do not re-run the model.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <torch/script.h>

#include <audio/AudioFeatures.h>
#include <audio/BurnoutAnalyzer.h>
#include <audio/BurnoutModels.h>
#include <audio/LibrosaFeatureExtractor.h>
#include <emotionai/Audio.h>
#include <logging/Logger.h>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct Args {
    std::string model;
    std::string out_dir;
    std::string baseline_path;  // optional
    std::vector<std::pair<std::string, std::string>> sets;  // name -> dir
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            a.model = next("--model");
        } else if (arg == "--out") {
            a.out_dir = next("--out");
        } else if (arg == "--baseline") {
            a.baseline_path = next("--baseline");
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown flag: " + arg);
        } else {
            auto eq = arg.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("expected name=dir, got: " + arg);
            }
            a.sets.emplace_back(arg.substr(0, eq), arg.substr(eq + 1));
        }
    }
    if (a.model.empty() || a.out_dir.empty() || a.sets.empty()) {
        throw std::runtime_error(
            "usage: burnout_harness --model M --out DIR [--baseline B.json] "
            "name=dir [name2=dir2 ...]");
    }
    return a;
}

bool isAudioFile(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ext == ".wav" || ext == ".mp3" || ext == ".flac" || ext == ".m4a" ||
           ext == ".ogg" || ext == ".opus" || ext == ".aac" || ext == ".wma";
}

std::vector<std::string> listAudioFiles(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && isAudioFile(entry.path())) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

double numOr(const json& j, const char* key, double def) {
    if (j.is_object() && j.contains(key) && j[key].is_number()) {
        return j[key].get<double>();
    }
    return def;
}

struct Row {
    std::string set;
    std::string file;
    std::string variant;
    std::string level;
    std::string state;
    std::string top_factor;
    double risk = 0.0;
    double confidence = 0.0;
    double voice_activity = 0.0;
    double max_prob = 0.0;
    std::unordered_map<std::string, double> components;
    std::string error;
};

// Aggregate description for one (set, variant) group.
struct Group {
    size_t n = 0;
    size_t insufficient = 0;
    size_t low = 0;
    size_t moderate = 0;
    size_t high = 0;
    size_t severe = 0;
    double risk_sum = 0.0;
    std::unordered_map<std::string, size_t> component_high;  // > 0.5
};

void writeReport(std::ostream& os,
                 const std::map<std::pair<std::string, std::string>, Group>& groups) {
    os << "=== aggregate report ===\n";
    for (const auto& [key, g] : groups) {
        const auto& [set_name, variant] = key;
        os << "\n[" << set_name << " / " << variant << "] n=" << g.n
           << " insufficient=" << g.insufficient
           << " low=" << g.low << " moderate=" << g.moderate
           << " high=" << g.high << " severe=" << g.severe
           << "  mean_risk=" << fmt::format("{:.3f}", g.n ? g.risk_sum / g.n : 0.0)
           << "  FPR(non-low)="
           << fmt::format("{:.1f}%", g.n ? 100.0 * (g.n - g.low) / g.n : 0.0)
           << "  high_rate="
           << fmt::format("{:.1f}%", g.n ? 100.0 * (g.high + g.severe) / g.n : 0.0)
           << "\n";
        os << "    component>0.5:";
        for (const char* c : {"exhaustion", "prosodic_flattening", "pause_tempo",
                              "negative_activation", "positive_affect_loss"}) {
            auto it = g.component_high.find(c);
            os << " " << c << "=" << (it == g.component_high.end() ? 0 : it->second)
               << "/" << g.n;
        }
        os << "\n";
    }
}

void dumpRow(std::ostream& os, const Row& r) {
    os << r.set << '\t' << r.file << '\t' << r.variant << '\t' << r.level << '\t'
       << r.state << '\t' << fmt::format("{:.4f}", r.risk) << '\t'
       << fmt::format("{:.4f}", r.confidence) << '\t'
       << fmt::format("{:.4f}", r.voice_activity) << '\t'
       << fmt::format("{:.4f}", r.max_prob) << '\t'
       << fmt::format("{:.4f}", numOr(json(r.components), "exhaustion", 0.0)) << '\t'
       << fmt::format("{:.4f}", numOr(json(r.components), "prosodic_flattening", 0.0)) << '\t'
       << fmt::format("{:.4f}", numOr(json(r.components), "pause_tempo", 0.0)) << '\t'
       << fmt::format("{:.4f}", numOr(json(r.components), "negative_activation", 0.0)) << '\t'
       << fmt::format("{:.4f}", numOr(json(r.components), "positive_affect_loss", 0.0)) << '\t'
       << r.top_factor << '\t' << r.error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "argument error: " << e.what() << "\n";
        return 2;
    }

    fs::create_directories(args.out_dir);
    Logger::instance().initialize(args.out_dir + "/logs", "BurnoutHarness",
                                  spdlog::level::warn);

    std::cout << "Loading audio model: " << args.model << "\n";
    std::unique_ptr<torch::jit::Module> model;
    try {
        model = std::make_unique<torch::jit::Module>(torch::jit::load(args.model));
    } catch (const std::exception& e) {
        std::cerr << "failed to load model: " << e.what() << "\n";
        return 1;
    }

    // Baseline variants: (name, json). Empty json = production inline default.
    std::vector<std::pair<std::string, json>> baselines;
    baselines.emplace_back("default", json::object());
    if (!args.baseline_path.empty()) {
        std::ifstream in(args.baseline_path);
        if (!in) {
            std::cerr << "cannot open baseline: " << args.baseline_path << "\n";
            return 1;
        }
        baselines.emplace_back("custom", json::parse(in));
    }

    const std::string tsv_path = args.out_dir + "/summary.tsv";
    std::ofstream tsv(tsv_path);
    tsv << "set\tfile\tvariant\tlevel\tstate\trisk\tconfidence\t"
           "voice_activity\tmax_prob\texhaustion\tprosodic_flattening\tpause_tempo\t"
           "negative_activation\tpositive_affect_loss\ttop_factor\terror\n";

    std::map<std::pair<std::string, std::string>, Group> groups;

    for (const auto& [set_name, dir] : args.sets) {
        std::vector<std::string> files;
        try {
            files = listAudioFiles(dir);
        } catch (const std::exception& e) {
            std::cerr << "cannot list " << dir << ": " << e.what() << "\n";
            continue;
        }
        std::cout << "[" << set_name << "] " << files.size() << " files from " << dir << "\n";

        size_t idx = 0;
        for (const auto& path : files) {
            ++idx;
            const std::string base = fs::path(path).filename().string();

            Audio audio(path);
            if (!audio.is_loaded()) {
                std::cerr << "  skip " << base << ": " << audio.get_error() << "\n";
                continue;
            }

            json emotion = audio.process_audio(model.get());
            audio::AcousticFeatures features = audio.extract_acoustic_features();

            double va = 0.0;
            {
                json fj = features.toJson();
                va = numOr(fj, "voice_activity_ratio", 0.0);
            }
            double max_prob = 0.0;
            if (emotion.contains("additional_probs") &&
                emotion["additional_probs"].is_object()) {
                for (const auto& [k, v] : emotion["additional_probs"].items()) {
                    try {
                        max_prob = std::max(max_prob, std::stod(v.get<std::string>()));
                    } catch (...) {
                    }
                }
            }

            json record = {
                {"set", set_name},
                {"file", base},
                {"duration_seconds", audio.get_duration()},
                {"voice_activity_ratio", va},
                {"max_emotion_prob", max_prob},
                {"emotion", emotion},
                {"acoustic_features", features.toJson()},
                {"results", json::object()},
            };

            for (const auto& [vname, baseline] : baselines) {
                json scored = Audio::add_burnout_analysis(emotion, baseline, &features);
                const json& b = scored.value("burnout_analysis", json::object());

                Row row;
                row.set = set_name;
                row.file = base;
                row.variant = vname;
                row.level = b.value("level", std::string("unknown"));
                row.state = b.value("state", std::string("INSUFFICIENT_DATA"));
                row.top_factor = b.value("top_factor", std::string(""));
                row.risk = b.value("risk", 0.0);
                row.confidence = b.value("confidence", 0.0);
                row.voice_activity = va;
                row.max_prob = max_prob;
                if (b.contains("components") && b["components"].is_object()) {
                    for (const auto& [k, v] : b["components"].items()) {
                        if (v.is_number()) row.components[k] = v.get<double>();
                    }
                }
                if (b.contains("error") && b["error"].is_string()) {
                    row.error = b["error"].get<std::string>();
                }

                record["results"][vname] = b;
                dumpRow(tsv, row);

                Group& g = groups[{set_name, vname}];
                g.n++;
                if (row.state == "INSUFFICIENT_DATA") {
                    ++g.insufficient;
                }
                if (row.level == "low") {
                    ++g.low;
                } else if (row.level == "moderate") {
                    ++g.moderate;
                } else if (row.level == "high") {
                    ++g.high;
                } else if (row.level == "severe") {
                    ++g.severe;
                }
                g.risk_sum += row.risk;
                for (const auto& [k, v] : row.components) {
                    if (v > 0.5) ++g.component_high[k];
                }
            }

            std::ofstream rec(args.out_dir + "/" + set_name + "__" + base + ".json");
            rec << record.dump(2);

            if (idx % 10 == 0) {
                std::cout << "  " << idx << "/" << files.size() << "\n" << std::flush;
            }
        }
    }
    tsv.flush();

    std::ofstream report(args.out_dir + "/report.txt");
    writeReport(std::cout, groups);
    writeReport(report, groups);

    std::cout << "\nsummary: " << tsv_path << "\nreport:  " << args.out_dir << "/report.txt\n";
    return 0;
}
