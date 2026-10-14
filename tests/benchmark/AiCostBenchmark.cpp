#include "AiCostBenchmark.h"

#include <chrono>

AiCostBenchmark::AiCostBenchmark(const Config &config)
    : config_(config)
{
    client_ = std::make_unique<HttpClient>(config_.server_host, config_.server_port, config_.timeout_seconds);
}

AiCostBenchmark::~AiCostBenchmark() = default;

const std::vector<std::string> &AiCostBenchmark::sampleTasks()
{
    static const std::vector<std::string> tasks = {
        "Подготовить презентацию для инвесторов о продукте EmotionAI",
        "Написать письмо начальнику с просьбой об отгуле",
        "Изучить материал по машинному обучению и сделать конспект",
        "Подготовить отчёт за квартал и оформить в Excel",
        "Провести интервью с кандидатом на позицию разработчика",
        "Составить план проекта на ближайший месяц",
        "Перевести документ с английского на русский и выверить терминологию",
        "Подготовить речь для конференции на 10 минут",
        "Разобрать bug в системе авторизации и написать фикс",
        "Составить бюджет на командировку в Москву",
    };
    return tasks;
}

AiCostBenchmark::CostResult AiCostBenchmark::callBreakdown(const std::string &task_text,
                                                            const std::string &duration,
                                                            const std::string &lang)
{
    CostResult result;

    nlohmann::json body = {
        {"text", task_text},
        {"duration", duration},
        {"lang", lang},
    };

    auto response = client_->post("/api/focus/breakdown", body);
    result.status_code = response.status_code;

    if (response.status_code != 200)
        return result;

    try
    {
        auto j = response.json();
        if (!j.contains("usage"))
            return result;

        const auto &u = j["usage"];
        result.prompt_tokens = u.value("prompt_tokens", 0);
        result.completion_tokens = u.value("completion_tokens", 0);
        result.total_tokens = u.value("total_tokens", 0);
        result.ok = true;
    }
    catch (...)
    {
    }

    return result;
}
