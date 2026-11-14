from datetime import date

from schengen_agent.agent import VisaAgent
from schengen_agent.config import ROOT
from schengen_agent.data import LocalData
from schengen_agent.providers import MockProvider, ModelReply, ToolCall
from schengen_agent.tools import FaultInjector, ToolExecutor


def make_agent(faults=None, steps=6):
    executor = ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14)), FaultInjector(faults))
    return VisaAgent(MockProvider(), executor, steps)


def test_greeting_needs_no_tools():
    result = make_agent().run("Привет!")
    assert result.trace == []
    assert result.steps == 1


def test_multi_tool_agent_flow():
    result = make_agent().run(
        "Франция Москва: какие документы, сколько ждать и есть похожие кейсы?"
    )
    assert [entry["tool"] for entry in result.trace] == [
        "get_visa_requirements",
        "get_visa_statistics",
        "search_visa_cases",
    ]
    assert "синтетические" in result.answer.lower()


def test_agent_handles_tool_failure():
    result = make_agent({"search_visa_cases": ["timeout", "timeout"]}).run(
        "Найди похожие кейсы подачи на визу в Италию"
    )
    assert any(entry["error_type"] == "timeout" for entry in result.trace)
    assert "Не удалось" in result.answer


def test_model_step_limit():
    result = make_agent(steps=1).run("Сколько ждать визу во Францию?")
    assert result.steps == 1
    assert "лимит" in result.answer.lower()


def test_repeated_identical_call_blocked():
    class LoopingProvider:
        def complete(self, messages, tools):
            return ModelReply(calls=[ToolCall("same", "get_visa_statistics", {"country": "france"})])
    executor = ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14)))
    result = VisaAgent(LoopingProvider(), executor, max_steps=4).run("Тест")
    assert len(result.trace) == 2  # third/fourth identical calls not executed
    assert "лимит" in result.answer.lower()


def test_provider_error_is_diagnostic_without_leaking_secrets():
    class FailingProvider:
        def complete(self, messages, tools):
            raise RuntimeError("Authorization SECRET_DO_NOT_LOG")
    executor = ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14)))
    result = VisaAgent(FailingProvider(), executor).run("Тест")
    assert "RuntimeError" in result.answer
    assert "SECRET_DO_NOT_LOG" not in result.answer
    assert result.trace == []


def test_real_model_cannot_omit_synthetic_data_notice():
    """A stub provider deliberately omits the disclaimer."""
    class UnreliableProvider:
        def complete(self, messages, tools):
            if not any(item["role"] == "tool" for item in messages):
                return ModelReply(calls=[
                    ToolCall("call-1", "search_visa_cases", {"country": "france"})
                ])
            return ModelReply(content="Одобрения были. Обычно ожидание 10 дней.")
    executor = ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14)))
    result = VisaAgent(UnreliableProvider(), executor).run(
        "Сколько ждать визу во Францию? Есть похожие случаи?"
    )
    assert result.trace[0]["tool"] == "search_visa_cases"
    assert result.answer.startswith("⚠️ Учебная демонстрация")
    assert "Сводная статистика" in result.answer


def test_greeting_is_not_labeled_with_data_warning():
    class GreetingProvider:
        def complete(self, messages, tools):
            return ModelReply(content="Привет!")
    agent = VisaAgent(GreetingProvider(), ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14))))
    assert agent.run("Привет").answer == "Привет!"



def test_country_only_search_does_not_need_city():
    """A broad query searches all cities; city is optional in Pydantic."""
    executor = ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14)))
    result = executor.execute("search_visa_cases", {"country": "italy"})
    assert "error" not in result
    assert result["matches"] == 12


def test_tool_contract_says_do_not_request_optional_city():
    from schengen_agent.schemas import TOOL_DESCRIPTIONS, SearchVisaCasesArgs
    assert "Do NOT ask the user for the city first" in TOOL_DESCRIPTIONS["search_visa_cases"]
    assert not SearchVisaCasesArgs.model_fields["city"].is_required()
    assert not SearchVisaCasesArgs.model_fields["had_schengen_before"].is_required()



def test_system_prompt_uses_exact_statistical_terms():
    from schengen_agent.agent import SYSTEM_PROMPT
    assert "НАБЛЮДАЕМАЯ ДОЛЯ" in SYSTEM_PROMPT
    assert "НЕ вероятность" in SYSTEM_PROMPT
    assert "МЕДИАНА" in SYSTEM_PROMPT


def test_statistics_are_medians_and_observed_shares():
    from schengen_agent.schemas import VisaStatisticsArgs
    data = LocalData(ROOT / "data", as_of=date(2026, 11, 14))
    result = data.get_visa_statistics(VisaStatisticsArgs(country="france"))
    assert result["case_count"] == 12
    assert result["known_decisions"] == 12
    assert result["median_processing_days"] == 19
    assert result["observed_approval_rate"] == 0.833
    assert result["difficulty_index_0_100"] == 31.3


def test_metric_response_with_no_tools_cannot_hallucinate():
    class HallucinatingProvider:
        def complete(self, messages, tools):
            return ModelReply(content="Во Франции 95% одобрений, срок 7 дней.")

    agent = VisaAgent(HallucinatingProvider(), ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14))))
    answer = agent.run("Сколько ждать и какая статистика одобрений во Франции?").answer
    assert "95%" not in answer
    assert "7 дней" not in answer
    assert "Сводная статистика не получена" in answer


def test_metric_response_is_composed_from_tools_not_llm_hallucinations():
    class MeanMistakeProvider:
        def complete(self, messages, tools):
            if not any(m["role"] == "tool" for m in messages):
                return ModelReply(calls=[ToolCall("id1", "get_visa_statistics", {"country": "france", "city": "moscow"})])
            return ModelReply(content="Средний срок — 7 дней, вероятность одобрения 99%.")

    agent = VisaAgent(MeanMistakeProvider(), ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14))))
    result = agent.run("Сколько ждать визу во Францию, Москва?")
    assert "Медиана ожидания записи: 18,5 дня." in result.answer
    assert "Медиана рассмотрения: 15 дней." in result.answer
    assert "99%" not in result.answer
    assert "Средний срок" not in result.answer
    assert "недостаточно данных" in result.answer
    assert result.trace[0]["tool"] == "get_visa_statistics"


def test_reused_agent_clears_previous_results_between_runs():
    agent = make_agent()
    first = agent.run("Франция: сколько ждать?")
    assert any(t["tool"] == "get_visa_statistics" for t in first.trace)
    second = agent.run("Привет!")
    assert second.trace == []


def test_early_model_final_is_nudged_to_request_missing_statistics():
    """GigaChat may stop after docs+cases; the model must get a recovery turn."""
    class PrematureProvider:
        def __init__(self):
            self.reminder_was_seen = False

        def complete(self, messages, tools):
            called = [m["name"] for m in messages if m["role"] == "tool"]
            if "get_visa_requirements" not in called:
                return ModelReply(calls=[ToolCall("docs", "get_visa_requirements", {"country": "france"})])
            if "search_visa_cases" not in called:
                return ModelReply(calls=[ToolCall("cases", "search_visa_cases", {"country": "france"})])
            if "get_visa_statistics" not in called:
                if "HARNESS: Исходный запрос" not in messages[0]["content"]:
                    return ModelReply(content="Документы и похожие кейсы нашлись.")
                self.reminder_was_seen = True
                assert [t["function"]["name"] for t in tools] == ["get_visa_statistics"]
                return ModelReply(calls=[ToolCall("stats", "get_visa_statistics", {"country": "france"})])
            return ModelReply(content="Сроки примерно 7 дней.")

    provider = PrematureProvider()
    agent = VisaAgent(provider, ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14))))
    result = agent.run("Франция: какие документы, сколько ждать и есть похожие кейсы?")
    assert provider.reminder_was_seen
    assert [event["tool"] for event in result.trace] == [
        "get_visa_requirements", "search_visa_cases", "get_visa_statistics"
    ]
    assert "Медиана рассмотрения: 19 дней" in result.answer
    assert "7 дней" not in result.answer
    assert result.steps == 5


def test_statistics_recovery_is_bounded_when_model_ignores_reminder():
    class NeverCallsTools:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            return ModelReply(content="Одобряют 99% заявок за 7 дней")

    provider = NeverCallsTools()
    agent = VisaAgent(provider, ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14))))
    result = agent.run("Сколько ждать визу во Францию?")
    assert provider.calls == 3  # initial + two recovery attempts, no infinite loop
    assert "99%" not in result.answer
    assert "Сводная статистика не получена" in result.answer


def test_statistics_recovery_skipped_if_statistics_was_already_attempted():
    class UnavailableStatsProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return ModelReply(calls=[ToolCall("stat", "get_visa_statistics", {
                    "country": "france", "city": "ekaterinburg", "period_days": 30,
                })])
            return ModelReply(content="Данных нет.")

    provider = UnavailableStatsProvider()
    agent = VisaAgent(provider, ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14))))
    result = agent.run("Франция Екатеринбург: сколько ждать визу?")
    assert provider.calls == 2
    assert result.trace[0]["tool"] == "get_visa_statistics"
    assert result.trace[0]["error_type"] == "no_data"
    assert "Сводная статистика не получена" in result.answer
