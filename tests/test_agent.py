from schengen_agent.agent import VisaAgent
from schengen_agent.config import ROOT
from schengen_agent.data import LocalData
from schengen_agent.providers import MockProvider, ModelReply, ToolCall
from schengen_agent.tools import FaultInjector, ToolExecutor


def make_agent(faults=None, steps=6):
    executor = ToolExecutor(LocalData(ROOT / "data"), FaultInjector(faults))
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
    executor = ToolExecutor(LocalData(ROOT / "data"))
    result = VisaAgent(LoopingProvider(), executor, max_steps=4).run("Тест")
    assert len(result.trace) == 2  # third/fourth identical calls not executed
    assert "лимит" in result.answer.lower()
