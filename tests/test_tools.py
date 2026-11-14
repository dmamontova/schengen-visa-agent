from datetime import date

"""Deterministic, network-free level-1 and failure-injection tests."""
from pathlib import Path

from schengen_agent.config import ROOT
from schengen_agent.data import LocalData
from schengen_agent.tools import FaultInjector, ToolExecutor


def executor(faults=None):
    return ToolExecutor(LocalData(ROOT / "data", as_of=date(2026, 11, 14)), FaultInjector(faults))


def test_validation_and_unknown_tool():
    service = executor()
    assert service.execute("get_visa_statistics", {"country": "france", "period_days": 999})["error"]["type"] == "validation"
    assert service.execute("get_visa_statistics", {"country": "france", "extra": 42})["error"]["type"] == "validation"
    assert service.execute("transfer_money", {})["error"]["type"] == "unknown_tool"


def test_statistics_are_computed_not_generated():
    result = executor().execute("get_visa_statistics", {"country": "france", "period_days": 90})
    assert result["source_type"] == "synthetic_demo"
    assert result["case_count"] == 12
    assert result["known_decisions"] == 12
    assert 0 <= result["observed_approval_rate"] <= 1
    assert 0 <= result["difficulty_index_0_100"] <= 100


def test_small_sample_has_no_approval_rate_or_index():
    result = executor().execute("get_visa_statistics", {
        "country": "france", "city": "ekaterinburg"
    })
    assert result["case_count"] == 2
    assert result["observed_approval_rate"] is None
    assert result["difficulty_index_0_100"] is None


def test_no_data_and_no_fabricated_links():
    service = executor()
    res = service.execute("search_visa_cases", {
        "country": "spain", "city": "ekaterinburg", "had_schengen_before": False
    })
    assert res["error"]["type"] == "no_data"
    res = service.execute("search_visa_cases", {"country": "italy", "limit": 2})
    assert len(res["cases"]) == 2
    assert all(case["source_url"] is None for case in res["cases"])


def test_timeout_is_returned_not_raised():
    service = executor({"search_visa_cases": ["timeout", "timeout"]})
    res = service.execute("search_visa_cases", {"country": "italy"})
    assert res["error"]["type"] == "timeout"
    assert service.call_log[-1]["error_type"] == "timeout"


def test_one_transient_timeout_is_retried_safely():
    service = executor({"search_visa_cases": ["timeout"]})
    res = service.execute("search_visa_cases", {"country": "italy"})
    assert "cases" in res


def test_missing_data_file_is_data_error_and_not_retried(tmp_path):
    service = ToolExecutor(LocalData(tmp_path, as_of=date(2026, 11, 14)))
    result = service.execute("get_visa_statistics", {"country": "france"})
    assert result["error"]["type"] == "data_error"
    assert service.call_log[-1]["attempts"] == 1


def test_transient_fault_and_retry_count():
    service = executor({"search_visa_cases": ["timeout", "timeout"]})
    result = service.execute("search_visa_cases", {"country": "italy"})
    assert result["error"]["type"] == "timeout"
    assert service.call_log[-1]["attempts"] == 2


def test_date_is_injectable_and_deterministic():
    from schengen_agent.schemas import VisaStatisticsArgs
    from schengen_agent.data import DataUnavailable
    import pytest

    old = LocalData(ROOT / "data", as_of=date(2026, 11, 14))
    assert old.get_visa_statistics(VisaStatisticsArgs(country="france"))["case_count"] == 12
    future = LocalData(ROOT / "data", as_of=date(2027, 4, 1))
    with pytest.raises(DataUnavailable):
        future.get_visa_statistics(VisaStatisticsArgs(country="france"))
