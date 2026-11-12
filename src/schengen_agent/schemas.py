"""Pydantic contracts used both as LLM tool schemas and runtime validators."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Country = Literal["france", "italy", "spain"]
City = Literal["moscow", "saint_petersburg", "ekaterinburg"]


class StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisaStatisticsArgs(StrictArguments):
    """Arguments for aggregated analytics on DEMONSTRATION cases."""

    country: Country = Field(description="Country slug: france, italy, spain.")
    city: City | None = Field(
        default=None,
        description="Optional filing city slug. Omit if unspecified: aggregate over ALL cities; do NOT ask for a city.",
    )
    period_days: int = Field(
        default=90, ge=30, le=365,
        description="Lookback period in days (30..365); default 90.",
    )


class VisaRequirementsArgs(StrictArguments):
    """Arguments for reference links and an unverified document checklist."""

    country: Country = Field(description="Country slug: france, italy, spain.")
    city: City | None = Field(
        default=None,
        description="Optional filing city slug. If not given, omit it; do NOT ask the user to clarify.",
    )


class SearchVisaCasesArgs(StrictArguments):
    """Arguments for searching DEMONSTRATION cases, not real Telegram posts."""

    country: Country = Field(description="Country slug: france, italy, spain.")
    city: City | None = Field(
        default=None,
        description="Optional filing city. When absent, search ALL cities in the country. Never ask just to fill this.",
    )
    had_schengen_before: bool | None = Field(
        default=None, description="Optional prior Schengen experience. Omit if unknown; search both values."
    )
    limit: int = Field(default=3, ge=1, le=5, description="Return 1..5 cases.")


TOOL_MODELS = {
    "get_visa_statistics": VisaStatisticsArgs,
    "get_visa_requirements": VisaRequirementsArgs,
    "search_visa_cases": SearchVisaCasesArgs,
}

TOOL_DESCRIPTIONS = {
    "get_visa_statistics": (
        "Calculate counts, medians, observed approval rate (only >=10 known decisions), "
        "and an illustrative difficulty index using SYNTHETIC cases. "
        "Use for waiting times, processing times or comparisons. Do not use to get "
        "official requirements or individual applicant stories. "
        "Accept country slug france/italy/spain, optional filing city and 30..365 days. "
        "Never call this real/current visa statistics."
    ),
    "get_visa_requirements": (
        "Return official portal links and an illustrative UNVERIFIED generic document "
        "checklist for a short-stay tourist visa. Use for questions about documents "
        "and application steps, not for forecasts, statistics or stories. Country-specific "
        "requirements and appointment availability are NOT checked live."
    ),
    "search_visa_cases": (
        "Search locally stored SYNTHETIC applicant examples by country, optional city "
        "and previous-visa experience; returns 1..5 examples and no genuine Telegram links. "
        "If country is specified, call this tool immediately even when city and past visa "
        "history are NOT provided: omit both optional filters and search all cases. "
        "Do NOT ask the user for the city first. Not official rules, "
        "not evidence of current outcomes and not for computing aggregate statistics."
    ),
}


def tool_specifications() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in TOOL_MODELS.items()
    ]
