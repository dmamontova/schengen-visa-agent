"""Read-only local data operations. All visa cases in this repository are synthetic."""
from __future__ import annotations

import json
import statistics
from datetime import date, timedelta
from pathlib import Path

from .schemas import SearchVisaCasesArgs, VisaRequirementsArgs, VisaStatisticsArgs


class DataUnavailable(Exception):
    """No observations match the supplied filters."""


class LocalData:
    def __init__(self, data_dir: Path, *, as_of: date | None = None):
        self.data_dir = Path(data_dir)
        # Inject the clock in tests; production continues to use the current day.
        self.as_of = as_of

    def _load(self, filename: str):
        with (self.data_dir / filename).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _filter(self, country: str, city: str | None = None):
        cases = self._load("cases.json")
        return [
            case for case in cases
            if case["country"] == country and (city is None or case["city"] == city)
        ]

    def get_visa_statistics(self, args: VisaStatisticsArgs) -> dict:
        today = self.as_of or date.today()
        cutoff = today - timedelta(days=args.period_days)
        records = [
            c for c in self._filter(args.country, args.city)
            if cutoff <= date.fromisoformat(c["case_date"]) <= today
        ]
        if not records:
            raise DataUnavailable("За выбранный период нет демонстрационных кейсов.")
        wait = [c["appointment_wait_days"] for c in records if c["appointment_wait_days"] is not None]
        process = [c["processing_days"] for c in records if c["processing_days"] is not None]
        known = [c for c in records if c["decision"] in {"approved", "refused"}]
        approval_rate = (
            round(sum(c["decision"] == "approved" for c in known) / len(known), 3)
            if len(known) >= 10 else None
        )
        median_wait = statistics.median(wait) if wait else None
        median_process = statistics.median(process) if process else None
        # Non-official illustrative index, computed only when all components available.
        difficulty = None
        if median_wait is not None and median_process is not None and approval_rate is not None:
            difficulty = round(
                100 * (
                    0.40 * min(median_wait / 60, 1)
                    + 0.35 * min(median_process / 45, 1)
                    + 0.25 * (1 - approval_rate)
                ),
                1,
            )
        return {
            "source_type": "synthetic_demo",
            "warning": "Синтетические данные: НЕ реальные сроки/доли одобрений.",
            "country": args.country,
            "city": args.city,
            "period_days": args.period_days,
            "case_count": len(records),
            "known_decisions": len(known),
            "median_appointment_wait_days": median_wait,
            "median_processing_days": median_process,
            "observed_approval_rate": approval_rate,
            "difficulty_index_0_100": difficulty,
            "index_formula": "100*(0.40*min(wait/60,1)+0.35*min(process/45,1)+0.25*refusal_rate)",
            "sample_rule": "Approval rate and index hidden if <10 known decisions.",
        }

    def get_visa_requirements(self, args: VisaRequirementsArgs) -> dict:
        item = self._load("requirements.json").get(args.country)
        if not item:
            raise DataUnavailable("Не найдена карточка страны.")
        return {
            "country": args.country,
            "city": args.city,
            "status": "unverified_reference_only",
            "warning": (
                "Общий перечень для ориентира; НЕ проверенный актуальный список "
                "документов для граждан РФ и конкретного консульства."
            ),
            "illustrative_documents": item["illustrative_documents"],
            "official_portal": item["official_portal"],
            "general_eu_guidance": "https://home-affairs.ec.europa.eu/policies/schengen/visa-policy/applying-schengen-visa_en",
            "requirements_verified_at": None,
            "hint": "Уточните требования и способ подачи непосредственно на официальном портале.",
        }

    def search_visa_cases(self, args: SearchVisaCasesArgs) -> dict:
        records = self._filter(args.country, args.city)
        if args.had_schengen_before is not None:
            records = [c for c in records if c["had_schengen_before"] == args.had_schengen_before]
        if not records:
            raise DataUnavailable("Нет похожих демонстрационных кейсов по этим фильтрам.")
        records.sort(key=lambda c: (c["case_date"], c["case_id"]), reverse=True)
        return {
            "source_type": "synthetic_demo",
            "warning": "Примеры вымышлены. Ссылок на Telegram нет.",
            "matches": len(records),
            "cases": [
                {
                    "case_id": c["case_id"],
                    "country": c["country"],
                    "city": c["city"],
                    "case_date": c["case_date"],
                    "had_schengen_before": c["had_schengen_before"],
                    "decision": c["decision"],
                    "processing_days": c["processing_days"],
                    "summary": c["summary"][:200],
                    "source_url": None,
                }
                for c in records[:args.limit]
            ],
        }
