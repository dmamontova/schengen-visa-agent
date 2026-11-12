"""Command line demonstration. Example: python -m schengen_agent.cli --fault-demo."""
from __future__ import annotations

import argparse
import json

from .agent import VisaAgent
from .config import Settings
from .data import LocalData
from .providers import make_provider
from .tools import FaultInjector, ToolExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="Schengen visa agent — synthetic demo")
    parser.add_argument("--question", "-q", default=None, help="Russian-language question")
    parser.add_argument("--fault-demo", action="store_true", help="Simulate two timeouts for case search")
    parser.add_argument("--trace", action="store_true", help="Print sanitized tool trace")
    args = parser.parse_args()
    settings = Settings.from_env()
    faults = FaultInjector(
        {"search_visa_cases": ["timeout", "timeout"]}
        if args.fault_demo else {}
    )
    executor = ToolExecutor(LocalData(settings.data_dir), faults)
    provider = make_provider(settings.provider, settings.model, settings.timeout_sec)
    agent = VisaAgent(provider, executor, settings.max_steps)
    question = args.question or input("Вопрос о визе: ").strip()
    outcome = agent.run(question)
    print("\n" + outcome.answer)
    if args.trace:
        print("\nTRACE:")
        print(json.dumps(outcome.trace, ensure_ascii=False, indent=2))
        print("Model steps:", outcome.steps)


if __name__ == "__main__":
    main()
