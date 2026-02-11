"""Smoke tests for fleet-triage-agent.

These tests are intentionally lightweight:
- Validate config.yaml parses and contains expected sections.
- Ensure core modules import without crashing.

This also prevents pytest exit code 5 ("no tests collected") in CI.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_yaml_parses_and_has_expected_sections() -> None:
    cfg_path = REPO_ROOT / "config.yaml"
    assert cfg_path.exists(), "config.yaml missing at repo root"

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)

    for key in ["project", "paths", "simulation", "modeling", "agent", "queue", "reporting", "app"]:
        assert key in cfg, f"Missing top-level key in config.yaml: {key}"


def test_core_modules_import() -> None:
    # Importing should not trigger heavy work; modules should guard main() with __name__ == "__main__".
    import src.data.simulate  # noqa: F401
    import src.data.validate  # noqa: F401
    import src.models.train  # noqa: F401
    import src.reporting.eda_stats  # noqa: F401
    import src.reporting.triage_snapshot  # noqa: F401
    import src.reporting.make_latest_aliases  # noqa: F401
    import src.reporting.make_auto_deck  # noqa: F401