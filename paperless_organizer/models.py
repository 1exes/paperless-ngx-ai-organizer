"""Data models and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionContext:
    """Collected runtime context for better document decisions."""
    employer_names: set[str] = field(default_factory=set)
    provider_names: set[str] = field(default_factory=set)
    top_work_paths: list[str] = field(default_factory=list)
    top_private_paths: list[str] = field(default_factory=list)
    profile_employment_lines: list[str] = field(default_factory=list)
    profile_private_vehicles: list[str] = field(default_factory=list)
    profile_company_vehicles: list[str] = field(default_factory=list)
    profile_context_text: str = ""
    notes: list[str] = field(default_factory=list)
