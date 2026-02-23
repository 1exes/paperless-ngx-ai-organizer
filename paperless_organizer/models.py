"""Data models and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import PaperlessClient
    from .db import LocalStateDB
    from .knowledge import KnowledgeDB
    from .learning import LearningExamples, LearningProfile
    from .llm import LocalLLMAnalyzer
    from .taxonomy import TagTaxonomy


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
    knowledge_context_text: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessingContext:
    """Shared context for document processing, replacing 12+ threaded parameters."""
    paperless: PaperlessClient
    analyzer: LocalLLMAnalyzer
    tags: list
    correspondents: list
    doc_types: list
    storage_paths: list
    taxonomy: TagTaxonomy | None = None
    decision_context: DecisionContext | None = None
    learning_profile: LearningProfile | None = None
    learning_examples: LearningExamples | None = None
    knowledge_db: KnowledgeDB | None = None
    run_db: LocalStateDB | None = None
    run_id: int | None = None

    def copy_master_data(self) -> ProcessingContext:
        """Return a copy with independent master-data lists (thread-safe)."""
        return ProcessingContext(
            paperless=self.paperless,
            analyzer=self.analyzer,
            tags=list(self.tags),
            correspondents=list(self.correspondents),
            doc_types=list(self.doc_types),
            storage_paths=list(self.storage_paths),
            taxonomy=self.taxonomy,
            decision_context=self.decision_context,
            learning_profile=self.learning_profile,
            learning_examples=self.learning_examples,
            knowledge_db=self.knowledge_db,
            run_db=self.run_db,
            run_id=self.run_id,
        )
