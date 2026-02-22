"""Learning profile and few-shot example memory."""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from .config import (
    ENABLE_LEARNING_PRIORS,
    LEARNING_EXAMPLE_LIMIT,
    LEARNING_MAX_EXAMPLES,
    LEARNING_PRIOR_MAX_HINTS,
    LEARNING_PRIOR_MIN_SAMPLES,
    ORGANIZER_OWNER_NAME,
    log,
)
from .constants import (
    COMPANY_VEHICLE_HINTS,
    EMPLOYER_HINTS,
    PRIVATE_VEHICLE_HINTS,
)
from .utils import (
    _content_fingerprint,
    _content_similarity,
    _normalize_tag_name,
    _normalize_text,
    _safe_iso_date,
    _strip_diacritics,
)


class LearningProfile:
    """Persistent lightweight learning profile for jobs/vehicles and stable context."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict = {}
        self.load()

    def _default_data(self) -> dict:
        return {
            "owner": ORGANIZER_OWNER_NAME,
            "jobs": [],
            "vehicles": {
                "private": [],
                "company": [],
            },
            "notes": [],
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.data = raw
                else:
                    self.data = self._default_data()
            except Exception as exc:
                log.debug("Learning-Profil konnte nicht geladen werden: %s", exc)
                self.data = self._default_data()
        else:
            self.data = self._default_data()
            self.save()

        self.data.setdefault("jobs", [])
        self.data.setdefault("vehicles", {})
        self.data["vehicles"].setdefault("private", [])
        self.data["vehicles"].setdefault("company", [])
        self.data.setdefault("notes", [])
        self.data.setdefault("owner", ORGANIZER_OWNER_NAME)
        self.data.setdefault("last_updated", datetime.now().isoformat(timespec="seconds"))

    def save(self):
        with self._lock:
            self.data["last_updated"] = datetime.now().isoformat(timespec="seconds")
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def employer_names(self) -> set[str]:
        names = set()
        for job in self.data.get("jobs", []):
            company = _normalize_text(str(job.get("company", "")))
            if not company:
                continue
            names.add(_normalize_tag_name(company))
        return names

    def private_vehicle_hints(self) -> list[str]:
        hints = list(PRIVATE_VEHICLE_HINTS)
        hints.extend(self.data.get("vehicles", {}).get("private", []))
        return list(dict.fromkeys(_normalize_tag_name(h) for h in hints if _normalize_text(h)))

    def company_vehicle_hints(self) -> list[str]:
        hints = list(COMPANY_VEHICLE_HINTS)
        hints.extend(self.data.get("vehicles", {}).get("company", []))
        return list(dict.fromkeys(_normalize_tag_name(h) for h in hints if _normalize_text(h)))

    def employment_lines(self) -> list[str]:
        jobs = self.data.get("jobs", [])
        normalized = []
        for job in jobs:
            company = _normalize_text(str(job.get("company", "")))
            if not company:
                continue
            start = _safe_iso_date(str(job.get("start", "")))
            end = _safe_iso_date(str(job.get("end", "")))
            if start and end:
                line = f"{start} bis {end}: {company}"
            elif start:
                line = f"seit {start}: {company}"
            else:
                line = company
            normalized.append((start or "0000-00-00", line))
        normalized.sort(key=lambda x: x[0], reverse=True)
        return [line for _, line in normalized[:10]]

    def prompt_context_text(self) -> str:
        owner = _normalize_text(str(self.data.get("owner", ORGANIZER_OWNER_NAME)))
        jobs = "; ".join(self.employment_lines()) or "keine bekannten Beschaeftigungen"
        private_vehicles = ", ".join(self.data.get("vehicles", {}).get("private", [])) or "keine"
        company_vehicles = ", ".join(self.data.get("vehicles", {}).get("company", [])) or "keine"
        notes = ", ".join(_normalize_text(n) for n in self.data.get("notes", []) if _normalize_text(n)) or "keine"
        if len(notes) > 240:
            notes = notes[:240] + "..."
        return (
            f"Person: {owner}.\n"
            f"Beschaeftigungsverlauf: {jobs}.\n"
            f"Privatfahrzeuge: {private_vehicles}.\n"
            f"Firmenfahrzeuge: {company_vehicles}.\n"
            f"Weitere Hinweise: {notes}."
        )

    @staticmethod
    def _extract_vehicle_candidates(text: str) -> list[str]:
        if not text:
            return []
        cleaned = _strip_diacritics(text.lower())
        out = []
        has_model_brand = set()

        pattern_with_model = r"\b(toyota|volkswagen|vw|audi|bmw|mercedes|skoda|ford|renault|opel|seat|tesla|hyundai|kia)\s+([a-z0-9\-]{2,20})\b"
        for m in re.finditer(pattern_with_model, cleaned):
            brand = m.group(1)
            model = m.group(2)
            if brand == "vw":
                brand = "volkswagen"
            has_model_brand.add(brand)
            candidate = _normalize_text(f"{brand} {model}").title()
            if candidate and candidate not in out:
                out.append(candidate)

        pattern_brand_only = r"\b(toyota|volkswagen|vw|audi|bmw|mercedes|skoda|ford|renault|opel|seat|tesla|hyundai|kia)\b"
        for m in re.finditer(pattern_brand_only, cleaned):
            brand = m.group(1)
            if brand == "vw":
                brand = "volkswagen"
            if brand in has_model_brand:
                continue
            candidate = _normalize_text(brand).title()
            if candidate and candidate not in out:
                out.append(candidate)
        return out[:8]

    def _learn_job(self, company: str, start_date: str):
        company_clean = _normalize_text(company)
        if not company_clean:
            return
        jobs = self.data.get("jobs", [])
        wanted_norm = _normalize_tag_name(company_clean)
        existing = None
        for job in jobs:
            if _normalize_tag_name(str(job.get("company", ""))) == wanted_norm:
                existing = job
                break

        if existing:
            if start_date:
                old_start = _safe_iso_date(str(existing.get("start", "")))
                if not old_start or start_date < old_start:
                    existing["start"] = start_date
            if not existing.get("company"):
                existing["company"] = company_clean
            return

        jobs.append({"company": company_clean, "start": start_date, "end": "", "source": "auto"})
        self.data["jobs"] = jobs

        if start_date:
            new_start = datetime.strptime(start_date, "%Y-%m-%d")
            for job in jobs:
                same = _normalize_tag_name(str(job.get("company", ""))) == wanted_norm
                if same:
                    continue
                old_end = _safe_iso_date(str(job.get("end", "")))
                old_start = _safe_iso_date(str(job.get("start", "")))
                if old_end or not old_start:
                    continue
                try:
                    old_start_dt = datetime.strptime(old_start, "%Y-%m-%d")
                except ValueError:
                    continue
                if old_start_dt <= new_start:
                    job["end"] = (new_start - timedelta(days=1)).strftime("%Y-%m-%d")

        if len(jobs) > 25:
            jobs.sort(
                key=lambda j: _safe_iso_date(str(j.get("start", ""))) or "0000-00-00",
                reverse=True,
            )
            self.data["jobs"] = jobs[:25]

    def _learn_vehicle(self, candidates: list[str], company_vehicle: bool):
        if not candidates:
            return
        vehicle_key = "company" if company_vehicle else "private"
        target = self.data.setdefault("vehicles", {}).setdefault(vehicle_key, [])
        known_norm = {_normalize_tag_name(v) for v in target}
        for cand in candidates:
            if _normalize_tag_name(cand) in known_norm:
                continue
            target.append(cand)
            known_norm.add(_normalize_tag_name(cand))

    def learn_from_document(self, document: dict, suggestion: dict):
        text = " ".join([
            str(document.get("title", "")),
            str(document.get("original_file_name", "")),
            str(document.get("content", ""))[:6000],
            str(suggestion.get("title", "")),
            str(suggestion.get("correspondent", "")),
            str(suggestion.get("storage_path", "")),
            " ".join(str(t) for t in (suggestion.get("tags") or [])),
        ]).lower()

        created = _safe_iso_date(str(document.get("created", "")))
        corr = _normalize_text(str(suggestion.get("correspondent", "")))
        path_lower = _normalize_text(str(suggestion.get("storage_path", ""))).lower()
        doc_type_lower = _normalize_text(str(suggestion.get("document_type", ""))).lower()
        tags_lower = {_normalize_tag_name(str(t)) for t in (suggestion.get("tags") or [])}

        employment_keywords = (
            "arbeitsvertrag",
            "arbeitsverhaltnis",
            "arbeitsverhältnis",
            "eintritt",
            "anstellung",
            "arbeitsbeginn",
            "gehaltsabrechnung",
        )
        corr_norm = _normalize_tag_name(corr)
        known_employer = corr_norm in EMPLOYER_HINTS or corr_norm in self.employer_names()
        explicit_employment = any(k in _strip_diacritics(text) for k in employment_keywords)
        strong_doc_type = doc_type_lower in {"vertrag", "gehaltsabrechnung", "zeugnis"}
        has_work_tag = ("arbeit" in tags_lower or "wbs" in tags_lower or "msg" in tags_lower)
        if corr and (
            (known_employer and path_lower.startswith("arbeit/") and (strong_doc_type or has_work_tag))
            or (explicit_employment and (strong_doc_type or path_lower.startswith("arbeit/")))
        ):
            self._learn_job(corr, created)

        vehicle_keywords = ("fahrzeug", "firmenwagen", "dienstwagen", "kfz", "zulassungsbescheinigung", "kennzeichen", "leasing")
        if any(k in _strip_diacritics(text) for k in vehicle_keywords):
            candidates = self._extract_vehicle_candidates(text)
            if candidates:
                company_signal = path_lower.startswith("arbeit/") or ("firmenwagen" in text) or ("dienstwagen" in text)
                private_signal = ("privatwagen" in text) or ("kfz-versicherung" in text)
                if company_signal and not private_signal:
                    self._learn_vehicle(candidates, company_vehicle=True)
                elif private_signal and not company_signal:
                    self._learn_vehicle(candidates, company_vehicle=False)
                else:
                    self._learn_vehicle(candidates, company_vehicle=path_lower.startswith("arbeit/"))


class LearningExamples:
    """Persistent few-shot memory for small models (JSONL)."""

    def __init__(self, path: str, max_examples: int = LEARNING_MAX_EXAMPLES):
        self.path = path
        self.max_examples = max(100, int(max_examples or 100))
        self._lock = threading.Lock()
        self._examples: list[dict] = []
        self.load()

    def _rewrite_file(self):
        """Full rewrite of the JSONL file (called only on compaction)."""
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in self._examples:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self):
        self._examples = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        self._examples.append(row)
            if len(self._examples) > self.max_examples:
                self._examples = self._examples[-self.max_examples:]
        except Exception as exc:
            log.debug("Learning-Beispiele konnten nicht geladen werden: %s", exc)
            self._examples = []

    def validate(self) -> dict:
        """Check integrity of learning data. Returns stats and issues found."""
        stats = {"total": 0, "valid": 0, "invalid_json": 0, "rejected": 0,
                 "missing_fields": 0, "duplicates": 0, "correspondents": set()}
        seen_keys = set()
        issues: list[str] = []
        if not os.path.exists(self.path):
            issues.append("Learning-Datei existiert nicht")
            return {"stats": stats, "issues": issues}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    stats["total"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats["invalid_json"] += 1
                        issues.append(f"Zeile {line_no}: Ungültiges JSON")
                        continue
                    if not isinstance(row, dict):
                        stats["invalid_json"] += 1
                        continue
                    if row.get("rejected"):
                        stats["rejected"] += 1
                    if not row.get("correspondent") and not row.get("document_type"):
                        stats["missing_fields"] += 1
                    key = (row.get("doc_title", ""), row.get("correspondent", ""),
                           row.get("document_type", ""), row.get("storage_path", ""))
                    if key in seen_keys:
                        stats["duplicates"] += 1
                    else:
                        seen_keys.add(key)
                    stats["valid"] += 1
                    if row.get("correspondent"):
                        stats["correspondents"].add(row["correspondent"])
        except Exception as exc:
            issues.append(f"Lesefehler: {exc}")
        stats["correspondents"] = len(stats["correspondents"])
        return {"stats": stats, "issues": issues}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9]{3,}", _strip_diacritics((text or "").lower()))
        stop = {"und", "der", "die", "das", "von", "mit", "fuer", "for", "the", "and", "ein", "eine"}
        return {w for w in words if w not in stop}

    def append(self, document: dict, suggestion: dict, rejected: bool = False):
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "doc_title": _normalize_text(str(document.get("title", ""))),
            "filename": _normalize_text(str(document.get("original_file_name", ""))),
            "correspondent": _normalize_text(str(suggestion.get("correspondent", ""))),
            "document_type": _normalize_text(str(suggestion.get("document_type", ""))),
            "storage_path": _normalize_text(str(suggestion.get("storage_path", ""))),
            "tags": [_normalize_text(str(t)) for t in (suggestion.get("tags") or []) if _normalize_text(str(t))],
        }
        if rejected:
            row["rejected"] = True
        if not row["correspondent"] and not row["document_type"] and not row["storage_path"]:
            return

        with self._lock:
            self._examples.append(row)
            if len(self._examples) > self.max_examples:
                self._examples = self._examples[-self.max_examples:]
                self._rewrite_file()
            else:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def get_rejected_patterns(self, correspondent: str, limit: int = 5) -> list[dict]:
        """Get recent rejected suggestions for a correspondent to avoid repeating mistakes."""
        corr_norm = _normalize_text(correspondent)
        if not corr_norm:
            return []
        rejected = []
        for entry in reversed(self._examples):
            if not entry.get("rejected"):
                continue
            if _normalize_text(str(entry.get("correspondent", ""))) == corr_norm:
                rejected.append(entry)
                if len(rejected) >= limit:
                    break
        return rejected

    def select(self, document: dict, limit: int = LEARNING_EXAMPLE_LIMIT) -> list[dict]:
        if not self._examples:
            return []
        query = " ".join([
            str(document.get("title", "")),
            str(document.get("original_file_name", "")),
            str(document.get("content", ""))[:1200],
        ])
        q_tokens = self._tokens(query)
        if not q_tokens:
            return []

        q_content = (document.get("content") or "")[:2000]
        q_fp = _content_fingerprint(q_content) if len(q_content) > 50 else set()

        scored = []
        for entry in self._examples[-500:]:
            if entry.get("rejected"):
                continue
            text = " ".join([
                str(entry.get("doc_title", "")),
                str(entry.get("filename", "")),
                str(entry.get("correspondent", "")),
                str(entry.get("document_type", "")),
                str(entry.get("storage_path", "")),
                " ".join(entry.get("tags") or []),
            ])
            e_tokens = self._tokens(text)
            overlap = len(q_tokens & e_tokens)
            if overlap <= 0:
                continue
            score = float(overlap)
            if q_fp and entry.get("doc_title"):
                e_text = str(entry.get("doc_title", "")) + " " + str(entry.get("filename", ""))
                e_fp = _content_fingerprint(e_text) if len(e_text) > 10 else set()
                if e_fp:
                    sim = _content_similarity(q_fp, e_fp)
                    score += sim * 3.0
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        unique = []
        seen_keys = set()
        for _, entry in scored:
            key = (
                entry.get("correspondent", ""),
                entry.get("document_type", ""),
                entry.get("storage_path", ""),
                tuple(entry.get("tags") or []),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(entry)
            if len(unique) >= max(0, int(limit)):
                break
        return unique

    @staticmethod
    def _entry_weight(entry: dict) -> float:
        """Age-based weight: recent examples (last 90 days) count 1.0, older decay to 0.5."""
        ts = entry.get("ts", "")
        if not ts:
            return 0.7
        try:
            entry_date = datetime.fromisoformat(ts)
            age_days = (datetime.now() - entry_date).days
            if age_days <= 90:
                return 1.0
            elif age_days <= 180:
                return 0.75
            else:
                return 0.5
        except (ValueError, TypeError):
            return 0.7

    def _build_correspondent_profiles(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for entry in self._examples:
            if entry.get("rejected"):
                continue
            corr = _normalize_text(str(entry.get("correspondent", "")))
            if not corr:
                continue
            key = _normalize_tag_name(corr)
            if not key:
                continue
            weight = self._entry_weight(entry)
            row = grouped.setdefault(
                key,
                {
                    "correspondent": corr,
                    "count": 0,
                    "doc_types": defaultdict(float),
                    "paths": defaultdict(float),
                    "tags": defaultdict(float),
                },
            )
            row["count"] += weight
            doc_type = _normalize_text(str(entry.get("document_type", "")))
            path = _normalize_text(str(entry.get("storage_path", "")))
            if doc_type:
                row["doc_types"][doc_type] += weight
            if path:
                row["paths"][path] += weight
            for t in entry.get("tags") or []:
                tag_name = _normalize_text(str(t))
                if tag_name:
                    row["tags"][tag_name] += weight

        grouped = self._merge_correspondent_variants(grouped)

        profiles = []
        for row in grouped.values():
            total = max(1, int(row["count"]))
            top_doc_type, top_doc_type_count = ("", 0)
            top_path, top_path_count = ("", 0)
            if row["doc_types"]:
                top_doc_type, top_doc_type_count = max(row["doc_types"].items(), key=lambda x: x[1])
            if row["paths"]:
                top_path, top_path_count = max(row["paths"].items(), key=lambda x: x[1])
            top_tags_sorted = sorted(row["tags"].items(), key=lambda x: x[1], reverse=True)[:4]
            tag_ratios = {name: (count / total) for name, count in top_tags_sorted}
            profiles.append(
                {
                    "correspondent": row["correspondent"],
                    "count": total,
                    "top_document_type": top_doc_type,
                    "document_type_ratio": (top_doc_type_count / total) if top_doc_type else 0.0,
                    "top_storage_path": top_path,
                    "storage_path_ratio": (top_path_count / total) if top_path else 0.0,
                    "top_tags": [name for name, _ in top_tags_sorted],
                    "tag_ratios": tag_ratios,
                }
            )
        return profiles

    @staticmethod
    def _merge_correspondent_variants(grouped: dict[str, dict], threshold: float = 0.85) -> dict[str, dict]:
        """Merge correspondent variants with similar names."""
        keys = list(grouped.keys())
        merged_into: dict[str, str] = {}
        for i, key_a in enumerate(keys):
            if key_a in merged_into:
                continue
            for key_b in keys[i + 1:]:
                if key_b in merged_into:
                    continue
                ratio = difflib.SequenceMatcher(None, key_a, key_b).ratio()
                if ratio < threshold:
                    continue
                row_a = grouped[key_a]
                row_b = grouped[key_b]
                if row_a["count"] >= row_b["count"]:
                    canonical_key, absorbed_key = key_a, key_b
                else:
                    canonical_key, absorbed_key = key_b, key_a
                canon = grouped[canonical_key]
                absorbed = grouped[absorbed_key]
                canon["count"] += absorbed["count"]
                for dt, cnt in absorbed["doc_types"].items():
                    canon["doc_types"][dt] += cnt
                for p, cnt in absorbed["paths"].items():
                    canon["paths"][p] += cnt
                for t, cnt in absorbed["tags"].items():
                    canon["tags"][t] += cnt
                merged_into[absorbed_key] = canonical_key
        for absorbed_key in merged_into:
            grouped.pop(absorbed_key, None)
        return grouped

    def routing_hints_for_document(self, document: dict, limit: int = LEARNING_PRIOR_MAX_HINTS) -> list[dict]:
        """Learns stable routing priors from confirmed examples."""
        if not ENABLE_LEARNING_PRIORS:
            return []
        if not self._examples:
            return []

        query = " ".join([
            str(document.get("title", "")),
            str(document.get("original_file_name", "")),
            str(document.get("content", ""))[:2200],
        ])
        query_norm = _normalize_tag_name(query)
        q_tokens = self._tokens(query)
        if not query_norm and not q_tokens:
            return []

        profiles = self._build_correspondent_profiles()
        scored = []
        for profile in profiles:
            count = int(profile.get("count", 0) or 0)
            if count < LEARNING_PRIOR_MIN_SAMPLES:
                continue
            corr = str(profile.get("correspondent", ""))
            corr_norm = _normalize_tag_name(corr)
            corr_tokens = self._tokens(corr)
            overlap = len(q_tokens & corr_tokens)
            exact = 1 if (corr_norm and corr_norm in query_norm) else 0
            if overlap <= 0 and exact == 0:
                continue
            score = float(overlap + (3 * exact)) + min(count, 20) * 0.08
            scored.append((score, profile))

        scored.sort(key=lambda x: x[0], reverse=True)
        unique = []
        seen = set()
        for _, profile in scored:
            corr = str(profile.get("correspondent", ""))
            corr_norm = _normalize_tag_name(corr)
            if not corr_norm or corr_norm in seen:
                continue
            seen.add(corr_norm)
            unique.append(profile)
            if len(unique) >= max(0, int(limit)):
                break
        return unique


# --- German-Character Detection ---
GERMAN_CHARS_RE = re.compile(r'[aeoeueAeOeUess]')
