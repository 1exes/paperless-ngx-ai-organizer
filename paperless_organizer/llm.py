"""Local LLM analyzer for document classification."""

from __future__ import annotations

import json
import random
import re
import time
from urllib.parse import urlparse, urlunparse

import requests

from . import config as _cfg
from .config import (
    LEARNING_EXAMPLE_LIMIT,
    LEARNING_PRIOR_MAX_HINTS,
    LLM_API_KEY,
    LLM_COMPACT_MAX_TOKENS,
    LLM_COMPACT_PROMPT_MAX_PATHS,
    LLM_COMPACT_PROMPT_MAX_TAGS,
    LLM_COMPACT_TIMEOUT,
    LLM_CONNECT_TIMEOUT,
    LLM_FALLBACK_AFTER_ERRORS,
    LLM_FALLBACK_MODEL,
    LLM_KEEP_ALIVE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_COUNT,
    LLM_SPEEDCHECK_TIMEOUT,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_URL,
    LLM_VERIFY_ON_LOW_CONFIDENCE,
    MAX_CONTEXT_EMPLOYER_HINTS,
    MAX_PROMPT_TAG_CHOICES,
    MAX_TAGS_PER_DOC,
    console,
    log,
)
from .constants import ALLOWED_DOC_TYPES, KNOWN_BRAND_HINTS
from .utils import _get_brand_hint
from .web_hints import (
    _collect_web_entity_hints,
    _fetch_web_hint,
    _web_search_entity,
)


class LocalLLMAnalyzer:
    """Analysiert Dokumente mit lokalem LLM-Endpunkt."""

    def __init__(self, url: str = LLM_URL, model: str = LLM_MODEL):
        self.url = self._normalize_url(url)
        self.model = (model or "").strip()
        self._original_model = self.model
        self._response_times: list[float] = []
        self._max_tracked = 50
        self._consecutive_errors = 0
        self._using_fallback = False
        self._prompt_cache: dict[str, str] = {}
        self._prompt_cache_key: str = ""

    def _check_model_fallback(self):
        """Switch to fallback model after too many consecutive errors."""
        if (
            LLM_FALLBACK_MODEL
            and not self._using_fallback
            and self._consecutive_errors >= LLM_FALLBACK_AFTER_ERRORS
        ):
            log.warning(
                f"[yellow]LLM-Modell-Fallback[/yellow]: {self.model} -> {LLM_FALLBACK_MODEL} "
                f"(nach {self._consecutive_errors} Fehlern)"
            )
            self.model = LLM_FALLBACK_MODEL
            self._using_fallback = True
            self._consecutive_errors = 0

    def _record_success(self):
        """Reset error counter on success; revert to original model after 10 consecutive successes."""
        self._consecutive_errors = 0
        if self._using_fallback:
            self._response_times_since_fallback = getattr(self, "_response_times_since_fallback", 0) + 1
            if self._response_times_since_fallback >= 10:
                log.info(f"[green]LLM-Modell zurueck auf Original[/green]: {LLM_FALLBACK_MODEL} -> {self._original_model}")
                self.model = self._original_model
                self._using_fallback = False
                self._response_times_since_fallback = 0

    def _record_error(self):
        """Track consecutive errors for fallback decision."""
        self._consecutive_errors += 1
        self._check_model_fallback()

    @property
    def avg_response_time(self) -> float:
        return sum(self._response_times) / len(self._response_times) if self._response_times else 0.0

    @property
    def p95_response_time(self) -> float:
        if not self._response_times:
            return 0.0
        sorted_times = sorted(self._response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse((url or "").strip())
        path = (parsed.path or "").strip()
        if path in ("", "/"):
            if parsed.port == 11434:
                parsed = parsed._replace(path="/api/chat")
            else:
                parsed = parsed._replace(path="/v1/chat/completions")
            return urlunparse(parsed)
        return url

    def _auth_headers(self) -> dict | None:
        return {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else None

    def _candidate_model_urls(self) -> list[str]:
        candidates: list[str] = []
        if "/v1/chat/completions" in self.url:
            candidates.append(self.url.replace("/v1/chat/completions", "/v1/models"))
        if "/api/v1/chat" in self.url:
            candidates.append(self.url.replace("/api/v1/chat", "/v1/models"))
            candidates.append(self.url.replace("/api/v1/chat", "/api/v1/models"))
        if "/api/chat" in self.url:
            candidates.append(self.url.replace("/api/chat", "/api/tags"))
            candidates.append(self.url.replace("/api/chat", "/v1/models"))
        return list(dict.fromkeys(candidates))

    # Model name patterns that indicate non-chat models (embedding, TTS, vision-only, etc.)
    _NON_CHAT_MODEL_PATTERNS = (
        "embed", "embedding", "tts", "whisper", "rerank",
        "text-embedding", "nomic-embed", "bge-", "e5-",
    )

    @classmethod
    def _is_chat_model(cls, model_id: str) -> bool:
        """Return True if model_id looks like a chat/completion model (not embedding/TTS/etc)."""
        lower = model_id.lower()
        return not any(pat in lower for pat in cls._NON_CHAT_MODEL_PATTERNS)

    def discover_available_models(self, chat_only: bool = True) -> list[str]:
        """Holt alle verfuegbaren Modelle vom Server (LM Studio + Ollama).

        Args:
            chat_only: If True, filters out embedding/TTS/non-chat models.
        """
        headers = self._auth_headers()
        discovered: list[str] = []
        for test_url in self._candidate_model_urls():
            try:
                resp = requests.get(test_url, headers=headers, timeout=5)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                models = data.get("data")
                if not isinstance(models, list):
                    models = data.get("models")
                if not isinstance(models, list):
                    continue
                for entry in models:
                    if not isinstance(entry, dict):
                        continue
                    model_id = entry.get("id") or entry.get("name") or entry.get("key") or entry.get("model")
                    if isinstance(model_id, str) and model_id.strip():
                        discovered.append(model_id.strip())
            except (requests.exceptions.RequestException, ValueError):
                continue
        unique = list(dict.fromkeys(discovered))
        if chat_only:
            unique = [m for m in unique if self._is_chat_model(m)]
        return unique

    def _discover_model(self, headers: dict | None) -> str:
        preferred = [
            "qwen2.5:14b", "qwen2.5:7b", "qwen3:8b",
            "qwen2.5-coder:14b", "qwen2.5-coder:7b",
            "gemma3:12b", "gemma3:4b",
            "google/gemma-3-12b", "google/gemma-3-4b", "gemma3:latest",
            "llama3.1:8b", "mistral:7b", "phi-4:14b", "llama3.2:3b",
        ]
        unique = self.discover_available_models()
        if not unique:
            return ""
        lower_map = {m.lower(): m for m in unique}
        for pref in preferred:
            hit = lower_map.get(pref.lower())
            if hit:
                return hit
        return unique[0]

    def verify_connection(self) -> bool:
        """Prueft ob der konfigurierte LLM-Server erreichbar ist."""
        log.info(f"Teste LLM-Verbindung: {self.url}")
        headers = self._auth_headers()
        try:
            for test_url in self._candidate_model_urls():
                resp = requests.get(test_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    if not self.model:
                        self.model = self._discover_model(headers)
                    model_label = self.model or "Server-Default"
                    log.info(f"[green]LLM verbunden[/green] - Modell: {model_label}")
                    return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException:
            pass
        try:
            probe = requests.post(
                self.url,
                headers=headers,
                json=self._build_payload("ping"),
                timeout=5,
            )
            if probe.status_code < 500 and probe.status_code != 404:
                if not self.model:
                    self.model = self._discover_model(headers)
                model_label = self.model or "Server-Default"
                log.info(f"[green]LLM-Endpunkt erreichbar[/green] - Modell: {model_label}")
                return True
        except requests.exceptions.RequestException:
            pass
        log.error(f"LLM-Server nicht erreichbar! ({self.url})")
        console.print("[yellow]Bitte LLM-Server starten und Modell laden.[/yellow]")
        return False

    def speedcheck(self, timeout: int | None = None) -> dict:
        """Sendet Mini-Klassifikations-Prompt, misst Antwortzeit.

        Returns dict with keys: ok, response_time, model, error
        """
        effective_timeout = timeout if timeout is not None else LLM_SPEEDCHECK_TIMEOUT
        test_prompt = (
            'Klassifiziere: "Rechnung Telekom Februar 2025"\n'
            'Antwort NUR JSON: {"document_type": "...", "correspondent": "..."}'
        )
        headers = self._auth_headers()
        payload = self._build_payload(test_prompt, max_tokens=60)
        model_label = self.model or "(server-default)"
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=(max(3, LLM_CONNECT_TIMEOUT), effective_timeout),
            )
            elapsed = time.perf_counter() - t0
            if resp.ok:
                # Validate response contains actual text (not embedding vectors)
                try:
                    text = self._extract_response_text(resp.json())
                    if not text or len(text) < 3:
                        return {
                            "ok": False, "response_time": round(elapsed, 2), "model": model_label,
                            "error": "Leere Antwort (kein Chat-Modell?)",
                        }
                except (ValueError, RuntimeError):
                    return {
                        "ok": False, "response_time": round(elapsed, 2), "model": model_label,
                        "error": "Kein Text in Antwort (kein Chat-Modell?)",
                    }
                return {"ok": True, "response_time": round(elapsed, 2), "model": model_label, "error": None}
            return {
                "ok": False, "response_time": round(elapsed, 2), "model": model_label,
                "error": f"HTTP {resp.status_code}",
            }
        except requests.exceptions.Timeout:
            elapsed = time.perf_counter() - t0
            return {
                "ok": False, "response_time": round(elapsed, 2), "model": model_label,
                "error": f"Timeout (>{effective_timeout}s)",
            }
        except requests.exceptions.RequestException as exc:
            return {"ok": False, "response_time": 0.0, "model": model_label, "error": str(exc)}

    def benchmark_available_models(
        self,
        max_acceptable_time: float = 10.0,
        timeout_per_model: int | None = None,
    ) -> dict:
        """Benchmarkt aktuelles + alle anderen Modelle, empfiehlt schnellstes.

        Returns dict with keys: results (list of dicts), best_model, best_time,
        current_model, current_ok
        """
        effective_timeout = timeout_per_model if timeout_per_model is not None else LLM_SPEEDCHECK_TIMEOUT
        available = self.discover_available_models()
        current_model = self.model or ""

        # Ensure current model is tested first
        models_to_test = []
        if current_model and current_model in available:
            models_to_test.append(current_model)
            models_to_test.extend(m for m in available if m != current_model)
        elif current_model:
            models_to_test.append(current_model)
            models_to_test.extend(available)
        else:
            models_to_test = list(available)

        results: list[dict] = []
        original_model = self.model
        try:
            for model_name in models_to_test:
                self.model = model_name
                result = self.speedcheck(timeout=effective_timeout)
                result["is_current"] = (model_name == current_model)
                results.append(result)
                log.info(
                    "  Speedcheck %s: %s (%.1fs)",
                    model_name,
                    "OK" if result["ok"] else result["error"],
                    result["response_time"],
                )
        finally:
            self.model = original_model

        # Find best model
        ok_results = [r for r in results if r["ok"]]
        if ok_results:
            best = min(ok_results, key=lambda r: r["response_time"])
        else:
            best = {"model": current_model, "response_time": 0.0}

        current_result = next((r for r in results if r.get("is_current")), None)
        return {
            "results": results,
            "best_model": best["model"],
            "best_time": best["response_time"],
            "current_model": current_model or "(server-default)",
            "current_ok": current_result["ok"] if current_result else False,
            "current_time": current_result["response_time"] if current_result else 0.0,
            "max_acceptable_time": max_acceptable_time,
        }

    def _parse_json_response(self, text: str) -> dict:
        """JSON aus LLM-Antwort extrahieren."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)

    def analyze(self, document: dict, existing_tags: list,
                existing_correspondents: list, existing_types: list,
                existing_paths: list, taxonomy=None,
                decision_context=None,
                few_shot_examples: list[dict] | None = None,
                learning_hints: list[dict] | None = None,
                compact_mode: bool = False,
                read_timeout_override: int | None = None,
                web_hint_override: str | None = None,
                doc_language: str = "de") -> dict:
        """Analysiert ein Dokument und gibt Organisationsvorschlag zurueck."""
        from .config import ENFORCE_TAG_TAXONOMY

        cache_key = f"{len(existing_correspondents)}:{len(existing_tags)}:{len(existing_paths)}:{compact_mode}"
        if cache_key != self._prompt_cache_key:
            self._prompt_cache.clear()
            self._prompt_cache_key = cache_key

        if "corr_names" in self._prompt_cache:
            corr_names = json.loads(self._prompt_cache["corr_names"])
            tag_choices = json.loads(self._prompt_cache["tag_choices"])
        else:
            top_corrs = sorted(existing_correspondents, key=lambda c: c.get("document_count", 0), reverse=True)
            max_corr_choices = 12 if compact_mode else 35
            corr_names = [c["name"] for c in top_corrs[:max_corr_choices]]
            if taxonomy and ENFORCE_TAG_TAXONOMY and taxonomy.canonical_tags:
                if compact_mode:
                    max_tag_choices = max(8, min(MAX_PROMPT_TAG_CHOICES, LLM_COMPACT_PROMPT_MAX_TAGS))
                else:
                    max_tag_choices = min(MAX_PROMPT_TAG_CHOICES, 90)
                tag_choices = taxonomy.prompt_tags(max_tag_choices)
            else:
                top_tags = sorted(existing_tags, key=lambda t: t.get("document_count", 0), reverse=True)
                if compact_mode:
                    max_tag_choices = max(8, min(MAX_PROMPT_TAG_CHOICES, LLM_COMPACT_PROMPT_MAX_TAGS))
                else:
                    max_tag_choices = min(MAX_PROMPT_TAG_CHOICES, 90)
                tag_choices = [t["name"] for t in top_tags[:max_tag_choices]]
            self._prompt_cache["corr_names"] = json.dumps(corr_names, ensure_ascii=False)
            self._prompt_cache["tag_choices"] = json.dumps(tag_choices, ensure_ascii=False)

        current_tags = [
            t["name"] for t in existing_tags
            if t["id"] in (document.get("tags") or [])
        ]
        current_corr = next(
            (c["name"] for c in existing_correspondents
             if c["id"] == document.get("correspondent")), "")
        current_type = next(
            (t["name"] for t in existing_types
             if t["id"] == document.get("document_type")), "")
        current_path = next(
            (p["name"] for p in existing_paths
             if p["id"] == document.get("storage_path")), "")

        content = document.get("content") or ""
        content_len = len(content)
        if compact_mode:
            if content_len > 1200:
                head = content[:400]
                tail = content[-200:] if content_len > 800 else ""
                mid_snippet = ""
                if content_len > 1600:
                    mid_start = content_len // 3
                    mid_block = content[mid_start:mid_start + 400]
                    if re.search(r"(rechnungs|invoice|betrag|amount|iban|datum|date)", mid_block, re.IGNORECASE):
                        mid_snippet = f"\n[Mitte:] {mid_block[:200]}\n"
                content_preview = head + mid_snippet + f"\n[...{content_len} Zeichen, Kompaktmodus...]\n" + tail
            else:
                content_preview = content[:600]
        else:
            if content_len > 5000:
                head = content[:1000]
                tail = content[-400:]
                mid_snippet = ""
                mid_start = content_len // 3
                mid_block = content[mid_start:mid_start + 600]
                if re.search(r"(rechnungs|invoice|betrag|amount|iban|datum|date|vertrag|contract)", mid_block, re.IGNORECASE):
                    mid_snippet = f"\n[Mitte:] {mid_block[:300]}\n"
                content_preview = head + mid_snippet + f"\n[...{content_len} Zeichen insgesamt...]\n" + tail
            elif content_len > 2000:
                content_preview = content[:1200] + f"\n[...{content_len} Zeichen...]\n" + content[-300:]
            else:
                content_preview = content

        brand_hint = _get_brand_hint(f"{document.get('title', '')} {document.get('original_file_name', '')} {content_preview[:1000]}")
        if web_hint_override:
            web_hint = web_hint_override
        elif _cfg.ENABLE_WEB_HINTS and not compact_mode:
            web_hint_primary = _fetch_web_hint(document.get("title", ""), content_preview)
            web_hint_entities = _collect_web_entity_hints(document, current_corr=current_corr)
            web_search_corr = ""
            if current_corr and current_corr not in KNOWN_BRAND_HINTS:
                web_search_corr = _web_search_entity(current_corr)
            web_hint = " | ".join([h for h in [web_hint_primary, web_hint_entities, web_search_corr] if h])
        else:
            web_hint = ""

        if decision_context:
            employer_list = sorted(decision_context.employer_names)[:max(1, MAX_CONTEXT_EMPLOYER_HINTS)]
            provider_list = sorted(decision_context.provider_names)[:10]
            employers_info = ", ".join(employer_list) if employer_list else "keine"
            providers_info = ", ".join(provider_list) if provider_list else "keine"
        else:
            employers_info = "keine"
            providers_info = "keine"
        work_paths_info = ", ".join(decision_context.top_work_paths) if decision_context else "keine"
        private_paths_info = ", ".join(decision_context.top_private_paths) if decision_context else "keine"
        # Prefer knowledge DB context over legacy profile context
        if decision_context and decision_context.knowledge_context_text:
            profile_context = decision_context.knowledge_context_text
        elif decision_context and decision_context.profile_context_text:
            profile_context = decision_context.profile_context_text
        else:
            profile_context = "Kein Profil-Kontext verfuegbar."
        examples_text = "keine"
        if few_shot_examples and not compact_mode:
            lines = []
            for idx, ex in enumerate(few_shot_examples[:LEARNING_EXAMPLE_LIMIT], 1):
                lines.append(
                    f"{idx}) Titel~{ex.get('doc_title', '?')} -> "
                    f"Korr={ex.get('correspondent', '?')}, "
                    f"Typ={ex.get('document_type', '?')}, "
                    f"Pfad={ex.get('storage_path', '?')}, "
                    f"Tags={', '.join(ex.get('tags') or []) or 'keine'}"
                )
            examples_text = "\n".join(lines)

        learning_hint_text = "keine"
        if learning_hints and not compact_mode:
            lines = []
            for idx, hint in enumerate(learning_hints[:LEARNING_PRIOR_MAX_HINTS], 1):
                corr = hint.get("correspondent", "?")
                count = int(hint.get("count", 0) or 0)
                top_type = hint.get("top_document_type", "") or "?"
                top_type_ratio = float(hint.get("document_type_ratio", 0.0) or 0.0)
                top_path = hint.get("top_storage_path", "") or "?"
                top_path_ratio = float(hint.get("storage_path_ratio", 0.0) or 0.0)
                top_tags = hint.get("top_tags") or []
                lines.append(
                    f"{idx}) {corr} (n={count}) -> Typ {top_type} ({top_type_ratio:.0%}), "
                    f"Pfad {top_path} ({top_path_ratio:.0%}), Tags {', '.join(top_tags) or 'keine'}"
                )
            learning_hint_text = "\n".join(lines)

        path_names = [p["name"] for p in existing_paths if "Duplikat" not in p["name"]]
        if compact_mode:
            path_names = path_names[:max(8, LLM_COMPACT_PROMPT_MAX_PATHS)]
            if doc_language == "en":
                prompt = f"""Paperless-NGX: Classify and assign document.
DOCUMENT #{document['id']}:
Title: {document.get('title', '?')} | File: {document.get('original_file_name', '?')}
Current: Tags={current_tags or 'none'} | Corr={current_corr or 'none'} | Type={current_type or 'none'} | Path={current_path or 'none'}
{f'WEB CONTEXT: {web_hint}' if web_hint else ''}

CONTENT (EXCERPT):
{content_preview}

ALLOWED CORRESPONDENTS: {', '.join(corr_names)}
ALLOWED DOCUMENT TYPES: {', '.join(ALLOWED_DOC_TYPES)}
ALLOWED STORAGE PATHS: {', '.join(path_names)}
ALLOWED TAGS: {', '.join(tag_choices)}

FIELDS (all required):
- title: Descriptive German title (e.g. "Rechnung Telekom Februar 2025")
- correspondent: The SPECIFIC sender (company/authority) from letterhead, MUST be from the list
- document_type: Document type, MUST be from the list
- storage_path: Category path EXACTLY from the list
- tags: 1-{MAX_TAGS_PER_DOC} tags ONLY from the list
- confidence: "high", "medium", or "low"
- reasoning: Brief explanation

JSON ONLY:
{{"title": "Title", "tags": ["Tag1"], "correspondent": "Company", "document_type": "Type", "storage_path_name": "Category", "storage_path": "Category/Sub", "confidence": "high", "reasoning": "Brief"}}"""
            else:
                prompt = f"""Paperless-NGX: Dokument klassifizieren und zuordnen.
DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')}
Aktuell: Tags={current_tags or 'keine'} | Korr={current_corr or 'keiner'} | Typ={current_type or 'keiner'} | Pfad={current_path or 'keiner'}
{f'WEB-KONTEXT: {web_hint}' if web_hint else ''}

INHALT (AUSZUG):
{content_preview}

ERLAUBTE KORRESPONDENTEN: {', '.join(corr_names)}
ERLAUBTE DOKUMENTTYPEN: {', '.join(ALLOWED_DOC_TYPES)}
ERLAUBTE SPEICHERPFADE: {', '.join(path_names)}
ERLAUBTE TAGS: {', '.join(tag_choices)}

FELDER (alle Pflichtfelder):
- title: Aussagekraeftiger deutscher Titel (z.B. "Rechnung Telekom Februar 2025")
- correspondent: Der KONKRETE Absender (Firma/Behoerde) aus dem Briefkopf, MUSS aus der Liste sein
- document_type: Dokumenttyp (z.B. Rechnung, Kontoauszug, Vertrag), MUSS aus der Liste sein
- storage_path: Kategorie-Pfad EXAKT aus der Liste (z.B. "Finanzen/Bank")
- tags: 1-{MAX_TAGS_PER_DOC} Tags NUR aus der Liste, keine neuen erfinden
- confidence: "high" (sicher), "medium" (wahrscheinlich), "low" (unsicher)
- reasoning: Kurze Begruendung

NUR JSON:
{{"title": "Titel", "tags": ["Tag1"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""
        else:
            prompt = f"""Paperless-NGX Dokument organisieren.
{profile_context}

DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')} | Erstellt: {document.get('created', '?')}
Tags: {current_tags or 'keine'} | Korr: {current_corr or 'keiner'} | Typ: {current_type or 'keiner'} | Pfad: {current_path or 'keiner'}

INHALT:
{content_preview}

KORRESPONDENTEN (bevorzuge vorhandene): {', '.join(corr_names)}

DOKUMENTTYPEN (NUR diese): {', '.join(ALLOWED_DOC_TYPES)}

SPEICHERPFADE (NUR diese Kategorienamen verwenden, NICHT den vollen Pfad mit Jahr/Titel!):
{', '.join(path_names)}
WICHTIG: storage_path muss EXAKT einem der obigen Namen entsprechen, z.B. "Auto/Unfall" oder "Finanzen/Bank"

ERLAUBTE TAGS (NUR aus dieser Liste waehlen, keine neuen erfinden):
{', '.join(tag_choices)}
WICHTIG: maximal {MAX_TAGS_PER_DOC} Tags und nur aus der obigen Liste.

BRAND-HINWEIS: {brand_hint or 'kein spezieller Hinweis'}
WEB-HINWEIS: {web_hint or 'kein externer Treffer'}
KONTEXT (vorher gesammelt):
- erkannte Arbeitgeber im Bestand: {employers_info}
- erkannte externe Anbieter: {providers_info}
- haeufige Arbeitspfade: {work_paths_info}
- haeufige Nicht-Arbeitspfade: {private_paths_info}
- bestaetigte aehnliche Beispiele:
{examples_text}
- lernende Korrespondent-Hinweise (aus bestaetigten Faellen):
{learning_hint_text}

ZUORDNUNGSREGELN - GENAU BEACHTEN:
- Arbeit/msg: NUR Dokumente die DIREKT von msg systems ag stammen (Arbeitsvertrag, Zeugnis, Gehaltsabrechnung MIT msg im Absender/Inhalt)
- Arbeit/WBS: NUR Dokumente die DIREKT von WBS TRAINING AG stammen (Arbeitsvertrag, Gehaltsabrechnung MIT WBS im Absender/Inhalt)
- NIEMALS Arbeitgeber als Korrespondent setzen, wenn im Dokument klar ein externer Anbieter steht (z.B. Google Cloud, GitHub, JetBrains, OpenAI, ElevenLabs)
- NICHT zu Arbeit: Software-Abos (Claude, GitHub, JetBrains, etc.), Weiterbildungen die privat bezahlt werden, private Cloud-Dienste, Technik-Kaeufe
- Software-Abos, KI-Dienste, Cloud-Dienste, Hosting -> Freizeit/IT oder Finanzen je nach Kontext
- Korrespondent muss der KONKRETE Absender sein, nicht ein Oberbegriff.
- Verwandte Organisationen NICHT zusammenziehen (z.B. Tochterfirma, Abteilung, Dienst, Klinik, Kreisverband, Wasserwacht, Blutspendedienst sind getrennte Absender).
- Bei Mehrdeutigkeit immer offiziellen Absender aus Briefkopf/Fusszeile bevorzugen.
- Wenn unklar: bestehenden Korrespondenten beibehalten statt neuen erfinden.
- Versicherungen -> Versicherungen/[Typ]
- Bank/Finanzen/Depot -> Finanzen/Bank oder Finanzen/Depot
- Arzt/Gesundheit/AOK -> Gesundheit
- VW Polo / Polo / Golf Polo = PRIVAT einordnen (nicht Arbeit)
- Toyota = Firmenwagen-Kontext (arbeitsbezogen), nicht automatisch privat einordnen
- Frage dich IMMER: Ist das ein ARBEITSDOKUMENT (direkt vom Arbeitgeber) oder PRIVAT?
- Im Zweifel: PRIVAT einordnen, nicht Arbeit!

WEITERE REGELN: Tags kurz und sinnvoll. Titel deutsch, aussagekraeftig. Korrespondent=Absender/Firma.
confidence: Wie sicher bist du dir bei der Zuordnung? "high", "medium" oder "low".
Antwort moeglichst kurz und strukturiert.

NUR JSON, kein anderer Text:
{{"title": "Titel", "tags": ["Tag1", "Tag2"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""

        effective_read_timeout = (
            int(read_timeout_override)
            if read_timeout_override is not None
            else (LLM_COMPACT_TIMEOUT if compact_mode else LLM_TIMEOUT)
        )

        schema_enums: dict | None = None
        if self._use_ollama_chat_api():
            enums: dict = {}
            if corr_names:
                enums["correspondent"] = corr_names + [""]
            if path_names:
                enums["storage_path"] = path_names
            if ALLOWED_DOC_TYPES:
                enums["document_type"] = list(ALLOWED_DOC_TYPES)
            if tag_choices:
                enums["tags"] = tag_choices
            if enums:
                schema_enums = enums

        response = self._call_llm(
            prompt,
            read_timeout=effective_read_timeout,
            retries=1 if compact_mode else LLM_RETRY_COUNT,
            max_tokens=LLM_COMPACT_MAX_TOKENS if compact_mode else LLM_MAX_TOKENS,
            schema_enums=schema_enums,
        )

        try:
            suggestion = self._parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            log.info("[yellow]JSON-Parse-Fehler, Retry mit Reasoning-Prompt...[/yellow]")
            retry_prompt = (
                "Analysiere das folgende Dokument Schritt fuer Schritt:\n"
                "1. Wer ist der Absender/Korrespondent?\n"
                "2. Welcher Dokumenttyp passt am besten?\n"
                "3. In welchen Speicherpfad gehoert es?\n"
                "4. Welche Tags sind relevant?\n"
                "5. Wie sicher bist du dir?\n\n"
                f"{prompt}\n\n"
                "WICHTIG: Antworte AUSSCHLIESSLICH mit einem einzigen JSON-Objekt. "
                "Kein Text davor oder danach."
            )
            response = self._call_llm(
                retry_prompt,
                read_timeout=effective_read_timeout,
                retries=1,
                max_tokens=LLM_COMPACT_MAX_TOKENS if compact_mode else LLM_MAX_TOKENS,
                schema_enums=schema_enums,
            )
            suggestion = self._parse_json_response(response)

        confidence = suggestion.get("confidence", "high").lower()
        if LLM_VERIFY_ON_LOW_CONFIDENCE and not compact_mode and confidence in ("low", "medium"):
            log.info(f"  [yellow]Konfidenz: {confidence}[/yellow] -> Verifiziere Zuordnung...")
            suggestion = self._verify_suggestion(document, suggestion, content_preview)

        return suggestion

    def _verify_suggestion(self, document: dict, suggestion: dict, content_preview: str) -> dict:
        """Zweiter LLM-Aufruf zur Verifikation bei unsicherer Zuordnung."""
        verify_prompt = f"""Ich habe folgendes Dokument analysiert und bin mir bei der Zuordnung unsicher.
Pruefe meine Zuordnung und korrigiere sie wenn noetig.

DOKUMENT #{document['id']}:
Titel: {document.get('title', '?')} | Datei: {document.get('original_file_name', '?')}

INHALT (Auszug):
{content_preview[:1000]}

MEINE ZUORDNUNG:
- Titel: {suggestion.get('title', '?')}
- Korrespondent: {suggestion.get('correspondent', '?')}
- Dokumenttyp: {suggestion.get('document_type', '?')}
- Speicherpfad: {suggestion.get('storage_path', '?')}
- Tags: {suggestion.get('tags', [])}
- Begruendung: {suggestion.get('reasoning', '?')}

WICHTIGE REGELN:
- Arbeit/msg oder Arbeit/WBS NUR wenn das Dokument DIREKT vom Arbeitgeber stammt (Vertrag, Zeugnis, Gehaltsabrechnung)
- Software-Abos (Claude, GitHub, JetBrains), Cloud-Dienste, Hosting = PRIVAT, nicht Arbeit
- Im Zweifel PRIVAT einordnen
- Ist der Korrespondent korrekt? Das muss der tatsaechliche Absender/die Firma sein die das Dokument erstellt hat.

Antworte NUR mit korrigiertem JSON (oder identischem wenn alles stimmt):
{{"title": "Titel", "tags": ["Tag1", "Tag2"], "correspondent": "Firma", "document_type": "Typ", "storage_path_name": "Kategorie", "storage_path": "Kategorie/Sub", "confidence": "high", "reasoning": "Kurz"}}"""

        response = self._call_llm(
            verify_prompt,
            read_timeout=LLM_COMPACT_TIMEOUT,
            retries=1,
            max_tokens=LLM_COMPACT_MAX_TOKENS,
        )
        try:
            verified = self._parse_json_response(response)
            if verified.get("storage_path") != suggestion.get("storage_path"):
                log.info(f"  [green]Korrigiert:[/green] {suggestion.get('storage_path')} -> {verified.get('storage_path')}")
            return verified
        except (json.JSONDecodeError, ValueError):
            log.info("  [yellow]Verifikation fehlgeschlagen, nutze Original[/yellow]")
            return suggestion

    def _use_simple_chat_api(self) -> bool:
        return self.url.rstrip("/").endswith("/api/v1/chat")

    def _use_ollama_chat_api(self) -> bool:
        return self.url.rstrip("/").endswith("/api/chat")

    def _build_payload(self, prompt: str, max_tokens: int | None = None,
                       schema_enums: dict | None = None) -> dict:
        token_limit = max(64, int(max_tokens if max_tokens is not None else LLM_MAX_TOKENS))
        if self._use_ollama_chat_api():
            messages = []
            if LLM_SYSTEM_PROMPT:
                messages.append({"role": "system", "content": LLM_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})
            corr_prop: dict = {"type": "string"}
            dtype_prop: dict = {"type": "string"}
            spath_prop: dict = {"type": "string"}
            tag_items: dict = {"type": "string"}
            if schema_enums:
                if schema_enums.get("correspondent"):
                    corr_prop = {"type": "string", "enum": schema_enums["correspondent"]}
                if schema_enums.get("document_type"):
                    dtype_prop = {"type": "string", "enum": schema_enums["document_type"]}
                if schema_enums.get("storage_path"):
                    spath_prop = {"type": "string", "enum": schema_enums["storage_path"]}
                if schema_enums.get("tags"):
                    tag_items = {"type": "string", "enum": schema_enums["tags"]}
            payload = {
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": token_limit,
                },
                "format": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "tags": {"type": "array", "items": tag_items},
                        "correspondent": corr_prop,
                        "document_type": dtype_prop,
                        "storage_path_name": {"type": "string"},
                        "storage_path": spath_prop,
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["title", "tags", "correspondent", "document_type", "storage_path", "confidence"],
                },
            }
            if self.model:
                payload["model"] = self.model
            if LLM_KEEP_ALIVE:
                payload["keep_alive"] = LLM_KEEP_ALIVE
            return payload
        if self._use_simple_chat_api():
            payload = {"input": prompt}
            if self.model:
                payload["model"] = self.model
            if LLM_SYSTEM_PROMPT:
                payload["system_prompt"] = LLM_SYSTEM_PROMPT
            return payload
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": token_limit,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _extract_response_text(self, data: dict) -> str:
        output = data.get("output")
        if isinstance(output, list) and output:
            for entry in output:
                if isinstance(entry, dict):
                    content = entry.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        if isinstance(output, str) and output.strip():
            return output.strip()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            text = first_choice.get("text")
            if isinstance(text, str):
                return text.strip()
        for key in ("output", "response", "text"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()
        message = data.get("message")
        if isinstance(message, str):
            return message.strip()
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        keys = ", ".join(sorted(data.keys())[:8])
        raise RuntimeError(f"Unbekanntes LLM-Antwortformat (keys: {keys})")

    @staticmethod
    def _error_snippet(resp: requests.Response) -> str:
        text = (resp.text or "").strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:300] + "..."
        return text

    def _post_with_retry(
        self,
        headers: dict | None,
        payload: dict,
        read_timeout: int | None = None,
        retries: int | None = None,
    ) -> requests.Response:
        last_resp = None
        last_exc: Exception | None = None
        connect_timeout = max(3, LLM_CONNECT_TIMEOUT)
        effective_read_timeout = max(5, int(read_timeout if read_timeout is not None else LLM_TIMEOUT))
        max_attempts = max(1, int(retries if retries is not None else LLM_RETRY_COUNT))
        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=(connect_timeout, effective_read_timeout),
                )
                last_resp = resp
                if resp.ok:
                    return resp
                transient_400 = resp.status_code == 400 and not (resp.text or "").strip()
                if resp.status_code in (429, 500, 502, 503, 504) or transient_400:
                    if attempt < max_attempts - 1:
                        backoff = min(30, (2 ** attempt) * 0.8 + random.uniform(0, 0.5))
                        time.sleep(backoff)
                        continue
                    continue
                return resp
            except requests.exceptions.Timeout:
                raise
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    backoff = min(30, (2 ** attempt) * 0.8 + random.uniform(0, 0.5))
                    time.sleep(backoff)
                    continue
        if last_exc is not None:
            raise last_exc
        return last_resp

    def _call_llm(
        self,
        prompt: str,
        read_timeout: int | None = None,
        retries: int | None = None,
        max_tokens: int | None = None,
        schema_enums: dict | None = None,
    ) -> str:
        """Sendet Prompt an LLM-Endpunkt und gibt Antwort zurueck."""
        headers = self._auth_headers()
        payload = self._build_payload(prompt, max_tokens=max_tokens, schema_enums=schema_enums)
        t0 = time.perf_counter()
        resp = self._post_with_retry(headers, payload, read_timeout=read_timeout, retries=retries)
        elapsed = time.perf_counter() - t0
        if resp.ok:
            self._response_times.append(elapsed)
            if len(self._response_times) > self._max_tracked:
                self._response_times = self._response_times[-self._max_tracked:]

        if resp.status_code in (400, 422) and self.model:
            model_backup = self.model
            self.model = ""
            retry_payload = self._build_payload(prompt, max_tokens=max_tokens, schema_enums=schema_enums)
            retry_resp = self._post_with_retry(headers, retry_payload, read_timeout=read_timeout, retries=1)
            if retry_resp.ok:
                resp = retry_resp
            else:
                self.model = model_backup

        if resp.status_code in (400, 422) and not self.model:
            detected_model = self._discover_model(headers)
            if detected_model:
                self.model = detected_model
                retry_payload = self._build_payload(prompt, max_tokens=max_tokens, schema_enums=schema_enums)
                retry_resp = self._post_with_retry(headers, retry_payload, read_timeout=read_timeout, retries=1)
                if retry_resp.ok:
                    resp = retry_resp

        if not resp.ok:
            self._record_error()
            detail = self._error_snippet(resp)
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for url: {self.url} | body: {detail}",
                response=resp,
            )
        self._record_success()
        return self._extract_response_text(resp.json())
