"""OpenAI-compatible chat client for the two planning steps.

Any provider that speaks the OpenAI Chat Completions API works (OpenAI, DeepSeek, a local
vLLM server, ...). Configure it with environment variables:

    WATERKG_LLM_API_KEY   (falls back to OPENAI_API_KEY)
    WATERKG_LLM_BASE_URL  (optional; default is the OpenAI endpoint)
    WATERKG_LLM_MODEL     (default: gpt-5-mini, the model used in the paper)
"""
from __future__ import annotations

import concurrent.futures
import json
import os

from .prompts import ENTITY_TYPES, LLM1_PROMPT, LLM2_PROMPT, LLM2_USER

DEFAULT_MODEL = "gpt-5-mini"


def _client():
    from openai import OpenAI

    key = os.environ.get("WATERKG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set WATERKG_LLM_API_KEY (or OPENAI_API_KEY) to run the LLM planning steps.")
    return OpenAI(api_key=key, base_url=os.environ.get("WATERKG_LLM_BASE_URL") or None, timeout=180)


def _parse_json(content: str) -> dict:
    s = (content or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    return json.loads(s)


def _chat_json(client, model, system, user, max_tokens, attempts=3) -> dict:
    last = None
    for _ in range(attempts):
        try:
            r = client.chat.completions.create(
                model=model, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens)
            return _parse_json(r.choices[0].message.content)
        except (json.JSONDecodeError, ValueError) as exc:
            last = exc
    raise RuntimeError(f"LLM did not return valid JSON after {attempts} attempts: {last}")


def normalize_groups(llm1: dict) -> list[dict]:
    """Keep well-formed groups; members inherit the group's entity type."""
    groups = []
    for g in llm1.get("entity_groups") or []:
        if not isinstance(g, dict):
            continue
        etype = g.get("entity_type") or ""
        members = []
        for m in g.get("members") or []:
            if not isinstance(m, dict):
                continue
            src = m.get("source_text") or m.get("search_text") or ""
            if src:
                members.append({"source_text": src,
                                "search_text": m.get("search_text") or src,
                                "entity_type": m.get("entity_type") or etype})
        if members and etype in ENTITY_TYPES:
            groups.append({"label": g.get("label", ""), "entity_type": etype, "members": members})
    return groups


def plan_query(question: str, taxonomy_text, model: str | None = None, max_parallel: int = 5) -> dict:
    """Run LLM1 once and LLM2 once per entity word (in parallel).

    ``taxonomy_text(entity_type)`` returns the "L3 / L2 / n entities" listing shown to LLM2.
    Returns {"llm1": raw LLM1 output, "groups": [...], "llm2": {entity word: LLM2 output}}.
    """
    client = _client()
    model = model or os.environ.get("WATERKG_LLM_MODEL") or DEFAULT_MODEL
    llm1 = _chat_json(client, model, LLM1_PROMPT, question, max_tokens=4096)
    groups = normalize_groups(llm1)

    words = {(m["source_text"], m["entity_type"]) for g in groups for m in g["members"]}

    def run(word_type):
        word, etype = word_type
        system = LLM2_PROMPT.replace("{entity_type}", etype).replace("{taxonomy_list}", taxonomy_text(etype))
        try:
            return word_type, _chat_json(client, model, system, LLM2_USER.format(word=word), max_tokens=1024)
        except RuntimeError as exc:
            return word_type, {"layer": "evidence", "matches": [], "reason": f"fallback: {exc}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        llm2 = dict(ex.map(run, sorted(words)))
    return {"llm1": llm1, "groups": groups,
            "llm2": {f"{etype}::{word}": out for (word, etype), out in llm2.items()}}
