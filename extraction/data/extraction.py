"""Shared extraction logic: prompt building, output parsing, validation.

Implements the guide.md extraction contract (project_overview / specs / design)
and the four input->output process types:
  new        - no existing state, prompt introduces a project
  update     - existing state, prompt changes requirements
  no_change  - existing state, prompt only asks/discusses
  no_project - no existing state, prompt introduces no project
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Guidelines (from guide.md, verbatim)
# ---------------------------------------------------------------------------

GUIDELINES = """# Extraction Guidelines

## Project Overview

- Output exactly 10-13 words.
- Use only words present in the user prompt.
- Represent the project's primary purpose and type.
- Exclude implementation details when possible.
- Exclude pronouns.
- Exclude filler words.
- Avoid new terminology.
- Preserve important nouns and quantifiable project characteristics.
- Produce one direct declarative phrase.

## Specs

- Output 2-12 entries.
- Use `"key": "value"` format.
- Keys and values should use prompt terminology.
- Preserve numbers, units, limits, platforms, technologies, interfaces, and prohibitions.
- Prefer measurable or categorical values.
- Minimize wording.
- Do not infer unspecified values.
- Do not convert qualitative statements into unsupported measurements.
- Combine closely related constraints when necessary.
- Use `"none"` only when the prompt explicitly prohibits or excludes something.

## Design Statements

- Output 2-4 statements.
- Each statement must contain fewer than 6 words.
- Use only words present in the user prompt.
- Represent implementation structure, component relationships, or required behavior.
- Avoid generic software terminology absent from the prompt.
- Exclude unsupported implementation decisions.
- Preserve technical nouns and relationships.
- Do not duplicate specs unless the relationship or architecture is represented.
- Prefer noun-based or action-based structures."""

SYSTEM_PROMPT = f"""You are a project memory extraction engine. You maintain a software project's memory in exactly three fields: project_overview, specs, and design.

You will receive:
1. EXTRACTION GUIDELINES
2. CURRENT PROJECT STATE (a JSON object, or the literal word "none" when no project exists yet)
3. A USER PROMPT

Determine what the user prompt does to the project, then return the UPDATED project state.

DECISION RULES:
- NEW PROJECT: No current project state exists AND the prompt introduces a software project. Extract the full project state from the prompt.
- UPDATE: A current project state exists AND the prompt changes project requirements. Update the state: add new specs or designs, modify changed values, remove deleted entries, and update the overview if the project's purpose changed.
- NO CHANGE: A current project state exists AND the prompt only asks a question or discusses the project without changing any requirement. Return the current state unchanged.
- NO PROJECT: No current project state exists AND the prompt does not introduce a software project. Return null for all three fields.

OUTPUT FORMAT: Return ONLY a single JSON object with exactly these keys:
{{"project_overview": <string or null>, "specs": <object or null>, "design": <array of strings or null>}}

No markdown, no code fences, no explanations, no extra text before or after the JSON.

HARD LIMITS (never exceed these):
- project_overview: exactly 10-13 words. If the user prompt is too short to reach 10 words without repeating words, prefer a shorter overview over repetition.
- specs: 2-12 entries maximum. Combine closely related constraints into one entry (e.g. "views": "3" instead of one entry per view). Do not invent keys for every sentence.
- design: 2-4 statements, each fewer than 6 words. Do not duplicate spec content; represent structure, relationships, or required behavior.

EXAMPLES (from the extraction guide):

Example 1 - prompt: "Build a desktop inventory tracker for a small warehouse. Python only. The interface needs three views: current inventory, incoming shipments, and outgoing shipments. Store everything locally using SQLite; no cloud services. Product records need SKU, name, quantity, supplier, and reorder threshold. Search should work by SKU or product name. When quantity reaches the reorder threshold, display a warning beside that product. The application should launch without internet access and remain usable on Windows 11 machines with less than 6GB RAM. Keep the database below 400MB. Use a black background throughout the interface."
Expected output:
{{"project_overview": "Desktop inventory tracker for warehouse using Python and SQLite", "specs": {{"language": "Python", "views": "3", "storage": "SQLite", "cloud": "none", "RAM": "under 6GB", "database": "under 400MB", "platform": "Windows 11", "background": "black"}}, "design": ["current inventory view", "incoming shipments view", "outgoing shipments view", "local SQLite storage"]}}

Example 2 - prompt: "I need a backend service, not a frontend. It should receive temperature readings from remote sensors through HTTP POST requests and expose historical readings through a REST API. Each reading contains sensor ID, timestamp, Celsius temperature, and battery percentage. PostgreSQL is required. Reject temperatures below -80 or above 120 Celsius. Reject battery percentages outside 0 through 100. The service must process at least 500 requests per second and return HTTP 400 for invalid readings. Authentication is unnecessary for this prototype. Package it with Docker and expose port 8080."
Expected output:
{{"project_overview": "Backend service receiving temperature readings through HTTP POST requests", "specs": {{"interface": "HTTP POST", "API": "REST", "storage": "PostgreSQL", "throughput": "500 requests/second", "temperature": "-80 to 120 Celsius", "battery": "0 to 100", "invalid": "HTTP 400", "authentication": "none", "port": "8080", "packaging": "Docker"}}, "design": ["sensor ID and timestamp", "Celsius temperature readings", "battery percentage readings", "historical REST API"]}}

EXTRACTION GUIDELINES:
{GUIDELINES}"""


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

EMPTY_STATE: dict[str, Any] = {
    "project_overview": None,
    "specs": None,
    "design": None,
}


def is_empty_state(state: Optional[dict[str, Any]]) -> bool:
    if not state:
        return True
    return all(v is None for v in (state.get("project_overview"),
                                   state.get("specs"), state.get("design")))


def _clean_none(value: Any) -> Any:
    """Treat null-like values (None, "none", "null", "") as None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return value


def normalize_state(state: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Coerce any parsed state into the canonical 3-key form."""
    if not state:
        return dict(EMPTY_STATE)
    out = {
        "project_overview": _clean_none(state.get("project_overview")),
        "specs": _clean_none(state.get("specs")),
        "design": _clean_none(state.get("design")),
    }
    if isinstance(out["specs"], dict) and not out["specs"]:
        out["specs"] = None
    if isinstance(out["design"], list) and not out["design"]:
        out["design"] = None
    return out


def states_equal(a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]) -> bool:
    return normalize_state(a) == normalize_state(b)


def classify_type(
    input_state: Optional[dict[str, Any]],
    output_state: Optional[dict[str, Any]],
) -> str:
    """Classify the input->output process type."""
    in_empty = is_empty_state(input_state)
    out_empty = is_empty_state(output_state)
    if in_empty and out_empty:
        return "no_project"
    if in_empty and not out_empty:
        return "new"
    if not in_empty and not out_empty:
        return "no_change" if states_equal(input_state, output_state) else "update"
    # Existing project that vanished (output empty) - treat as update edge case.
    return "update"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def state_to_text(state: Optional[dict[str, Any]]) -> str:
    if is_empty_state(state):
        return "none"
    return json.dumps(normalize_state(state), ensure_ascii=False, indent=2)


def build_user_prompt(
    user_prompt: str,
    state: Optional[dict[str, Any]] = None,
) -> str:
    return (
        "CURRENT PROJECT STATE:\n"
        f"{state_to_text(state)}\n\n"
        "USER PROMPT:\n"
        f"{user_prompt}"
    )


def build_messages(
    user_prompt: str,
    state: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(user_prompt, state)},
    ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Pull the first balanced JSON object out of a model response."""
    text = text.strip()
    # Strip markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


def parse_model_output(text: str) -> Optional[dict[str, Any]]:
    """Parse a model response into a normalized state dict (or None)."""
    if not text or not text.strip():
        return None
    obj = _extract_json_object(text)
    if obj is None:
        return None
    return normalize_state(obj)


# ---------------------------------------------------------------------------
# Validation (guide.md compliance checks)
# ---------------------------------------------------------------------------

def validate_output(
    state: Optional[dict[str, Any]],
    prompt: Optional[str] = None,
) -> list[str]:
    """Return a list of guideline violations (empty list == compliant).

    ``prompt`` is optional; when supplied, the overview word-count rule is
    relaxed for very short prompts (a 10-13 word overview cannot be built
    from a 6-word prompt without repeating words).
    """
    violations: list[str] = []
    if is_empty_state(state):
        return violations
    s = normalize_state(state)

    prompt_words = len((prompt or "").split()) if prompt else 999

    overview = s.get("project_overview")
    if not isinstance(overview, str) or not overview.strip():
        violations.append("project_overview missing or not a string")
    else:
        words = overview.split()
        # Guide says 10-13 words, but the guide's own samples use 9-word
        # overviews; accept 9-13 to stay consistent with the samples.
        # For very short prompts, allow shorter overviews (no repetition).
        min_words = 9 if prompt_words >= 12 else 4
        if not (min_words <= len(words) <= 13):
            violations.append(
                f"project_overview has {len(words)} words "
                f"(need {min_words}-13): {overview!r}"
            )

    specs = s.get("specs")
    if not isinstance(specs, dict) or not specs:
        violations.append("specs missing or not a non-empty object")
    else:
        if not (2 <= len(specs) <= 12):
            violations.append(f"specs has {len(specs)} entries (need 2-12)")
        for k, v in specs.items():
            if not isinstance(k, str) or not isinstance(v, str):
                violations.append(f"spec entry not string->string: {k!r}: {v!r}")

    design = s.get("design")
    if not isinstance(design, list) or not design:
        violations.append("design missing or not a non-empty array")
    else:
        if not (2 <= len(design) <= 4):
            violations.append(f"design has {len(design)} statements (need 2-4)")
        for stmt in design:
            if not isinstance(stmt, str):
                violations.append(f"design entry not a string: {stmt!r}")
            elif len(stmt.split()) >= 6:
                violations.append(
                    f"design statement has >=6 words: {stmt!r}"
                )
    return violations


def format_state(state: Optional[dict[str, Any]]) -> str:
    return json.dumps(normalize_state(state), ensure_ascii=False, indent=2)