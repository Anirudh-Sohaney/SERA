"""
Transition engine.

Combines state matching and rule classification to produce
explicit :class:`Transition` objects for every memory operation.

The transition engine is the top-level orchestrator that ties together
the :class:`StateMatcher` (match finding) and the
:class:`TransitionRuleEngine` (transition classification) into a single
``process_candidates`` call.

Processing order:
    1. REMOVE / REJECT transitions are applied first to avoid conflicts
       when an old value is being replaced.
    2. ADD / MODIFY transitions are applied next.
    3. NO_CHANGE transitions are appended last (for audit logging).

The engine does **not** persist state to disk; the caller is responsible
for calling ``state.save(...)`` after all transitions have been applied.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.memory.schema import (
    MemoryCandidate,
    MemoryCategory,
    MemoryItem,
    MemoryStatus,
    ProjectState,
    Transition,
    TransitionType,
    _new_id,
    _now_iso,
)
from src.memory.matcher import MatchResult, StateMatcher
from src.memory.rules import TransitionRuleEngine, detect_replacement_context, detect_negation_context


# ---------------------------------------------------------------------------
# Known technology lookup tables for category inference
# ---------------------------------------------------------------------------

KNOWN_LANGUAGES: set = {
    "python", "javascript", "typescript", "java", "c", "c++", "c#", "go",
    "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
    "matlab", "perl", "haskell", "elixir", "erlang", "clojure", "lisp",
    "fortran", "cobol", "assembly", "asm", "bash", "shell", "zsh", "powershell",
    "sql", "plsql", "t-sql", "transact-sql", "dart", "lua", "groovy",
    "objective-c", "objc", "ocaml", "f#", "fsharp", "julia", "nim",
    "crystal", "zig", "v", "odin", "gleam", "supercollider", "max",
    "Processing", "glsl", "hlsl", "wgsl", "solidity", "move", "cairo",
}

KNOWN_FRAMEWORKS: set = {
    "react", "react.js", "reactjs", "next.js", "nextjs", "next",
    "vue", "vue.js", "vuejs", "nuxt", "nuxt.js", "nuxtjs",
    "angular", "angularjs", "svelte", "sveltekit",
    "django", "flask", "fastapi", "fast-api", "starlette",
    "express", "express.js", "expressjs", "koa", "hapi", "nestjs",
    "spring", "spring-boot", "springboot", "springboot",
    "rails", "ruby on rails", "sinatra",
    "laravel", "symfony", "codeigniter", "cakephp",
    "asp.net", "aspnet", "blazor", "grpc",
    "graphql", "apollo", "relay", "urql",
    "tailwind", "tailwindcss", "bootstrap", "material-ui", "mui",
    "chakra-ui", "ant-design", "radix", "shadcn",
    "pytest", "unittest", "nose", "jest", "mocha", "chai", "vitest",
    "cypress", "playwright", "selenium",
    "tensorflow", "pytorch", "torch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "langchain", "llamaindex", "openai", "anthropic", "transformers",
    "celery", "redis", "rabbitmq", "kafka",
    "terraform", "ansible", "pulumi", "cdk",
    "nginx", "apache", "caddy", "traefik",
    "grafana", "prometheus", "datadog", "newrelic", "sentry",
    "jest", "vitest", "pyunit", "unittest", "rspec", "minitest",
    "webpack", "vite", "esbuild", "rollup", "parcel", "turbopack",
    "babel", "postcss", "sass", "less", "stylus",
    "electron", "tauri", "capacitor", "cordova",
    "redis", "memcached", "elasticsearch", "opensearch",
    "prisma", "sequelize", "typeorm", "knex", "sqlalchemy", "alembic",
    "mongoose", "typeorm", "drizzle",
    "jwt", "oauth", "passport", "bcrypt", "argon2",
    "stripe", "twilio", "sendgrid", "mailgun",
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
    "jupyter", "jupyterlab", "notebook",
    "ipython", "bpython", "ptpython",
    "poetry", "pipenv", "conda", "virtualenv", "venv",
    "npm", "yarn", "pnpm", "bun",
    "cargo", "pip", "gem", "brew",
}

KNOWN_TOOLS: set = {
    "docker", "docker-compose", "dockerhub",
    "kubernetes", "k8s", "helm",
    "terraform", "ansible", "pulumi", "cdk",
    "jenkins", "github actions", "gitlab ci", "circleci", "travis",
    "gitpod", "codespaces",
    "make", "cmake", "gradle", "maven", "ant",
    "webpack", "vite", "esbuild", "rollup", "parcel", "turbopack",
    "babel", "postcss", "sass", "less",
    "eslint", "prettier", "black", "ruff", "flake8", "mypy", "pylint",
    "pytest", "jest", "vitest", "mocha", "cypress", "playwright",
    "curl", "wget", "httpie", "postman",
    "git", "svn", "hg",
    "vim", "neovim", "emacs", "vscode", "intellij",
    "gdb", "valgrind", "strace",
    "redis-cli", "psql", "mysql", "mongosh",
}

KNOWN_DATABASES: set = {
    "postgresql", "postgres", "mysql", "mariadb", "sqlite", "sqlite3",
    "mongodb", "mongo", "redis", "memcached", "cassandra", "dynamodb",
    "couchdb", "couchbase", "neo4j", "influxdb", "timescaledb",
    "cockroachdb", "yugabyte", "clickhouse", "bigquery", "snowflake",
    "redshift", "firestore", "supabase", "planetscale", "tiDB",
    "oracle", "mssql", "ms sql", "sql server", "db2", "informix",
    "etcd", "zookeeper", "consul",
}

KNOWN_PLATFORMS: set = {
    "aws", "amazon web services", "gcp", "google cloud", "google cloud platform",
    "azure", "microsoft azure", "heroku", "vercel", "netlify", "digitalocean",
    "digital ocean", "linode", "vultr", "cloudflare", "fly.io", "flyio",
    "render", "railway", "cyclic", "openshift", "firebase",
    "cloud run", "cloud functions", "lambda", "ecs", "eks", "fargate",
    "app engine", "cloud foundry", "docker", "dockerhub",
    "github actions", "gitlab ci", "circleci", "travis", "jenkins",
    "gitpod", "codespaces", "replit",
    "ios", "android", "windows", "linux", "macos", "mac os",
    "chrome", "firefox", "safari", "edge",
    "raspberry pi", "raspberry", "arduino", "esp32", "esp8266",
}

# File / directory path heuristics
_PATH_INDICATORS = re.compile(
    r"^(?:~?/|[./]|[a-zA-Z]:\\|\.\.?/)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

def _infer_category_from_text(text: str) -> MemoryCategory:
    """Heuristically infer a MemoryCategory from raw text.

    This function is used when no field-level category information is
    available (e.g. spans extracted without alignment metadata).

    Args:
        text: The extracted text span.

    Returns:
        The inferred :class:`MemoryCategory`.
    """
    if not text:
        return MemoryCategory.REQUIREMENT

    lower = text.strip().lower()

    # File / directory path heuristics
    if "/" in lower or "\\" in lower or lower.startswith("."):
        if re.search(r"\.\w{1,5}$", lower) and "/" not in lower:
            return MemoryCategory.FILE
        return MemoryCategory.DIRECTORY

    if lower in KNOWN_LANGUAGES:
        return MemoryCategory.LANGUAGE
    if lower in KNOWN_TOOLS:
        return MemoryCategory.TOOL
    if lower in KNOWN_FRAMEWORKS:
        return MemoryCategory.FRAMEWORK
    if lower in KNOWN_DATABASES:
        return MemoryCategory.DATABASE
    if lower in KNOWN_PLATFORMS:
        return MemoryCategory.PLATFORM

    # Project name heuristic: CamelCase or PascalCase without spaces
    # and doesn't match any known technology
    if (len(text) > 1 and 
        text[0].isupper() and 
        not text.isupper() and  # Not all caps (like AWS)
        ' ' not in text and
        not re.search(r'[0-9]', text)):  # No digits
        return MemoryCategory.PROJECT

    return MemoryCategory.REQUIREMENT


# Field name → category mapping for alignment-based inference
FIELD_CATEGORY_MAP: Dict[str, MemoryCategory] = {
    "specs.language": MemoryCategory.LANGUAGE,
    "specs.purpose": MemoryCategory.REQUIREMENT,
    "specs.input": MemoryCategory.INPUT,
    "specs.output": MemoryCategory.OUTPUT,
    "specs.type": MemoryCategory.CONSTRAINT,
    "specs.application": MemoryCategory.PROJECT,
    "specs.platform": MemoryCategory.PLATFORM,
    "specs.servers": MemoryCategory.DEPLOYMENT,
    "specs.code": MemoryCategory.LANGUAGE,
    "specs.source": MemoryCategory.INPUT,
    "specs.target": MemoryCategory.OUTPUT,
    "specs.command": MemoryCategory.TOOL,
    "specs.exposure": MemoryCategory.CONSTRAINT,
    "specs.resources": MemoryCategory.CONSTRAINT,
    "specs.notifications": MemoryCategory.REQUIREMENT,
    "project_overview": MemoryCategory.PROJECT,
}


def _category_from_field(field_name: str) -> Optional[MemoryCategory]:
    """Resolve a field name to a MemoryCategory.

    Handles exact matches and the ``design[i]`` pattern (where ``i``
    is an index).

    Args:
        field_name: The alignment field name (e.g. ``"specs.language"``).

    Returns:
        The resolved :class:`MemoryCategory`, or ``None`` if unrecognised.
    """
    if not field_name:
        return None
    # Direct lookup
    if field_name in FIELD_CATEGORY_MAP:
        return FIELD_CATEGORY_MAP[field_name]
    # design[i] pattern
    if re.match(r"^design\[\d+\]$", field_name):
        return MemoryCategory.DESIGN
    return None


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------

def build_memory_candidates(
    spans: List[Dict[str, Any]],
    prompt_text: str,
    turn_number: int,
) -> List[MemoryCandidate]:
    """Convert raw extracted spans into typed :class:`MemoryCandidate` objects.

    Each span dict is expected to have at least a ``"text"`` key.  Optional
    keys include ``"start"``, ``"end"``, ``"confidence"``, and
    ``"field"`` (the alignment field name used for category inference).

    Category resolution order:
        1. ``field`` key → ``FIELD_CATEGORY_MAP`` lookup.
        2. ``category`` key in span (if already a valid MemoryCategory).
        3. Heuristic text-based inference.

    Args:
        spans:        List of raw span dicts from the SERA extractor.
        prompt_text:  The full prompt text these spans were extracted from.
        turn_number:  The conversation turn number.

    Returns:
        A list of :class:`MemoryCandidate` objects, one per span.
    """
    candidates: List[MemoryCandidate] = []
    for span in spans:
        text = str(span.get("text", "")).strip()
        if not text:
            continue

        # Resolve category
        category: Optional[MemoryCategory] = None

        # 1. Field-based inference
        field_name = span.get("field")
        if field_name:
            category = _category_from_field(field_name)

        # 2. Explicit category in span
        if category is None:
            raw_cat = span.get("category")
            if raw_cat:
                try:
                    category = MemoryCategory(raw_cat)
                except (ValueError, KeyError):
                    category = None

        # 3. Text-based heuristic
        if category is None:
            category = _infer_category_from_text(text)

        candidates.append(
            MemoryCandidate(
                text=text,
                category=category,
                start=int(span.get("start", 0)),
                end=int(span.get("end", 0)),
                prompt_text=prompt_text,
                confidence=float(span.get("confidence", 1.0)),
                turn_number=turn_number,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Transition engine
# ---------------------------------------------------------------------------

class TransitionEngine:
    """Top-level transition engine.

    Combines the :class:`StateMatcher` and :class:`TransitionRuleEngine`
    to produce a list of :class:`Transition` objects for a batch of
    candidates.

    Processing order:
        1. REMOVE / REJECT (to clear space for replacements).
        2. ADD / MODIFY.
        3. NO_CHANGE (audit trail).

    The engine mutates the ``state`` in-place (appending transitions and
    updating memories) but does **not** persist to disk.

    Args:
        state: The current project state.  Will be modified in-place.
    """

    def __init__(self, state: ProjectState) -> None:
        self._state = state
        self._matcher = StateMatcher(state)
        self._rules = TransitionRuleEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_candidates(
        self, candidates: List[MemoryCandidate]
    ) -> List[Transition]:
        """Process a batch of candidates and apply transitions to state.

        For each candidate the engine:
            1. Runs the matcher to find existing matches.
            2. Runs the rule engine to classify the transition.
            3. Creates a :class:`Transition` record.
            4. Applies the transition to the in-memory state.

        Transitions are returned in the order they were applied (which
        respects the priority ordering: REMOVE/REJECT → ADD/MODIFY →
        NO_CHANGE).

        Args:
            candidates: Memory candidates to process.

        Returns:
            A list of :class:`Transition` objects recording every
            decision made.
        """
        if not candidates:
            return []

        # Step 0: Check for replacement patterns in the prompt
        # If detected, create a synthetic REMOVE transition for the old value
        replacement_transitions: List[Transition] = []
        removed_values: set = set()
        if candidates and candidates[0].prompt_text:
            prompt_text = candidates[0].prompt_text
            
            # Check for explicit replacement patterns
            replacement = detect_replacement_context(prompt_text)
            if replacement is not None:
                old_value = replacement["old_value"].rstrip('.')  # Remove trailing punctuation
                removed_values.add(old_value.lower())
                # Find and remove the old value from state
                for mem in list(self._state.active_memories):
                    if (mem.value.lower() == old_value.lower() or 
                        old_value.lower() in mem.value.lower() or
                        mem.value.lower() in old_value.lower()):
                        turn = candidates[0].turn_number or self._state.current_turn
                        transition = self._state.remove_memory(
                            memory_id=mem.memory_id,
                            turn=turn,
                        )
                        if transition is not None:
                            replacement_transitions.append(transition)
            
            # Check for explicit replacement patterns that imply removal
            # Only "Replace X with Y" and "Use X instead of Y" patterns
            # trigger removal. Other patterns like "switch to" or bare
            # "actually" do NOT imply removal — they just signal a change.
            conflict_patterns = [
                (r"\breplace\s+\w+\s+with\b", ["replace"]),
                (r"\binstead\s+of\b", ["instead of"]),
            ]
            new_values = {c.text.lower() for c in candidates}
            new_categories = {c.category for c in candidates}
            for pattern, keywords in conflict_patterns:
                if re.search(pattern, prompt_text, re.IGNORECASE):
                    # Remove existing memories that are:
                    # 1. In the SAME category as new candidates
                    # 2. NOT in the new candidates (exact or partial match)
                    for mem in list(self._state.active_memories):
                        if mem.category not in new_categories:
                            continue  # Don't touch other categories
                        is_in_new_values = mem.value.lower() in new_values
                        is_partial_match = any(
                            mem.value.lower() in nv or nv in mem.value.lower()
                            for nv in new_values
                        )
                        if not is_in_new_values and not is_partial_match:
                            removed_values.add(mem.value.lower())
                            turn = candidates[0].turn_number or self._state.current_turn
                            transition = self._state.remove_memory(
                                memory_id=mem.memory_id,
                                turn=turn,
                            )
                            if transition is not None:
                                replacement_transitions.append(transition)
                    break  # Only apply first matching conflict pattern

        # Filter out candidates that match removed values
        candidates = [c for c in candidates if c.text.lower() not in removed_values]

        # Step 1: Match all candidates
        match_results = self._matcher.find_matches(candidates)

        # Step 2: Classify all candidates
        classified: List[
            tuple[MemoryCandidate, MatchResult, Any]
        ] = []
        for candidate, match_result in zip(candidates, match_results):
            rule_result = self._rules.classify(
                candidate=candidate,
                match_result=match_result,
                state=self._state,
                full_prompt=candidate.prompt_text,
            )
            classified.append((candidate, match_result, rule_result))

        # Step 3: Sort by processing priority
        classified.sort(
            key=lambda x: self._transition_priority(x[2].transition_type)
        )

        # Step 4: Apply transitions
        transitions: List[Transition] = []
        for candidate, match_result, rule_result in classified:
            transition = self._apply_transition(
                candidate=candidate,
                match_result=match_result,
                rule_result=rule_result,
            )
            if transition is not None:
                transitions.append(transition)

        # Prepend replacement transitions (REMOVE of old value)
        return replacement_transitions + transitions

    # ------------------------------------------------------------------
    # Internal: apply a single transition
    # ------------------------------------------------------------------

    def _apply_transition(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        rule_result: Any,
    ) -> Optional[Transition]:
        """Apply a single classified transition to the state.

        Dispatches to the appropriate ``ProjectState`` method based on
        ``rule_result.transition_type``.

        Args:
            candidate:   The candidate that was classified.
            match_result: The match result for this candidate.
            rule_result:  The rule engine's classification result.

        Returns:
            The :class:`Transition` that was recorded, or ``None`` if
            the operation failed silently.
        """
        tt = rule_result.transition_type
        turn = candidate.turn_number or self._state.current_turn

        if tt == TransitionType.ADD:
            return self._apply_add(candidate, turn, rule_result)

        if tt == TransitionType.MODIFY:
            return self._apply_modify(candidate, match_result, turn, rule_result)

        if tt == TransitionType.REMOVE:
            return self._apply_remove(candidate, match_result, turn, rule_result)

        if tt == TransitionType.REJECT:
            return self._apply_reject(candidate, match_result, turn, rule_result)

        if tt == TransitionType.NO_CHANGE:
            return self._apply_no_change(candidate, match_result, turn, rule_result)

        return None

    def _apply_add(
        self,
        candidate: MemoryCandidate,
        turn: int,
        rule_result: Any,
    ) -> Transition:
        """Apply an ADD transition.

        Creates a new :class:`MemoryItem` and adds it to the state.

        Args:
            candidate:   The candidate to add.
            turn:        The current turn number.
            rule_result: The rule engine result (for metadata).

        Returns:
            The recorded :class:`Transition`.
        """
        item = MemoryItem(
            memory_id=_new_id(),
            category=candidate.category,
            value=candidate.text,
            source_text=candidate.text,
            source_start=candidate.start,
            source_end=candidate.end,
            prompt_text=candidate.prompt_text,
            status=MemoryStatus.ACTIVE,
            created_turn=turn,
            updated_turn=turn,
            confidence=candidate.confidence,
            metadata={
                "rule_id": rule_result.rule_id,
                "reason": rule_result.reason,
            },
        )
        transition = self._state.add_memory(item)
        self._state.current_turn = max(self._state.current_turn, turn)
        return transition

    def _apply_modify(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        turn: int,
        rule_result: Any,
    ) -> Optional[Transition]:
        """Apply a MODIFY transition.

        Updates the matched memory's value to the candidate's text.

        Args:
            candidate:   The candidate with the new value.
            match_result: Must have a ``matched_memory``.
            turn:        The current turn number.
            rule_result: The rule engine result (for metadata).

        Returns:
            The recorded :class:`Transition`, or ``None`` if no match.
        """
        memory = match_result.matched_memory
        if memory is None:
            # Fallback: treat as ADD
            return self._apply_add(candidate, turn, rule_result)

        transition = self._state.modify_memory(
            memory_id=memory.memory_id,
            new_value=candidate.text,
            new_source=candidate.text,
            new_source_start=candidate.start,
            new_source_end=candidate.end,
            new_prompt=candidate.prompt_text,
            turn=turn,
            confidence=candidate.confidence,
        )
        self._state.current_turn = max(self._state.current_turn, turn)
        return transition

    def _apply_remove(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        turn: int,
        rule_result: Any,
    ) -> Optional[Transition]:
        """Apply a REMOVE transition.

        Marks the matched memory as REMOVED.

        Args:
            candidate:   The candidate (used for logging/metadata).
            match_result: Must have a ``matched_memory``.
            turn:        The current turn number.
            rule_result: The rule engine result (for metadata).

        Returns:
            The recorded :class:`Transition`, or ``None`` if no match.
        """
        memory = match_result.matched_memory
        if memory is None:
            return None

        transition = self._state.remove_memory(
            memory_id=memory.memory_id,
            turn=turn,
        )
        self._state.current_turn = max(self._state.current_turn, turn)
        return transition

    def _apply_reject(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        turn: int,
        rule_result: Any,
    ) -> Optional[Transition]:
        """Apply a REJECT transition.

        If a match exists, marks it as REJECTED.  If no match exists,
        creates a rejected memory item (recorded in all_memories with
        REJECTED status).

        Args:
            candidate:   The candidate to reject.
            match_result: The match result.
            turn:        The current turn number.
            rule_result: The rule engine result (for metadata).

        Returns:
            The recorded :class:`Transition`.
        """
        memory = match_result.matched_memory
        if memory is not None:
            transition = self._state.reject_memory(
                memory_id=memory.memory_id,
                turn=turn,
            )
            self._state.current_turn = max(self._state.current_turn, turn)
            return transition

        # No existing memory — create a rejected item for audit trail
        item = MemoryItem(
            memory_id=_new_id(),
            category=candidate.category,
            value=candidate.text,
            source_text=candidate.text,
            source_start=candidate.start,
            source_end=candidate.end,
            prompt_text=candidate.prompt_text,
            status=MemoryStatus.REJECTED,
            created_turn=turn,
            updated_turn=turn,
            confidence=candidate.confidence,
            metadata={
                "rule_id": rule_result.rule_id,
                "reason": rule_result.reason,
            },
        )
        self._state.all_memories.append(item)

        transition = Transition(
            transition_type=TransitionType.REJECT,
            category=candidate.category,
            value=candidate.text,
            memory_id=item.memory_id,
            source_text=candidate.text,
            source_start=candidate.start,
            source_end=candidate.end,
            prompt_text=candidate.prompt_text,
            turn_number=turn,
            confidence=candidate.confidence,
            metadata={
                "rule_id": rule_result.rule_id,
                "reason": rule_result.reason,
            },
        )
        self._state.transition_log.append(transition)
        self._state.current_turn = max(self._state.current_turn, turn)
        return transition

    def _apply_no_change(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        turn: int,
        rule_result: Any,
    ) -> Optional[Transition]:
        """Apply a NO_CHANGE transition.

        Records an audit entry without modifying state.  Refreshes the
        matched memory's ``updated_turn``.

        Args:
            candidate:   The candidate.
            match_result: Must have a ``matched_memory``.
            turn:        The current turn number.
            rule_result: The rule engine result (for metadata).

        Returns:
            The recorded :class:`Transition`, or ``None`` if no match.
        """
        memory = match_result.matched_memory
        if memory is None:
            return None

        memory.updated_turn = turn
        transition = Transition(
            transition_type=TransitionType.NO_CHANGE,
            category=candidate.category,
            value=candidate.text,
            memory_id=memory.memory_id,
            source_text=candidate.text,
            source_start=candidate.start,
            source_end=candidate.end,
            prompt_text=candidate.prompt_text,
            turn_number=turn,
            confidence=candidate.confidence,
            metadata={
                "rule_id": rule_result.rule_id,
                "reason": rule_result.reason,
            },
        )
        self._state.transition_log.append(transition)
        self._state.current_turn = max(self._state.current_turn, turn)
        return transition

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transition_priority(tt: TransitionType) -> int:
        """Return a numeric priority for processing order.

        Lower values are processed first.  REMOVE/REJECT go first to
        avoid conflicts when replacing values.

        Args:
            tt: The transition type.

        Returns:
            Integer priority (0 = first).
        """
        return {
            TransitionType.REMOVE: 0,
            TransitionType.REJECT: 1,
            TransitionType.ADD: 2,
            TransitionType.MODIFY: 3,
            TransitionType.NO_CHANGE: 4,
        }.get(tt, 5)
