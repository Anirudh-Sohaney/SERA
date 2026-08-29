"""
E6 Enhanced Augmentation Generator

Uses actual false-positive patterns from error analysis to generate
realistic training examples that teach the model:
1. What NOT to extract (description language = O)
2. What TO extract (explicit requirements = B/I)
"""

import json
import hashlib
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ─── False-Positive Patterns from Error Analysis ─────────────────────────────

FP_PATTERNS = {
    "COMMON_WORDS": [
        "given", "each", "takes", "all", "only", "two", "no", "first", "any",
        "write", "using", "handle", "find", "check", "new", "original", "empty",
        "input", "output", "number", "numbers", "sum", "length", "difference",
        "maximum", "minimum", "largest", "smallest", "between", "among", "from",
        "to", "with", "without", "where", "when", "what", "how", "why", "which",
        "that", "this", "these", "those", "it", "its", "they", "them", "their",
        "there", "then", "than", "but", "and", "or", "not", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "must", "shall",
        "need", "want", "like", "make", "let", "get", "put", "take", "give", "keep",
        "set", "run", "go", "come", "see", "look", "seem", "feel", "think", "know",
        "understand", "believe", "expect", "hope", "wish", "at", "on", "in", "of",
        "for", "about", "over", "under", "above", "below", "through", "during",
        "before", "after", "since", "until", "while", "although", "though",
        "because", "if", "unless", "except", "besides", "instead", "rather", "than",
        "just", "also", "too", "very", "quite", "really", "actually", "certainly",
        "definitely", "probably", "possibly", "maybe", "perhaps", "almost", "nearly",
        "barely", "hardly", "simply", "merely", "purely", "entirely", "completely",
        "totally", "absolutely", "exactly", "precisely", "specifically", "particularly",
        "especially", "generally", "usually", "typically", "normally", "commonly",
        "frequently", "often", "sometimes", "rarely", "seldom", "never", "always",
        "once", "twice", "thrice", "one", "a", "use", "total", "text", "data",
        "time", "code", "size", "user", "program", "index", "default", "same",
        "provided", "specified", "returned", "multiple", "target", "corresponding",
        "numbered", "otherwise", "determines", "checks", "contains", "various",
        "starting", "basic", "efficient",
    ],
    "DATA_TYPES": [
        "string", "integer", "int", "float", "bool", "list", "array", "dict",
        "dictionary", "tuple", "set", "vector", "matrix", "elements", "characters",
        "words", "items", "values", "keys", "pairs", "strings", "integers", "chars",
        "tokens", "three", "scores", "word", "character", "even numbers",
        "lowercase letters", "two numbers",
    ],
    "FUNCTION_PATTERNS": [
        "python function", "returns the", "takes a", "def function", "function(",
        "returns a", "return value", "input parameter", "output parameter",
        "function call", "method invocation", "callback function", "anonymous function",
        "lambda expression", "recursive function", "function to", "find the",
    ],
    "IO_SPECS": [
        "list of", "list of integers", "number of", "return an", "ascending order",
        "list of strings", "list of tuples", "array of integers", "sum of",
        "length of the", "return a", "return the", "empty list", "new list",
        "original list", "input string", "given string", "string of",
        "string containing", "sequence of", "sorted in", "descending order",
        "largest possible", "check if", "total of", "count of",
    ],
    "CODE_SYNTAX": [
        "return", "returns", "def", "sum", "Returns", "def", "import", "print",
        "return 0", "from",
    ],
    "CODE_ANNOTATIONS": [
        "n:", "List[int]", "List[", "int]) ->", "nums:", "arr:", "str)", "->",
        "int]]) ->", "List[int]) ->",
    ],
    "PUNCTUATION": ["'", "(", "[", '"', "$"],
    "BOOLEAN_NULL": ["True", "False", "None", "YES", "NO"],
    "ADJECTIVES": [
        "simple", "optimal", "competitive", "valid", "specific", "sorted",
        "ascending", "descending", "strictly", "increasing", "decreasing",
        "unique", "distinct", "duplicate", "repeated", "consecutive", "adjacent",
        "neighboring", "previous", "next", "last", "final", "initial", "second",
        "third", "fourth", "fifth", "nth", "arbitrary", "random", "deterministic",
        "predictable", "consistent", "stable", "unstable", "robust", "fragile",
        "strong", "weak", "powerful", "lightweight", "heavy", "fast", "slow",
        "quick", "rapid", "gradual", "sudden", "immediate", "delayed", "early",
        "late", "old", "fresh", "stale", "modern", "legacy", "deprecated",
        "obsolete", "current", "upcoming", "past", "present", "future", "common",
        "rare", "typical", "atypical", "normal", "abnormal", "standard", "custom",
        "optional", "required", "mandatory", "automatic", "manual", "explicit",
        "implicit", "direct", "indirect", "linear", "nonlinear", "parallel",
        "serial", "synchronous", "asynchronous", "concurrent", "sequential",
        "atomic", "pure", "functional", "imperative", "declarative", "generic",
        "abstract", "concrete", "virtual", "logical", "remote", "local", "global",
        "distributed", "centralized", "secure", "insecure", "encrypted",
        "authenticated", "authorized", "trusted", "untrusted", "safe", "reliable",
        "unreliable", "available", "unavailable", "accessible", "visible",
        "invisible", "transparent", "opaque", "clear", "obvious", "subtle",
        "hidden", "secret", "named", "labeled", "longest", "largest", "basic",
        "efficient", "starting", "various",
    ],
    "ACTION_VERBS": [
        "find", "create", "calculate", "test", "check", "sort", "remove", "add",
        "insert", "update", "delete", "modify", "change", "get", "send", "receive",
        "request", "call", "invoke", "execute", "start", "stop", "pause", "resume",
        "continue", "break", "exit", "quit", "close", "open", "connect",
        "disconnect", "authenticate", "authorize", "validate", "verify", "confirm",
        "reject", "accept", "approve", "deny", "allow", "block", "filter", "search",
        "lookup", "match", "compare", "analyze", "evaluate", "measure", "debug",
        "trace", "log", "monitor", "track", "record", "report", "document",
        "describe", "explain", "summarize", "enumerate", "average", "normalize",
        "scale", "transform", "convert", "parse", "format", "encode", "decode",
        "compress", "encrypt", "decrypt", "hash", "sign", "generate", "determine",
        "solve", "implement", "build", "construct", "develop", "design", "plan",
        "organize", "arrange", "order", "sequence", "prioritize", "categorize",
        "classify", "group", "cluster", "segment", "partition", "divide", "split",
        "merge", "combine", "join", "link", "associate", "relate", "map", "pair",
        "bind", "attach", "detach", "separate", "isolate", "extract", "select",
        "choose", "pick", "move", "contains", "determines", "checks",
    ],
}


# ─── Positive Contrast Templates ────────────────────────────────────────────

POSITIVE_TECHNOLOGIES = [
    # Languages
    "Python", "Java", "Rust", "Go", "JavaScript", "TypeScript", "C++", "C#",
    "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R", "MATLAB", "Perl", "Haskell",
    "Elixir", "Clojure", "F#", "OCaml", "Lua", "Julia", "Dart", "Groovy",
    
    # Frameworks
    "React", "Vue.js", "Angular", "Node.js", "Express", "FastAPI", "Django",
    "Flask", "Spring Boot", "Laravel", "Rails", "Sinatra", "Gin", "Echo",
    "Fiber", "Actix", "Rocket", "Axum", "Tide", "Warp",
    
    # Databases
    "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Elasticsearch",
    "Cassandra", "DynamoDB", "Neo4j", "CouchDB", "MariaDB", "InfluxDB",
    "TimescaleDB", "ClickHouse", "Snowflake", "BigQuery", "Redshift",
    
    # Infrastructure
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Ansible",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI",
    "Prometheus", "Grafana", "ELK Stack", "Datadog", "New Relic",
    
    # Protocols
    "GraphQL", "gRPC", "REST", "WebSocket", "MQTT", "AMQP", "Kafka",
    "RabbitMQ", "NATS", "ZeroMQ", "Protocol Buffers", "Avro",
    
    # Libraries
    "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "OpenCV",
    "NLTK", "spaCy", "Matplotlib", "Seaborn", "Plotly", "D3.js",
    "Bootstrap", "Tailwind CSS", "Material UI", "Ant Design",
]


# ─── Negative Prompt Templates ───────────────────────────────────────────────

NEGATIVE_TEMPLATES = [
    # Task descriptions (should be O)
    "Create a {thing} that {action}.",
    "Build a {thing} which {action}.",
    "Implement a {thing} that {action}.",
    "Write a {thing} to {action}.",
    "Develop a {thing} for {action}.",
    "Design a {thing} that {action}.",
    "Make a {thing} which {action}.",
    "Produce a {thing} for {action}.",
    "Generate a {thing} that {action}.",
    "Construct a {thing} to {action}.",
    
    # Function descriptions (should be O)
    "The function should {action}.",
    "The method needs to {action}.",
    "It should return {output}.",
    "It takes {input} and returns {output}.",
    "The input is {input} and output is {output}.",
    
    # I/O descriptions (should be O)
    "Given a {type}, return {output}.",
    "For each {type}, {action}.",
    "The {type} contains {content}.",
    "Process the {type} to {action}.",
    "Read the {type} and {action}.",
    
    # Algorithm descriptions (should be O)
    "Write code to {action}.",
    "The program should {action}.",
    "The algorithm must {action}.",
    "This function will {action}.",
    "The solution needs to {action}.",
    
    # Data structure descriptions (should be O)
    "Use a {type} to store {content}.",
    "The {type} should {action}.",
    "Initialize the {type} with {content}.",
    "The {type} contains {content}.",
    "Process each {type} in the {type}.",
    
    # More FP patterns
    "The function takes a {type} as input.",
    "Return the {type} as output.",
    "The {type} is given as input.",
    "For each element in the {type}, {action}.",
    "The {type} should be {adjective}.",
]

THING_WORDS = [
    "application", "program", "tool", "service", "system", "module", "component",
    "script", "utility", "library", "framework", "interface", "API", "dashboard",
    "console", "CLI", "web app", "mobile app", "desktop app", "backend", "frontend",
    "server", "client", "database", "cache", "queue", "pipeline", "workflow",
    "algorithm", "function", "method", "class", "module", "package",
]

ACTION_WORDS = [
    "process data", "handle requests", "manage users", "validate input",
    "transform data", "aggregate results", "filter records", "sort items",
    "search content", "cache responses", "log events", "queue tasks",
    "schedule jobs", "monitor performance", "track metrics", "analyze patterns",
    "detect anomalies", "generate reports", "send notifications", "backup data",
    "compress files", "encrypt messages", "parse input", "format output",
    "validate schema", "check integrity", "resolve conflicts", "merge changes",
]

TYPE_WORDS = [
    "list", "array", "string", "number", "dictionary", "object", "file",
    "stream", "buffer", "queue", "stack", "tree", "graph", "node", "edge",
    "record", "tuple", "set", "map", "hash", "table", "matrix", "vector",
]

OUTPUT_WORDS = [
    "a list", "a string", "a number", "a boolean", "the result", "the count",
    "the sum", "the maximum", "the minimum", "the average", "the sorted list",
    "the filtered list", "the transformed data", "the aggregated result",
]

INPUT_WORDS = [
    "a list", "a string", "a number", "an array", "a dictionary", "a file",
    "a stream", "a buffer", "a queue", "a stack", "a tree", "a graph",
]

CONTENT_WORDS = [
    "data", "values", "elements", "items", "records", "entries", "keys",
    "pairs", "tokens", "characters", "words", "numbers", "integers", "strings",
]

ADJECTIVE_WORDS = [
    "sorted", "filtered", "transformed", "validated", "compressed", "encrypted",
    "cached", "logged", "monitored", "tracked", "analyzed", "aggregated",
]


def generate_negative_prompt() -> Tuple[str, List[str]]:
    """Generate a negative prompt with extracted FP patterns."""
    template = random.choice(NEGATIVE_TEMPLATES)
    
    prompt = template
    if "{thing}" in prompt:
        prompt = prompt.replace("{thing}", random.choice(THING_WORDS))
    if "{action}" in prompt:
        prompt = prompt.replace("{action}", random.choice(ACTION_WORDS))
    if "{type}" in prompt:
        prompt = prompt.replace("{type}", random.choice(TYPE_WORDS))
    if "{output}" in prompt:
        prompt = prompt.replace("{output}", random.choice(OUTPUT_WORDS))
    if "{input}" in prompt:
        prompt = prompt.replace("{input}", random.choice(INPUT_WORDS))
    if "{content}" in prompt:
        prompt = prompt.replace("{content}", random.choice(CONTENT_WORDS))
    if "{adjective}" in prompt:
        prompt = prompt.replace("{adjective}", random.choice(ADJECTIVE_WORDS))
    
    return prompt, []


def generate_positive_contrast(negative_prompt: str) -> Tuple[str, List[Dict]]:
    """Generate a positive contrast by adding a technology requirement."""
    tech = random.choice(POSITIVE_TECHNOLOGIES)
    
    # Strategy 1: Add "Use {tech} for..." at the beginning
    strategies = [
        f"Use {tech} for the backend.",
        f"Implement with {tech}.",
        f"Build using {tech}.",
        f"The project uses {tech}.",
        f"Technology stack: {tech}.",
        f"Preferred technology: {tech}.",
        f"We use {tech} for this.",
        f"Choose {tech} for implementation.",
        f"Selected: {tech}.",
        f"{tech} is required.",
    ]
    
    prompt = random.choice(strategies)
    start = prompt.find(tech)
    
    spans = [{
        "text": tech,
        "start": start,
        "end": start + len(tech),
    }]
    
    return prompt, spans


def generate_concise_positive() -> Tuple[str, List[Dict]]:
    """Generate a concise positive example."""
    tech = random.choice(POSITIVE_TECHNOLOGIES)
    
    templates = [
        f"Use {tech}.",
        f"{tech} backend.",
        f"{tech} frontend.",
        f"{tech} database.",
        f"{tech} server.",
        f"{tech} client.",
        f"{tech} API.",
        f"{tech} framework.",
        f"{tech} library.",
        f"{tech} tool.",
    ]
    
    prompt = random.choice(templates)
    start = prompt.find(tech)
    
    spans = [{
        "text": tech,
        "start": start,
        "end": start + len(tech),
    }]
    
    return prompt, spans


def validate_augmented_record(prompt: str, spans: List[Dict]) -> bool:
    """Validate that spans are exact substrings of the prompt."""
    for span in spans:
        start = span["start"]
        end = span["end"]
        text = span["text"]
        
        if start < 0 or end > len(prompt):
            return False
        
        if prompt[start:end] != text:
            return False
    
    return True


def build_augmented_dataset(
    output_path: str,
    manifest_path: str,
    base_records_path: str,
    seed: int = 42,
    augmentation_ratio: float = 0.10,
) -> Dict:
    """Build the E6 augmented dataset with realistic examples."""
    random.seed(seed)
    
    # Load base records
    base_count = 0
    base_prompts = set()
    with open(base_records_path) as f:
        for line in f:
            record = json.loads(line)
            base_prompts.add(record.get("prompt", ""))
            base_count += 1
    
    target_augmentations = int(base_count * augmentation_ratio)
    print(f"Base dataset: {base_count} records")
    print(f"Target augmentations: {target_augmentations} ({augmentation_ratio*100:.0f}%)")
    
    augmented_records = []
    rejected_count = 0
    max_attempts_per_category = target_augmentations * 3  # Allow 3x retries
    
    # Generate negative examples
    negative_count = 0
    attempts = 0
    while negative_count < target_augmentations // 3 and attempts < max_attempts_per_category:
        attempts += 1
        prompt, spans = generate_negative_prompt()
        
        if prompt in base_prompts:
            rejected_count += 1
            continue
        
        if not validate_augmented_record(prompt, spans):
            rejected_count += 1
            continue
        
        augmented_records.append({
            "prompt": prompt,
            "spans": spans,
            "source": "e6_negative_template",
            "category": "NEGATIVE",
        })
        base_prompts.add(prompt)
        negative_count += 1
    
    # Generate positive contrasts
    positive_count = 0
    attempts = 0
    while positive_count < target_augmentations // 3 and attempts < max_attempts_per_category:
        attempts += 1
        prompt, spans = generate_positive_contrast("")
        
        if prompt in base_prompts:
            rejected_count += 1
            continue
        
        if not validate_augmented_record(prompt, spans):
            rejected_count += 1
            continue
        
        augmented_records.append({
            "prompt": prompt,
            "spans": spans,
            "source": "e6_positive_contrast",
            "category": "POSITIVE",
        })
        base_prompts.add(prompt)
        positive_count += 1
    
    # Generate concise positives
    concise_count = 0
    attempts = 0
    while concise_count < target_augmentations // 3 and attempts < max_attempts_per_category:
        attempts += 1
        prompt, spans = generate_concise_positive()
        
        if prompt in base_prompts:
            rejected_count += 1
            continue
        
        if not validate_augmented_record(prompt, spans):
            rejected_count += 1
            continue
        
        augmented_records.append({
            "prompt": prompt,
            "spans": spans,
            "source": "e6_concise_positive",
            "category": "CONCISE",
        })
        base_prompts.add(prompt)
        concise_count += 1
    
    # Shuffle
    random.shuffle(augmented_records)
    
    # Write augmented records
    with open(output_path, "w") as f:
        for record in augmented_records:
            f.write(json.dumps(record) + "\n")
    
    # Calculate dataset hash
    with open(base_records_path, "rb") as f:
        base_hash = hashlib.md5(f.read()).hexdigest()
    
    # Write manifest
    manifest = {
        "base_dataset_hash": base_hash,
        "augmentation_count": len(augmented_records),
        "augmentation_categories": {
            "NEGATIVE": negative_count,
            "POSITIVE": positive_count,
            "CONCISE": concise_count,
        },
        "generation_method": "template_based_with_validation",
        "validation_method": "substring_equality_check",
        "rejected_count": rejected_count,
        "accepted_count": len(augmented_records),
        "random_seed": seed,
        "augmentation_ratio": augmentation_ratio,
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nAugmentation complete:")
    print(f"  Negative examples: {negative_count}")
    print(f"  Positive contrasts: {positive_count}")
    print(f"  Concise positives: {concise_count}")
    print(f"  Total accepted: {len(augmented_records)}")
    print(f"  Total rejected: {rejected_count}")
    print(f"  Augmentation ratio: {len(augmented_records)/base_count*100:.1f}%")
    
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="E6 Enhanced Augmentation Generator")
    parser.add_argument("--base-records", default="data/processed/aligned_records.jsonl")
    parser.add_argument("--output", default="data/processed/e6_targeted_augmented.jsonl")
    parser.add_argument("--manifest", default="data/processed/e6_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.10)
    args = parser.parse_args()
    
    manifest = build_augmented_dataset(
        output_path=args.output,
        manifest_path=args.manifest,
        base_records_path=args.base_records,
        seed=args.seed,
        augmentation_ratio=args.ratio,
    )
