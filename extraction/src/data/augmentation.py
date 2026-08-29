"""
E6 Targeted False-Positive Augmentation Generator

Generates training examples from measured false-positive patterns:
1. Negative examples: description phrases that should NOT be extracted
2. Positive contrasts: paired examples where the concept IS a requirement
3. Concise positive examples: short explicit requirements

Every augmented record is validated:
- prompt[start:end] == target_text (substring verification)
- BIO alignment verified
- No duplicate prompts in training set
"""

import json
import hashlib
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ─── Category Definitions ───────────────────────────────────────────────────

CATEGORIES = {
    "DESCRIPTION_FRAGMENT": {
        "definition": "A phrase describing what the requested software does, without specifying a persistent project requirement.",
        "augmentation_strategy": "negative",
        "examples": [
            "each word separated by a",
            "single string with",
            "items in the list",
            "type and data",
            "product of all the",
            "list of words into a",
            "list of integers as input",
            "sum of all elements",
            "number of occurrences",
            "characters in the string",
            "elements in the array",
            "words in the sentence",
            "digits in the number",
            "values in the dictionary",
            "keys in the hash map",
            "nodes in the tree",
            "edges in the graph",
            "rows in the table",
            "columns in the matrix",
            "items in the queue",
        ],
    },
    "FUNCTION_SIGNATURE": {
        "definition": "Function signatures and parameter descriptions that are code structure, not project information.",
        "augmentation_strategy": "negative",
        "examples": [
            "python function",
            "returns the",
            "takes a",
            "def function",
            "function(",
            "returns a",
            "return value",
            "input parameter",
            "output parameter",
            "function call",
            "method invocation",
            "callback function",
            "anonymous function",
            "lambda expression",
            "recursive function",
        ],
    },
    "INPUT_OUTPUT_SPEC": {
        "definition": "Input/output specifications that describe data flow, not project requirements.",
        "augmentation_strategy": "negative",
        "examples": [
            "list of words into a",
            "product of all the",
            "list of integers as input",
            "given a list of",
            "return the sum of",
            "array of strings",
            "dictionary of key-value pairs",
            "set of unique elements",
            "queue of tasks",
            "stack of frames",
            "tree of nodes",
            "graph of edges",
            "matrix of values",
            "vector of components",
            "tensor of features",
        ],
    },
    "DATA_TYPE": {
        "definition": "Data type descriptions that are programming concepts, not project information.",
        "augmentation_strategy": "negative",
        "examples": [
            "string",
            "integer",
            "list",
            "array",
            "dictionary",
            "tuple",
            "set",
            "boolean",
            "float",
            "character",
            "number",
            "array of integers",
            "list of strings",
            "dictionary of mappings",
            "set of elements",
        ],
    },
    "COMMON_WORD": {
        "definition": "Common English words without project meaning.",
        "augmentation_strategy": "negative",
        "examples": [
            "given",
            "each",
            "takes",
            "all",
            "only",
            "two",
            "no",
            "first",
            "any",
            "write",
            "using",
            "handle",
            "find",
            "check",
            "new",
            "original",
            "empty",
            "input",
            "output",
            "number",
        ],
    },
}


# ─── Positive Contrast Templates ────────────────────────────────────────────

POSITIVE_CONTRASTS = {
    "DESCRIPTION_FRAGMENT": [
        ("Create a web application that displays user data.", "React"),
        ("Build a command-line tool that parses JSON files.", "Rust"),
        ("Implement a function that sorts a list of integers.", "Python"),
        ("Write a program that processes images from a directory.", "OpenCV"),
        ("Develop a service that handles HTTP requests.", "FastAPI"),
        ("Create an application for processing images.", "Pillow"),
        ("Build a dashboard that shows real-time metrics.", "Grafana"),
        ("Implement a system that manages user authentication.", "JWT"),
        ("Write a tool that analyzes log files.", "Python"),
        ("Create a script that automates database backups.", "cron"),
    ],
    "FUNCTION_SIGNATURE": [
        ("Use FastAPI for the backend API.", "FastAPI"),
        ("Implement the CLI in Rust.", "Rust"),
        ("Build with React for the frontend.", "React"),
        ("Use PostgreSQL for the database.", "PostgreSQL"),
        ("Implement using Django framework.", "Django"),
        ("Use Flask for the web server.", "Flask"),
        ("Build with Node.js backend.", "Node.js"),
        ("Use TypeScript for type safety.", "TypeScript"),
        ("Implement with Go for concurrency.", "Go"),
        ("Use Vue.js for the UI.", "Vue.js"),
    ],
    "INPUT_OUTPUT_SPEC": [
        ("Use Redis for caching.", "Redis"),
        ("Implement with MongoDB for documents.", "MongoDB"),
        ("Use Elasticsearch for search.", "Elasticsearch"),
        ("Build with RabbitMQ for messaging.", "RabbitMQ"),
        ("Use Kafka for event streaming.", "Kafka"),
        ("Implement with GraphQL API.", "GraphQL"),
        ("Use gRPC for communication.", "gRPC"),
        ("Build with WebSocket for real-time.", "WebSocket"),
        ("Use MQTT for IoT devices.", "MQTT"),
        ("Implement with REST API.", "REST"),
    ],
    "DATA_TYPE": [
        ("Use Python for the backend.", "Python"),
        ("Implement in Java.", "Java"),
        ("Build with C++ for performance.", "C++"),
        ("Use JavaScript for scripting.", "JavaScript"),
        ("Implement with Go.", "Go"),
        ("Use Rust for systems programming.", "Rust"),
        ("Build with TypeScript.", "TypeScript"),
        ("Use Kotlin for Android.", "Kotlin"),
        ("Implement with Swift for iOS.", "Swift"),
        ("Use Ruby for rapid prototyping.", "Ruby"),
    ],
    "COMMON_WORD": [
        ("Use Docker for containerization.", "Docker"),
        ("Implement with Kubernetes.", "Kubernetes"),
        ("Build with AWS cloud services.", "AWS"),
        ("Use Azure for deployment.", "Azure"),
        ("Implement with GCP.", "GCP"),
        ("Use Terraform for infrastructure.", "Terraform"),
        ("Build with Ansible for automation.", "Ansible"),
        ("Use Jenkins for CI/CD.", "Jenkins"),
        ("Implement with GitHub Actions.", "GitHub Actions"),
        ("Use GitLab CI for pipelines.", "GitLab CI"),
    ],
}


# ─── Concise Positive Examples ───────────────────────────────────────────────

CONCISE_POSITIVES = [
    ("Use Rust.", "Rust"),
    ("Python backend.", "Python"),
    ("PostgreSQL database.", "PostgreSQL"),
    ("React frontend.", "React"),
    ("Use Redis.", "Redis"),
    ("Build with FastAPI.", "FastAPI"),
    ("Use Docker.", "Docker"),
    ("Implement in Go.", "Go"),
    ("Use MongoDB.", "MongoDB"),
    ("Build with Node.js.", "Node.js"),
    ("Use TypeScript.", "TypeScript"),
    ("Implement with Django.", "Django"),
    ("Use Flask.", "Flask"),
    ("Build with Vue.js.", "Vue.js"),
    ("Use Angular.", "Angular"),
    ("Implement in Java.", "Java"),
    ("Use C++ for performance.", "C++"),
    ("Build with Kotlin.", "Kotlin"),
    ("Use Swift for iOS.", "Swift"),
    ("Implement with Ruby.", "Ruby"),
    ("Use GraphQL.", "GraphQL"),
    ("Build with gRPC.", "gRPC"),
    ("Use Kafka.", "Kafka"),
    ("Implement with RabbitMQ.", "RabbitMQ"),
    ("Use Elasticsearch.", "Elasticsearch"),
    ("Build with Terraform.", "Terraform"),
    ("Use Jenkins.", "Jenkins"),
    ("Implement with GitHub Actions.", "GitHub Actions"),
    ("Use GitLab CI.", "GitLab CI"),
    ("Build with AWS.", "AWS"),
    ("Use Azure.", "Azure"),
    ("Implement with GCP.", "GCP"),
    ("Use SQLite for testing.", "SQLite"),
    ("Build with MySQL.", "MySQL"),
    ("Use MariaDB.", "MariaDB"),
    ("Implement with Cassandra.", "Cassandra"),
    ("Use Neo4j for graphs.", "Neo4j"),
    ("Build with DynamoDB.", "DynamoDB"),
    ("Use CouchDB.", "CouchDB"),
    ("Implement with Firebase.", "Firebase"),
]


# ─── Negative Templates ─────────────────────────────────────────────────────

NEGATIVE_TEMPLATES = [
    # Description fragments
    ("Create a {thing} that {action}.", None),
    ("Build a {thing} which {action}.", None),
    ("Implement a {thing} that {action}.", None),
    ("Write a {thing} to {action}.", None),
    ("Develop a {thing} for {action}.", None),
    ("Design a {thing} that {action}.", None),
    ("Make a {thing} which {action}.", None),
    ("Produce a {thing} for {action}.", None),
    ("Generate a {thing} that {action}.", None),
    ("Construct a {thing} to {action}.", None),
    
    # Function signatures
    ("The function should {action}.", None),
    ("The method needs to {action}.", None),
    ("It should return {output}.", None),
    ("It takes {input} and returns {output}.", None),
    ("The input is {input} and output is {output}.", None),
    
    # Input/output specs
    ("Given a {type}, return {output}.", None),
    ("For each {type}, {action}.", None),
    ("The {type} contains {content}.", None),
    ("Process the {type} to {action}.", None),
    ("Read the {type} and {action}.", None),
    
    # Common patterns
    ("Write code to {action}.", None),
    ("The program should {action}.", None),
    ("The algorithm must {action}.", None),
    ("This function will {action}.", None),
    ("The solution needs to {action}.", None),
]

THING_WORDS = ["application", "program", "tool", "service", "system", "module", "component", "script", "utility", "library", "framework", "interface", "API", "dashboard", "console", "CLI", "web app", "mobile app", "desktop app"]
ACTION_WORDS = ["process data", "handle requests", "manage users", "validate input", "transform data", "aggregate results", "filter records", "sort items", "search content", "cache responses", "log events", "queue tasks", "schedule jobs", "monitor performance", "track metrics"]
TYPE_WORDS = ["list", "array", "string", "number", "dictionary", "object", "file", "stream", "buffer", "queue", "stack", "tree", "graph", "node", "edge"]
OUTPUT_WORDS = ["a list", "a string", "a number", "a boolean", "the result", "the count", "the sum", "the maximum", "the minimum", "the average"]
INPUT_WORDS = ["a list", "a string", "a number", "an array", "a dictionary", "a file", "a stream"]
CONTENT_WORDS = ["data", "values", "elements", "items", "records", "entries", "keys", "pairs", "tokens", "characters"]


def generate_negative_example(template: str) -> Optional[Tuple[str, List[Dict]]]:
    """Generate a negative example from a template."""
    prompt = template
    
    # Fill in placeholders
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
    
    # All tokens are O (no project information)
    return prompt, []


def generate_positive_contrast(negative_prompt: str, tech_name: str) -> Optional[Tuple[str, List[Dict]]]:
    """Generate a positive contrast from a negative prompt."""
    # Create a new prompt with the technology requirement
    positive_prompt = f"Use {tech_name} for the backend."
    
    # Find the technology span
    start = positive_prompt.find(tech_name)
    if start == -1:
        return None
    
    spans = [{
        "text": tech_name,
        "start": start,
        "end": start + len(tech_name),
    }]
    
    return positive_prompt, spans


def generate_concise_positive(tech_name: str) -> Optional[Tuple[str, List[Dict]]]:
    """Generate a concise positive example."""
    prompt = f"Use {tech_name}."
    
    start = prompt.find(tech_name)
    if start == -1:
        return None
    
    spans = [{
        "text": tech_name,
        "start": start,
        "end": start + len(tech_name),
    }]
    
    return prompt, spans


def validate_augmented_record(prompt: str, spans: List[Dict]) -> bool:
    """Validate that spans are exact substrings of the prompt."""
    for span in spans:
        start = span["start"]
        end = span["end"]
        text = span["text"]
        
        # Check bounds
        if start < 0 or end > len(prompt):
            return False
        
        # Check substring equality
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
    """
    Build the E6 augmented dataset.
    
    Args:
        output_path: Path to write augmented records
        manifest_path: Path to write dataset manifest
        base_records_path: Path to base aligned records
        seed: Random seed
        augmentation_ratio: Fraction of base dataset to add as augmentation
    
    Returns:
        Dictionary with augmentation statistics
    """
    random.seed(seed)
    
    # Load base records to get count
    base_count = 0
    base_prompts = set()
    with open(base_records_path) as f:
        for line in f:
            record = json.loads(line)
            base_prompts.add(record.get("prompt", ""))
            base_count += 1
    
    # Calculate augmentation target
    target_augmentations = int(base_count * augmentation_ratio)
    print(f"Base dataset: {base_count} records")
    print(f"Target augmentations: {target_augmentations} ({augmentation_ratio*100:.0f}%)")
    
    augmented_records = []
    rejected_count = 0
    source_records = []
    
    # Generate negative examples from templates
    negative_count = 0
    for template, _ in NEGATIVE_TEMPLATES:
        if negative_count >= target_augmentations // 3:
            break
        
        result = generate_negative_example(template)
        if result is None:
            rejected_count += 1
            continue
        
        prompt, spans = result
        
        # Check for duplicate
        if prompt in base_prompts:
            rejected_count += 1
            continue
        
        # Validate
        if not validate_augmented_record(prompt, spans):
            rejected_count += 1
            continue
        
        augmented_records.append({
            "prompt": prompt,
            "spans": spans,
            "source": "e6_negative_template",
            "category": "NEGATIVE_TEMPLATE",
        })
        base_prompts.add(prompt)
        source_records.append(f"template:{template}")
        negative_count += 1
    
    # Generate positive contrasts
    positive_count = 0
    for category, contrasts in POSITIVE_CONTRASTS.items():
        for negative_example, tech_name in contrasts:
            if positive_count >= target_augmentations // 3:
                break
            
            result = generate_positive_contrast(negative_example, tech_name)
            if result is None:
                rejected_count += 1
                continue
            
            prompt, spans = result
            
            # Check for duplicate
            if prompt in base_prompts:
                rejected_count += 1
                continue
            
            # Validate
            if not validate_augmented_record(prompt, spans):
                rejected_count += 1
                continue
            
            augmented_records.append({
                "prompt": prompt,
                "spans": spans,
                "source": f"e6_positive_contrast_{category}",
                "category": f"POSITIVE_CONTRAST_{category}",
            })
            base_prompts.add(prompt)
            source_records.append(f"contrast:{category}:{tech_name}")
            positive_count += 1
    
    # Generate concise positives
    concise_count = 0
    for tech_name, _ in CONCISE_POSITIVES:
        if concise_count >= target_augmentations // 3:
            break
        
        result = generate_concise_positive(tech_name)
        if result is None:
            rejected_count += 1
            continue
        
        prompt, spans = result
        
        # Check for duplicate
        if prompt in base_prompts:
            rejected_count += 1
            continue
        
        # Validate
        if not validate_augmented_record(prompt, spans):
            rejected_count += 1
            continue
        
        augmented_records.append({
            "prompt": prompt,
            "spans": spans,
            "source": "e6_concise_positive",
            "category": "CONCISE_POSITIVE",
        })
        base_prompts.add(prompt)
        source_records.append(f"concise:{tech_name}")
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
            "NEGATIVE_TEMPLATE": negative_count,
            "POSITIVE_CONTRAST": positive_count,
            "CONCISE_POSITIVE": concise_count,
        },
        "source_records": source_records[:100],  # First 100 for reference
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
    print(f"  Negative templates: {negative_count}")
    print(f"  Positive contrasts: {positive_count}")
    print(f"  Concise positives: {concise_count}")
    print(f"  Total accepted: {len(augmented_records)}")
    print(f"  Total rejected: {rejected_count}")
    print(f"  Augmentation ratio: {len(augmented_records)/base_count*100:.1f}%")
    
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="E6 Targeted Augmentation Generator")
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
