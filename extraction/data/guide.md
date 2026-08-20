# Project Memory Extraction Dataset

## Sample 1

### Prompt

Build a desktop inventory tracker for a small warehouse. Python only. The interface needs three views: current inventory, incoming shipments, and outgoing shipments. Store everything locally using SQLite; no cloud services. Product records need SKU, name, quantity, supplier, and reorder threshold. Search should work by SKU or product name. When quantity reaches the reorder threshold, display a warning beside that product. The application should launch without internet access and remain usable on Windows 11 machines with less than 6GB RAM. Keep the database below 400MB. Use a black background throughout the interface.

### Expected Output

{
  "project_overview": "Desktop inventory tracker for warehouse using Python and SQLite",
  "specs": {
    "language": "Python",
    "views": "3",
    "storage": "SQLite",
    "cloud": "none",
    "RAM": "under 6GB",
    "database": "under 400MB",
    "platform": "Windows 11",
    "background": "black"
  },
  "design": [
    "current inventory view",
    "incoming shipments view",
    "outgoing shipments view",
    "local SQLite storage"
  ]
}


## Sample 2

### Prompt

I need a backend service, not a frontend. It should receive temperature readings from remote sensors through HTTP POST requests and expose historical readings through a REST API. Each reading contains sensor ID, timestamp, Celsius temperature, and battery percentage. PostgreSQL is required. Reject temperatures below -80 or above 120 Celsius. Reject battery percentages outside 0 through 100. The service must process at least 500 requests per second and return HTTP 400 for invalid readings. Authentication is unnecessary for this prototype. Package it with Docker and expose port 8080.

### Expected Output

{
  "project_overview": "Backend service receiving temperature readings through HTTP POST requests",
  "specs": {
    "interface": "HTTP POST",
    "API": "REST",
    "storage": "PostgreSQL",
    "throughput": "500 requests/second",
    "temperature": "-80 to 120 Celsius",
    "battery": "0 to 100",
    "invalid": "HTTP 400",
    "authentication": "none",
    "port": "8080",
    "packaging": "Docker"
  },
  "design": [
    "sensor ID and timestamp",
    "Celsius temperature readings",
    "battery percentage readings",
    "historical REST API"
  ]
}


## Sample 3

### Prompt

Make a small 2D game. Player movement is keyboard-based: WASD moves the character and Space activates the ability. Enemies spawn from the top edge and move downward. The player has three lives. Reaching zero lives ends the game. Each enemy destroyed adds 10 points. Display score and lives at the top-left. There should be exactly one playable level, a 1280x720 window, and a fixed 60 FPS update rate. Do not add multiplayer, sound, networking, inventory, or procedural generation. Keep the project contained in one executable with no external services.

### Expected Output

{
  "project_overview": "2D game with keyboard movement enemies lives score one level",
  "specs": {
    "movement": "WASD",
    "ability": "Space",
    "lives": "3",
    "enemy_direction": "downward",
    "enemy_score": "10 points",
    "levels": "1",
    "window": "1280x720",
    "FPS": "60",
    "multiplayer": "none",
    "sound": "none",
    "networking": "none",
    "inventory": "none"
  },
  "design": [
    "keyboard-based player movement",
    "enemies spawn from top",
    "score and lives top-left",
    "one playable level"
  ]
}


## Sample 4

### Prompt

Write a command-line backup utility for Linux. It takes a source directory and destination directory as arguments, recursively copies files, preserves directory structure, and skips files unchanged since the previous backup. Large files should stream instead of loading completely into memory. Add a dry-run option that prints planned operations without modifying anything. The program must return exit code 0 after success and exit code 1 when copying fails. Do not create archives. Do not upload files. Do not delete source files. Python standard library only.

### Expected Output

{
  "project_overview": "Command-line backup utility for Linux copying directory files",
  "specs": {
    "platform": "Linux",
    "inputs": "source and destination",
    "copy": "recursive",
    "unchanged": "skip",
    "large_files": "stream",
    "dry_run": "no modifications",
    "success": "exit code 0",
    "failure": "exit code 1",
    "archives": "none",
    "uploads": "none",
    "deletion": "none",
    "dependencies": "Python standard library"
  },
  "design": [
    "preserve directory structure",
    "skip unchanged files",
    "stream large files",
    "dry-run planned operations"
  ]
}


## Sample 5

### Prompt

Create an AI coding assistant that runs entirely on a local machine. Users submit coding tasks through a terminal interface. The assistant can read files, edit files, execute tests, and inspect command output. Use a local language model through Ollama. Keep conversation memory limited to the active task, but permanently retain project constraints, architecture decisions, successful debugging solutions, and important project facts. Store retained information as Markdown files. Never send project files or prompts to external APIs. Limit tool output retained in active context to the latest 6 results.

### Expected Output

{
  "project_overview": "AI coding assistant running locally with limited conversation memory",
  "specs": {
    "execution": "local machine",
    "interface": "terminal",
    "model": "Ollama",
    "memory": "active task",
    "retention": "project information",
    "storage": "Markdown files",
    "external_APIs": "none",
    "tool_results": "latest 6"
  },
  "design": [
    "read files",
    "edit files",
    "execute tests",
    "inspect command output"
  ]
}


# Extraction Guidelines

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
- Prefer noun-based or action-based structures.
