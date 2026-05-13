# CLAUDE.md

## Code Style

- **Functions must be short** — if a function exceeds ~20 lines, split it.
- **Files must be under 100 lines** — if a file grows past 100 lines, split it into focused modules.
- **Aggressive separation of concerns** — one file = one responsibility. No file should do two things.
- **No comments that describe what the code does** — only comment the non-obvious *why*.
- **No docstrings** — clear naming is sufficient.
- **No unnecessary abstractions** — don't generalize until there are 3+ concrete use cases.

## Project Structure

Keep modules flat and focused:
- `envs/` — environment wrappers only, no reward logic
- `rewards/` — reward computation only, no env logic
- Top-level scripts (`train.py`, `eval.py`, `extract_features.py`) stay thin — just arg parsing and orchestration

## Dependencies

Use the `dancing-agent2` conda environment:
```bash
conda activate dancing-agent2
```
