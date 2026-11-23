---
name: semantic-parking-detector-guidelines
description: Project coding and collaboration rules for the Semantic Parking Detector (Python 3.13, OpenCV, PKLot, YOLO).
---

# Semantic Parking Detector — Development Rules

## Purpose and scope

Use these rules whenever generating, modifying, reviewing, or explaining code for the **Semantic Parking Detector** university IOC project.

Project stack: **Python 3.13**, **PyCharm**, **OpenCV**, **NumPy**, **Ultralytics YOLO / PyTorch**, **pytest**, and the **PKLot** dataset.

Priority: **correctness > clarity > simplicity > performance > cleverness**. Prefer focused, maintainable changes over unnecessary complexity.

## 1. General implementation principles

- Follow the **existing repository structure and conventions** before introducing new patterns.
- Read the relevant existing modules before suggesting replacements or new files.
- Implement only what the task requires. Do not add speculative features, abstractions, frameworks, or dependencies.
- Keep changes small, incremental, and easy to review. Avoid unrelated refactoring.
- Prefer **simple, explicit Python** over clever one-liners or overly generic abstractions.
- Reuse existing functions and modules instead of duplicating logic.
- Do not silently change public function signatures, return formats, configuration keys, or CLI arguments. If a change is necessary, update callers and tests together.
- When requirements are ambiguous, use the least disruptive reasonable assumption and state it briefly.
- Do not claim code has been run, validated, or measured unless it actually has.

## 2. Language, naming, and formatting

- **All identifiers must be in English**: variables, functions, classes, constants, module names, parameters, and test names.
- **All code comments, docstrings, exception messages, and log messages must be in English.**
- Use `snake_case` for functions, variables, and Python file names; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants.
- Prefer meaningful names such as `parking_spaces`, `vehicle_boxes`, `occupancy_threshold`, and `image_height`; avoid vague names such as `data1`, `temp`, or `result2`.
- Follow **PEP 8**, use four spaces for indentation, and group imports as standard library / third-party / local modules.
- Add type hints to new public functions and nontrivial internal functions. Use Python 3.13 built-in generics, e.g. `list[tuple[int, int, int, int]]`.
- Use `pathlib.Path` for filesystem paths when practical.
- Keep functions focused on one responsibility. Extract helpers only when they meaningfully improve readability or reuse.
- Avoid wildcard imports, mutable default arguments, and hardcoded machine-specific paths.

## 3. Comments and documentation — keep them minimal

- **Do not add random, decorative, or redundant comments.**
- **Do not comment every line or restate what the code already says.**
- Add a short comment only when it explains **why** a non-obvious decision was made, a mathematical assumption, a tricky edge case, or an external constraint.
- Use docstrings for reusable public APIs or algorithms whose inputs, outputs, and assumptions are not immediately clear. Keep them concise.
- Do not insert large banner comments, numbered step comments, TODO spam, or AI-style explanations inside source files.
- If an explanation is useful for teammates, put it in the README, implementation plan, or response instead of cluttering the code.
- Remove obsolete comments when the underlying code changes.

**Preferred:**

```python
# PKLot polygons and YOLO boxes must use the same image coordinate system.
if image_size != annotation_size:
    raise ValueError("Image and annotation dimensions do not match")
```

**Avoid:**

```python
# Loop over all parking spaces
for parking_space in parking_spaces:
    # Get the identifier
    space_id = parking_space["id"]
```

## 4. Project architecture

Keep responsibilities separated using the existing layout:

| Location | Responsibility |
| --- | --- |
| `src/data/` | PKLot image/annotation loading, parsing, dataset utilities |
| `src/detection/` | YOLO or other AI inference; vehicle detections |
| `src/parking/` | Parking-space geometry, overlap, occupancy decisions |
| `src/visualization/` | Drawing overlays and saving readable results |
| `src/evaluation/` | Evaluation metrics and dataset-level comparison (when added) |
| `scripts/` | Small command-line entry points for experiments and evaluation |
| `tests/` | Unit and integration tests |
| `configs/` | Lightweight, versioned configuration files |
| `main.py` | Thin orchestration and CLI entry point; avoid placing core algorithms here |

- Modules should communicate through documented inputs/outputs, not hidden global state.
- Do not couple `src/parking/` to Ultralytics model objects. Pass plain geometry/detection data instead.
- Preserve the current baseline contract unless intentionally migrated:
  - A detection box: `(x1, y1, x2, y2)` in image pixel coordinates.
  - A parking space: `{"id": "A1", "polygon": [[x, y], ...]}`.
  - Occupancy output: `dict[str, bool]` with `True` meaning **occupied** and `False` meaning **free**.
- Keep ground-truth labels separate from model inputs and inferred occupancy. **Never use the `occupied` annotation to generate a prediction.**

## 5. OpenCV, AI, and dataset-specific rules

- Treat OpenCV images as NumPy arrays in **BGR** order; convert explicitly when RGB is required.
- State and preserve coordinate conventions: bounding boxes use `(x1, y1, x2, y2)`; `image.shape[:2]` is `(height, width)`.
- Validate image loading, annotation parsing, polygon size, threshold ranges, and image/annotation dimension consistency.
- Guard against empty detections, missing files, invalid polygons, out-of-bounds coordinates, and division by zero.
- Keep preprocessing transformations consistent for images, boxes, polygons, and masks. Never resize an image without adapting coordinates.
- Prefer deterministic, reproducible experiments: record model name, configuration, thresholds, dataset split, and random seeds when applicable.
- Use pretrained models for the initial baseline. Introduce training/fine-tuning only as an explicit task with a measurable benefit.
- Describe detection, instance segmentation, semantic segmentation, and parking-occupancy classification accurately; they are different tasks.
- Do not present a heuristic overlap threshold (e.g. `0.15`) as validated until it has been evaluated.
- Prevent dataset leakage: split PKLot by **capture day/session or camera** when appropriate; avoid near-duplicate images across training and testing.
- Use ground-truth occupancy labels **only for training an explicitly supervised occupancy model or for validation/evaluation**, never as hidden inputs to the inference pipeline.
- Report metrics only from real experiments; never fabricate accuracy, F1, mAP, or runtime results.

## 6. Configuration, dependencies, and files

- Prefer configuration values or CLI arguments to unexplained hardcoded paths and thresholds.
- Use **Python 3.13** and the existing `.venv` setup. Do not suggest Python version changes without a concrete compatibility issue.
- Keep dependencies minimal and update `requirements.txt` only when necessary.
- Do not commit `.venv/`, downloaded PKLot images, large model weights (`.pt`, `.pth`, etc.), run artifacts, or generated outputs.
- Keep `.gitkeep` in otherwise empty directories that need to appear in the repository.
- Never commit credentials, API keys, absolute personal paths, or sensitive local configuration.
- Keep existing `README.md` and `IMPLEMENTATION_PLAN.md` consistent with substantive changes to setup, CLI usage, or workflow.

## 7. Testing and quality checks

- Add or update **pytest** tests for new nontrivial logic and bug fixes.
- Prefer fast, synthetic unit tests for geometry, occupancy rules, validation, and parsers; they should not require downloading models or the full PKLot dataset.
- Keep AI integration tests separate from pure unit tests when model weights or large datasets are required.
- Include relevant edge cases: no vehicles, multiple detections, partial overlap, zero-area/invalid polygons, mismatched dimensions, malformed XML, and absent ground truth.
- Run the most relevant tests after a change when execution is available. Report what passed and what could not be tested.
- Do not remove tests just to make a failing build pass.

## 8. Collaboration and Git workflow

- Make task-focused changes that can be reviewed independently by either team member.
- Respect ownership boundaries: AI/detection work should not require simultaneous changes to occupancy code, and vice versa, unless integration is explicitly the goal.
- Use descriptive branch names such as `feature/pklot-parser`, `feature/yolo-evaluation`, or `feature/occupancy-metrics`.
- Use concise commit messages, e.g. `feat: parse PKLot parking annotations` or `test: cover polygon overlap edge cases`.
- Do not modify unrelated files or rewrite another contributor's work without a clear reason.
- Before merging, confirm interface compatibility, run tests, and review the changes.
- Do not commit, push, or merge on the user's behalf unless explicitly requested.

## 9. How the assistant should respond to coding tasks

When implementing a requested task:

1. Briefly identify the relevant existing files and the intended behavior.
2. Make the **smallest complete change** that satisfies the request.
3. Provide usable code or an exact patch, with correct file paths.
4. Explain only non-obvious technical decisions, integration impacts, and limitations.
5. Include the command to run or test the change when useful.
6. Clearly distinguish implemented features from proposals or future improvements.

Avoid:

- Adding entire new architectures for a small task.
- Creating multiple utility classes when a simple function is sufficient.
- Excessive comments, docstrings, logs, print statements, or boilerplate.
- Unrequested GUI, web services, databases, Docker, or cloud integrations.
- Changing formatting throughout unrelated files.
- Inventing files, model outputs, performance results, or dataset characteristics.

## 10. Definition of done

A task is complete only when:

- Its behavior matches the requested requirements and the agreed module interface.
- Its code follows the naming, structure, and minimal-comment rules above.
- Relevant edge cases and failures are handled.
- Appropriate tests are present or the absence of tests is explained.
- The integration impact and any necessary setup/documentation updates are clear.
- Any unverified assumptions or remaining limitations are explicitly identified.
