# Sports2D PR #41: Temporal Left/Right Swap Correction Investigation

## Project Context
- **Repository**: `davidpagnon/Sports2D` (Fork: `Mrudula-itsjuzme/Sports2D`, branch `fix/temporal-lower-limb-swaps`).
- **PR**: #41 ("Temporal left/right keypoint swap correction").
- **Maintainer**: David Pagnon (`@davidpagnon`).
- **Objective**: Improve bilateral keypoint swap detection, resolve persistent false swaps, explore offline Viterbi filtering before interpolation, and build a reproducible benchmark for David to review.

## Key Engineering Rules
1. **Never fabricate improvements**: Distinguish verified identity corrections from coordinate alterations or trajectory smoothing.
2. **Inspect before editing**: Understand architectural contracts, coordinate frames, and skeleton hierarchies across supported models (HALPE_26, COCO_17, BODY_25, etc.).
3. **Strict Non-Interference**: When `handle_LR_swap=False`, pipeline behavior must remain bit-for-bit identical to upstream baseline.
4. **No Git Remote Pushes**: Never push, open PRs, or transmit private data without explicit user approval.
5. **Agent Memory Protocol**: Maintain append-only notes in [`docs/agent-notes.md`](file:///home/mrudula/Downloads/mocap/Sports2D/docs/agent-notes.md). Record attempts, metrics, mistakes, fixes, and next actions (3–6 lines per entry).

## Workflow Guidance
- **Python Environment**: `/home/mrudula/Downloads/mocap/.venv_sports2d/bin/python`
- **Tests**: `pytest` or `python Sports2D/Utilities/tests.py`
- **Documentation**:
  - Investigation Report: [`docs/swap_correction_investigation.md`](file:///home/mrudula/Downloads/mocap/Sports2D/docs/swap_correction_investigation.md)
  - Agent Notes: [`docs/agent-notes.md`](file:///home/mrudula/Downloads/mocap/Sports2D/docs/agent-notes.md)
