# Nested Slow Update marker fix

Commit: `fix: nested Slow Update block parsing and protected-range handling`

## Context

The Qwen2.5-32B SearchQA run's final Skill contained duplicate nested
`SLOW_UPDATE_START` and `SLOW_UPDATE_END` markers. The archived raw output is
preserved as generated. In that particular Skill there was no prose between
the inner and outer end markers, so its existing guidance remained protected.

The previous parser paired the first start marker with the first end marker.
That could pass a nested marker into the next Slow Update prompt and could
leave text after an inner end marker outside the protected range.

## Changes

- Added balanced, depth-aware marker matching in `skillopt/marker_utils.py`.
- Updated Slow Update extraction and replacement to remove nested structural
  markers and keep exactly one canonical marker pair on write.
- Updated step-edit protected-range checks and semantic-density metadata
  removal to honor nested boundaries.
- Added regression coverage in `tests/test_nested_slow_update_markers.py`.
- Kept the original experiment Skill unchanged and saved a normalized copy as
  `outputs/experiment_archive_20261017/experiment_2_qwen25_32b_nf4_bf16/skill_snapshots/final_skill_v0040_for_benchmark.md`.

## Verification

Run the focused regression tests with:

```bash
python -m unittest tests.test_nested_slow_update_markers -v
```

The tests cover nested matching, protection of text after an inner end marker,
exclusion of protected guidance from semantic-density scoring, multiple
separate regions, and unclosed markers.
