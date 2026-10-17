#!/usr/bin/env bash
# Paired repeated SearchQA evaluation of the initial and selected skill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/skillopt/bin/python}"
OUT_ROOT="${1:-${PROJECT_ROOT}/outputs/fixed_skill_repeats_$(date +%Y%m%d_%H%M%S)}"
SEEDS=(31415 31416 31417)

cd "$PROJECT_ROOT"
mkdir -p "$OUT_ROOT"

for seed in "${SEEDS[@]}"; do
    for skill in initial best; do
        if [[ "$skill" == "initial" ]]; then
            skill_path="${PROJECT_ROOT}/skillopt/envs/searchqa/skills/initial.md"
        else
            skill_path="${PROJECT_ROOT}/outputs/skillopt_searchqa_Qwen2.5-7B-Instruct_20261016_174915/best_skill.md"
        fi
        run_dir="${OUT_ROOT}/seed_${seed}/${skill}"
        echo "[fixed-skill-eval] seed=${seed} skill=${skill} out=${run_dir}"
        "$PYTHON_BIN" scripts/eval_only.py \
            --config configs/searchqa/default.yaml \
            --skill "$skill_path" \
            --split valid_unseen \
            --out_root "$run_dir" \
            --cfg-options "model.transformers_local_seed=${seed}"
    done
done

echo "[fixed-skill-eval] all runs finished: ${OUT_ROOT}"
