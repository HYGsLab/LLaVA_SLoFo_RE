#!/usr/bin/env bash
set -euo pipefail
ROOT=/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo
REPO="$ROOT/repo"
ENV=/labmount/users/202400390068/envs/HYG_LLaVA_SLoFo/llava-slofo-py310
EXPECTED_PROJECT=c452a0763d7215075ccaaaeff1c17311155b4922
EXPECTED_LLAVA=c121f0432da27facab705978f83c4ada465e46fd

pass() { echo "[PASS] $*"; }
die() { echo "[FAIL] $*" >&2; exit 1; }

[[ "$(stat -c '%a' "$ROOT")" == "700" ]] || die "project root is not mode 700"
pass "private project root"

git -C "$REPO" merge-base --is-ancestor "$EXPECTED_PROJECT" HEAD \
  || die "required project commit is not in current history"
[[ "$(git -C "$REPO/LLaVA" rev-parse HEAD)" == "$EXPECTED_LLAVA" ]] || die "LLaVA commit mismatch"
[[ -z "$(git -C "$REPO" status --short)" ]] || die "repo working tree has unexpected untracked/modified files"
pass "fixed Git commits and clean working tree"

[[ "$(readlink -f "$REPO/models")" == "$ROOT/models" ]] || die "models symlink mismatch"
[[ "$(readlink -f "$REPO/benchmarks")" == "$ROOT/benchmarks" ]] || die "benchmarks symlink mismatch"
pass "runtime symlinks"

check_size() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || die "missing $path"
  [[ "$(stat -c '%s' "$path")" == "$expected" ]] || die "size mismatch: $path"
}
check_size "$ROOT/models/llava-v1.5-7b/pytorch_model-00001-of-00002.bin" 9976634558
check_size "$ROOT/models/llava-v1.5-7b/pytorch_model-00002-of-00002.bin" 3542276251
check_size "$ROOT/models/clip-vit-large-patch14-336/pytorch_model.bin" 1711974081
grep -Fq "$ROOT/models/clip-vit-large-patch14-336" "$ROOT/models/llava-v1.5-7b/config.json" || die "vision tower is not pinned to local path"
pass "model files and local vision tower"

source "$ROOT/activate_slurm.sh" >/dev/null
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import torch
import transformers
import llava

assert torch.__version__ == "2.1.2+cu121", torch.__version__
assert transformers.__version__ == "4.37.2", transformers.__version__
assert np.__version__ == "1.26.4", np.__version__

root = Path("/labmount/users/202400390068/projects/HYG_LLaVA_SLoFo/benchmarks/official")
specs = [
    ("textvqa_subset_512.json", 512, root / "data/textvqa/train_images"),
    ("gqa_subset_512.json", 512, root / "data/gqa/images"),
    ("pope_stratified_600.json", 600, root / "data/pope/val2014"),
]
for name, expected, image_root in specs:
    obj = json.loads((root / "manifests" / name).read_text())
    records = obj["records"]
    assert len(records) == expected, (name, len(records))
    missing = [r["image_file"] for r in records if not (image_root / r["image_file"]).is_file()]
    assert not missing, (name, missing[:3])
print("[PASS] Python imports and 1624/1624 benchmark references")
PY

for f in "$ROOT"/slurm/*.sbatch "$ROOT"/slurm/preflight_empty_gpu.sh; do
  bash -n "$f" || die "shell syntax: $f"
done
pass "all Slurm templates pass bash -n"

cd "$REPO"
python -m pytest -q tests
pass "CPU unit tests"

for f in   /labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/llava_baseline.json   /labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/slofo_focus/result.json   /labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/textvqa_batch_smoke/batch_summary.json   /labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/gqa_batch_smoke/batch_summary.json   /labmount/users/202400390068/results/HYG_LLaVA_SLoFo/slurm_acceptance/pope_batch_smoke/batch_summary.json
do
  [[ -s "$f" ]] || die "missing acceptance output: $f"
done
pass "GPU acceptance outputs"
echo "SLURM_READINESS_OK"
