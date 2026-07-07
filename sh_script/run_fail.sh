#!/bin/bash
set -euo pipefail

# Usage:
#   bash sh_script/run_fail.sh [source_model_name|all] [bcos|bcosify]
# Example:
#   bash sh_script/run_fail.sh densenet121
#   bash sh_script/run_fail.sh resnet50
#   bash sh_script/run_fail.sh vit_b_16
#   bash sh_script/run_fail.sh all
#   bash sh_script/run_fail.sh all bcos
#   bash sh_script/run_fail.sh all bcosify

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_MODEL_TYPE="torchvision"
SOURCE_MODEL_NAME="${1:-all}"
TARGET_BACKEND="${2:-bcos}"
EPSILONS=(0.03 0.05 0.1 0.2)
SOURCE_MODEL_NAMES=("resnet50" "densenet121" "vit_b_16")
TARGET_MODEL_NAMES=("resnet50" "densenet121" "simple_vit_b_patch16_224")
SAMPLE_SIZE=100
SEED=42

ATTACK_ROOT="/datastore/elo/khoatn/On-Adversarial-Bcos/attack_result"
IMAGENET_VAL_DIR="/datastore/elo/quanphm/dataset/ImageNet1K/val"
ANNOTATIONS_FILE="script/id_2_classname.json"
OUTPUT_LOCALIZED_DIR="failed_transfer"
TRANSFER_RESULT_DIR="transfer_result"

validate_source_model() {
  local src="$1"
  if [ "$src" = "all" ]; then
    return 0
  fi
  for allowed in "${SOURCE_MODEL_NAMES[@]}"; do
    if [ "$src" = "$allowed" ]; then
      return 0
    fi
  done
  echo "Unsupported source model: $src" >&2
  echo "Supported: all, resnet50, densenet121, vit_b_16" >&2
  exit 1
}

validate_backend() {
  local backend="$1"
  case "$backend" in
    bcos|bcosify)
      return 0
      ;;
    *)
      echo "Unsupported backend: $backend" >&2
      echo "Supported: bcos, bcosify" >&2
      exit 1
      ;;
  esac
}

validate_source_model "$SOURCE_MODEL_NAME"
validate_backend "$TARGET_BACKEND"

if [ "$SOURCE_MODEL_NAME" = "all" ]; then
  RUN_SOURCE_MODELS=("${SOURCE_MODEL_NAMES[@]}")
else
  RUN_SOURCE_MODELS=("$SOURCE_MODEL_NAME")
fi

epsilon_to_tag() {
  local eps="$1"
  # 0.03 -> 0p03 for path-safe readable tag
  echo "${eps/./p}"
}

echo "===================================================="
echo "Source type: ${SOURCE_MODEL_TYPE}"
echo "Source models: ${RUN_SOURCE_MODELS[*]}"
echo "Target backend: ${TARGET_BACKEND}"
echo "Target models: ${TARGET_MODEL_NAMES[*]}"
echo "Mode: build failed transfer only (no transfer.py run)"
echo "===================================================="

for SRC_MODEL in "${RUN_SOURCE_MODELS[@]}"; do
  TARGETS=()
  for TARGET_MODEL in "${TARGET_MODEL_NAMES[@]}"; do
    TARGETS+=("${TARGET_BACKEND}:${TARGET_MODEL}")
  done
  echo "---- Source model: ${SRC_MODEL} | targets: ${TARGETS[*]}"

  for target in "${TARGETS[@]}"; do
    TARGET_MODEL_TYPE="${target%%:*}"
    TARGET_MODEL_NAME="${target#*:}"
    TRANSFER_TAG="from_${SOURCE_MODEL_TYPE}_${SRC_MODEL}__to__${TARGET_MODEL_TYPE}_${TARGET_MODEL_NAME}"
    TRANSFER_JSON="${TRANSFER_RESULT_DIR}/transfer_${SOURCE_MODEL_TYPE}_${SRC_MODEL}__to__${TARGET_MODEL_TYPE}_${TARGET_MODEL_NAME}.json"

    if [ ! -f "$TRANSFER_JSON" ]; then
      echo "[skip] Missing transfer JSON: $TRANSFER_JSON"
      continue
    fi

    for EPSILON in "${EPSILONS[@]}"; do
      EPS_TAG="$(epsilon_to_tag "$EPSILON")"
      OUTPUT_PAIR_DIR="${OUTPUT_LOCALIZED_DIR}/${TRANSFER_TAG}/epsilon_${EPS_TAG}"
      mkdir -p "$OUTPUT_PAIR_DIR"

      echo "[run] ${TRANSFER_TAG} | epsilon=${EPSILON}"
      python script/select_nosucess.py \
        --transfer-json "$TRANSFER_JSON" \
        --attack-root "$ATTACK_ROOT" \
        --epsilon "$EPSILON" \
        --sample-size "$SAMPLE_SIZE" \
        --seed "$SEED" \
        --imagenet-val-dir "$IMAGENET_VAL_DIR" \
        --annotations-file "$ANNOTATIONS_FILE" \
        --output "$OUTPUT_PAIR_DIR" \
        --output-root "./transfer_result/NES_PGD"
    done
  done
done

echo "===================================================="
echo "Done failed-transfer selection for source models: ${RUN_SOURCE_MODELS[*]}."
