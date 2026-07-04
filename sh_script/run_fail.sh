#!/bin/bash
set -euo pipefail

# Usage:
#   bash sh_script/test.sh [source_model_name]
# Example:
#   bash sh_script/test.sh densenet121
#   bash sh_script/test.sh resnet50
#   bash sh_script/test.sh vit_b_16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_MODEL_TYPE="torchvision"
SOURCE_MODEL_NAME="${1:-densenet121}"
EPSILONS=(0.03)
SAMPLE_SIZE=100
SEED=42

ATTACK_ROOT="/datastore/elo/khoatn/On-Adversarial-Bcos/attack_result"
IMAGENET_VAL_DIR="/datastore/elo/quanphm/dataset/ImageNet1K/val"
ANNOTATIONS_FILE="script/id_2_classname.json"
OUTPUT_LOCALIZED_DIR="localized/failed_transfer"
TRANSFER_RESULT_DIR="transfer_result"

resolve_targets_for_source() {
  local src="$1"
  case "$src" in
    resnet50)
      echo "bcos:resnet50 bcosify:resnet50"
      ;;
    densenet121)
      echo "bcos:densenet121 bcosify:densenet121"
      ;;
    vit_b_16)
      echo "bcos:simple_vit_b_patch16_224 bcosify:simple_vit_b_patch16_224"
      ;;
    *)
      echo "Unsupported source model: $src" >&2
      echo "Supported: resnet50, densenet121, vit_b_16" >&2
      exit 1
      ;;
  esac
}

TARGETS=( $(resolve_targets_for_source "$SOURCE_MODEL_NAME") )

echo "===================================================="
echo "Source: ${SOURCE_MODEL_TYPE}/${SOURCE_MODEL_NAME}"
echo "Targets: ${TARGETS[*]}"
echo "Mode: build failed transfer only (no transfer.py run)"
echo "===================================================="

for target in "${TARGETS[@]}"; do
  TARGET_MODEL_TYPE="${target%%:*}"
  TARGET_MODEL_NAME="${target#*:}"
  TRANSFER_TAG="from_${SOURCE_MODEL_TYPE}_${SOURCE_MODEL_NAME}__to__${TARGET_MODEL_TYPE}_${TARGET_MODEL_NAME}"
  TRANSFER_JSON="${TRANSFER_RESULT_DIR}/transfer_${SOURCE_MODEL_TYPE}_${SOURCE_MODEL_NAME}__to__${TARGET_MODEL_TYPE}_${TARGET_MODEL_NAME}.json"
  OUTPUT_PAIR_DIR="${OUTPUT_LOCALIZED_DIR}/${TRANSFER_TAG}"

  if [ ! -f "$TRANSFER_JSON" ]; then
    echo "[skip] Missing transfer JSON: $TRANSFER_JSON"
    continue
  fi

  mkdir -p "$OUTPUT_PAIR_DIR"

  echo "[run] Building failed-transfer sample set from ${TRANSFER_JSON}"
  python script/select_nosucess.py \
    --transfer-json "$TRANSFER_JSON" \
    --attack-root "$ATTACK_ROOT" \
    --epsilon "${EPSILONS[0]}" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --imagenet-val-dir "$IMAGENET_VAL_DIR" \
    --annotations-file "$ANNOTATIONS_FILE" \
    --output "$OUTPUT_PAIR_DIR"
done

echo "===================================================="
echo "Done failed-transfer selection for source ${SOURCE_MODEL_NAME}."
