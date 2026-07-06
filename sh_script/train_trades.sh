#!/bin/bash
#SBATCH --job-name=trades_train
#SBATCH --output=trades_train/mps_%j.out
#SBATCH --error=trades_train/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

REQUIRED_VRAM=70000

# =========================================================
# CHUAN BI MOI TRUONG (for sbatch)
# =========================================================
module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack
echo "ENV: $CONDA_DEFAULT_ENV"
echo "PREFIX: $CONDA_PREFIX"
which python
python -c "import sys; print(sys.executable)"

mkdir -p trades_train

# If running inside Slurm, auto-select GPU with enough free VRAM.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  unset CUDA_VISIBLE_DEVICES
  CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
  elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
  fi
  BEST_GPU=$CHECK_OUT
  echo "Job $SLURM_JOB_ID starts on GPU: $BEST_GPU"

  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
  rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

  export CUDA_VISIBLE_DEVICES=$BEST_GPU
fi

# Usage:
#   bash sh_script/train_trades.sh [torchvision|bcos|bcosify] [model_name]
# Examples:
#   bash sh_script/train_trades.sh torchvision resnet50
#   bash sh_script/train_trades.sh bcos simple_vit_b_patch16_224

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [ ! -f "$PROJECT_ROOT/script/const.py" ]; then
  echo "Cannot find script/const.py under PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "Hint: submit job from project root with: sbatch sh_script/train_trades.sh" >&2
  exit 1
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"



# You asked to use this ImageNet train root.
IMAGENET_TRAIN_DIR="/datastore/elo/quanphm/dataset/ImageNet1K/train"
IMAGENET_VAL_DIR="/datastore/elo/quanphm/dataset/ImageNet1K/val"
ANNOTATIONS_FILE="$PROJECT_ROOT/script/id_2_classname.json"

EPOCHS=10
BATCH_SIZE=64
VAL_BATCH_SIZE=64
NUM_WORKERS=8
LR=0.01
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
EPSILON=0.01 # epsilon=4/255
STEP_SIZE=0.00075 # step_size=2/255
NUM_STEPS=10
BETA=0.1
DISTANCE="l_inf"
TRAIN_OBJECTIVE="trades"
SUPERVISED_LOSS="auto"
BCE_OFF_LABEL=""
DEVICE="cuda"
SEED=42
DEBUG_GRAD_EVERY_STEPS=50

OUTPUT_ROOT="$PROJECT_ROOT/checkpoints/trades"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

MODEL_TYPE="${1:-bcos}"
MODEL_NAME="${2:-resnet50}"


run_train() {
  local model_type="$1"
  local model_name="$2"
  local out_dir="$OUTPUT_ROOT/$model_type/$model_name"
  mkdir -p "$out_dir"

  echo "===================================================="
  echo "TRADES train: $model_type/$model_name"
  echo "Train dir: $IMAGENET_TRAIN_DIR"
  echo "Val dir: $IMAGENET_VAL_DIR"
  echo "Output: $out_dir"

  local cmd=(
    python TRADES/train_trades_imagenet.py
    --model-type "$model_type"
    --model-name "$model_name"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --train-dir "$IMAGENET_TRAIN_DIR"
    --val-dir "$IMAGENET_VAL_DIR"
    --annotations-file "$ANNOTATIONS_FILE"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --val-batch-size "$VAL_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --momentum "$MOMENTUM"
    --weight-decay "$WEIGHT_DECAY"
    --epsilon "$EPSILON"
    --step-size "$STEP_SIZE"
    --num-steps "$NUM_STEPS"
    --beta "$BETA"
    --distance "$DISTANCE"
    --train-objective "$TRAIN_OBJECTIVE"
    --supervised-loss "$SUPERVISED_LOSS"
    --debug-grad-every-steps "$DEBUG_GRAD_EVERY_STEPS"
    --device "$DEVICE"
    --seed "$SEED"
    --output-dir "$out_dir"
  )

  if [ -n "$BCE_OFF_LABEL" ]; then
    cmd+=(--bce-off-label "$BCE_OFF_LABEL")
  fi

  "${cmd[@]}"
}



run_train "$MODEL_TYPE" "$MODEL_NAME"


echo "===================================================="
echo "Done TRADES training run."