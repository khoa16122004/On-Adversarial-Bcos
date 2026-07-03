#!/bin/bash
#SBATCH --job-name=BCOS_SimBA
#SBATCH --output=SimBA/mps_%j.out
#SBATCH --error=SimBA/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2 # Khong khai bao GPU; --gres=mps:l40:2 (card L40); --gres=mps:a100:2 (card A100)
#SBATCH --mem=4G
#SBATCH --time=72:00:00
REQUIRED_VRAM=40000
# =========================================================
# CHUAN BI MOI TRUONG
# =========================================================
module clear -f
cd /datastore/elo/khoatn/BCOS_ATTACK/script
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack
echo "ENV:" $CONDA_DEFAULT_ENV
echo "PREFIX:" $CONDA_PREFIX
which python
python -c "import sys; print(sys.executable)"

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

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# CHAY CODE
# =========================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

ATTACK_JSON="$PROJECT_ROOT/classification_result/attack_1k.json"
OUTPUT_ROOT="$PROJECT_ROOT/attack_result"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
DEVICE="cuda"
STEPS=100
EPSILONS=(0.5 1.0 2.0 5.0)
ORDER="rand"
FREQ_DIMS=14
STRIDE=7
LINF_BOUND=0.0
IMAGE_SIZE=224

run_simba() {
    local model_type="$1"
    local model_name="$2"
    local checkpoint="${3:-}"

    echo "===================================================="
    echo "Running SimBA: model_type=$model_type | model_name=$model_name"

    local cmd=(
        python SimBA_attack.py
        --model-type "$model_type"
        --model-name "$model_name"
        --attack-json "$ATTACK_JSON"
        --output-root "$OUTPUT_ROOT"
        --checkpoint-dir "$CHECKPOINT_DIR"
        --steps "$STEPS"
        --device "$DEVICE"
        --order "$ORDER"
        --freq-dims "$FREQ_DIMS"
        --stride "$STRIDE"
        --linf-bound "$LINF_BOUND"
        --image-size "$IMAGE_SIZE"
        --epsilons "${EPSILONS[@]}"
    )

    if [ -n "$checkpoint" ]; then
        cmd+=(--checkpoint "$checkpoint")
    fi

    "${cmd[@]}"
}

run_simba "torchvision" "resnet50"
run_simba "torchvision" "densenet121"
run_simba "torchvision" "vit_b_16"

run_simba "bcos" "resnet50"
run_simba "bcos" "densenet121"
run_simba "bcos" "simple_vit_b_patch16_224"

run_simba "bcosify" "resnet50"
run_simba "bcosify" "densenet121"
run_simba "bcosify" "simple_vit_b_patch16_224" \
    "$PROJECT_ROOT/checkpoints/bcosify/bcosifyv2_bcos_simple_vit_b_patch16_224_0.001_lrWarmup_gapReorder.ckpt"

echo "===================================================="
echo "Done running SimBA for all 9 models. Output: $OUTPUT_ROOT"
