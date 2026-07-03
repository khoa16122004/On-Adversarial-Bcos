#!/bin/bash
#SBATCH --job-name=BCOS_PGD
#SBATCH --output=PGD/mps_%j.out
#SBATCH --error=PGD/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2 # Không khai báo GPU; --gres=mps:l40:2 (card L40); --gres=mps:a100:2 (card A100, ưu tiên các job dùng > 40GB vRAM)
#SBATCH --mem=4G
#SBATCH --time=72:00:00
REQUIRED_VRAM=40000  # Quan trọng - Số vRAM cần dùng (để tìm GPU phù hợp)
# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
# *** Kích hoạt venv (Sửa đường dẫn / môi trường theo user)
cd /datastore/elo/khoatn/BCOS_ATTACK/script
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack
echo "ENV:" $CONDA_DEFAULT_ENV
echo "PREFIX:" $CONDA_PREFIX
which python
python -c "import sys; print(sys.executable)"

# Xóa biến môi trường Slurm để tự chọn GPU
unset CUDA_VISIBLE_DEVICES
# --- GỌI HELPER --- (Quan trọng, cần gọi hàm này (có sẵn) để tìm GPU có vRAM trống >= REQUIRED_VRAM, nếu không tìn thấy GPU đủ vRAM thì hàm CHECK_OUT sẽ đưa job vào lại hàng đợi để chờ tìm slot khác; sau 5 lần requeue mà vẫn chưa tìm được slot thì sẽ trả về mã lỗi để kết thúc job)
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
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"


export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU 

# =========================================================
# CHẠY CODE
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
BATCH_SIZE=32
EPSILONS=(0.03 0.05 0.1 0.2)

run_pgd() {
    local model_type="$1"
    local model_name="$2"
    local checkpoint="${3:-}"

    echo "===================================================="
    echo "Running PGD: model_type=$model_type | model_name=$model_name"

    local cmd=(
        python PGD_attack.py
        --model-type "$model_type"
        --model-name "$model_name"
        --attack-json "$ATTACK_JSON"
        --output-root "$OUTPUT_ROOT"
        --checkpoint-dir "$CHECKPOINT_DIR"
        --steps "$STEPS"
        --batch-size "$BATCH_SIZE"
        --device "$DEVICE"
        --epsilons "${EPSILONS[@]}"
    )

    if [ -n "$checkpoint" ]; then
        cmd+=(--checkpoint "$checkpoint")
    fi

    "${cmd[@]}"
}

# # 9 models = 3 model types x 3 model names
# run_pgd "torchvision" "resnet50"
# run_pgd "torchvision" "densenet121"
# run_pgd "torchvision" "vit_b_16"

run_pgd "bcos" "resnet_50"
run_pgd "bcos" "densenet_121"
run_pgd "bcos" "simple_vit_b_patch16_224"

run_pgd "bcosify" "resnet_50"
run_pgd "bcosify" "densenet_121"
run_pgd "bcosify" "simple_vit_b_patch16_224" \
    "$PROJECT_ROOT/checkpoints/bcosify/bcosifyv2_bcos_simple_vit_b_patch16_224_0.001_lrWarmup_gapReorder.ckpt"

echo "===================================================="
echo "Done running PGD for all 9 models. Output: $OUTPUT_ROOT"