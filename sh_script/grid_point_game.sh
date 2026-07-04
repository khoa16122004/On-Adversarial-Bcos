#!/bin/bash
#SBATCH --job-name=grid_point_game
#SBATCH --output=grid_point_game/mps_%j.out
#SBATCH --error=grid_point_game/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2
#SBATCH --mem=8G
#SBATCH --time=72:00:00

REQUIRED_VRAM=20000

module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack

echo "ENV: $CONDA_DEFAULT_ENV"
echo "PREFIX: $CONDA_PREFIX"
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$PROJECT_ROOT/grid_point_game"

SEEDS=(42)
INPUT_BASE="$PROJECT_ROOT/localized/failed_transfer"

if [ ! -d "$INPUT_BASE" ]; then
    echo "Input folder not found: $INPUT_BASE"
    exit 1
fi

mapfile -t INPUT_JSONS < <(find "$INPUT_BASE" -type f -name "transfer_failed_*.json" | sort)

if [ ${#INPUT_JSONS[@]} -eq 0 ]; then
    echo "No transfer_failed JSON found under: $INPUT_BASE"
    exit 1
fi

echo "Found ${#INPUT_JSONS[@]} failed-transfer JSON files."

for input_json in "${INPUT_JSONS[@]}"; do
    rel_path="${input_json#${INPUT_BASE}/}"
    transfer_tag="${rel_path%%/*}"

    remain_path="${rel_path#*/}"
    epsilon_tag="${remain_path%%/*}"

    target_part="${transfer_tag##*__to__}"
    target_model_type="${target_part%%_*}"
    target_model_name="${target_part#*_}"

    explain_method="bcos-explain"
    if [ "$target_model_type" = "torchvision" ]; then
        explain_method="simple-gradient"
    fi

    output_dir="$PROJECT_ROOT/grid_point_game/$transfer_tag/$epsilon_tag"
    mkdir -p "$output_dir"

    echo "===================================================="
    echo "Input JSON      : $input_json"
    echo "Transfer tag    : $transfer_tag"
    echo "Epsilon tag     : $epsilon_tag"
    echo "Target model    : $target_model_type/$target_model_name"
    echo "Explain method  : $explain_method"
    echo "Output dir      : $output_dir"

    for seed in "${SEEDS[@]}"; do
        echo "Running grid_point_game with seed=$seed"
        python script/grid_point_game.py \
            --input-json "$input_json" \
            --model-type "$target_model_type" \
            --model-name "$target_model_name" \
            --explain-method "$explain_method" \
            --device cuda \
            --output-dir "$output_dir" \
            --seed "$seed"
    done
done

echo "Done grid point game for all transfer pairs."