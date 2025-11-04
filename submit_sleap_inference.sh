#!/bin/bash

# submit_sleap_inference.sh
# Submits a SLURM job to run SLEAP inference
# Usage: ./submit_sleap_inference.sh -s 001 -d 20200317 -m /path/to/model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/run_sleap_inference.sh"

if [[ ! -f "$INFERENCE_SCRIPT" ]]; then
    echo "Error: Inference script not found at $INFERENCE_SCRIPT"
    exit 1
fi

# Parse email
EMAIL="${USER}@ucl.ac.uk"

# Create temporary SLURM script
SLURM_SCRIPT=$(mktemp /tmp/slurm_sleap_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << 'SLURM_EOF'
#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem 32G
#SBATCH -n 4
#SBATCH -t 0-12:00
#SBATCH --gres gpu:1
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
SLURM_EOF

echo "#SBATCH --mail-user=${EMAIL}" >> "$SLURM_SCRIPT"

cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'

# Load required modules
module load cuda/12.1
module load SLEAP/2025-09-30

# Change to script directory
cd "SCRIPT_DIR_PLACEHOLDER"

# Run inference with all arguments
bash run_sleap_inference.sh "${ARGS[@]}"
exit $?
SLURM_EOF

# Replace placeholder (works on both Mac and Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" "$SLURM_SCRIPT"
else
    sed -i "s|SCRIPT_DIR_PLACEHOLDER|${SCRIPT_DIR}|g" "$SLURM_SCRIPT"
fi

chmod +x "$SLURM_SCRIPT"

# Store arguments
ARGS=("$@")

# Submit job with arguments
echo "Submitting SLURM job..."
echo "Arguments: $@"
echo ""

JOB_ID=$(sbatch --export=ARGS="$(printf '%q ' "${ARGS[@]}")" "$SLURM_SCRIPT" | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
    echo "Error: Failed to submit job"
    rm "$SLURM_SCRIPT"
    exit 1
fi

echo "✓ Job submitted! Job ID: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f slurm.*.${JOB_ID}.out"
echo ""
rm "$SLURM_SCRIPT"