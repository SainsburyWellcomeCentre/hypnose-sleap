#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sleap

# run_sleap_inference_local.sh
#
# Usage examples:
#   bash run_sleap_inference_local.sh -s 038 -d 20251029
#   bash run_sleap_inference_local.sh -s 038 039 -d 20251029 20251030
#   bash run_sleap_inference_local.sh -s 038 -d 20251029 -m /custom/model/path -b /custom/base
#

# ======================================
# DEFAULT CONFIGURATION
# ======================================

DEFAULT_MODEL="/Users/joschua/repos/harris_lab/sleap-hypnose/models/251031_100645.single_instance.n=160"
DEFAULT_BASE_DIR="/Volumes/harris/hypnose"
VIDEO_EXTENSIONS="*.avi"

MODEL="$DEFAULT_MODEL"
BASE_DIR="$DEFAULT_BASE_DIR"

# ======================================
# ARGUMENT PARSING
# ======================================

SUBJECTS=()
DATES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--subject)
            SUBJECTS+=("$2")
            shift 2
            ;;
        -d|--date)
            while [[ $# -gt 1 && ! "$2" =~ ^- ]]; do
                DATES+=("$2")
                shift
            done
            shift
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -b|--base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ${#SUBJECTS[@]} -eq 0 ]] || [[ ${#DATES[@]} -eq 0 ]]; then
    echo "Usage: bash run_sleap_inference_local.sh -s <SUBJ> [SUBJ2 ...] -d <DATE> [DATE2 ...] [-m <MODEL>] [-b <BASE_DIR>]"
    echo "Defaults: MODEL=$DEFAULT_MODEL, BASE_DIR=$DEFAULT_BASE_DIR"
    exit 1
fi

# ======================================
# MAIN LOGIC
# ======================================

for SUBJ in "${SUBJECTS[@]}"; do
    for DATE in "${DATES[@]}"; do
        echo ">>> Processing subject $SUBJ, date $DATE"
        SEARCH_PATH="${BASE_DIR}/rawdata/sub-${SUBJ}_*"
        VIDEOS_TO_PROCESS=()

        for SUBJ_DIR in $SEARCH_PATH; do
            [[ ! -d "$SUBJ_DIR" ]] && continue

            SESSION_DIR=$(find "$SUBJ_DIR" -maxdepth 1 -type d -name "ses-*_date-${DATE}" 2>/dev/null | head -1)
            [[ -z "$SESSION_DIR" ]] && continue

            BEHAV_DIR="$SESSION_DIR/behav"
            [[ ! -d "$BEHAV_DIR" ]] && continue

            for TS_DIR in "$BEHAV_DIR"/*T*; do
                [[ ! -d "$TS_DIR" ]] && continue

                VIDEO_DIR="$TS_DIR/VideoData"
                [[ ! -d "$VIDEO_DIR" ]] && continue

                for VIDEO in "$VIDEO_DIR"/$VIDEO_EXTENSIONS; do
                    [[ -f "$VIDEO" ]] && VIDEOS_TO_PROCESS+=("$VIDEO")
                done
            done
        done

        if [[ ${#VIDEOS_TO_PROCESS[@]} -eq 0 ]]; then
            echo "⚠️ No videos found for sub-${SUBJ} / date ${DATE}."
            continue
        fi

        echo "Found ${#VIDEOS_TO_PROCESS[@]} video(s)."

        for VIDEO in "${VIDEOS_TO_PROCESS[@]}"; do
            echo "→ Processing: $VIDEO"

            # Extract subject/session folder names
            SUBJ_DIR_NAME=$(basename "$(dirname "$(dirname "$VIDEO")")")
            SESS_DIR_NAME=$(basename "$(dirname "$(dirname "$(dirname "$VIDEO")")")")

            OUTPUT_DIR="${BASE_DIR}/derivatives/${SUBJ_DIR_NAME}/${SESS_DIR_NAME}/saved_analysis_results"
            mkdir -p "$OUTPUT_DIR"

            BASENAME=$(basename "${VIDEO%.avi}")
            OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}.predictions.slp"

            if sleap-track "$VIDEO" -m "$MODEL" -o "$OUTPUT_FILE"; then
                echo "✓ Completed: $(basename "$VIDEO") → $OUTPUT_FILE"
            else
                echo "✗ FAILED: $(basename "$VIDEO")"
            fi
        done

        echo "✅ Finished subject $SUBJ, date $DATE"
    done
done
