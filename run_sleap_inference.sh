#!/bin/bash

# Usage: ./run_sleap_inference.sh -s 001 -d 20200317 20200318 -m /path/to/model

# Don't exit on error - handle failures individually
set -o pipefail

SUBJECTS=()
DATES=()
MODEL=""
BASE_DIR="/ceph/harris/hypnose"
VIDEO_EXTENSIONS="*.avi"

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

if [[ ${#SUBJECTS[@]} -eq 0 ]] || [[ ${#DATES[@]} -eq 0 ]] || [[ -z "$MODEL" ]]; then
    echo "Error: Missing required arguments"
    exit 1
fi

if [[ ! -d "$MODEL" ]]; then
    echo "Error: Model directory not found: $MODEL"
    exit 1
fi

declare -a VIDEOS_TO_PROCESS
FAILED_VIDEOS=()
SUCCESSFUL_VIDEOS=()

for SUBJ in "${SUBJECTS[@]}"; do
    for DATE in "${DATES[@]}"; do
        SEARCH_PATH="${BASE_DIR}/rawdata/sub-${SUBJ}_*"
        
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
    done
done

if [[ ${#VIDEOS_TO_PROCESS[@]} -eq 0 ]]; then
    echo "No videos found matching criteria."
    exit 1
fi

echo "Found ${#VIDEOS_TO_PROCESS[@]} videos to process"

VIDEO_COUNT=1
for VIDEO in "${VIDEOS_TO_PROCESS[@]}"; do
    BASENAME=$(basename "$VIDEO")
    SUBJ_DIR=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$VIDEO")")")")")
    SESS_DIR=$(basename "$(dirname "$(dirname "$(dirname "$VIDEO")")")")
    
    OUTPUT_DIR="${BASE_DIR}/derivatives/${SUBJ_DIR}/${SESS_DIR}/saved_analysis_results"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME%.*}.predictions.slp"
    
    mkdir -p "$OUTPUT_DIR"
    
    echo "[${VIDEO_COUNT}/${#VIDEOS_TO_PROCESS[@]}] Processing: $VIDEO"
    
    if sleap-track "$VIDEO" -m "$MODEL" --gpu auto -o "$OUTPUT_FILE" --verbosity json --no-empty-frames 2>&1; then
        echo "  ✓ Completed"
        SUCCESSFUL_VIDEOS+=("$VIDEO")
    else
        echo "  ✗ FAILED"
        FAILED_VIDEOS+=("$VIDEO")
    fi
    
    ((VIDEO_COUNT++))
done

echo ""
echo "=========================================="
echo "Summary: ${#SUCCESSFUL_VIDEOS[@]} successful, ${#FAILED_VIDEOS[@]} failed"
if [[ ${#FAILED_VIDEOS[@]} -gt 0 ]]; then
    echo "Failed videos:"
    printf '  - %s\n' "${FAILED_VIDEOS[@]}"
fi
echo "=========================================="

exit 0