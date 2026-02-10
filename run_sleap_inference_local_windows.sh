#!/bin/bash

# Windows Git Bash conda activation
eval "$(conda shell.bash hook)"
conda activate sleap-gpu 2>/dev/null || conda activate sleap

# run_sleap_inference_local.sh
#
# Usage examples:
#   bash run_sleap_inference_local.sh -s 038 -d 20251119
#   bash run_sleap_inference_local.sh -s 038 039 -d 20251029 20251030
#   bash run_sleap_inference_local.sh -s 038 -d 20251029 -m /custom/model/path -b /custom/base
#

# ======================================
# DEFAULT CONFIGURATION
# ======================================

DEFAULT_MODEL="C:/Users/HarrisLab/Desktop/Repos/sleap-models/sleap_v1.5.5_new_model/models/run_trial2_251205_181823.single_instance.n=150"
DEFAULT_BASE_DIR="Z:/hypnose"
DEFAULT_DERIV_DIR="${DEFAULT_BASE_DIR}/derivatives"
DEFAULT_BATCH_SIZE=64
VIDEO_EXTENSIONS="*.avi"

MODEL="$DEFAULT_MODEL"
BASE_DIR="$DEFAULT_BASE_DIR"
DERIV_DIR="$DEFAULT_DERIV_DIR"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"

# ======================================
# HELPERS
# ======================================

is_valid_date() {
    # Validate YYYYMMDD using date
    date -d "${1:0:4}-${1:4:2}-${1:6:2}" +%Y%m%d >/dev/null 2>&1
}

add_dates_from_token() {
    local token="$1"
    if [[ $token =~ ^[0-9]{8}$ ]]; then
        # Single date
        if ! is_valid_date "$token"; then
            echo "Invalid date: $token" >&2
            exit 1
        fi
        DATES+=("$token")
    elif [[ $token =~ ^([0-9]{8})-([0-9]{8})$ ]]; then
        # Range
        local start="${BASH_REMATCH[1]}"
        local end="${BASH_REMATCH[2]}"
        if ! is_valid_date "$start" || ! is_valid_date "$end"; then
            echo "Invalid date range: $token" >&2
            exit 1
        fi
        local cur="$start"
        while true; do
            DATES+=("$cur")
            [[ "$cur" == "$end" ]] && break
            cur=$(date -d "${cur:0:4}-${cur:4:2}-${cur:6:2} + 1 day" +%Y%m%d)
        done
    else
        echo "Invalid date token: $token (use YYYYMMDD or YYYYMMDD-YYYYMMDD)" >&2
        exit 1
    fi
}

# ======================================
# ARGUMENT PARSING
# ======================================

SUBJECTS=()
DATES=()
SUBJECT_ORDER=()
declare -A VIDEO_COUNTS

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--subject)
            SUBJECTS+=("$2")
            shift 2
            ;;
        -d|--date)
            while [[ $# -gt 1 && ! "$2" =~ ^- ]]; do
                add_dates_from_token "$2"
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
        -bz|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    echo "Usage: bash run_sleap_inference_local.sh -s <SUBJ> [SUBJ2 ...] [-d <DATE|DATE_RANGE> ...] [-m <MODEL>] [-b <BASE_DIR>] [-bz <BATCH_SIZE>]"
    echo "Dates can be YYYYMMDD or YYYYMMDD-YYYYMMDD (inclusive range). If -d is omitted, all dates found for each subject are processed."
    echo "Defaults: MODEL=$DEFAULT_MODEL, BASE_DIR=$DEFAULT_BASE_DIR, DERIV_DIR=$DEFAULT_DERIV_DIR, BATCH_SIZE=$DEFAULT_BATCH_SIZE"
    exit 1
fi

# Prefer a local E: drive copy when present (e.g., E:/rawdata)
if [[ "$BASE_DIR" == "$DEFAULT_BASE_DIR" && -d "E:/rawdata" ]]; then
    echo "Detected local copy at E:/rawdata; switching BASE_DIR to E:/"
    BASE_DIR="E:"
fi

# ======================================
# MAIN LOGIC
# ======================================

for SUBJ in "${SUBJECTS[@]}"; do
    # Pad subject number with leading zeros (e.g., 40 -> 040)
    SUBJ_PADDED=$(printf "%03d" "$SUBJ")
    if [[ ! " ${SUBJECT_ORDER[*]} " =~ " ${SUBJ_PADDED} " ]]; then
        SUBJECT_ORDER+=("$SUBJ_PADDED")
    fi

    # Determine dates: use provided DATES, or discover all session dates if none specified
    EFFECTIVE_DATES=()
    if [[ ${#DATES[@]} -eq 0 ]]; then
        SEARCH_PATH="${BASE_DIR}/rawdata/sub-${SUBJ_PADDED}_*"
        for SUBJ_DIR in $SEARCH_PATH; do
            [[ ! -d "$SUBJ_DIR" ]] && continue
            while IFS= read -r sess; do
                DATE_PART=$(basename "$sess" | sed -E 's/^ses-[0-9]+_date-([0-9]{8}).*/\1/')
                [[ -n "$DATE_PART" ]] && EFFECTIVE_DATES+=("$DATE_PART")
            done < <(find "$SUBJ_DIR" -maxdepth 1 -type d -name "ses-*_date-*" 2>/dev/null)
        done
        # deduplicate
        if [[ ${#EFFECTIVE_DATES[@]} -gt 0 ]]; then
            mapfile -t EFFECTIVE_DATES < <(printf "%s\n" "${EFFECTIVE_DATES[@]}" | sort -u)
        fi
    else
        EFFECTIVE_DATES=("${DATES[@]}")
    fi

    for DATE in "${EFFECTIVE_DATES[@]}"; do
        echo ">>> Processing subject $SUBJ_PADDED, date $DATE"
        SEARCH_PATH="${BASE_DIR}/rawdata/sub-${SUBJ_PADDED}_*"
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
            VIDEO_COUNTS["${SUBJ_PADDED}|${DATE}"]=0
            continue
        fi

        echo "Found ${#VIDEOS_TO_PROCESS[@]} video(s)."

        for VIDEO in "${VIDEOS_TO_PROCESS[@]}"; do
            echo "→ Processing: $VIDEO"

            # Extract subject/session folder names (relative to rawdata hierarchy)
            SUBJ_DIR_NAME=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$VIDEO")")")")")")
            SESS_DIR_NAME=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$VIDEO")")")")")

            # Extract behav folder name directly from the video path to avoid cross-folder reuse
            # VIDEO: .../behav/<behav_ts>/VideoData/<file>.avi
            BEHAV_DIR_NAME=$(basename "$(dirname "$(dirname "$VIDEO")")")

            OUTPUT_DIR="${DERIV_DIR}/${SUBJ_DIR_NAME}/${SESS_DIR_NAME}/saved_analysis_results"
            mkdir -p "$OUTPUT_DIR"

            BASENAME=$(basename "${VIDEO%.avi}")
            SAFE_PREFIX="${BEHAV_DIR_NAME}__${BASENAME}"
            OUTPUT_FILE="${OUTPUT_DIR}/${SAFE_PREFIX}.predictions.slp"

            # Set PyTorch precision and run inference with batch size
            if python -c "import torch; torch.set_float32_matmul_precision('high')" && \
               sleap-track "$VIDEO" -m "$MODEL" -o "$OUTPUT_FILE" --batch_size "$BATCH_SIZE"; then
                echo "✓ Completed: $(basename "$VIDEO") → $OUTPUT_FILE"
            else
                echo "✗ FAILED: $(basename "$VIDEO")"
            fi
        done

        echo "✅ Finished subject $SUBJ, date $DATE"
        VIDEO_COUNTS["${SUBJ_PADDED}|${DATE}"]=${#VIDEOS_TO_PROCESS[@]}
    done
done

echo ""
echo "=== Summary ==="
for SUBJ in "${SUBJECT_ORDER[@]}"; do
    echo "Subject ${SUBJ}"
    for DATE in "${DATES[@]}"; do
        key="${SUBJ}|${DATE}"
        if [[ -n "${VIDEO_COUNTS[$key]+set}" ]]; then
            echo "    ${DATE} (${VIDEO_COUNTS[$key]})"
        fi
    done
done
