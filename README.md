# SLEAP Inference Local Runner

This Repository contains a lightweight bash utility to automatically run SLEAP inference with a pre-trained model across multiple subjects and sessions.

Output predictions are saved per video in the respective derivatives folder. 

## Quick Start

clone this repository 

create a sleap environment using v1.3.3.

conda creae -n sleap -c conda-forge -c sleap sleap=1.3.3

Make the script executable: 
chmod +x run_sleap_inference_local.sh

Create a symlink for global function call
sudo ln -s ~/path/to/directory/run_sleap_inference_local.sh /usr/local/bin/run_sleap_inference_local

Adjust Base Directory and Model Path to your directory

## Convert videos from .avi to .mp4 for SLEAP (model training and labeling)

On Analysis PC, powershell automatically loads the conversion script via the PowerShell Profile
- To configure or change path, run 'notepad $PROFILE' and update path to the file. Refresh profile with '. $PROFILE' or restart powershell
- Once configured, the function is available from anywhere in PowerShell
- Run convert-avitomp4 "file\path\to\file.avi". Works with single or multiple files ("file\one.avi" "file\two.avi")
- Outputs are currently written to E:\videos_sleap_models
- Encoding uses NVIDIA NVENC (h264_nvenc, preset p7, CQ18)

## Run Inference

run_sleap_inference_local -s XXX -d YYYYMMDD 

For Windows: 
In GitBash Terminal run: $ bash run_sleap_inference_local_windows.sh -s XX -b E: -bz 64

### Defaults
- Model: `C:/Users/HarrisLab/Desktop/Repos/sleap-models/sleap_v1.5.5_new_model/models/run_trial2_251205_181823.single_instance.n=150`
- Base data root: `Z:/hypnose`
- Derivatives root: `${BASE_DIR}/derivatives`
- Batch size: `64`
- Video glob: `*.avi`
- If `BASE_DIR` is the default and `E:/rawdata` exists, it automatically switches to `E:/` to use a local copy.


### Arguments
- `-s, --subject <SUBJ ...>`: One or more subject IDs (e.g., `40` or `038`). Padding to three digits is handled internally. Repeat -s for each subject (-s 38 -s 40)
- `-d, --date <DATE|DATE_RANGE ...>`: Zero or more dates. Accepts `YYYYMMDD` or ranges `YYYYMMDD-YYYYMMDD` (inclusive). If omitted, the script discovers all dates present for each subject under `<BASE_DIR>/rawdata/sub-XXX_*`.
- `-m, --model <PATH>`: Override model path.
- `-b, --base-dir <PATH>`: Override base data root (expects `rawdata/...`).
- `-bz, --batch-size <N>`: Override batch size for `sleap-track`.

### Usage Examples
- Single subject/date with defaults:
    - `bash run_sleap_inference_local_windows.sh -s 038 -d 20251119`
- Multiple subjects and dates:
    - `bash run_sleap_inference_local_windows.sh -s 038 039 -d 20251029 20251030`
- Date range and custom model/base:
    - `bash run_sleap_inference_local_windows.sh -s 040 -d 20251101-20251105 -m D:/models/custom.slp -b E:`
- Custom batch size:
    - `bash run_sleap_inference_local_windows.sh -s 040 -d 20251125 -bz 32`


## For HPC: 

Clone repo on the HPC. Make scripts executable using:
    chmod +x run_sleap_inference.sh
    chmod +x submit_sleap_inference.sh

Submit job: 
    - cd into the sleap-hypnose folder containing the scripts
    - run: ./submit_sleap_inference.sh -s 40 -d 20251128 -m ./models/251031_100645.single_instance.n=160

## Transfer SLEAP results locally

Use the PowerShell helper [transfer_sleap_results.ps1] to copy tracking CSVs from a local run (defaults to E:/derivatives) to mounted server (defaults to Z:/hypnose/derivatives) while preserving the folder structure.

- Matches files containing sleap_tracking_video or combined_sleap_tracking_timestamps
- Preserves sub-XXX/ses-XXX_date-YYYYMMDD/saved_analysis_results layout
- Optional filters: subjects via -Sub, dates or date ranges via -Date

Examples (run from this folder in PowerShell):

```
# Preview without copying and show what files are skipped (usually .slp files) 
./transfer_sleap_results.ps1 -DryRun -ShowSkipped

# Copy everything matching the patterns
./transfer_sleap_results.ps1

# Copy only specific subjects
./transfer_sleap_results.ps1 -Sub 45 46 47

# Copy specific dates and/or a range (inclusive)
./transfer_sleap_results.ps1 -Date 20260213 20260217-20260220

# Combine subject and date filters
./transfer_sleap_results.ps1 -Sub 45 -Date 20260213-20260220
```
