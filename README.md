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
- `-s, --subject <SUBJ ...>`: One or more subject IDs (e.g., `40` or `038`). Padding to three digits is handled internally.
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
