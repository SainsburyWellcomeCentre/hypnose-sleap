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

--> can run inference for multiple subjids and dates

## For HPC: 

Clone repo on the HPC. Make scripts executable using:
    chmod +x run_sleap_inference.sh
    chmod +x submit_sleap_inference.sh

Submit job: 
    - cd into the sleap-hypnose folder containing the scripts
    - run: ./submit_sleap_inference.sh -s 40 -d 20251128 -m ./models/251031_100645.single_instance.n=160
