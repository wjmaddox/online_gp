#!/bin/bash
OMP_NUM_THREADS=4

script_name='experiments/classification.py'
datasets=("svm_guide_1")
models=("exact_gpd")
num_trials=3

for dataset in ${datasets[@]}
do
  for model in ${models[@]}
  do
    for ((trial_id=0; trial_id<${num_trials}; trial_id++))
    do
      echo "launching trial ${trial_id}"
      python ${script_name} dataset=${dataset} model=${model} trial_id=${trial_id} logger=s3 stem=linear seed=${trial_id}
      done
    done
done
							      

