#!/bin/bash

# Configurable Parameters
data_dir="/data/" # Your data folder
model_file="/models/dummy_model.pt"  # The model path
output_channels=2  # model output channel 
spacing_x=1.5
spacing_y=1.5
spacing_z=2.0
hu_min=-500
hu_max=500

python run_segmentation.py \
    --roi_x 128 \
    --roi_y 128 \
    --roi_z 128 \
    --space_x "${spacing_x}" \
    --space_y "${spacing_y}" \
    --space_z "${spacing_z}" \
    --a_min "${hu_min}" \
    --a_max "${hu_max}" \
    --scale_intensity \
    --data_dir "${data_dir}/nii_image/" \
    --out_channels "${output_channels}" \
    --load_weight_name "${model_file}" \
    --save_folder "${data_dir}/nii_seg/" \
    --model_feature 128