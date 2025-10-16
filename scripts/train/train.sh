# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#!/bin/bash

# Parse command line arguments
LIGHT_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --light)
            LIGHT_FLAG="--light"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

#set REPO_DIR
REPO_DIR="/home/azureuser/repos/TNC/tnc.rice-irrigation-mapping-s1s2/"
OUTPUT_DIR="/mnt/ephemeral_raid/rice_mapping/training_output"

# Here we run

#1. SOWING: Jun-Sep
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro1/HC_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro1/HC_P_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro1/HC_SE_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro1/HC_P_SE_jun-sep --TASK=DSR $LIGHT_FLAG

python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro2/HC_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro2/HC_P_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro2/HC_SE_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro2/HC_P_SE_jun-sep --TASK=DSR $LIGHT_FLAG

python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro3/HC_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro3/HC_P_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro3/HC_SE_jun-sep --TASK=DSR $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro3/HC_P_SE_jun-sep --TASK=DSR $LIGHT_FLAG

#2. Irrigation: May-Dec
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro1/HC_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro1/HC_P_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro1/HC_SE_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro1/HC_P_SE_may-dec --TASK=AWD $LIGHT_FLAG

python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro2/HC_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro2/HC_P_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro2/HC_SE_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro2/HC_P_SE_may-dec --TASK=AWD $LIGHT_FLAG

python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro3/HC_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro3/HC_P_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro3/HC_SE_may-dec --TASK=AWD $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/05-01_12-15_f10d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro3/HC_P_SE_may-dec --TASK=AWD $LIGHT_FLAG


#3. 3-way: Jun-Sep
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro1/HC_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro1/HC_P_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro1/HC_SE_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro1/HC_P_SE_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG

python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro2/HC_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro2/HC_P_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro2/HC_SE_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro2/HC_P_SE_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG

python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC.parquet --output_dir=$OUTPUT_DIR/repro3/HC_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P.parquet --output_dir=$OUTPUT_DIR/repro3/HC_P_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_SE.parquet --output_dir=$OUTPUT_DIR/repro3/HC_SE_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG
python train_model.py --train_ft_path=$REPO_DIR/data/features/06-01_09-05_f4d/train_HC_P_SE.parquet --output_dir=$OUTPUT_DIR/repro3/HC_P_SE_jun-sep --TASK=ALL_TASKS $LIGHT_FLAG