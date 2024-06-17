#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1
#SBATCH --mem=120G  
#SBATCH --job-name="cite"
module purge
module load miniconda
source activate yfhpc

NAME_ROUND="cite"

MODEL_NAME="bert-base-uncased"
#MODEL_NAME="microsoft/deberta-base"
#MODEL_NAME="roberta-base"

SAMPLES_SIZE=15
SAMPLE_TYPE='mix-n-w'
BATCH_SIZE=64
TEMPERATURE=0.3
MAX_SEQUENCE_LENGTH=512
NUM_POS_SAMPLES=6

OUTPUT_DIR="/gpfsnyu/scratch/yf2722/FAST/MixModels/Adjust_Version/${NAME_ROUND}"

# 创建输出文件夹，如果不存在的话
mkdir -p ${OUTPUT_DIR}

python -m core.trainMix \
    --round_name $NAME_ROUND \
    --model_name $MODEL_NAME \
    --samples_size $SAMPLES_SIZE \
    --sample_type $SAMPLE_TYPE \
    --batch_size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --max_sequence_length $MAX_SEQUENCE_LENGTH \
    --num_pos_samples $NUM_POS_SAMPLES >> ${OUTPUT_DIR}/MixC.out 2>&1