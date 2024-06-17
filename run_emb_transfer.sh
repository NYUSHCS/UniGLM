#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1
#SBATCH --mem=120G  
#SBATCH --job-name="bert_emb"
module purge
module load miniconda
source activate yfhpc


# 定义一个数组包含所有需要的round names
#NAME_ROUNDS=("bert-mix-ori")  # 可以根据需要增加更多round name
NAME_ROUNDS=("original")
for NAME_ROUND in "${NAME_ROUNDS[@]}"; do
    OUTPUT_DIR="/gpfsnyu/scratch/yf2722/FAST/MixModels/transfer/${NAME_ROUND}"

    # 创建输出文件夹，如果不存在的话
    mkdir -p ${OUTPUT_DIR}

    # 运行python模块，并把输出重定向到日志文件
    python -m core.generate_emb_transfer \
        --round_name $NAME_ROUND >> ${OUTPUT_DIR}/Embs.out 2>&1
done