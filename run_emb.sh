#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:1
#SBATCH --mem=120G  
module purge
module load miniconda
source activate yfhpc


# 定义一个数组包含所有需要的round names
#NAME_ROUNDS=("both" "none")  # 可以根据需要增加更多round name
#NAME_ROUNDS=("IDR-mix-512-64-15-6")
NAME_ROUNDS=("dirty")
# 遍历数组中的每个元素
for NAME_ROUND in "${NAME_ROUNDS[@]}"; do
    OUTPUT_DIR="/gpfsnyu/scratch/yf2722/FAST/MixModels/Adjust_Version/${NAME_ROUND}"

    # 创建输出文件夹，如果不存在的话
    mkdir -p ${OUTPUT_DIR}

    # 运行python模块，并把输出重定向到日志文件
    python -m core.generate_emb_adjust \
        --round_name $NAME_ROUND >> ${OUTPUT_DIR}/Embs.out 2>&1
done