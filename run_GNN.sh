#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:1

module purge
module load miniconda
source activate yfhpc

#pubmed' 'ogbn-arxiv' 'ogbn-product' 'arxivTA'
#'Amazon-Children' 'Amazon-Computers' 'Amazon-Fitness' 'Amazon-History' 'Amazon-Photo' 'VideoGames'
for dataset in 'pubmed'
do 
    for feature_type in 'bert'
    do
        for split in 'semi-supervised' #'supervised'
        #for split in 'supervised' #'supervised'
        do
            mkdir -p "results/out/$dataset/$feature_type/$split"
            # 如果数据集名称包含'ogb'并且split为'supervised'，则跳过此次循环
            if [[ $dataset == *"ogb"* && $split == "supervised" ]]; then
                continue
            fi
            # 运行模型训练
            #python -m core.trainGNNs gnn.model.name MLP gnn.train.feature_type $feature_type dataset $dataset split $split >> "results/out/$dataset/$feature_type/$split/mlp.out"
            #python -m core.trainGNNs gnn.model.name GCN gnn.train.feature_type $feature_type dataset $dataset split $split >> "results/out/$dataset/$feature_type/$split/gcn.out"
            #python -m core.trainGNNs gnn.model.name SAGE gnn.train.feature_type $feature_type dataset $dataset split $split >> "results/out/$dataset/$feature_type/$split/sage.out"
            python -m core.trainGNNs gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.5 gnn.train.feature_type $feature_type dataset $dataset split $split >> "results/out/$dataset/$feature_type/$split/revgat.out"
        done
    done
done
