# UniGLM
Official code for "[UniGLM: Training One Unified Language Model for Text-Attributed Graphs](https://arxiv.org/abs/2406.12052)". 

## Pipeline of the UniGLM
Representation learning on text-attributed graphs (TAGs), where nodes are represented by textual descriptions, is crucial for textual and relational knowledge systems and recommendation systems. Currently, state-of-the-art embedding methods for TAGs primarily focus on fine-tuning language models (e.g., BERT) using structure-aware training signals. While effective, these methods are tailored for individual TAG and cannot generalize across various graph scenarios. Given the shared textual space, leveraging multiple TAGs for joint fine-tuning, aligning text and graph structure from different aspects, would be more beneficial. Motivated by this, we introduce a novel Unified Graph Language Model (UniGLM) framework, the first graph embedding model that generalizes well to both in-domain and cross-domain TAGs. Specifically, UniGLM is trained over multiple TAGs with different domains and scales using self-supervised contrastive learning. UniGLM includes an adaptive positive sample selection technique for identifying structurally similar nodes and a lazy contrastive module that is devised to accelerate training by minimizing repetitive encoding calculations. Extensive empirical results across 9 benchmark TAGs demonstrate UniGLM's efficacy against leading embedding baselines in terms of generalization (various downstream tasks and backbones) and transfer learning (in and out of domain scenarios).
![architecture](https://github.com/NYUSHCS/UniGLM/blob/main/img/UniGLMpipeline.png)

## 🚀Quick Start
To train and evaluate UniGLM with your own datasets, there are three steps.

### Model Training
Use run_Mix.sh to train your own model. Datasets and folder names can be changed.

### Embedding Generation
Use run_emb.sh and run_emb_transfer.sh to generate embeddings.

### GNNs Evaluation
Use run_GNN.sh to run evaluations on different GNNs.

## Reproducibility 
Model weight is available at [deltayf/UniGLM](https://huggingface.co/deltayf/UniGLM).
