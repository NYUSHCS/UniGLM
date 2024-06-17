import numpy as np
import torch
import random
import json

def split_graph_Ratio(SEED, num_nodes, train_ratio=0.6, val_ratio=0.2):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    train_id = np.sort(node_id[:int(num_nodes * train_ratio)])
    val_id = np.sort(node_id[int(num_nodes * train_ratio):int(num_nodes * (train_ratio + val_ratio))])
    test_id = np.sort(node_id[int(num_nodes * (train_ratio + val_ratio)):])

    train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])
    return train_id, val_id, test_id, train_mask, val_mask, test_mask



def split_graph_Shot(SEED, num_nodes, labels, k = 20):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    num_classes = labels.max() + 1
    all_indices = np.arange(len(labels))

    train_indices = []
    labels = labels.cpu()
    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        if len(class_indices) < k:
            print(f"Not enough samples in class {i} for {k}shot learning!")
            break
        else: 
            class_train_indices = np.random.choice(class_indices, k, replace=False)
        train_indices.extend(class_train_indices)

    all_indices = np.setdiff1d(all_indices, train_indices)

    val_indices = []

    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        class_indices = np.setdiff1d(class_indices, train_indices)  # remove already chosen train_indices
        
        #! if val is not sufficient , use rest as val
        class_val_indices = np.random.choice(class_indices, len(class_indices) if len(class_indices)<30 else 30, replace=False)
        val_indices.extend(class_val_indices)

    val_indices = np.array(val_indices)
    all_indices = np.setdiff1d(all_indices, val_indices)

    # All remaining indices will be for testing
    test_indices = all_indices

    train_id, val_id, test_id = train_indices, val_indices, test_indices
    train_mask = np.isin(np.arange(len(labels)), train_indices)
    val_mask = np.isin(np.arange(len(labels)), val_indices)
    test_mask = np.isin(np.arange(len(labels)), test_indices)

    return train_id, val_id, test_id, train_mask, val_mask, test_mask