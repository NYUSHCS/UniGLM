from core.GNNs.utils import init_path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch

def get_gnn_trainer(model):
    if model in ['GCN', 'RevGAT', 'SAGE']:
        from GNNs.PYTGNN_Trainer import PYTGNNTrainer
    else:
        raise ValueError(f'GNN-Trainer for model {model} is not defined')
    return PYTGNNTrainer


class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}

class Evaluator_LP:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):

        y_true = input_dict["y_true"].detach().cpu().numpy().astype(np.int32)  # 真实标签
        y_scores = input_dict["y_pred"].detach().cpu().numpy() # 模型输出的预测概率

        # 计算 AUC-ROC
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = float('nan')  # 如果样本中只有一类标签存在，roc_auc_score会抛出错误

        # Average Precision (AP)
        avg_precision = average_precision_score(y_true, y_scores)

        return {'auc': auc,'ap':avg_precision}


"""
Early stop modified from DGL implementation
"""


class EarlyStopping:
    def __init__(self, patience=10, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        if isinstance(path, list):
            self.path = [init_path(p) for p in path]
        else:
            self.path = init_path(path)

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        es_str = f'{self.counter:02d}/{self.patience:02d} | BestVal={self.best_score:.4f}@E{self.best_epoch}'
        return self.early_stop, es_str

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        if isinstance(model, list):
            for i, m in enumerate(model):
                torch.save(m.state_dict(), self.path[i])
        else:
            torch.save(model.state_dict(), self.path)
