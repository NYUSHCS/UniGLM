import torch

from core.GNNs.PYTGNN_Link_Trainer import PYTGNN_LP_Trainer
from core.GNNs.DGLGNN_Trainer import DGLGNNTrainer
from core.data_utils.data_loader import load_data
from torch_geometric.utils import negative_sampling, remove_self_loops
LOG_FREQ = 10

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_edges(data_edge_index, val_ratio=0.05, test_ratio=0.1, seed=42):
    seed_everything(seed)

    num_total_edges = data_edge_index.size(1)  # 使用负样本边的总数

    # 计算验证和测试集的边数
    num_val_edges = int(num_total_edges * val_ratio)
    num_test_edges = int(num_total_edges * test_ratio)

    # 将负样本边随机打乱
    perm = torch.randperm(num_total_edges)

    # 划分负样本边
    val_edge_index = data_edge_index[:, perm[:num_val_edges]]
    test_edge_index = data_edge_index[:, perm[num_val_edges:num_val_edges + num_test_edges]]
    train_edge_index = data_edge_index[:, perm[num_val_edges + num_test_edges:]]

    return train_edge_index, val_edge_index, test_edge_index


def prep_edges(data):
    # 移除自环
    edge_index, _ = remove_self_loops(data.edge_index)

    # 创建验证集、测试集和训练集的边索引
    data.train_pos_edge_index,data.val_pos_edge_index,data.test_pos_edge_index = split_edges(edge_index)
    data.negative_edge_index = negative_sampling(edge_index=edge_index, num_nodes=data.num_nodes, num_neg_samples=edge_index.size(1))
    data.train_neg_edge_index,data.val_neg_edge_index,data.test_neg_edge_index = split_edges(data.negative_edge_index)
    return data


class GNN_LP_Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers

        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = cfg.gnn.train.feature_type
        self.epochs = cfg.gnn.train.epochs
        self.weight_decay = cfg.gnn.train.weight_decay

        # ! Load data
        data, _ = load_data(self.dataset_name, seed = cfg.seed, split = cfg.split)
        data = prep_edges(data)
        self.data = data.to(self.device)

        from core.GNNs.GNN_utils import Evaluator_LP
        self._evaluator = Evaluator_LP(name=self.dataset_name)
        self.evaluator_auc = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.view(-1, 1),
             "y_true": labels.view(-1, 1)}
        )["auc"]
        self.evaluator_ap = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.view(-1, 1),
             "y_true": labels.view(-1, 1)}
        )["ap"]

        if cfg.gnn.model.name == 'RevGAT':
            self.TRAINER = DGLGNNTrainer
        else:
            self.TRAINER = PYTGNN_LP_Trainer

    def train(self):
        all_pred = []
        all_auc = {}
        all_ap = {}
        feature_types = self.feature_type.split('_')
        for feature_type in feature_types:
            trainer = self.TRAINER(self.cfg, self.data, feature_type)
            trainer.train()
            pred, auc, ap = trainer.eval_and_save()
            all_pred.append(pred)
            all_auc[feature_type] = auc
            all_ap[feature_type] = ap
        return all_auc,all_ap
