import os
import torch
import numpy as np 
import dgl
import pandas as pd 
from torch_geometric.utils import add_self_loops,remove_self_loops
from torch_geometric.data import Data
from core.data_utils.dgl_transfer import CustomDGLDataset
from pecos.utils import smat_util
from torch_geometric.utils.sparse import to_edge_index
from core.data_utils.eval_utils import split_graph_Ratio,split_graph_Shot
import pandas as pd
import gzip

def load_ori_data(dataset_name, use_dgl=False):
    if dataset_name == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
    elif dataset_name == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    elif dataset_name == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
    elif dataset_name == 'arxivTA':
        from core.data_utils.load_arxivTA import get_raw_text_arxiv_2023 as get_raw_text
    elif dataset_name == 'ogbn-product':
        from core.data_utils.load_product import get_raw_text_products as get_raw_text
    else:
        exit(f'Error: Dataset {dataset_name} not supported')
    data, text = get_raw_text(use_text=True)
    if dataset_name in ['ogbn-arxiv','ogbn-product']:
        data.edge_index,_ = to_edge_index(data.edge_index)

    if use_dgl:
        data = CustomDGLDataset(dataset_name, data)
    
    return data,text
    

def build_pygData_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    category_list = df['category']
    text_list = df['text']
    label_list = df['label']
    neighbour_list = df['neighbour']

    # 创建标签到类别的映射字典
    label_to_category = dict(zip(label_list, category_list))
     
    # 获取所有类别
    category_names = [label_to_category[label] for label in sorted(label_to_category.keys())]
    x = None
    y = torch.tensor(label_list.values, dtype=torch.float)
    edge_index_list = []
    for index, neighbours in enumerate(neighbour_list):
        # 将邻居列表转换为整数
        neighbours = neighbours.strip('[]').split(',')
        neighbours = [int(neighbour.strip()) for neighbour in neighbours if neighbour.strip()]
        # 创建边索引对
        edges = [(index, neighbour) for neighbour in neighbours]
        if (index,index) not in edges:
            edges.append((index,index))
        edge_index_list.extend(edges)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    num_nodes = len(text_list)
    data = Data(x=x, edge_index=edge_index, y=y, category_names = category_names, num_nodes = num_nodes)

    return data,text_list

def build_dgl_graph_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # 获取各列数据
    text_list = df['text']
    label_list = df['label']
    neighbour_list = df['neighbour']

    # 创建图
    g = dgl.DGLGraph()
    g.add_nodes(len(text_list))
    
    # 添加边
    edge_index_list = []
    for i, neighbours in enumerate(neighbour_list):
        neighbours = neighbours.strip('[]').split(',')
        neighbours = [int(neighbour.strip()) for neighbour in neighbours if neighbour.strip()]
        for neighbour in neighbours:
            edge_index_list.append((i, neighbour))
    src, dst = tuple(zip(*edge_index_list))
    g.add_edges(src, dst)

    # 添加节点特征，初始化为全零
    num_nodes = len(text_list)
    g.ndata['feat'] = torch.zeros(num_nodes)
    g.ndata['label'] = torch.tensor(label_list, dtype=torch.float)

    return g, text_list


def load_csv_data(dataset_name, use_dgl):
    #dataset_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # 上一级目录
    dataset_dir = '/gpfsnyu/scratch/yf2722/'
    if dataset_name == "Amazon-Children" or dataset_name == "amazon-children":
        csv_path = os.path.join(dataset_dir, "dataset", "Amazon-Children", "Children_Final.csv")
    elif dataset_name == "Amazon-Computers" or dataset_name == "amazon-computers":
        csv_path = os.path.join(dataset_dir, "dataset", "Amazon-Computers", "Computers.csv")
    elif dataset_name == "Amazon-Fitness" or dataset_name == "amazon-fitness":
        csv_path = os.path.join(dataset_dir, "dataset", "Amazon-Fitness", "Sports_Fitness_Final.csv")
    elif dataset_name == "Amazon-History" or dataset_name == "amazon-history":
        csv_path = os.path.join(dataset_dir, "dataset", "Amazon-History", "History_Final.csv")
    elif dataset_name == "Amazon-Photo" or dataset_name == "amazon-photo":
        csv_path = os.path.join(dataset_dir, "dataset", "Amazon-Photo", "Photo_Final.csv")
    '''
    elif dataset_name == "Goodreads":
        csv_path = os.path.join(parent_dir, "dataset", "Goodreads", "Final_Goodreads.csv")
    elif dataset_name == "DBLP":
        csv_path = os.path.join(parent_dir, "dataset", "DBLP-Citation", "Citation-2015.csv")
    '''
    
    if use_dgl:
        data, text = build_dgl_graph_from_csv(csv_path)
    else:
        data, text = build_pygData_from_csv(csv_path)
    text = text.tolist()
    return data, text

def load_new_data(dataset_name,use_dgl=False):
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    if dataset_name == 'VideoGames':
        file = '/gpfsnyu/home/yf2722/FAST/dataset/VideoGames/meta_Video_Games.json.gz'
    df = getDF(file)
    df = df.drop_duplicates(subset=['description'], keep='first')
    df = df.dropna(subset=['description'])
    df['bought_together'] = df['related'].apply(lambda x: x.get('bought_together', []) if isinstance(x, dict) else [])
    df['also_viewed'] = df['related'].apply(lambda x: x.get('also_viewed', []) if isinstance(x, dict) else [])

    df.reset_index(drop=True, inplace=True)
    text = df['description'].tolist()
    asin_to_index = {asin: idx for idx, asin in enumerate(df['asin'].unique())}

    # 节点特征（可以根据需要选择，这里暂时忽略）
    num_nodes = len(asin_to_index)
    x = None

    train_edges = []
    val_edges = []
    test_edges = []

    for _, row in df.iterrows():
        src = asin_to_index[row['asin']]  # 源节点索引
        if isinstance(row['bought_together'], list):  # 确保bought_together是列表
            for tgt_asin in row['bought_together']:
                tgt = asin_to_index.get(tgt_asin)  # 目标节点索引
                if tgt is not None:
                    test_edges.append([src, tgt])  # 为每个bought_together添加边
        if isinstance(row['also_viewed'], list):
            for tgt_asin in row['also_viewed']:
                tgt = asin_to_index.get(tgt_asin)  # 目标节点索引
                if tgt is not None:
                    train_edges.append([src, tgt])  # 为每个bought_together添加边

    # 转换edges列表为tensor
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()   
    val_edge_index = torch.tensor(val_edges, dtype=torch.long).t().contiguous()   
    # 创建图数据对象
    data = Data(x=x, edge_index=train_edge_index,test_edge_index = test_edge_index)
    print(data)
    return data, text


def split_data_pyg(dataset_name,data,seed,split):
    num_nodes = len(data.y)

    if dataset_name == 'arxivTA' and split == 'semi-supervised':
        data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.2, val_ratio=0.2)
    elif dataset_name in ['cora', 'pubmed', 'arxivTA'] and split == 'supervised':
        data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.6, val_ratio=0.2)
    elif dataset_name in ['cora', 'pubmed'] and split == 'semi-supervised':
        data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Shot(seed, num_nodes, labels = data.y, k = 20)
    elif 'Amazon' in dataset_name and split == 'supervised':
        data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.6, val_ratio=0.2)
    elif 'Amazon' in dataset_name and split == 'semi-supervised':
        data.train_id, data.val_id, data.test_id, data.train_mask, data.val_mask, data.test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.05, val_ratio=0.2)
    elif dataset_name in ['ogbn-arxiv', 'ogbn-product']:
        pass
    return data

def split_data_dgl(dataset_name,data,seed,split):
    num_nodes = data[0].ndata['label'].squeeze().shape[0]
    labels = data[0].ndata['label'].squeeze()
    if dataset_name == 'arxivTA' and split == 'semi-supervised':
        _,_,_, train_mask, val_mask, test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.2, val_ratio=0.2)
    elif dataset_name in ['cora', 'pubmed', 'arxivTA'] and split == 'supervised':
        _,_,_, train_mask, val_mask, test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.6, val_ratio=0.2)
    elif dataset_name in ['cora', 'pubmed'] and split == 'semi-supervised':
        _,_,_, train_mask, val_mask, test_mask =  split_graph_Shot(seed, num_nodes, labels = labels, k = 20)
    elif 'Amazon' in dataset_name and split == 'supervised':
        _,_,_, train_mask, val_mask, test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.6, val_ratio=0.2)
    elif 'Amazon' in dataset_name and split == 'semi-supervised':
        _,_,_, train_mask, val_mask, test_mask =  split_graph_Ratio(seed, num_nodes, train_ratio=0.05, val_ratio=0.2)
    elif dataset_name in ['ogbn-arxiv', 'ogbn-product']:
        return data.train_mask, data.val_mask, data.test_mask
    return torch.tensor(train_mask),torch.tensor(val_mask),torch.tensor(test_mask)

def load_data(dataset_name, feature_type=None, use_dgl=False, device=0, seed = 0, split = 'supervised'):
    if dataset_name in ('cora', 'pubmed', 'ogbn-arxiv', 'ogbn-product', 'arxivTA'):
        data,text = load_ori_data(dataset_name,use_dgl=use_dgl)
    elif "Amazon" in dataset_name or dataset_name == "Goodreads" or dataset_name == "DBLP" or 'amazon' in dataset_name:
        data,text = load_csv_data(dataset_name,use_dgl)
    elif dataset_name in ['VideoGames']:
        data,text = load_new_data(dataset_name)
        return data,text
    else:
        raise ValueError(dataset_name)

    ## Update Data Split
    if not use_dgl:
        data = split_data_pyg(dataset_name, data, seed, split)
    else:
        train_mask, val_mask, test_mask = split_data_dgl(dataset_name, data, seed, split)
    ## Update Node Feature
    if feature_type == None:
        return data, text
    elif feature_type in "GIA":
        LM_emb_path = f"dataset/All_Embs/{dataset_name}/{feature_type}.emb"
        features = torch.from_numpy(smat_util.load_matrix(LM_emb_path).astype(np.float32))
    elif feature_type == "BoW":
        LM_emb_path = f"dataset/All_Embs/{dataset_name}/{feature_type}.emb"
        features = torch.load(LM_emb_path).to(torch.float)
    elif feature_type == "MixGIA-2":
        LM_emb_path = f"dataset/All_Embs/{dataset_name}/{feature_type}.emb"
        features = torch.from_numpy(smat_util.load_matrix(LM_emb_path).astype(np.float32))
    elif feature_type == "MixGIA-3":
        LM_emb_path = f"dataset/All_Embs/{dataset_name}/{feature_type}.emb"
        features = torch.from_numpy(smat_util.load_matrix(LM_emb_path).astype(np.float32))
    elif 'MixContrast' in feature_type:
        LM_emb_path = f"dataset/All_Embs/{dataset_name}/{feature_type}.emb"
        features = torch.load(LM_emb_path)
    elif 'GIA' in feature_type:
        LM_emb_path = f'/gpfsnyu/home/yf2722/FAST/dataset/All_Embs/{dataset_name}/{feature_type}.emb'
        features = torch.from_numpy(smat_util.load_matrix(LM_emb_path).astype(np.float32))
    elif 'patton' in feature_type:
        LM_emb_path = f'/gpfsnyu/home/yf2722/FAST/dataset/All_Embs/{dataset_name}/{feature_type}.npy'
        features = torch.from_numpy(np.load(LM_emb_path).astype(np.float32))
    else:
        LM_emb_path = f'/gpfsnyu/home/yf2722/FAST/dataset/All_Embs/{dataset_name}/{feature_type}.emb'
        features = torch.load(LM_emb_path)
        features = torch.stack(features, dim=0)
        
    if use_dgl:
        return data,text, features, train_mask, val_mask, test_mask
    else:
        data.x = features.to(device)  # to fit dgl , use to() instead of .cuda()
        data = data.to(device)
        
    return data,text