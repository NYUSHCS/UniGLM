from transformers import AutoTokenizer,AutoModel,AutoConfig,BertModel,RobertaModel,DebertaModel
from torch_geometric.data import DataLoader
from core.LLMs.utils import neighbor_sampler,init_path,neighbor_sampler_PR,neighbor_sampler_Mix,neighbor_sampler_Mix_Three,neighbor_sampler_Mix_New,ablation_study_one_no_Degree,ablation_study_two_no_Dataset
import random
import torch
from tqdm import tqdm
from core.LLMs.DataReader import MixDataset,Dataset
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool
import os
import pickle
import json
import random
import time


def random_sample_from_data(data, k): #data： [tensor,tensor,tensor(batch_size)]
    data = [tensor.tolist() for tensor in data]
    data = [[row[i] for row in data] for i in range(len(data[0]))] #(batch_size, 15)
    selected_data = [random.sample(array, k) for array in data] #(batch_size, 6)
    transposed_selected_data = [[row[i] for row in selected_data] for i in range(k)]
    # 将每个子列表转换为张量
    tensors = [torch.tensor(data) for data in transposed_selected_data]
    return tensors

def random_sample_from_data_new(data, k): 
    # 转换成列表以便操作
    data = [tensor.tolist() for tensor in data]
    # 转置数据，以确保一行一个 batch
    data = [[row[i] for row in data] for i in range(len(data[0]))]
    # 按照所需采样数量进行采样
    selected_data = []
    for array in data:
        filtered_array = [item for item in array if item != -1]
        if len(filtered_array) < k:
            samples = []
            while len(samples) < k:
                samples.extend(random.sample(filtered_array, min(len(filtered_array), k - len(samples))))
            selected_data.append(samples)
        else:
            selected_data.append(random.sample(filtered_array, k))
    transposed_selected_data = [[row[i] for row in selected_data] for i in range(k)]
    # 转换成张量
    tensors = [torch.tensor(data) for data in transposed_selected_data]
    return tensors

def weighed_sample_from_data(data, k, weights):
    # 确保数据是张量列表
    data = [tensor for tensor in data]
    weights = [tensor for tensor in weights]
    # 转置数据
    data_transposed = torch.stack(data, dim=1)
    weights_transposed = torch.stack(weights, dim=1)
    new_data = []  # 保存新数据
    # 从每一行中抽样 k 个元素
    for i in range(data_transposed.shape[0]):
        row = data_transposed[i]
        weight = weights_transposed[i]
        # 确保权重和行长度一致
        assert len(row) == len(weight), f"row length: {len(row)}, weight length: {len(weight)}"
        # 根据权重进行采样
        try:
            sampled_values = torch.tensor(np.random.choice(row.numpy(), k, p=weight.numpy(), replace=False))
        except ValueError as e:
            print("Error occurred during sampling:")
            print("row:", row)
            print("weight:", weight)
            print("sum:", torch.sum(weight))
            raise e
        new_data.append(sampled_values)  # 添加采样结果到 new_data
    # 重新转置数据，使其恢复到原始格式
    data_final = torch.stack(new_data, dim=1)
    return data_final

def weighed_sample_from_data_new(data, k, weights):
    # 确保数据是张量列表
    data = [tensor for tensor in data]
    weights = [tensor for tensor in weights]
    # 转置数据
    data_transposed = torch.stack(data, dim=1)
    weights_transposed = torch.stack(weights, dim=1)
    new_data = []  # 保存新数据
    # 从每一行中抽样 k 个元素
    for i in range(data_transposed.shape[0]):
        row = data_transposed[i]
        weight = weights_transposed[i]
        # 保留有效的索引
        valid_indices = [j for j in range(len(weight)) if weight[j] > 0]
        # 根据有效索引提取行和权重
        cleaned_row = row[valid_indices]
        cleaned_weight = weight[valid_indices]

        # 确保权重和行长度一致
        assert len(row) == len(weight), f"row length: {len(row)}, weight length: {len(weight)}"
        # 根据权重进行采样
        if torch.sum(weight) == 0:
            raise ValueError("Weights cannot all be zero")
        sampled_values = []
        if len(cleaned_row) < k:
            # 有放回采样以确保采样足够
            sampled_values = np.random.choice(
                cleaned_row.numpy(),
                k,
                p=cleaned_weight.numpy(),
                replace=True
            )
        else:
            # 无放回采样
            sampled_values = np.random.choice(
                cleaned_row.numpy(),
                k,
                p=cleaned_weight.numpy(),
                replace=False
            )

        new_data.append(torch.tensor(sampled_values))  # 添加采样结果到 new_data
    # 重新转置数据，使其恢复到原始格式
    data_final = torch.stack(new_data, dim=1)
    return data_final

class MixDatasetsBERTModel(BertModel):
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = outputs['last_hidden_state'][:, 0, :]
        return cls

class MixDatasetsRoBERTaModel(RobertaModel):
    def forward(self, input_ids, attention_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return cls

class MixDatasetsDeBERTaModel(DebertaModel):
    def forward(self, input_ids, attention_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return cls
    
class MixCLTrainer():
    def __init__(self, train_datas, train_texts, train_dataset_names, samples_size, name_of_round, model_name, sample_type, batch_size, temperature, max_sequence_length, num_pos_samples, device):
        #train_params
        self.sample_type = sample_type
        self.epochs = 3
        self.batch_size = batch_size
        self.lr = 2e-5
        self.dropout = 0.05
        self.num_pos_samples = num_pos_samples
        self.max_sequence_length = max_sequence_length
        self.temperature = temperature

        self.train_datas = train_datas
        self.train_texts = train_texts
        self.train_dataset_names = train_dataset_names
        self.samples_size = samples_size
        self.device = device
        self.name_of_round = name_of_round

        self.emb_dir = f'/gpfsnyu/scratch/yf2722/dataset/All_Embs/'
        self.checkpoints_dir = f'/gpfsnyu/scratch/yf2722/FAST/MixModels/Adjust_Version/{name_of_round}'
 
        self.model_name = model_name
        if model_name == 'bert-base-uncased':
            self.model = MixDatasetsBERTModel.from_pretrained(self.model_name)
        elif model_name == 'microsoft/deberta-base':
            self.model = MixDatasetsDeBERTaModel.from_pretrained(self.model_name)
        elif model_name == 'roberta-base':
            self.model = MixDatasetsRoBERTaModel.from_pretrained(self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.config.hidden_dropout_prob = self.dropout
        
        self.model = self.model.to(self.device)
        self.generate_dataset()

    #construct mix dataset
    def generate_dataset(self):
        if self.sample_type == 'random':
            self.dataset_indexes = neighbor_sampler(self.train_datas,samples_size = self.samples_size)
        elif self.sample_type == 'pagerank':
            self.dataset_indexes = neighbor_sampler_PR(self.train_datas,samples_size = self.samples_size)
        elif self.sample_type == 'mix':
            self.dataset_indexes = neighbor_sampler_Mix(self.train_datas,samples_size = self.samples_size)
        elif 'mix-3' in self.sample_type:
            self.dataset_indexes,one_hop_percentages = neighbor_sampler_Mix_Three(self.train_datas,samples_pool_size = self.samples_size)
        elif 'mix-n' in self.sample_type:
            self.dataset_indexes = neighbor_sampler_Mix_New(self.train_datas,samples_pool_size = self.samples_size)
        elif self.sample_type == 'ablation-1':
            self.dataset_indexes = ablation_study_one_no_Degree(self.train_datas,samples_pool_size = self.samples_size)
        elif self.sample_type == 'ablation-2':
            self.dataset_indexes = ablation_study_two_no_Dataset(self.train_datas,samples_pool_size = self.samples_size)
        


        shuffle_dataset_indexes = self.dataset_indexes[:]
        
        random.shuffle(shuffle_dataset_indexes)
        shuffle_all_text_list = []
        shuffle_dataset_index_list = []
        shuffle_neighbors_index_list = []
        shuffle_node_index_list = []
        shuffle_sample_weights_list = []
        for dataset_index, node_index, neighbors_index, sample_weights in shuffle_dataset_indexes:
            shuffle_all_text_list.append(self.train_texts[dataset_index][node_index])
            shuffle_node_index_list.append(node_index)
            shuffle_dataset_index_list.append(dataset_index)
            shuffle_neighbors_index_list.append(neighbors_index)
            assert np.isclose(np.sum(sample_weights), 1, atol=1e-6)
            shuffle_sample_weights_list.append(sample_weights)

        tokenized_text = self.tokenizer(shuffle_all_text_list, padding=True, truncation=True, max_length=self.max_sequence_length)
        dataset = MixDataset(tokenized_text,shuffle_dataset_index_list,shuffle_node_index_list,shuffle_neighbors_index_list,shuffle_sample_weights_list)

        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
    
    @staticmethod
    def compute_loss(batch_cls_embedding, batch_neighbor_cls_embeddings, temperature=0.5):
        batch_size, emb_size = batch_cls_embedding.shape
        _, k, _ = batch_neighbor_cls_embeddings.shape

        # 正样本相似度
        positive_samples = batch_neighbor_cls_embeddings.view(-1, emb_size)
        positive_similarity = F.cosine_similarity(batch_cls_embedding.unsqueeze(1).expand(-1, k, -1).reshape(-1, emb_size),positive_samples).view(batch_size, k)

        positive_similarity = torch.exp(positive_similarity)  # 应用温度参数

        # 负样本相似度
        negative_samples = batch_cls_embedding.unsqueeze(1).repeat(1, batch_size, 1)
        negative_similarity = F.cosine_similarity(negative_samples, batch_cls_embedding.unsqueeze(0).repeat(batch_size, 1, 1), dim=2)
        negative_similarity = negative_similarity / temperature
        negative_similarity = torch.exp(negative_similarity)
        negative_similarity = negative_similarity.sum(dim=1) + 1e-8
        
        # 计算损失，使用log的形式
        loss = -torch.log(positive_similarity.sum(dim=1) / negative_similarity).mean()
        return loss


    def fetch_batch_neighbor_embedding(self, neighbors, dataset_index, all_embeddings):
        dataset_index_list = dataset_index.tolist()
        neighbors_list_list = [[neighbor[i].item() for neighbor in neighbors] for i in range(len(neighbors[0]))]
        batch_neighbors_embedding = []

        assert len(dataset_index_list) == len(neighbors_list_list), "The lengths of dataset_index_list and neighbors_list_list must be equal."

        for dataset_idx, node_idx_list in zip(dataset_index_list, neighbors_list_list):
            dataset_embeddings = all_embeddings[dataset_idx]

            # 初始化一个空的邻居嵌入向量列表
            neighbors_embedding = []
            try:
                # 为每个邻居节点获取嵌入向量
                neighbors_embedding = [dataset_embeddings[i].clone().to(self.device) for i in node_idx_list]
            except IndexError as e:
                print("An IndexError occurred:", e)
                print("dataset index:", dataset_idx)
                print("node_idx_list", node_idx_list)
                print("Number of embeddings available:", len(dataset_embeddings))
                # 如果出现异常，可以在这里添加额外的处理逻辑

            # 将单个节点的所有邻居嵌入向量堆叠成一个tensor
            if neighbors_embedding:  # 确保列表不为空
                neighbors_embedding = torch.stack(neighbors_embedding).to(self.device)
                batch_neighbors_embedding.append(neighbors_embedding)
            else:
                # 处理邻居嵌入向量列表为空的情况，例如可以添加一个零向量或特定的占位符
                print(f"No embeddings found for node with dataset index {dataset_idx} and node_idx_list {node_idx_list}")

        # 将所有节点的邻居嵌入向量堆叠成一个更大的tensor
        if batch_neighbors_embedding:
            return torch.stack(batch_neighbors_embedding).to(self.device)
        else:
            # 返回一个空tensor或其他适当的默认值，如果没有找到任何邻居嵌入向量
            return torch.tensor([]).to(self.device)  # 根据实际情况可能需要调整这里

    

    def compute_all_cls_embeddings(self,dataset_names, texts, save_flag):
        all_cls_embeddings = []
        self.model.eval()  # Set the model to evaluation mode
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        for dataset_index, (dataset_name, text) in enumerate(zip(dataset_names, texts)):
            dataset_embedding = []
            X_c = tokenizer(text, padding=True, truncation=True, max_length=512)
            dataset = Dataset(X_c)
            dataloader = DataLoader(dataset, batch_size=256)
            with torch.no_grad():
                dataset_embedding = []
                total_batches = len(dataloader)
                for i, batch in enumerate(tqdm(dataloader, desc=f"Encoding dataset {dataset_name}", total=total_batches)):
                    batch_input_ids = batch['input_ids'].to(self.device)
                    batch_attention_mask = batch['attention_mask'].to(self.device)
                    if self.model_name == 'bert-base-uncased':
                        batch_token_type_ids = batch['token_type_ids'].to(self.device)
                        embeddings = self.model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    else:
                        embeddings = self.model(batch_input_ids, batch_attention_mask)
                    for emb in embeddings:
                        emb_cpu = emb.cpu().detach()  #添加了detach()
                        dataset_embedding.append(emb_cpu)
            all_cls_embeddings.append(dataset_embedding)
        if not save_flag:
            return all_cls_embeddings
        else:
            for dataset_index, dataset_name in enumerate(dataset_names):
                dataset_emb_path = os.path.join(self.emb_dir, dataset_name)
                dataset_emb_name = f'MixC-{self.name_of_round}.emb'
                dataset_emb_path = os.path.join(dataset_emb_path,dataset_emb_name)
                dataset_emb = torch.stack(all_cls_embeddings[dataset_index])
                torch.save(dataset_emb, dataset_emb_path)


    def update_all_cls_embeddings_with_batch(self, batch_node_embeddings, dataset_index, node_index):
        # 将嵌入转移到 CPU
        batch_node_embeddings = batch_node_embeddings.cpu().detach() #修改这里
        dataset_index_list = dataset_index.tolist()
        node_index_list = node_index.tolist()
        for Inbatch_index, (dataset_index, node_index) in enumerate(zip(dataset_index_list,node_index_list)):
            try:
                self.all_cls_embeddings[dataset_index][node_index] =  batch_node_embeddings[Inbatch_index].detach() #修改这里 删除了require grad
            except IndexError:
                print("IndexError occurred:")
                print("dataset_index:", dataset_index)
                print("node_index:", node_index)
                raise  # 重新引发异常以停止程序执行并显示完整的错误信息


    def save_model(self,epoch):
        model_path = os.path.join(self.checkpoints_dir, f"model-{epoch}.pt")
        
        hyperparams_path = os.path.join(self.checkpoints_dir, "train_params.json")
        torch.save(self.model.state_dict(), init_path(model_path))
        print(f'Epoch {epoch} checkpoint saved to {model_path}')

        hyperparameters = self.model.config.to_dict()
        
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        print(f'Training Hyperparameters saved to {hyperparams_path}')


    
    def save_losses(self,epoch,loss):
        losses_path = os.path.join(self.checkpoints_dir, f"loss-{epoch}.pkl")
        with open(losses_path, 'wb') as f:
            pickle.dump(loss, f)
        print(f'Losses saved to {losses_path}')


    ##首先计算所有的embedding，并保存为dict形式
    def train(self):
        torch.autograd.set_detect_anomaly(True)
        all_losses = []
        self.model.to(self.device)
        save_points = [int(len(self.dataloader) * 1/3), int(len(self.dataloader) * 2/3),  len(self.dataloader)]
        save_count = 0
        self.all_cls_embeddings = self.compute_all_cls_embeddings(dataset_names = self.train_dataset_names, texts = self.train_texts, save_flag=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print("Start training!")
        start_time = time.time()
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_losses = []
            pbar = tqdm(total=len(self.dataloader), desc=f"Training epoch {epoch}")
            for bat_cnt, batch in enumerate(self.dataloader):
                optimizer.zero_grad()
                neighbors = batch['neighbors']
                sample_weights = batch['sample_weights']
                dataset_index = batch['dataset_index']
                node_index = batch['node_index']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                if self.model_name == 'bert-base-uncased':
                    token_type_ids = batch['token_type_ids'].to(self.device)

                if self.sample_type in ['random','mix-2-r','mix-3-r']:
                    neighbors = random_sample_from_data(neighbors, self.num_pos_samples)
                elif self.sample_type in ['pagerank','mix','mix-2-w','mix-3-w']:
                    neighbors = weighed_sample_from_data(neighbors, self.num_pos_samples, sample_weights)
                elif self.sample_type == 'mix-n-r':
                    neighbors = random_sample_from_data_new(neighbors, self.num_pos_samples)
                elif self.sample_type == 'mix-n-w':
                    neighbors = weighed_sample_from_data_new(neighbors, self.num_pos_samples,sample_weights)
                elif self.sample_type in ['ablation-1','ablation-2']:
                    neighbors = weighed_sample_from_data_new(neighbors, self.num_pos_samples,sample_weights)

                batch_neighbor_cls_embeddings = self.fetch_batch_neighbor_embedding(neighbors,dataset_index,self.all_cls_embeddings)
                
                if self.model_name == 'bert-base-uncased':
                    batch_cls_embeddings = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    batch_cls_embeddings = self.model(input_ids, attention_mask)
                
                self.update_all_cls_embeddings_with_batch(batch_cls_embeddings,dataset_index,node_index)
                batch_cls_embeddings = batch_cls_embeddings.to(self.device)
                batch_neighbor_cls_embeddings = batch_neighbor_cls_embeddings.to(self.device)



                loss = self.compute_loss(batch_cls_embeddings,batch_neighbor_cls_embeddings, self.temperature)
                epoch_losses.append(loss.item())
                
                loss.backward()
                optimizer.step()

                # 更新进度条描述，显示当前loss值
                pbar.set_description(f"Training epoch {epoch} - Loss: {loss.item():.4f}")
                pbar.update(1)

                if (bat_cnt + 1) in save_points:
                    save_count += 1
                    self.save_model(save_count)
                    self.save_losses(save_count,epoch_losses)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Duration of processing epoch {epoch}: {duration:.2f} seconds")
            all_losses.append(epoch_losses)
            pbar.close()  # 确保结束时关闭进度条
        self.all_losses = all_losses
        self.save_losses(99,self.all_losses)
        print("Done Training!")


    def load_model(self,model_path):
        self.model.load_state_dict(torch.load(model_path))
        print(f'LM loaded from {model_path}')
        

    def encode_and_save(self, texts, dataset_names):
        self.compute_all_cls_embeddings(texts=texts,dataset_names=dataset_names,save_flag=True)
        

 