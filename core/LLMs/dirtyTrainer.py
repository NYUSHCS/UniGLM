from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel, RobertaModel, DebertaModel
from torch_geometric.data import DataLoader
from core.LLMs.utils import (neighbor_sampler, init_path, neighbor_sampler_PR, neighbor_sampler_Mix, 
                             neighbor_sampler_Mix_Three, neighbor_sampler_Mix_New, ablation_study_one_no_Degree, 
                             ablation_study_two_no_Dataset)
import random
import torch
from tqdm import tqdm
from core.LLMs.DataReader import MixDataset, Dataset
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool
import os
import pickle
import json
import time

# Sampling methods
def random_sample_from_data(data, k):
    data = [tensor.tolist() for tensor in data]
    data = [[row[i] for row in data] for i in range(len(data[0]))]
    selected_data = [random.sample(array, k) for array in data]
    transposed_selected_data = [[row[i] for row in selected_data] for i in range(k)]
    tensors = [torch.tensor(data) for data in transposed_selected_data]
    return tensors

def random_sample_from_data_new(data, k):
    data = [tensor.tolist() for tensor in data]
    data = [[row[i] for row in data] for i in range(len(data[0]))]
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
    tensors = [torch.tensor(data) for data in transposed_selected_data]
    return tensors

def weighed_sample_from_data(data, k, weights):
    data = [tensor for tensor in data]
    weights = [tensor for tensor in weights]
    data_transposed = torch.stack(data, dim=1)
    weights_transposed = torch.stack(weights, dim=1)
    new_data = []
    for i in range(data_transposed.shape[0]):
        row = data_transposed[i]
        weight = weights_transposed[i]
        assert len(row) == len(weight), f"row length: {len(row)}, weight length: {len(weight)}"
        try:
            sampled_values = torch.tensor(np.random.choice(row.numpy(), k, p=weight.numpy(), replace=False))
        except ValueError as e:
            print("Error occurred during sampling:")
            print("row:", row)
            print("weight:", weight)
            print("sum:", torch.sum(weight))
            raise e
        new_data.append(sampled_values)
    data_final = torch.stack(new_data, dim=1)
    return data_final

def weighed_sample_from_data_new(data, k, weights):
    data = [tensor for tensor in data]
    weights = [tensor for tensor in weights]
    data_transposed = torch.stack(data, dim=1)
    weights_transposed = torch.stack(weights, dim=1)
    new_data = []
    for i in range(data_transposed.shape[0]):
        row = data_transposed[i]
        weight = weights_transposed[i]
        valid_indices = [j for j in range(len(weight)) if weight[j] > 0]
        cleaned_row = row[valid_indices]
        cleaned_weight = weight[valid_indices]
        assert len(row) == len(weight), f"row length: {len(row)}, weight length: {len(weight)}"
        if torch.sum(weight) == 0:
            raise ValueError("Weights cannot all be zero")
        sampled_values = []
        if len(cleaned_row) < k:
            sampled_values = np.random.choice(
                cleaned_row.numpy(),
                k,
                p=cleaned_weight.numpy(),
                replace=True
            )
        else:
            sampled_values = np.random.choice(
                cleaned_row.numpy(),
                k,
                p=cleaned_weight.numpy(),
                replace=False
            )
        new_data.append(torch.tensor(sampled_values))
    data_final = torch.stack(new_data, dim=1)
    return data_final

# Custom models
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
    
# Trainer class
class MixCLTrainer():
    def __init__(self, train_datas, train_texts, train_dataset_names, samples_size, name_of_round, model_name, sample_type, batch_size, temperature, max_sequence_length, num_pos_samples, device):
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

    # Construct mix dataset
    def generate_dataset(self):
        if self.sample_type == 'random':
            self.dataset_indexes = neighbor_sampler(self.train_datas, samples_size=self.samples_size)
        elif self.sample_type == 'pagerank':
            self.dataset_indexes = neighbor_sampler_PR(self.train_datas, samples_size=self.samples_size)
        elif self.sample_type == 'mix':
            self.dataset_indexes = neighbor_sampler_Mix(self.train_datas, samples_size=self.samples_size)
        elif 'mix-3' in self.sample_type:
            self.dataset_indexes, one_hop_percentages = neighbor_sampler_Mix_Three(self.train_datas, samples_pool_size=self.samples_size)
        elif 'mix-n' in self.sample_type:
            self.dataset_indexes = neighbor_sampler_Mix_New(self.train_datas, samples_pool_size=self.samples_size)
        elif self.sample_type == 'ablation-1':
            self.dataset_indexes = ablation_study_one_no_Degree(self.train_datas, samples_pool_size=self.samples_size)
        elif self.sample_type == 'ablation-2':
            self.dataset_indexes = ablation_study_two_no_Dataset(self.train_datas, samples_pool_size=self.samples_size)

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
        dataset = MixDataset(tokenized_text, shuffle_dataset_index_list, shuffle_node_index_list, shuffle_neighbors_index_list, shuffle_sample_weights_list)

        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
    
    @staticmethod
    def compute_loss(batch_cls_embedding, batch_neighbor_cls_embeddings, temperature=0.5):
        batch_size, emb_size = batch_cls_embedding.shape
        _, k, _ = batch_neighbor_cls_embeddings.shape

        print("batch_cls_embedding shape:", batch_cls_embedding.shape)
        print("batch_neighbor_cls_embeddings shape:", batch_neighbor_cls_embeddings.shape)

        positive_samples = batch_neighbor_cls_embeddings.view(-1, emb_size)
        expanded_batch_cls_embedding = batch_cls_embedding.unsqueeze(1).expand(-1, k, -1).reshape(-1, emb_size)

        print("expanded_batch_cls_embedding shape:", expanded_batch_cls_embedding.shape)
        print("positive_samples shape:", positive_samples.shape)

        positive_similarity = F.cosine_similarity(expanded_batch_cls_embedding, positive_samples).view(batch_size, k)
        positive_similarity = torch.exp(positive_similarity)

        negative_samples = batch_cls_embedding.unsqueeze(1).repeat(1, batch_size, 1)
        negative_similarity = F.cosine_similarity(negative_samples, batch_cls_embedding.unsqueeze(0).repeat(batch_size, 1, 1), dim=2)
        negative_similarity = negative_similarity / temperature
        negative_similarity = torch.exp(negative_similarity)
        negative_similarity = negative_similarity.sum(dim=1) + 1e-8

        loss = -torch.log(positive_similarity.sum(dim=1) / negative_similarity).mean()
        return loss


    def calculate_batch_neighbor_embedding(self, neighbors, dataset_index, all_texts):
        dataset_index_list = dataset_index.tolist()
        neighbors_list_list = [[neighbor[i].item() for neighbor in neighbors] for i in range(len(neighbors[0]))]
        batch_neighbors_embedding = []

        assert len(dataset_index_list) == len(neighbors_list_list), "The lengths of dataset_index_list and neighbors_list_list must be equal."

        pbar = tqdm(total=len(dataset_index_list), desc="Calculating neighbor embeddings")

        for dataset_idx, node_idx_list in zip(dataset_index_list, neighbors_list_list):
            dataset_texts = all_texts[dataset_idx]

            neighbors_embedding = []
            text_list = [dataset_texts[i] for i in node_idx_list if i < len(dataset_texts)]
            tokenized_text = self.tokenizer(
                    text_list, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_sequence_length, 
                    return_tensors='pt'
                ).to(self.device)

            with torch.no_grad():
                if self.model_name == 'bert-base-uncased':
                    outputs = self.model(
                        input_ids=tokenized_text['input_ids'],
                        attention_mask=tokenized_text['attention_mask'],
                        token_type_ids=tokenized_text['token_type_ids']
                    )
                else:
                    outputs = self.model(
                        input_ids=tokenized_text['input_ids'],
                        attention_mask=tokenized_text['attention_mask']
                    )
                neighbors_embedding = outputs

            if len(neighbors_embedding) > 0:
                batch_neighbors_embedding.append(neighbors_embedding)
            else:
                placeholder_embedding = torch.zeros((1, self.model.config.hidden_size), device=self.device)
                batch_neighbors_embedding.append(placeholder_embedding)
                print(f"No valid texts found for node with dataset index {dataset_idx} and node_idx_list {node_idx_list}")

            pbar.update(1)

        pbar.close()

        if batch_neighbors_embedding:
            return torch.stack(batch_neighbors_embedding)
        else:
            return torch.tensor([], device=self.device)


    def save_model(self, epoch):
        model_path = os.path.join(self.checkpoints_dir, f"model-{epoch}.pt")
        hyperparams_path = os.path.join(self.checkpoints_dir, "train_params.json")
        torch.save(self.model.state_dict(), init_path(model_path))
        print(f'Epoch {epoch} checkpoint saved to {model_path}')

        hyperparameters = self.model.config.to_dict()
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        print(f'Training Hyperparameters saved to {hyperparams_path}')

    def save_losses(self, epoch, loss):
        losses_path = os.path.join(self.checkpoints_dir, f"loss-{epoch}.pkl")
        with open(losses_path, 'wb') as f:
            pickle.dump(loss, f)
        print(f'Losses saved to {losses_path}')

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        all_losses = []
        self.model.to(self.device)
        save_points = [int(len(self.dataloader) * 1/3), int(len(self.dataloader) * 2/3), len(self.dataloader)]
        save_count = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print("Start training!")
        
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_losses = []
            total_batches = len(self.dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}")

            for bat_cnt, batch in enumerate(self.dataloader):
                batch_start_time = time.time()
                optimizer.zero_grad()
                neighbors = batch['neighbors']
                sample_weights = batch['sample_weights']
                dataset_index = batch['dataset_index']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.model_name == 'bert-base-uncased':
                    token_type_ids = batch['token_type_ids'].to(self.device)
                
                if self.sample_type in ['random', 'mix-2-r', 'mix-3-r']:
                    neighbors = random_sample_from_data(neighbors, self.num_pos_samples)
                elif self.sample_type == 'mix-n-r':
                    neighbors = random_sample_from_data_new(neighbors, self.num_pos_samples)
                elif self.sample_type == 'mix-n-w' or self.sample_type in ['ablation-1', 'ablation-2']:
                    neighbors = weighed_sample_from_data_new(neighbors, self.num_pos_samples, sample_weights)
                
                if self.model_name == 'bert-base-uncased':
                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                print(f"Model outputs shape: {outputs.shape}")

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                batch_cls_embeddings = outputs
                batch_neighbor_cls_embeddings = self.calculate_batch_neighbor_embedding(neighbors, dataset_index, self.train_texts)
                
                batch_cls_embeddings = batch_cls_embeddings.to(self.device)
                batch_neighbor_cls_embeddings = batch_neighbor_cls_embeddings.to(self.device)

                loss = self.compute_loss(batch_cls_embeddings, batch_neighbor_cls_embeddings, self.temperature)
                epoch_losses.append(loss.item())
                
                loss.backward()
                optimizer.step()

                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                print(f"Batch {bat_cnt + 1}/{total_batches} - Loss: {loss.item():.4f} - Batch Time: {batch_duration:.2f} seconds")

                if (bat_cnt + 1) in save_points:
                    save_count += 1
                    self.save_model(save_count)
                    self.save_losses(save_count, epoch_losses)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"Duration of processing epoch {epoch + 1}: {duration:.2f} seconds")
            all_losses.append(epoch_losses)
        
        self.all_losses = all_losses
        self.save_losses(99, self.all_losses)
        print("Done Training!")


    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print(f'LM loaded from {model_path}')

    def encode_and_save(self, texts, dataset_names):
        self.compute_all_cls_embeddings(texts=texts, dataset_names=dataset_names, save_flag=True)
