import argparse
import torch
from core.data_utils.data_loader import load_data
import json
import os
from transformers import AutoTokenizer,AutoModel,AutoConfig,BertModel,BertTokenizer,RobertaModel,DebertaModel
from core.LLMs.DataReader import MixDataset,Dataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from natsort import natsorted
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
    
class Emb_Generater():
    def __init__(self,round_name,eval_dataset_names,eval_texts, device):
        self.round_name = round_name
        self.eval_dataset_names = eval_dataset_names
        self.eval_texts = eval_texts
        self.device = device
        self.emb_dir = '/gpfsnyu/home/yf2722/FAST/dataset/All_Embs/'
        self.models_path = f'/gpfsnyu/scratch/yf2722/FAST/MixModels/Adjust_Version/{self.round_name}'
        if 'bert' in self.round_name:
            self.model_name = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
            self.model = MixDatasetsBERTModel.from_pretrained(self.model_name)
        elif 'deb' in self.round_name:
            self.model_name = 'microsoft/deberta-base'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
            self.model = MixDatasetsDeBERTaModel.from_pretrained(self.model_name)
        elif 'rob' in self.round_name:
            self.model_name = 'roberta-base'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
            self.model = MixDatasetsRoBERTaModel.from_pretrained(self.model_name)
        else:
            self.model_name = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
            self.model = MixDatasetsBERTModel.from_pretrained(self.model_name)



    def generate_emb(self):
        # First, tokenize all texts from eval datasets
        tokenized_texts = [self.tokenizer(text, padding=True, truncation=True, max_length=512) for text in self.eval_texts]
        for model_index in [9]:
            dataset_embeddings = [[] for _ in range(len(self.eval_dataset_names))]
            model_path = os.path.join(self.models_path, f'model-{model_index}.pt')
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                for dataset_index, (dataset_name, tokenized_text) in enumerate(zip(self.eval_dataset_names, tokenized_texts)):
                    dataset = Dataset(tokenized_text)
                    dataloader = DataLoader(dataset, batch_size=512)
                    total_batches = len(dataloader) 
                    for i, batch in enumerate(tqdm(dataloader, desc=f"Encoding dataset {dataset_name} for model-{model_index}", total=total_batches)):
                        batch_input_ids = batch['input_ids'].to(self.device)
                        batch_attention_mask = batch['attention_mask'].to(self.device)
                        if self.model_name == 'bert-base-uncased':
                            batch_token_type_ids = batch['token_type_ids'].to(self.device)
                            embeddings = self.model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                        else:
                            embeddings = self.model(batch_input_ids, batch_attention_mask)
                        for emb in embeddings:
                            #print(emb)
                            emb_cpu = emb.cpu()
                            dataset_embeddings[dataset_index].append(emb_cpu)
            
            # Saving dataset embeddings for each model
            for dataset_index, dataset_name in enumerate(self.eval_dataset_names):
                dataset_emb_path = os.path.join(self.emb_dir, dataset_name)
                dataset_emb_name = f'{self.round_name}-model-{model_index}.emb'
                dataset_emb_path = os.path.join(dataset_emb_path, dataset_emb_name)
                torch.save(dataset_embeddings[dataset_index], dataset_emb_path)

def get_IDR_text(dataset_name):
    texts = []
    if 'Amazon' in dataset_name:
        if dataset_name == 'Amazon-Computers':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/amazon-computers-Mixtral"
            n = 87229
        elif dataset_name == 'Amazon-Photo':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/amazon-photo-Mixtral"
            n = 48362
        elif dataset_name == 'Amazon-Fitness':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/amazon-fitness-Mixtral"
            n = 173055
        elif dataset_name == 'Amazon-History':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/amazon-history-Mixtral"
            n = 41551
        elif dataset_name == 'Amazon-Children':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/amazon-children-Mixtral"
            n = 76875
        files = os.listdir(directory)
        files = natsorted(files)

        for i in range(n):
            file = f'generated_{i}.txt'
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                response = f.read().strip()
                texts.append(response)
    else:
        texts = []
        if dataset_name == 'ogbn-product':
            texts = []
            return texts
        if dataset_name == 'pubmed':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/pubmed-gpt"
            n = 19717
        elif dataset_name == 'ogbn-arxiv':
            directory = f"/gpfsnyu/scratch/yf2722/augs/Node_Feature/IDR/ogbn-arxiv-gpt"
            n = 169343
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                response = json_data['choices'][0]['message']['content']
                texts.append(response)
    return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mix Contrastive Learning Model')
    parser.add_argument('--round_name', type=str, default='999', help='Number of training rounds')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.cuda.reset_peak_memory_stats()
    

    #eval_dataset_names = ['pubmed','Amazon-History']
    eval_dataset_names = ['Amazon-Children','Amazon-Computers','Amazon-Fitness','Amazon-History','Amazon-Photo','pubmed','ogbn-arxiv','ogbn-product']
    #eval_dataset_names = ['Amazon-Children']
    print("Training Datasets:",eval_dataset_names)
    eval_datas = []
    eval_texts = []
    '''
    for dataset_name in eval_dataset_names:
        if dataset_name != 'ogbn-product':
            concatenated_texts = []
            texts_IDR = get_IDR_text(dataset_name)
            data, texts_ORI = load_data(dataset_name)
            eval_datas.append(data)
            assert len(texts_IDR) == len(texts_ORI),f'{dataset_name}, IDR length:{len(texts_IDR)}'
            for text1, text2 in zip(texts_ORI, texts_IDR):
                text = f"{text1} Explaination:{text2}"
                concatenated_texts.append(text)
            eval_texts.append(concatenated_texts)
        else:
            data, text = load_data(dataset_name)
            eval_texts.append(text)
            eval_datas.append(data)
    '''
    for dataset_name in eval_dataset_names:
        data, text = load_data(dataset_name)
        eval_datas.append(data)
        eval_texts.append(text)
    

    trainer = Emb_Generater(args.round_name,eval_dataset_names,eval_texts,device)
    trainer.generate_emb()