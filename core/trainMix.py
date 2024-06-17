from core.LLMs.MixTrainer import MixCLTrainer
import argparse
import torch
from core.data_utils.data_loader import load_data
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mix Contrastive Learning Model')
    parser.add_argument('--round_name', type=str, default='999', help='Number of training rounds')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Model name')
    parser.add_argument('--samples_size', type=int, default=5, help='Neighbor size for contrastive learning')
    parser.add_argument('--sample_type', type=str, default='random', help='Sampling type')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--num_pos_samples', type=int, default=6)

    args = parser.parse_args()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #torch.cuda.reset_peak_memory_stats()
    

    #train_dataset_names = ['pubmed']
    #train_dataset_names = ['Amazon-Children','Amazon-Computers','Amazon-Fitness','Amazon-History','Amazon-Photo','ogbn-product']
    train_dataset_names = ['pubmed', 'ogbn-arxiv']
    print("Training Datasets:",train_dataset_names)
    train_datas = []
    train_texts = []
    for dataset_name in train_dataset_names:
        data, text = load_data(dataset_name)
        train_datas.append(data)
        train_texts.append(text)

    parameters = {
        'train_dataset_names': train_dataset_names,
        'samples_size': args.samples_size,
        'round_name': args.round_name,
        'model_name': args.model_name,
        'sample_type': args.sample_type,
        'batch_size':args.batch_size,
        'temperature':args.temperature,
        'max_sequence_length':args.max_sequence_length
    }

    output_dir = '/gpfsnyu/scratch/yf2722/FAST/MixModels/Adjust_Version'
    directory_path = os.path.join(output_dir, str(args.round_name))
    file_path = os.path.join(directory_path, 'parameters.json')
    os.makedirs(directory_path, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(parameters, file, indent=4)
    print(f'Parameters saved to {file_path}')


    trainer = MixCLTrainer(train_datas, train_texts, train_dataset_names, args.samples_size, args.round_name, args.model_name, args.sample_type, args.batch_size, args.temperature,
                            args.max_sequence_length, args.num_pos_samples, device)
    trainer.train()
    #model_load_path = os.path.join(directory_path, 'model-31.pt')
    #trainer.load_model(model_load_path)
    #trainer.encode_and_save(train_texts, train_dataset_names)