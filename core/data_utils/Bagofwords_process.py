import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import torch
from data_loader import load_data

def generate_bow_embeddings(text_list, k=768):
    nltk.download('stopwords')
    top_words_count = k
    
    # 初始化结果列表
    embeddings = []

    # 初始化全局词频统计字典
    global_word_freq = Counter()

    # 迭代每一个文本并预处理
    for index, paragraph in enumerate(text_list):
        review = re.sub('[^a-zA-Z]', ' ', paragraph)
        review = review.lower()
        review = review.split()
        review = [word for word in review if word not in set(stopwords.words('english'))]
        local_word_freq = Counter(review)
        global_word_freq += local_word_freq
        print(f"Processed text {index}")

    # 取出现率最高的前 top_words_count 个词
    top_words = [word for word, _ in global_word_freq.most_common(top_words_count)]

    # 再次迭代每个文本并创建嵌入向量
    for index, paragraph in enumerate(text_list):
        review = re.sub('[^a-zA-Z]', ' ', paragraph)
        review = review.lower()
        review = review.split()
        review = set([word for word in review if word not in set(stopwords.words('english'))])
        embedding = [1 if word in review else 0 for word in top_words]
        embeddings.append(embedding)
        print(f"Processed text {index}")

    # 将嵌入向量转换为张量格式
    embeddings_tensor = torch.tensor(embeddings)

    return embeddings_tensor

def process(dataset_name):
    data,text = load_data(dataset_name,feature_type=None)
    embeddings = generate_bow_embeddings(text)
    base_path = '../dataset/All_Embs/'
    save_path = base_path + dataset_name + '/BoW.emb'
    print(save_path)
    torch.save(embeddings, save_path)


dataset_names = ['Amazon-Children','Amazon-Fitness']
for dataset_name in dataset_names:
    process(dataset_name)
