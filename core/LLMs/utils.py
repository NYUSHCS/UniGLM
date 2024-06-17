import random
import os
import numpy as np
import time
import datetime
import pytz
from collections import defaultdict,deque
from scipy.sparse import csr_matrix
from tqdm import tqdm

class MultipleOptimizer(object):
    def __init__(self,*op):
        self.optimizers = op
        self.param_groups=[]
        for op in self.optimizers:
            self.param_groups.extend(op.param_groups) 

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
            
def mkdir_p(path, log=True):

    import errno
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file




def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper





def build_adjacency_list(edge_index):
    adjacency_list = defaultdict(set)
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if src != dst:  # Exclude self-loops
            adjacency_list[src].add(dst)
            adjacency_list[dst].add(src)
    return adjacency_list

def get_neighbors(node, adjacency_list, exclude):
    # Safely return neighbors or an empty list if the node has no entries in the adjacency list
    return [n for n in adjacency_list.get(node, []) if n not in exclude]



import numpy as np

def page_rank(adjacency_matrix, teleportation_probability, max_iterations=100, tolerance=1e-6):
    # 初始化 PageRank 得分为均匀分布
    num_nodes = adjacency_matrix.shape[0]
    page_rank_scores = np.ones(num_nodes) / num_nodes

    for _ in tqdm(range(max_iterations)):
        # 计算新的 PageRank 得分
        new_page_rank_scores = adjacency_matrix.dot(page_rank_scores)
        # 将矩阵-向量乘法的结果归一化
        new_page_rank_scores = (1 - teleportation_probability) * new_page_rank_scores
        # 加上随机跳转的概率
        new_page_rank_scores += teleportation_probability / num_nodes
        # 对新的 PageRank 得分进行归一化
        new_page_rank_scores /= new_page_rank_scores.sum()

        # 检查是否收敛
        if np.allclose(page_rank_scores, new_page_rank_scores, atol=tolerance):
            break
        page_rank_scores = new_page_rank_scores

    return page_rank_scores


def collect_neighbors(node, adjacency_list, max_depth):
    current_level = {node}
    visited = set(current_level)
    all_neighbors = []

    for _ in range(max_depth):
        next_level = set()
        for n in current_level:
            for neighbor in adjacency_list.get(n, []):
                if neighbor not in visited:
                    next_level.add(neighbor)
                    visited.add(neighbor)
                    all_neighbors.append(neighbor)
        current_level = next_level
    return all_neighbors


def compute_degrees(adjacency_list):
    degrees = [len(neighbors) for node, neighbors in adjacency_list.items()]
    return degrees

# list of tuple [(dataset_index, node_index, [ neighbors_index ])] shuffled
# [(0, 2186, [2321, 2187, 2370, 531, 2314]), (1, 6904, [7919, 10703, 2959, 18049, 9386]))
def neighbor_sampler(data_list, samples_size):
    tuples_list = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = len(data.y)  
        adjacency_list = build_adjacency_list(edge_index)
        for node_index in range(num_nodes):
            if node_index not in adjacency_list:  # Skip isolated nodes
                #tuples_list.append((dataset_index, node_index, []))
                continue
            all_neighbors = collect_neighbors(node_index, adjacency_list, 3)
            if len(all_neighbors) > samples_size:
                samples = all_neighbors[:samples_size]
            else:
                samples = all_neighbors
            sampling_probabilities = [1/len(samples) for sample in samples]
            tuples_list.append((dataset_index, node_index, samples, sampling_probabilities))
    return tuples_list

def softmax(values):
    # 确保 values 是 numpy 数组
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # 获取 values 中的最大值
    max_value = np.max(values)
    if max_value != 0:
        scale_factor = 1 / abs(max_value)  # 这只是一个例子，你可以根据需要调整
    else:
        scale_factor = 1.0  # 防止零除
    # 应用标度因子并计算 softmax
    scaled_values = values * scale_factor
    shifted_values = scaled_values - np.max(scaled_values)  # 防止溢出
    exp_values = np.exp(shifted_values)
    softmax_output = exp_values / np.sum(exp_values)

    return softmax_output.tolist()

def neighbor_sampler_PR(data_list, samples_size, teleportation_probability=0.1, max_iterations=40):
    tuples_list = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = max(edge_index.max() + 1, len(data.y))  # Assuming data.x contains node features
        # Create adjacency matrix
        adjacency_matrix = csr_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        # Compute PageRank scores
        pg_scores = page_rank(adjacency_matrix, teleportation_probability, max_iterations)
        adjacency_list = build_adjacency_list(edge_index)
        for node_index in range(num_nodes):
            if node_index not in adjacency_list:  # Skip isolated nodes
                #tuples_list.append((dataset_index, node_index, []))
                continue

            all_neighbors = collect_neighbors(node_index, adjacency_list, 3)
            if len(all_neighbors) < samples_size:
                all_neighbors = collect_neighbors(node_index, adjacency_list, 4)

            # 将邻居加入 tuples_list
            samples = sorted(all_neighbors, key=lambda x: pg_scores[x], reverse=True)[:samples_size]
            neighbor_pagerank_scores = [pg_scores[sample] for sample in samples]
            sampling_probabilities = softmax(neighbor_pagerank_scores)
            tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))

            assert len(samples) == len(sampling_probabilities), f"Sample and probability lengths differ for node {node_index} in dataset {dataset_index}, samples length:{len(samples)}, score length:{len(sampling_probabilities)}"
    return tuples_list


def neighbor_sampler_Mix(data_list, samples_size, teleportation_probability=0.1, max_iterations=40):
    tuples_list = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = max(edge_index.max() + 1, len(data.y))  # Assuming data.x contains node features
        # Create adjacency matrix
        adjacency_matrix = csr_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        # Compute PageRank scores
        pg_scores = page_rank(adjacency_matrix, teleportation_probability, max_iterations)
        adjacency_list = build_adjacency_list(edge_index)
        for node_index in range(num_nodes):
            if node_index not in adjacency_list:  # Skip isolated nodes
                #tuples_list.append((dataset_index, node_index, []))
                continue
            
            one_neighbors = get_neighbors(node_index, adjacency_list, {node_index})
            all_neighbors = collect_neighbors(node_index, adjacency_list, 3)

            higher_order_neighbors = [n for n in all_neighbors if n not in one_neighbors]

            half_sample_size = int(samples_size // 2)
            if len(one_neighbors) >= half_sample_size:
                samples_from_one = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)[:half_sample_size]
                remaining_samples_needed = samples_size - len(samples_from_one)
                samples_from_higher = sorted(higher_order_neighbors, key=lambda x: pg_scores[x], reverse=True)[:remaining_samples_needed]
                samples = samples_from_one + samples_from_higher

                while len(samples) < samples_size:
                    additional_samples = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)[:samples_size - len(samples)]
                    samples.extend(additional_samples)

                neighbor_pagerank_scores = [pg_scores[sample] * 1.3 if sample in one_neighbors else pg_scores[sample]for sample in samples]
                sampling_probabilities = softmax(neighbor_pagerank_scores)
                tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))

            else:
                page_rank_sample_num = samples_size - len(one_neighbors)
                page_rank_samples = sorted(higher_order_neighbors, key=lambda x: pg_scores[x], reverse=True)[:page_rank_sample_num]
                samples = one_neighbors + page_rank_samples

                while len(samples) < samples_size:
                    additional_samples = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)[:samples_size - len(samples)]
                    samples.extend(additional_samples)

                neighbor_pagerank_scores = [pg_scores[sample] * 1.3 if sample in one_neighbors else pg_scores[sample]for sample in samples]
                sampling_probabilities = softmax(neighbor_pagerank_scores)
                tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))

            assert len(samples) == len(sampling_probabilities), f"Sample and probability lengths differ for node {node_index} in dataset {dataset_index}, samples length:{len(samples)}, score length:{len(sampling_probabilities)}"
    return tuples_list


def neighbor_sampler_Mix_Three(data_list, samples_pool_size = 15, one_hop_samples_size = 10, teleportation_probability=0.15, max_iterations=80,):
    tuples_list = []
    one_hop_percentages = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = max(edge_index.max() + 1, len(data.y))  # Assuming data.x contains node features
        
        adjacency_matrix = csr_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        
        pg_scores = page_rank(adjacency_matrix, teleportation_probability, max_iterations)
        adjacency_list = build_adjacency_list(edge_index)
        degrees = compute_degrees(adjacency_list)
        avg_degree = sum(degrees) / len(degrees)

        one_hop_counts = np.zeros(num_nodes, dtype=int)  # 初始化1阶邻居统计

        for node_index in range(num_nodes):
            one_neighbors = get_neighbors(node_index, adjacency_list, {node_index})
            two_hop_neighbors = []
            three_hop_neighbors = []
            degree = len(one_neighbors)
            
            if node_index not in adjacency_list:  # Skip isolated nodes
                continue
            if degree >= samples_pool_size:  #degree >= 15
                samples = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)[:samples_pool_size]
                assert len(samples) == samples_pool_size,'1'

            elif one_hop_samples_size <= degree < samples_pool_size:  # 10 < degree < 15 认为1阶邻居数量够了
                left_samples_size = samples_pool_size - degree
                one_two_hop_neighbors = collect_neighbors(node_index, adjacency_list, 2)
                two_hop_neighbors = [n for n in one_two_hop_neighbors if n not in one_neighbors]
                if len(two_hop_neighbors) >= left_samples_size:
                    two_hop_samples = sorted(two_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_samples_size]
                    samples = one_neighbors + two_hop_samples
                else:
                    samples = one_neighbors + two_hop_samples
                    samples = samples + samples
                    samples = samples[:samples_pool_size]
                
                assert len(samples) == samples_pool_size,f'2, length samples:{len(samples)}'
                

            elif degree < one_hop_samples_size and avg_degree > 5:  #degree <= 10  #平均度数都比较大,说明2-3阶信息噪声多,所以尽量都拿1阶邻居来填充
                one_hop_samples = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)
                samples = []
                while len(samples) < samples_pool_size:
                    samples = samples + one_hop_samples
                samples = samples[:samples_pool_size]
                assert len(samples) == samples_pool_size,'3'


            elif avg_degree <= degree < one_hop_samples_size and avg_degree <= 5: #degree <= 10  #平均度数都很小,说明2-3阶信息噪声不多,先2后3,实在不够再拿1阶
                samples = one_neighbors
                left = samples_pool_size - len(samples)
                one_two_hop_neighbors = collect_neighbors(node_index, adjacency_list, 2)
                two_hop_neighbors = [n for n in one_two_hop_neighbors if n not in one_neighbors]
                if len(two_hop_neighbors) >= left:
                    two_samples = sorted(two_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left]
                    samples = samples + two_samples
                    assert len(samples) == samples_pool_size,'4'
                else:
                    samples = samples + two_hop_neighbors
                    left = samples_pool_size - len(samples)
                    assert left > 0
                    one_two_three_hop_neighbors = collect_neighbors(node_index, adjacency_list, 3)
                    three_hop_neighbors = [n for n in one_two_three_hop_neighbors if n not in one_two_hop_neighbors]
                    if len(three_hop_neighbors) >= left:
                        three_hop_samples = sorted(three_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left]
                        samples = samples + three_hop_samples
                        assert len(samples) == samples_pool_size,'5'
                    else:
                        samples = samples + three_hop_neighbors
                        while len(samples) < samples_pool_size:
                            samples = samples + samples
                        samples = samples[:samples_pool_size]
                        assert len(samples) == samples_pool_size,'6'

            elif degree < avg_degree and avg_degree <= 5: #degree <= 10  #平均度数都很小,说明2-3阶信息噪声不多,先2后3,实在不够再拿1阶
                left_samples_size = samples_pool_size - degree
                samples = one_neighbors
                assert left_samples_size > 0
                if left_samples_size <= degree:
                    one_hop_samples = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_samples_size]
                    samples = one_neighbors + one_hop_samples
                    assert len(samples) == samples_pool_size,'10'
                else:
                    left_samples_size = samples_pool_size - 2*degree
                    samples = one_neighbors + one_neighbors
                    one_two_hop_neighbors = collect_neighbors(node_index, adjacency_list, 2)
                    two_hop_neighbors = [n for n in one_two_hop_neighbors if n not in one_neighbors]
                    if len(two_hop_neighbors) >=  left_samples_size:   #如果填充2阶邻居就够了，那就只1 2阶
                        two_hop_samples = sorted(two_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_samples_size]
                        samples = samples + two_hop_samples
                        assert len(samples) == samples_pool_size,f'7,left_samples_size:{left_samples_size}'
                    else:           #2阶邻居还不够，那就填充3阶
                        samples = samples + two_hop_neighbors
                        left_samples_size = samples_pool_size - len(samples)
                        one_two_three_hop_neighbors = collect_neighbors(node_index, adjacency_list, 3)
                        three_hop_neighbors = [n for n in one_two_three_hop_neighbors if n not in one_two_hop_neighbors]
                        if len(three_hop_neighbors) >= left_samples_size:
                            three_hop_samples = sorted(three_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_samples_size]
                            samples = samples + three_hop_samples
                            assert len(samples) == samples_pool_size,'8'
                        else:  #3阶都不够，重复1阶 2阶 3阶
                            samples = samples + three_hop_samples
                            while len(samples) < samples_pool_size:
                                samples = samples + one_neighbors
                                if len(samples) < samples_pool_size:
                                    samples = samples + two_hop_neighbors
                                    if len(samples) < samples_pool_size:
                                        samples = samples + three_hop_neighbors
                            samples = samples[:samples_pool_size]
                            assert len(samples) == samples_pool_size, '9'
            # 对1阶邻居进行统计
            one_hop_count = len([s for s in samples if s in one_neighbors])
            one_hop_counts[node_index] = one_hop_count  # 记录1阶邻居的数量
            assert len(samples) == samples_pool_size, f'degree:{degree},avg degree:{avg_degree}'
            neighbor_pagerank_scores = [pg_scores[sample] * 1.2 if sample in one_neighbors else pg_scores[sample]for sample in samples]
            sampling_probabilities = softmax(neighbor_pagerank_scores)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6),f'neighbor_pagerank_scores:{neighbor_pagerank_scores}, sampling_probabilities:{sampling_probabilities}, sum:{sum(sampling_probabilities)}'
            tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))
        one_hop_percentages.append(one_hop_counts)  # 添加到统计结果中
    return tuples_list,one_hop_percentages


def neighbor_sampler_Mix_New(data_list, samples_pool_size = 15, positive_sizes= 6, teleportation_probability=0.15, max_iterations=80,):
    #### 这个版本中我们以samples_pool_size为最高的采样数 而不是每个点都要采样到这么多 ####
    tuples_list = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = max(edge_index.max() + 1, len(data.y))
        adjacency_matrix = csr_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        pg_scores = page_rank(adjacency_matrix, teleportation_probability, max_iterations)
        adjacency_list = build_adjacency_list(edge_index)
        degrees = compute_degrees(adjacency_list)
        avg_degree = sum(degrees) / len(degrees)

        for node_index in range(num_nodes):
            one_neighbors = get_neighbors(node_index, adjacency_list, {node_index})
            two_hop_neighbors = []
            three_hop_neighbors = []
            degree = len(one_neighbors)

            if node_index not in adjacency_list:  # Skip isolated nodes
                continue

            if degree >= samples_pool_size: 
                one_neighbors = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)
                samples = one_neighbors[:samples_pool_size]

            elif degree < samples_pool_size and avg_degree > positive_sizes:
                samples = one_neighbors
            
            elif degree < samples_pool_size and avg_degree <= positive_sizes:
                one_two_hop_neighbors = collect_neighbors(node_index, adjacency_list, 2)
                left_size = positive_sizes - degree
                two_hop_neighbors = [n for n in one_two_hop_neighbors if n not in one_neighbors]
                two_hop_neighbors = sorted(two_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)
                if len(two_hop_neighbors) >= left_size:
                    two_hop_samples = two_hop_neighbors[:left_size]
                    samples = one_neighbors + two_hop_samples
                else:
                    samples = one_neighbors + two_hop_neighbors
                    left_size = positive_sizes - len(samples)
                    one_two_three_hop_neighbors = collect_neighbors(node_index, adjacency_list, 3)
                    three_hop_neighbors = [n for n in one_two_three_hop_neighbors if n not in one_two_hop_neighbors]
                    three_hop_samples = sorted(three_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_size]
                    samples = samples + three_hop_samples
                samples = one_neighbors[:samples_pool_size]
            assert isinstance(samples, list), f'node:{node_index}, one_hop:{one_neighbors}'
            neighbor_pagerank_scores = [pg_scores[sample] * 1.2 if sample in one_neighbors else pg_scores[sample]for sample in samples]
            sampling_probabilities = softmax(neighbor_pagerank_scores)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6)
            k = len(samples)
            if k < samples_pool_size:
                samples += [-1] * (samples_pool_size - k)
                sampling_probabilities += [0.0] * (samples_pool_size - k)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6)
            assert len(samples) == len(sampling_probabilities) == samples_pool_size, f'{len(samples)}, {len(sampling_probabilities)}, {samples_pool_size}'
            tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))
    return tuples_list


def ablation_study_two_no_Dataset(data_list, samples_pool_size = 15, positive_sizes= 6, teleportation_probability=0.15, max_iterations=80,):
    #### 这个版本中我们以samples_pool_size为最高的采样数 而不是每个点都要采样到这么多 ####
    tuples_list = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = max(edge_index.max() + 1, len(data.y))
        adjacency_matrix = csr_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        pg_scores = page_rank(adjacency_matrix, teleportation_probability, max_iterations)
        adjacency_list = build_adjacency_list(edge_index)
        degrees = compute_degrees(adjacency_list)
        avg_degree = sum(degrees) / len(degrees)

        for node_index in range(num_nodes):
            one_neighbors = get_neighbors(node_index, adjacency_list, {node_index})
            two_hop_neighbors = []
            three_hop_neighbors = []
            degree = len(one_neighbors)

            if node_index not in adjacency_list:  # Skip isolated nodes
                continue

            if degree >= samples_pool_size: 
                one_neighbors = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)
                samples = one_neighbors[:samples_pool_size]
            
            elif positive_sizes <= degree < samples_pool_size:
                samples = one_neighbors

            elif degree <= positive_sizes: #平均节点度数小于positiva_sizes，
                one_two_hop_neighbors = collect_neighbors(node_index, adjacency_list, 2)
                left_size = positive_sizes - degree
                two_hop_neighbors = [n for n in one_two_hop_neighbors if n not in one_neighbors]
                two_hop_neighbors = sorted(two_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)
                if len(two_hop_neighbors) >= left_size:
                    two_hop_samples = two_hop_neighbors[:left_size]
                    samples = one_neighbors + two_hop_samples
                else:
                    samples = one_neighbors + two_hop_neighbors
                    left_size = positive_sizes - len(samples)
                    one_two_three_hop_neighbors = collect_neighbors(node_index, adjacency_list, 3)
                    three_hop_neighbors = [n for n in one_two_three_hop_neighbors if n not in one_two_hop_neighbors]
                    three_hop_samples = sorted(three_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_size]
                    samples = samples + three_hop_samples
                samples = samples[:samples_pool_size]
            assert isinstance(samples, list), f'node:{node_index}, one_hop:{one_neighbors}'
            neighbor_pagerank_scores = [pg_scores[sample] * 1.2 if sample in one_neighbors else pg_scores[sample]for sample in samples]
            sampling_probabilities = softmax(neighbor_pagerank_scores)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6)
            k = len(samples)
            if k < samples_pool_size:
                samples += [-1] * (samples_pool_size - k)
                sampling_probabilities += [0.0] * (samples_pool_size - k)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6)
            assert len(samples) == len(sampling_probabilities) == samples_pool_size, f'{len(samples)}, {len(sampling_probabilities)}, {samples_pool_size}'
            tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))
    return tuples_list


def ablation_study_one_no_Degree(data_list, samples_pool_size = 15, positive_sizes= 6, teleportation_probability=0.15, max_iterations=80,):
    #### 这个版本中我们以samples_pool_size为最高的采样数 而不是每个点都要采样到这么多 ####
    tuples_list = []
    for dataset_index, data in enumerate(data_list):
        edge_index = data.edge_index
        num_nodes = max(edge_index.max() + 1, len(data.y))
        adjacency_matrix = csr_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        pg_scores = page_rank(adjacency_matrix, teleportation_probability, max_iterations)
        adjacency_list = build_adjacency_list(edge_index)
        degrees = compute_degrees(adjacency_list)
        avg_degree = sum(degrees) / len(degrees)

        for node_index in range(num_nodes):
            one_neighbors = get_neighbors(node_index, adjacency_list, {node_index})
            two_hop_neighbors = []
            three_hop_neighbors = []
            degree = len(one_neighbors)

            if node_index not in adjacency_list:  # Skip isolated nodes
                continue

            if degree >= samples_pool_size: 
                one_neighbors = sorted(one_neighbors, key=lambda x: pg_scores[x], reverse=True)
                samples = one_neighbors[:samples_pool_size]
            
            elif degree < samples_pool_size:
                one_two_hop_neighbors = collect_neighbors(node_index, adjacency_list, 2)
                left_size = positive_sizes - degree
                two_hop_neighbors = [n for n in one_two_hop_neighbors if n not in one_neighbors]
                two_hop_neighbors = sorted(two_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)
                if len(two_hop_neighbors) >= left_size:
                    two_hop_samples = two_hop_neighbors[:left_size]
                    samples = one_neighbors + two_hop_samples
                else:
                    samples = one_neighbors + two_hop_neighbors
                    left_size = positive_sizes - len(samples)
                    one_two_three_hop_neighbors = collect_neighbors(node_index, adjacency_list, 3)
                    three_hop_neighbors = [n for n in one_two_three_hop_neighbors if n not in one_two_hop_neighbors]
                    three_hop_samples = sorted(three_hop_neighbors, key=lambda x: pg_scores[x], reverse=True)[:left_size]
                    samples = samples + three_hop_samples
                samples = samples[:samples_pool_size]
            assert isinstance(samples, list), f'node:{node_index}, one_hop:{one_neighbors}'
            neighbor_pagerank_scores = [pg_scores[sample] * 1.2 if sample in one_neighbors else pg_scores[sample]for sample in samples]
            sampling_probabilities = softmax(neighbor_pagerank_scores)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6)
            k = len(samples)
            if k < samples_pool_size:
                samples += [-1] * (samples_pool_size - k)
                sampling_probabilities += [0.0] * (samples_pool_size - k)
            assert np.isclose(np.sum(sampling_probabilities), 1, atol=1e-6)
            assert len(samples) == len(sampling_probabilities) == samples_pool_size, f'{len(samples)}, {len(sampling_probabilities)}, {samples_pool_size}'
            tuples_list.append((dataset_index, node_index, samples,sampling_probabilities))
    return tuples_list
