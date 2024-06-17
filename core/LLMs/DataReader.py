import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class Dataset_CL(torch.utils.data.Dataset):
    def __init__(self, encodings, neighbors):
        self.encodings = encodings
        self.neighbors = neighbors  # Add this line

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        item['neighbors'] = self.neighbors[idx]  # Return neighbors for the node
        #print(item)  # Debugging: Print the item to check its contents

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
class MixDataset(torch.utils.data.Dataset):
    def __init__(self,encodings, dataset_index, node_index, neighbors,sample_weights):
        self.encodings = encodings
        self.dataset_index = dataset_index 
        self.node_index = node_index
        self.neighbors = neighbors  
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        item['dataset_index'] = self.dataset_index[idx] #a int
        item['node_index'] = self.node_index[idx]
        item['neighbors'] = self.neighbors[idx] # a list [2039,2103]
        item['sample_weights'] = self.sample_weights[idx] # a list [2039,2103]   
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
