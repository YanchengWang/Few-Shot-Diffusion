import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ml_collections

data_config_default = ml_collections.ConfigDict()
data_config_default.data_dir = '/data-drive/backup/changyu/expe/gge/graphvae_gat_cora_freeze_enc_feat_map_lr2.4/cora_latents_2708_gvae_50000_64_encode.npy'
#data_config_default.data_labels_dir = '/home/local/ASUAD/changyu2/few-shot-generate-graph-embedding/gcl_embeddings/cora_64d/all_gt_labels.npy'
data_config_default.data_labels_dir = '/home/local/ASUAD/changyu2/few-shot-generate-graph-embedding/gcl_embeddings/cora_64d/all_gt_labels.npy'
data_config_default.train_batchsize = 128
data_config_default.numworkers = 8
data_config_default.norm = 0



class Graph_fs_Dataset(Dataset):
    def __init__(self, emb_dir, label_dir, norm, classes=None, sample_size=5):
        self.sample_size = sample_size
        self.norm = norm
        self.emb = np.load(emb_dir)        
        self.max = self.emb.max()
        self.min = self.emb.min()
        if self.norm:
            self.emb = self.normalize(self.emb)
        self.labels = np.load(label_dir)

        #select data within the classes
        if classes is not None:
            in_classes = np.isin(self.labels, classes)
            self.emb = self.emb[in_classes]
            self.labels = self.labels[in_classes]

        #self.emb = torch.from_numpy(self.emb).float()
        ##self.labels = torch.from_numpy(self.labels)
        #B, D = self.emb.shape
        #self.emb = self.emb.unsqueeze(1)
        #self.labels = self.labels.squeeze(1)

        #transform the embs to a dict. key is the label, value is the emb
        self.embedding_dict = self.trans_to_dict()
        self.unique_labels = list(self.embedding_dict.keys())
        
        
    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        #first sample a label from the unique labels
        label = np.random.choice(self.unique_labels)
        
        #sample sample_size+1 nodes with the label 
        emb = self.embedding_dict[label]
        idx = np.random.choice(emb.shape[0], self.sample_size+1, replace=False)   #If the labl has less than sample_size+1 nodes, then sample with replacement
        emb = emb[idx]
        return emb, label

    def normalize(self, data):
        return (data-self.min)/(self.max - self.min)
    
    def transback(self, data):
        return data*(self.max - self.min) + self.min

    def trans_to_dict(self):
        embedding_dict = {}
        
        for i in range(len(self.emb)):
            if self.labels[i].item() in embedding_dict.keys():
                embedding_dict[self.labels[i].item()].append(self.emb[i])
            else:
                embedding_dict[self.labels[i].item()] = [self.emb[i]]

        #stack the embeddings with the same label with numpy
        for key in embedding_dict.keys():
            embedding_dict[key] = np.stack(embedding_dict[key], axis=0)

        return embedding_dict


def load_data(params):
    data_config = data_config_default
    data_config.train_batchsize = params.batchsize
    data_config.data_dir = params.datadir
    data_config.data_labels_dir = params.labeldir
    data_config.norm = params.norm

    train_dataset = Graph_fs_Dataset(emb_dir=data_config.data_dir, label_dir=data_config.data_labels_dir, norm = data_config.norm, sample_size=params.sample_size)
    trainloader = DataLoader(
                    train_dataset,
                    batch_size = data_config.train_batchsize,
                    num_workers = data_config.numworkers,
                )

    return trainloader, train_dataset


class Decode_Dataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


if __name__ == '__main__':
    ds = Graph_fs_Dataset(emb_dir=data_config_default.data_dir, label_dir=data_config_default.data_labels_dir, norm=1)
    print(ds[0])
    print(ds[1])
    print(ds[2])
    print(ds[3])
    print(ds[4])
