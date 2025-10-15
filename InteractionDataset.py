import numpy as np
import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    def __init__(self, user_artists, num_items):
        self.user_ids = user_artists['userID'].values
        self.item_ids = user_artists['artistID'].values
        self.labels = torch.ones(len(user_artists))

        negatives = []
        for u in np.unique(self.user_ids):
            all_items = set(range(num_items))
            pos_items = set(user_artists[user_artists['userID'] == u]['artistID'])
            neg_candidates = list(all_items - pos_items)
            if len(neg_candidates) == 0:
                continue
            neg_sample = np.random.choice(neg_candidates, size=min(len(pos_items), len(neg_candidates)), replace=False)
            for ni in neg_sample:
                negatives.append((u, ni))

        neg_users = [u for u, _ in negatives]
        neg_items = [i for _, i in negatives]
        neg_labels = torch.zeros(len(negatives))

        self.user_ids = torch.tensor(np.concatenate([self.user_ids, neg_users]), dtype=torch.long)
        self.item_ids = torch.tensor(np.concatenate([self.item_ids, neg_items]), dtype=torch.long)
        self.labels = torch.cat([self.labels, neg_labels])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
