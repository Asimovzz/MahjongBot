from bisect import bisect_right
from pathlib import Path
import json

import numpy as np
from torch.utils.data import Dataset

from .paths import DATA_DIR

FEATURE_NUM = 70

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin=0, end=1, augment=False, data_dir=None):
        data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        count_path = data_dir / "count.json"
        sample_dir = data_dir / ("augmented" if augment else "preprocessed")

        with count_path.open(encoding="utf-8") as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0: print('loading', i)
            
            if augment:
                path = sample_dir / f"{i + self.begin}_augmented_{FEATURE_NUM}.npz"
            else:
                path = sample_dir / f"{i + self.begin}.npz"
            d = np.load(path)
            for k in d:
                self.cache[k].append(d[k])
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], self.cache['act'][match_id][sample_id]
