import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, w2i, max_x_len=100, max_y_len=100, isTrain=False):
        self.data = data
        self.w2i = w2i
        self.max_x_len = max_x_len
        self.max_y_len = max_y_len
        self.isTrain = isTrain
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.isTrain:
            summary, text = self.data[idx]
        else:
            text = self.data[idx]

        # Trim to max_len (account for extra eos and sos)
        text = [self.w2i["<sos>"]] + text[:self.max_x_len - 2] + [self.w2i["<eos>"]]
        if self.isTrain:
            summary = [self.w2i["<sos>"]] + summary[:self.max_y_len - 2] + [self.w2i["<eos>"]]
        
        if self.isTrain:
            return torch.Tensor(text), torch.Tensor(summary)
        else:
            return torch.Tensor(text)

def index2Sent(prediction, i2w):
    return " ".join([i2w[tkn] for tkn in prediction])