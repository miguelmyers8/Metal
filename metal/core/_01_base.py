import numpy as np
from .samplers import Sampler, BatchSampler

# gets index from dataset of data and labels
class Dataset():
    def __init__(self,x,y):self.x,self.y = x, y
    def __len__(self):return len(self.x)
    def __getitem__(self, i):return self.x[i], self.y[i]

def collate(b):
    xs,ys = zip(*b)
    return np.stack(xs),np.stack(ys)

# takes dataset and batch size and returns a generator of the dataset
class DataLoader():
    def __init__(self, dataset, sampler, batch_sampler=None,batch_size=None ,collate_fn=collate):
        self.dataset,self.sampler,self.batch_sampler,self.batch_size,self.collate_fn = dataset,sampler,batch_sampler,batch_size,collate_fn
        if batch_size is not None and batch_sampler is None: self.batch_sampler = BatchSampler(sampler,batch_size,False)

    def __len__(self):
        if self.batch_sampler:
            return len(self.batch_sampler)
        else:
            return len(self.sampler)

    def __iter__(self):
        if self.batch_sampler:
            for s in self.batch_sampler: yield self.collate_fn([self.dataset[i] for i in s])
        else:
            for s in self.sampler: yield self.collate_fn([self.dataset[i] for i in s])


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl,self.valid_dl,self.c = train_dl, valid_dl, c

    @property
    def train_dataset(self): return self.train_dl.dataset

    @property
    def valid_dataset(self): return self.valid_dl.dataset

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data= model, opt, loss_func, data
