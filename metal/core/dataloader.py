class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = len(ds),bs,shuffle

    def __iter__(self):
        self.idxs = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]


# gets index from dataset of data and labels
class Dataset():
    def __init__(self,x,y):
        self.x,self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

def collate(b):
    xs,ys = zip(*b)
    return np.stack(xs),np.stack(ys)

# takes dataset and batch size and returns a generator of the dataset
class DataLoader():
    def __init__(self, ds, sampler, collate_fn=collate):
        self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn

    def __iter__(self):
        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c

    @property
    def train_ds(self): return self.train_dl

    @property
    def valid_ds(self): return self.valid_dl
