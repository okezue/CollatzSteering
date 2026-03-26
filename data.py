import torch
from torch.utils.data import IterableDataset,Dataset
import random
from collatz import kappa,kv,kpv
def enc(n,b,ml):
    if n==0:return[0]
    ds=[]
    while n>0:ds.append(n%b);n//=b
    ds.reverse()
    return ds[:ml]
def dec(ds,b):
    r=0
    for d in ds:r=r*b+d
    return r
def BOS(b):return b
def EOS(b):return b+1
def PAD(b):return b+2
def VSZ(b):return b+3
class CStream(IterableDataset):
    def __init__(self,base,msl,nmax,seed=42):
        self.b=base;self.ml=msl;self.nm=nmax;self.sd=seed
    def __iter__(self):
        wi=torch.utils.data.get_worker_info()
        s=self.sd+(wi.id*7919 if wi else 0)
        rng=random.Random(s)
        while True:
            n=rng.randrange(1,self.nm,2)
            kn=kappa(n)
            sr=[BOS(self.b)]+enc(n,self.b,self.ml-2)+[EOS(self.b)]
            tg=[BOS(self.b)]+enc(kn,self.b,self.ml-2)+[EOS(self.b)]
            yield{'s':sr,'t':tg,'n':n,'kn':kn,'k':kv(n),'kp':kpv(n)}
class CFixed(Dataset):
    def __init__(self,sz,base,msl,nmax,seed=99):
        self.b=base;self.ml=msl;self.data=[]
        rng=random.Random(seed)
        for _ in range(sz):
            n=rng.randrange(1,nmax,2)
            kn=kappa(n);self.data.append((n,kn,kv(n),kpv(n)))
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        n,kn,k,kp=self.data[i]
        sr=[BOS(self.b)]+enc(n,self.b,self.ml-2)+[EOS(self.b)]
        tg=[BOS(self.b)]+enc(kn,self.b,self.ml-2)+[EOS(self.b)]
        return{'s':sr,'t':tg,'n':n,'kn':kn,'k':k,'kp':kp}
class CTargeted(Dataset):
    def __init__(self,base,msl,nmax,k_range,kp_range,per_group=500,seed=33):
        self.b=base;self.ml=msl;self.data=[]
        rng=random.Random(seed)
        for tk in k_range:
            for tkp in kp_range:
                found=0
                for _ in range(per_group*200):
                    if found>=per_group:break
                    n=rng.randrange(1,nmax,2)
                    if kv(n)==tk and kpv(n)==tkp:
                        kn=kappa(n);self.data.append((n,kn,tk,tkp));found+=1
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        n,kn,k,kp=self.data[i]
        sr=[BOS(self.b)]+enc(n,self.b,self.ml-2)+[EOS(self.b)]
        tg=[BOS(self.b)]+enc(kn,self.b,self.ml-2)+[EOS(self.b)]
        return{'s':sr,'t':tg,'n':n,'kn':kn,'k':k,'kp':kp}
def collate(batch,base,msl):
    pv=PAD(base)
    ss=[x['s'] for x in batch];ts=[x['t'] for x in batch]
    sl=min(max(len(s) for s in ss),msl)
    tl=min(max(len(t) for t in ts),msl)
    sp=torch.full((len(batch),sl),pv,dtype=torch.long)
    tp=torch.full((len(batch),tl),pv,dtype=torch.long)
    for i,(s,t) in enumerate(zip(ss,ts)):
        s=s[:sl];t=t[:tl]
        sp[i,:len(s)]=torch.tensor(s)
        tp[i,:len(t)]=torch.tensor(t)
    return{'s':sp,'t':tp,'sm':sp!=pv,'tm':tp!=pv,
           'k':torch.tensor([x['k'] for x in batch]),
           'kp':torch.tensor([x['kp'] for x in batch]),
           'n':[x['n'] for x in batch],'kn':[x['kn'] for x in batch]}
