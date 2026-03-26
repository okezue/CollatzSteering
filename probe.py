import torch,torch.nn as nn,json,os
from torch.utils.data import DataLoader
from functools import partial
from config import C
from data import CFixed,collate
def collect_acts(m,dl,cfg):
    m.eval()
    acts={l:[]for l in range(cfg.nel+1)}
    ks=[];kps=[]
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            hs=m.enc_acts(s,sm)
            mk=sm.unsqueeze(-1).float()
            denom=mk.sum(1).clamp(min=1)
            for l in range(cfg.nel+1):
                mp=((hs[l]*mk).sum(1)/denom).cpu()
                acts[l].append(mp)
            ks.append(bat['k']);kps.append(bat['kp'])
    for l in acts:acts[l]=torch.cat(acts[l])
    return acts,torch.cat(ks),torch.cat(kps)
class LP(nn.Module):
    def __init__(self,d,nc):
        super().__init__()
        self.fc=nn.Linear(d,nc)
    def forward(self,x):return self.fc(x)
class MLP(nn.Module):
    def __init__(self,d,nc,hd=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,hd),nn.ReLU(),nn.Linear(hd,nc))
    def forward(self,x):return self.net(x)
def train_probe(acts,labels,nc,d,lr=1e-3,ep=50,dev='cuda',mlp=False):
    n=len(acts);tr=int(0.8*n)
    idx=torch.randperm(n)
    xa=acts[idx[:tr]].to(dev);ya=labels[idx[:tr]].clamp(0,nc-1).long().to(dev)
    xv=acts[idx[tr:]].to(dev);yv=labels[idx[tr:]].clamp(0,nc-1).long().to(dev)
    p=(MLP(d,nc)if mlp else LP(d,nc)).to(dev)
    opt=torch.optim.Adam(p.parameters(),lr=lr)
    ce=nn.CrossEntropyLoss()
    best=0
    for e in range(ep):
        p.train()
        for i in range(0,len(xa),512):
            xb=xa[i:i+512];yb=ya[i:i+512]
            loss=ce(p(xb),yb)
            opt.zero_grad();loss.backward();opt.step()
        p.eval()
        with torch.no_grad():
            acc=(p(xv).argmax(1)==yv).float().mean().item()
        if acc>best:best=acc
    return best
def probe_residual_bits(acts,labels_k,labels_kp,kmax,d,dev='cuda'):
    from collatz import v2
    results={}
    for tgt_name,labels in[('k',labels_k),('kp',labels_kp)]:
        for l in acts:
            a=train_probe(acts[l],labels,kmax,d,dev=dev)
            results[f"{tgt_name}_L{l}"]=a
    return results
def run_probes(cfg,base,ckpts):
    from model import CTF
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CFixed(cfg.pts+cfg.pvs,base,cfg.msl,cfg.nmax,seed=77)
    dl=DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
    results=[]
    for cp in ckpts:
        m=CTF(cfg,base).to(cfg.dev)
        sd=torch.load(cp,map_location=cfg.dev,weights_only=False)
        m.load_state_dict(sd['model'])
        ep=sd['ep']
        acts,ks,kps=collect_acts(m,dl,cfg)
        for l in range(cfg.nel+1):
            ka_lin=train_probe(acts[l],ks,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev)
            kpa_lin=train_probe(acts[l],kps,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev)
            ka_mlp=train_probe(acts[l],ks,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev,mlp=True)
            kpa_mlp=train_probe(acts[l],kps,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev,mlp=True)
            r={'ep':ep,'layer':l,
               'k_lin':ka_lin,'kp_lin':kpa_lin,
               'k_mlp':ka_mlp,'kp_mlp':kpa_mlp}
            results.append(r)
            print(f"ep={ep} L={l} k_lin={ka_lin:.4f} kp_lin={kpa_lin:.4f} "
                  f"k_mlp={ka_mlp:.4f} kp_mlp={kpa_mlp:.4f}")
        del m;torch.cuda.empty_cache()if cfg.dev=='cuda' else None
    return results
if __name__=='__main__':
    import argparse,glob
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=int,default=32)
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='output')
    ap.add_argument('--ckpt',type=str,default=None)
    a=ap.parse_args()
    cfg=C(dev=a.dev,out=a.out)
    if a.ckpt:ckpts=[a.ckpt]
    else:ckpts=sorted(glob.glob(f"{cfg.out}/b{a.base}/ck_*.pt"))
    r=run_probes(cfg,a.base,ckpts)
    od=f"{cfg.out}/b{a.base}"
    os.makedirs(od,exist_ok=True)
    with open(f"{od}/probe_results.json",'w')as f:json.dump(r,f)
