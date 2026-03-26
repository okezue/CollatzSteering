import torch,torch.nn as nn,os,json
from collections import defaultdict
from functools import partial
from config import C
from data import CFixed,collate
class JumpReLU(nn.Module):
    def __init__(self,d,th=0.01,bw=0.001):
        super().__init__()
        self.th=nn.Parameter(torch.ones(d)*th)
        self.bw=bw
    def forward(self,x):
        return x*torch.sigmoid((x-self.th)/self.bw)
class CLT(nn.Module):
    def __init__(self,din,dout,df,nl):
        super().__init__()
        self.encs=nn.ModuleList([nn.Linear(din,df)for _ in range(nl)])
        self.act=JumpReLU(df)
        self.decs=nn.ModuleList([nn.Linear(df,dout)for _ in range(nl)])
        self.nl=nl;self.df=df
    def forward(self,xs):
        zs=[self.act(e(x))for e,x in zip(self.encs,xs)]
        z=sum(zs)
        outs=[d(z)for d in self.decs]
        return outs,z
    def get_features(self,xs):
        zs=[self.act(e(x))for e,x in zip(self.encs,xs)]
        return sum(zs)
def cache_acts(m,dl,cfg):
    m.eval()
    ins=defaultdict(list);outs=defaultdict(list)
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            hs=m.enc_acts(s,sm)
            mk=sm.unsqueeze(-1).float()
            denom=mk.sum(1).clamp(min=1)
            for l in range(len(hs)-1):
                ai=((hs[l]*mk).sum(1)/denom).cpu()
                ao=((hs[l+1]*mk).sum(1)/denom).cpu()
                ins[l].append(ai);outs[l].append(ao)
    for l in ins:ins[l]=torch.cat(ins[l]);outs[l]=torch.cat(outs[l])
    return ins,outs
def train_clt(cfg,base,ckpt):
    from model import CTF
    m=CTF(cfg,base).to(cfg.dev)
    sd=torch.load(ckpt,map_location=cfg.dev,weights_only=False)
    m.load_state_dict(sd['model'])
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CFixed(cfg.pts,base,cfg.msl,cfg.nmax,seed=66)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
    print("caching activations...")
    ins,outs=cache_acts(m,dl,cfg)
    nl=cfg.nel
    print(f"cached {len(ins[0])} samples across {nl} layers")
    clt=CLT(cfg.d,cfg.d,cfg.tcd,nl).to(cfg.dev)
    opt=torch.optim.Adam(clt.parameters(),lr=cfg.tclr)
    n=len(ins[0]);bs=min(256,n)
    for ep in range(cfg.tcep):
        idx=torch.randperm(n);el=0;ec=0
        for i in range(0,n,bs):
            bi=idx[i:i+bs]
            xs=[ins[l][bi].to(cfg.dev)for l in range(nl)]
            yt=[outs[l][bi].to(cfg.dev)for l in range(nl)]
            ps,z=clt(xs)
            rl=sum(nn.functional.mse_loss(p,y)for p,y in zip(ps,yt))/nl
            sl=cfg.tcl1*z.abs().mean()
            alive=(z>0).float().mean(0)
            dl_=cfg.tcl0*nn.functional.relu(0.05-alive).mean()
            loss=rl+sl+dl_
            opt.zero_grad();loss.backward();opt.step()
            el+=loss.item();ec+=1
        sp=(z>0).float().mean().item()
        af=(alive>0.01).float().mean().item()
        print(f"CLT ep={ep+1:3d} loss={el/ec:.6f} sparsity={sp:.4f} alive={af:.4f}")
    return clt
def analyze_features(clt,ins,outs,ks,kps,cfg):
    nl=cfg.nel;n=len(ins[0])
    xs=[ins[l].to(cfg.dev)for l in range(nl)]
    with torch.no_grad():
        z=clt.get_features(xs)
    z=z.cpu()
    k_corrs=[];kp_corrs=[]
    for f in range(z.shape[1]):
        fv=z[:,f]
        if fv.std()<1e-8:continue
        kc=torch.corrcoef(torch.stack([fv,ks.float()]))[0,1].item()
        kpc=torch.corrcoef(torch.stack([fv,kps.float()]))[0,1].item()
        k_corrs.append((f,kc));kp_corrs.append((f,kpc))
    k_corrs.sort(key=lambda x:-abs(x[1]))
    kp_corrs.sort(key=lambda x:-abs(x[1]))
    return{'k_top':k_corrs[:20],'kp_top':kp_corrs[:20]}
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=int,default=32)
    ap.add_argument('--ckpt',type=str,required=True)
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='output')
    a=ap.parse_args()
    cfg=C(dev=a.dev,out=a.out)
    clt=train_clt(cfg,a.base,a.ckpt)
    od=f"{cfg.out}/b{a.base}"
    os.makedirs(od,exist_ok=True)
    torch.save(clt.state_dict(),f"{od}/clt.pt")
