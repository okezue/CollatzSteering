import torch,torch.nn as nn,os,json,time
from torch.utils.data import DataLoader
from functools import partial
from config import C
from data import CStream,CFixed,collate,PAD,VSZ
from model import CTF
from evaluate import eval_acc
def train(cfg,base):
    od=f"{cfg.out}/b{base}"
    os.makedirs(od,exist_ok=True)
    m=CTF(cfg,base).to(cfg.dev)
    np_=sum(p.numel() for p in m.parameters())
    print(f"base={base} params={np_:,}")
    opt=torch.optim.Adam(m.parameters(),lr=cfg.lr)
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CStream(base,cfg.msl,cfg.nmax,cfg.seed)
    dl=DataLoader(ds,batch_size=cfg.bs,collate_fn=cf,num_workers=cfg.nw,
                  pin_memory=(cfg.dev!='cpu'))
    tds=CFixed(cfg.tst,base,cfg.msl,cfg.nmax,seed=99)
    tdl=DataLoader(tds,batch_size=cfg.bs,collate_fn=cf,
                   pin_memory=(cfg.dev!='cpu'))
    ce=nn.CrossEntropyLoss(ignore_index=PAD(base))
    spe=cfg.epe//cfg.bs
    log=[];best_acc=-1
    it=iter(dl);step=0;t0=time.time()
    for ep in range(1,cfg.ep+1):
        m.train();el=0;ec=0
        for _ in range(spe):
            bat=next(it)
            s=bat['s'].to(cfg.dev);t=bat['t'].to(cfg.dev)
            sm=bat['sm'].to(cfg.dev);tm=bat['tm'].to(cfg.dev)
            out=m(s,t[:,:-1],sm,tm[:,:-1])
            loss=ce(out.reshape(-1,m.vs),t[:,1:].reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
            el+=loss.item();ec+=1;step+=1
        al=el/ec
        acc=eval_acc(m,tdl,base,cfg)
        dt=time.time()-t0
        log.append({'ep':ep,'loss':al,'acc':acc,'step':step,'time':dt})
        print(f"ep={ep:4d} loss={al:.4f} acc={acc:.4f} t={dt:.0f}s")
        if acc>best_acc:
            best_acc=acc
            torch.save({'model':m.state_dict(),'ep':ep,'acc':acc},
                       f"{od}/best.pt")
        if ep%cfg.ckf==0:
            torch.save({'model':m.state_dict(),'opt':opt.state_dict(),
                        'ep':ep,'acc':acc,'cfg':vars(cfg)},
                       f"{od}/ck_{ep:04d}.pt")
        with open(f"{od}/log.json",'w')as f:json.dump(log,f)
    return m
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=int,default=32)
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='output')
    ap.add_argument('--ep',type=int,default=None)
    ap.add_argument('--bs',type=int,default=None)
    ap.add_argument('--nw',type=int,default=None)
    a=ap.parse_args()
    cfg=C(dev=a.dev,out=a.out)
    if a.ep:cfg.ep=a.ep
    if a.bs:cfg.bs=a.bs
    if a.nw is not None:cfg.nw=a.nw
    train(cfg,a.base)
