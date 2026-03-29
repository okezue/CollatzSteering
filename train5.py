import torch,torch.nn as nn,os,json,time,random,argparse,glob
from torch.utils.data import IterableDataset,Dataset,DataLoader
from functools import partial
from dataclasses import dataclass
from collatz5 import kappa5,kv5,kpv5,check_odd
from data import enc,dec,BOS,EOS,PAD,VSZ,collate as _collate_orig
from model import CTF
from probe import collect_acts as _ca_orig,train_probe
@dataclass
class C5:
    d:int=512;nh:int=8;nel:int=4;ndl:int=1;ff:int=2048;do:float=0.1
    msl:int=16;bs:int=256;lr:float=3e-5;ep:int=300;epe:int=300000
    tst:int=50000;ckf:int=10;nw:int=4;seed:int=42;nmax:int=10**12
    plr:float=1e-3;pep:int=50;pts:int=50000;pvs:int=10000;kmax:int=20
    dev:str="cuda";out:str="output5";base:int=32
class S5(IterableDataset):
    def __init__(self,base,msl,nmax,seed=42):
        self.b=base;self.ml=msl;self.nm=nmax;self.sd=seed
    def __iter__(self):
        wi=torch.utils.data.get_worker_info()
        s=self.sd+(wi.id*7919 if wi else 0)
        rng=random.Random(s)
        while True:
            n=rng.randrange(1,self.nm,2)
            kn=kappa5(n)
            sr=[BOS(self.b)]+enc(n,self.b,self.ml-2)+[EOS(self.b)]
            tg=[BOS(self.b)]+enc(kn,self.b,self.ml-2)+[EOS(self.b)]
            yield{'s':sr,'t':tg,'n':n,'kn':kn,'k':kv5(n),'kp':kpv5(n)}
class F5(Dataset):
    def __init__(self,sz,base,msl,nmax,seed=99):
        self.b=base;self.ml=msl;self.data=[]
        rng=random.Random(seed)
        for _ in range(sz):
            n=rng.randrange(1,nmax,2)
            kn=kappa5(n);self.data.append((n,kn,kv5(n),kpv5(n)))
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        n,kn,k,kp=self.data[i]
        sr=[BOS(self.b)]+enc(n,self.b,self.ml-2)+[EOS(self.b)]
        tg=[BOS(self.b)]+enc(kn,self.b,self.ml-2)+[EOS(self.b)]
        return{'s':sr,'t':tg,'n':n,'kn':kn,'k':k,'kp':kp}
def collate5(batch,base,msl):
    return _collate_orig(batch,base,msl)
def decode_pred(pred,base,i):
    p=pred[i].tolist()
    bi=BOS(base)
    if bi in p:p=p[p.index(bi)+1:]
    ei=EOS(base)
    if ei in p:p=p[:p.index(ei)]
    p=[x for x in p if 0<=x<base]
    return dec(p,base)if p else -1
def eval5(m,dl,base,cfg):
    m.eval();cor=0;tot=0
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=decode_pred(pred,base,i)
                if pn==bat['kn'][i]:cor+=1
                tot+=1
    return cor/tot if tot>0 else 0
def collect_acts5(m,dl,cfg):
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
def train5(cfg):
    base=cfg.base;od=cfg.out
    os.makedirs(od,exist_ok=True)
    m=CTF(cfg,base).to(cfg.dev)
    np_=sum(p.numel() for p in m.parameters())
    print(f"5x+1 base={base} params={np_:,}")
    opt=torch.optim.Adam(m.parameters(),lr=cfg.lr)
    cf=partial(collate5,base=base,msl=cfg.msl)
    ds=S5(base,cfg.msl,cfg.nmax,cfg.seed)
    dl=DataLoader(ds,batch_size=cfg.bs,collate_fn=cf,num_workers=cfg.nw,
                  pin_memory=(cfg.dev!='cpu'))
    tds=F5(cfg.tst,base,cfg.msl,cfg.nmax,seed=99)
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
        acc=eval5(m,tdl,base,cfg)
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
def run_probes5(cfg):
    od=cfg.out;base=cfg.base
    ckpts=sorted(glob.glob(f"{od}/ck_*.pt"))
    if not ckpts:print("no checkpoints");return
    cf=partial(collate5,base=base,msl=cfg.msl)
    pds=F5(cfg.pts+cfg.pvs,base,cfg.msl,cfg.nmax,seed=77)
    pdl=DataLoader(pds,batch_size=cfg.bs,collate_fn=cf)
    results=[]
    for cp in ckpts:
        m=CTF(cfg,base).to(cfg.dev)
        sd=torch.load(cp,map_location=cfg.dev,weights_only=False)
        m.load_state_dict(sd['model'])
        ep=sd['ep']
        acts,ks,kps=collect_acts5(m,pdl,cfg)
        for l in range(cfg.nel+1):
            ka_lin=train_probe(acts[l],ks,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev)
            kpa_lin=train_probe(acts[l],kps,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev)
            ka_mlp=train_probe(acts[l],ks,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev,mlp=True)
            kpa_mlp=train_probe(acts[l],kps,cfg.kmax,cfg.d,cfg.plr,cfg.pep,cfg.dev,mlp=True)
            r={'ep':ep,'layer':l,
               'k5_lin':ka_lin,'kp5_lin':kpa_lin,
               'k5_mlp':ka_mlp,'kp5_mlp':kpa_mlp}
            results.append(r)
            print(f"ep={ep} L={l} k5_lin={ka_lin:.4f} kp5_lin={kpa_lin:.4f} "
                  f"k5_mlp={ka_mlp:.4f} kp5_mlp={kpa_mlp:.4f}")
        del m;torch.cuda.empty_cache()if cfg.dev=='cuda' else None
    with open(f"{od}/probe_results.json",'w')as f:json.dump(results,f)
    return results
def test_5x1():
    check_odd(2000)
    assert kappa5(1)%2==1
    assert kappa5(5)%2==1
    assert kappa5(7)%2==1
    for n in range(1,500,2):
        kn=kappa5(n)
        assert kn%2==1,f"kappa5({n})={kn} even"
        a=kv5(n);b=kpv5(n)
        assert a>=1;assert b>=1
    print("test_5x1 math: PASS")
    od='/tmp/test5x1'
    cfg=C5(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,
           bs=16,ep=2,epe=64,tst=32,ckf=1,nw=0,
           nmax=10**6,dev='cpu',out=od,kmax=8,
           pts=50,pvs=10,pep=3)
    m=train5(cfg)
    assert os.path.exists(f"{od}/log.json")
    assert os.path.exists(f"{od}/ck_0001.pt")
    assert os.path.exists(f"{od}/best.pt")
    print("test_5x1 train: PASS")
    cfg2=C5(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,
            bs=16,nw=0,nmax=10**6,dev='cpu',out=od,
            kmax=8,pts=50,pvs=10,pep=3)
    r=run_probes5(cfg2)
    assert len(r)>0
    assert os.path.exists(f"{od}/probe_results.json")
    print("test_5x1 probes: PASS")
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='output5')
    ap.add_argument('--ep',type=int,default=None)
    ap.add_argument('--bs',type=int,default=None)
    ap.add_argument('--nw',type=int,default=None)
    ap.add_argument('--test',action='store_true')
    a=ap.parse_args()
    if a.test:
        test_5x1()
    else:
        cfg=C5(dev=a.dev,out=a.out)
        if a.ep:cfg.ep=a.ep
        if a.bs:cfg.bs=a.bs
        if a.nw is not None:cfg.nw=a.nw
        m=train5(cfg)
        print("running probes...")
        run_probes5(cfg)
