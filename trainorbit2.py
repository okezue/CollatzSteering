import torch,torch.nn as nn,os,json,time,random,argparse
from torch.utils.data import IterableDataset,Dataset,DataLoader
from model import Enc,Dec
from probe import train_probe
from orbit import random_perm,orbit,cycle_len
def compose(s,t):
    return[s[t[i]]for i in range(len(s))]
def enc_pair(s,t,x):
    return[11]+list(s)+[10]+list(t)+[10]+[x]+[12]
SEP,BOS,EOS,PAD=10,11,12,13
VS=14;MSL=24
class O2Stream(IterableDataset):
    def __init__(self,n=8,seed=42):
        self.n=n;self.sd=seed
    def __iter__(self):
        wi=torch.utils.data.get_worker_info()
        s=self.sd+(wi.id*7919 if wi else 0)
        rng=random.Random(s)
        while True:
            sg=random_perm(self.n,rng);tg=random_perm(self.n,rng)
            c=compose(sg,tg);x=rng.randrange(self.n)
            orb=orbit(c,x);cl=cycle_len(c,x)
            src=enc_pair(sg,tg,x);tgt=[BOS]+list(orb)+[EOS]
            yield{'s':src,'t':tgt,'cl':cl,'n1':sg,'n2':tg,'x':x}
class O2Fixed(Dataset):
    def __init__(self,sz,n=8,seed=99):
        self.data=[];rng=random.Random(seed)
        for _ in range(sz):
            sg=random_perm(n,rng);tg=random_perm(n,rng)
            c=compose(sg,tg);x=rng.randrange(n)
            orb=orbit(c,x);cl=cycle_len(c,x)
            self.data.append((enc_pair(sg,tg,x),[BOS]+list(orb)+[EOS],cl))
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        s,t,cl=self.data[i]
        return{'s':s,'t':t,'cl':cl}
def collate2(batch):
    ss=[x['s']for x in batch];ts=[x['t']for x in batch]
    sl=max(len(s)for s in ss);tl=max(len(t)for t in ts)
    sl=min(sl,MSL);tl=min(tl,MSL)
    sp=torch.full((len(batch),sl),PAD,dtype=torch.long)
    tp=torch.full((len(batch),tl),PAD,dtype=torch.long)
    for i,(s,t)in enumerate(zip(ss,ts)):
        s=s[:sl];t=t[:tl]
        sp[i,:len(s)]=torch.tensor(s);tp[i,:len(t)]=torch.tensor(t)
    return{'s':sp,'t':tp,'sm':sp!=PAD,'tm':tp!=PAD,
           'cl':torch.tensor([x['cl']for x in batch])}
class O2Model(nn.Module):
    def __init__(self,d=512,nh=8,nel=4,ndl=1,ff=2048,do=0.1):
        super().__init__()
        self.enc=Enc(VS,d,nh,nel,ff,do,MSL)
        self.dec=Dec(VS,d,nh,ndl,ff,do,MSL)
        self.vs=VS
    def forward(self,src,tgt,sm=None,tm=None,rh=False):
        r=self.enc(src,sm,rh=rh)
        if rh:mem,hs=r
        else:mem=r;hs=None
        out=self.dec(tgt,mem,sm,tm)
        return(out,hs)if rh else out
    def generate(self,src,sm=None,ml=MSL):
        mem=self.enc(src,sm)
        b=src.size(0);dev=src.device
        ys=torch.full((b,1),BOS,dtype=torch.long,device=dev)
        for _ in range(ml-1):
            out=self.dec(ys,mem,sm)
            nx=out[:,-1,:].argmax(-1,keepdim=True)
            ys=torch.cat([ys,nx],1)
            if(nx==EOS).all():break
        return ys
    def enc_acts(self,src,sm=None):
        _,hs=self.enc(src,sm,rh=True)
        return hs
def eval_acc(m,dl,dev):
    m.eval();cor=0;tot=0
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(dev);sm=bat['sm'].to(dev)
            pred=m.generate(s,sm,ml=MSL)
            t=bat['t']
            for i in range(s.size(0)):
                p=pred[i].tolist();tv=t[i].tolist()
                if BOS in p:p=p[p.index(BOS)+1:]
                if EOS in p:p=p[:p.index(EOS)]
                p=[x for x in p if 0<=x<8]
                if BOS in tv:tv=tv[tv.index(BOS)+1:]
                if EOS in tv:tv=tv[:tv.index(EOS)]
                tv=[x for x in tv if 0<=x<8]
                if p==tv:cor+=1
                tot+=1
    return cor/tot if tot>0 else 0
def collect_acts(m,dl,dev,nel=4):
    m.eval()
    acts={l:[]for l in range(nel+1)};cls=[]
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(dev);sm=bat['sm'].to(dev)
            hs=m.enc_acts(s,sm)
            mk=sm.unsqueeze(-1).float();dn=mk.sum(1).clamp(min=1)
            for l in range(nel+1):
                mp=((hs[l]*mk).sum(1)/dn).cpu()
                acts[l].append(mp)
            cls.append(bat['cl'])
    for l in acts:acts[l]=torch.cat(acts[l])
    return acts,torch.cat(cls)
def run_probes(m,dl,dev,d=512,nel=4,kmax=9,ep=50):
    acts,cls=collect_acts(m,dl,dev,nel)
    results=[]
    for l in range(nel+1):
        cl_lin=train_probe(acts[l],cls,kmax,d,ep=ep,dev=dev)
        cl_mlp=train_probe(acts[l],cls,kmax,d,ep=ep,dev=dev,mlp=True)
        results.append({'layer':l,'cl_lin':cl_lin,'cl_mlp':cl_mlp})
    return results
def train(args):
    dev=args.dev;od=args.out
    os.makedirs(od,exist_ok=True)
    m=O2Model().to(dev)
    print(f"compose-orbit params={sum(p.numel()for p in m.parameters()):,}")
    opt=torch.optim.Adam(m.parameters(),lr=3e-5)
    ds=O2Stream(n=8,seed=42)
    dl=DataLoader(ds,batch_size=256,collate_fn=collate2,num_workers=4,
                  pin_memory=(dev!='cpu'))
    tds=O2Fixed(50000,n=8,seed=99)
    tdl=DataLoader(tds,batch_size=256,collate_fn=collate2,pin_memory=(dev!='cpu'))
    ce=nn.CrossEntropyLoss(ignore_index=PAD)
    spe=300000//256;log=[];best=-1
    it=iter(dl);step=0;t0=time.time()
    for ep in range(1,args.ep+1):
        m.train();el=0;ec=0
        for _ in range(spe):
            bat=next(it)
            s=bat['s'].to(dev);t=bat['t'].to(dev)
            sm=bat['sm'].to(dev);tm=bat['tm'].to(dev)
            out=m(s,t[:,:-1],sm,tm[:,:-1])
            loss=ce(out.reshape(-1,VS),t[:,1:].reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
            el+=loss.item();ec+=1;step+=1
        al=el/ec;acc=eval_acc(m,tdl,dev)
        dt=time.time()-t0
        log.append({'ep':ep,'loss':al,'acc':acc,'step':step,'time':dt})
        print(f"ep={ep:4d} loss={al:.4f} acc={acc:.4f} t={dt:.0f}s")
        if acc>best:
            best=acc
            torch.save({'model':m.state_dict(),'ep':ep,'acc':acc},f"{od}/best.pt")
        if ep%10==0:
            torch.save({'model':m.state_dict(),'opt':opt.state_dict(),
                        'ep':ep,'acc':acc},f"{od}/ck_{ep:04d}.pt")
        with open(f"{od}/log.json",'w')as f:json.dump(log,f)
    print("\nrunning probes on all checkpoints...")
    import glob
    pds=O2Fixed(30000,n=8,seed=77)
    pdl=DataLoader(pds,batch_size=256,collate_fn=collate2)
    all_probes=[]
    for cp in sorted(glob.glob(f"{od}/ck_*.pt")):
        sd=torch.load(cp,map_location=dev,weights_only=False)
        m.load_state_dict(sd['model'])
        pr=run_probes(m,pdl,dev)
        for r in pr:r['ep']=sd['ep']
        all_probes.extend(pr)
        l2=[r for r in pr if r['layer']==2][0]
        print(f"ep={sd['ep']:4d} L2 cl_lin={l2['cl_lin']:.4f} cl_mlp={l2['cl_mlp']:.4f}")
    with open(f"{od}/probe_results.json",'w')as f:json.dump(all_probes,f)
    print("done.")
def test():
    s1=[1,0,3,2,5,4,7,6];s2=[2,3,0,1,6,7,4,5]
    c=compose(s1,s2)
    assert c==[3,2,1,0,7,6,5,4]
    assert orbit(c,0)==[0,3]
    assert cycle_len(c,0)==2
    assert orbit(c,1)==[1,2]
    assert orbit(c,4)==[4,7]
    print("compose math: PASS")
    m=O2Model(d=64,nh=4,nel=2,ndl=1,ff=128)
    s=torch.randint(0,14,(4,20));sm=torch.ones(4,20,dtype=torch.bool)
    out=m(s,s[:,:5],sm,sm[:,:5])
    assert out.shape==(4,5,14)
    print("model: PASS")
    ds=O2Fixed(100,n=8,seed=42)
    dl=DataLoader(ds,batch_size=16,collate_fn=collate2)
    bat=next(iter(dl))
    assert bat['s'].shape[0]==16
    print("data: PASS")
    print("running 2-epoch smoke test...")
    od2='/tmp/orbit2_test'
    os.makedirs(od2,exist_ok=True)
    m2=O2Model(d=64,nh=4,nel=2,ndl=1,ff=128).to('cpu')
    opt=torch.optim.Adam(m2.parameters(),lr=3e-5)
    ce=nn.CrossEntropyLoss(ignore_index=PAD)
    ds2=O2Stream(n=8,seed=42)
    dl2=DataLoader(ds2,batch_size=16,collate_fn=collate2,num_workers=0)
    it=iter(dl2)
    for ep in range(1,3):
        m2.train();el=0
        for _ in range(4):
            bat=next(it)
            out=m2(bat['s'],bat['t'][:,:-1],bat['sm'],bat['tm'][:,:-1])
            loss=ce(out.reshape(-1,VS),bat['t'][:,1:].reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
            el+=loss.item()
        print(f"  smoke ep={ep} loss={el/4:.4f}")
    print("smoke test: PASS")
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='outputorbit2')
    ap.add_argument('--ep',type=int,default=300)
    ap.add_argument('--test',action='store_true')
    a=ap.parse_args()
    if a.test:test()
    else:train(a)
